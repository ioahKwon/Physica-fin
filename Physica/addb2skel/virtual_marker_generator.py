"""
Virtual Marker Generator for SKEL body mesh.

Generates BSM (Biomechanical Skeleton Model) markers on SKEL skin_verts,
computes bone rigging via skin_weights, and saves markers in multiple formats.

Reuses:
- bsm_markers.yaml: 106 BSM markers → SMPL vertex indices (= SKEL skin_verts indices)
- mapping.py: compute_mapping() / apply_mapping() for triangle-frame offsets
- nimblephysics: saveTRC() for AddBiomechanics-compatible .trc files
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import yaml


class VirtualMarkerGenerator:
    """
    Generate BSM virtual markers from fitted SKEL body mesh.

    Pipeline:
    1. Extract marker positions from skin_verts using BSM vertex indices
    2. Compute bone rigging from skel.skin_weights
    3. Compute triangle-frame offsets for pose-invariant reconstruction
    4. Save in .pkl, .trc, .npy formats
    """

    def __init__(
        self,
        skel_interface,
        bsm_markers_path: str,
        device: torch.device,
    ):
        """
        Args:
            skel_interface: SKELInterface instance.
            bsm_markers_path: Path to bsm_markers.yaml.
            device: Torch device.
        """
        self.skel = skel_interface
        self.device = device

        # Load BSM marker definitions
        with open(bsm_markers_path) as f:
            self.bsm_markers: Dict[str, int] = yaml.safe_load(f)

        self.marker_names: List[str] = list(self.bsm_markers.keys())
        self.vertex_indices: List[int] = list(self.bsm_markers.values())
        self.num_markers: int = len(self.marker_names)

        # Will be populated after generation
        self.marker_trajectories: Optional[np.ndarray] = None  # [T, K, 3]
        self.bone_rigging: Optional[np.ndarray] = None  # [K] bone index per marker
        self.triangle_offsets: Optional[Dict] = None  # triangle-frame mapping

    def generate_markers(
        self,
        betas: torch.Tensor,
        poses: torch.Tensor,
        trans: torch.Tensor,
    ) -> np.ndarray:
        """
        Extract BSM marker trajectories from fitted SKEL skin_verts.

        Args:
            betas: Shape parameters [10] or [B, 10].
            poses: Pose parameters [T, 46].
            trans: Translations [T, 3].

        Returns:
            marker_trajectories: [T, K, 3] marker positions in meters.
        """
        T = poses.shape[0]
        vidx = torch.tensor(self.vertex_indices, dtype=torch.long, device=self.device)

        with torch.no_grad():
            if betas.dim() == 1:
                betas_exp = betas.unsqueeze(0).expand(T, -1)
            else:
                betas_exp = betas

            verts, _, _ = self.skel.forward(betas_exp, poses, trans)
            # Extract marker vertices: [T, K, 3]
            markers = verts[:, vidx, :].cpu().numpy()

        self.marker_trajectories = markers
        return markers

    def compute_bone_rigging(self) -> np.ndarray:
        """
        Compute bone rigging for each marker using SKEL skin_weights.

        Each marker vertex is assigned to its dominant bone (highest skin weight).

        Returns:
            bone_rigging: [K] array of bone indices (0-23).
        """
        skin_weights = self.skel.model.skin_weights  # [6890, 24] sparse or dense
        if hasattr(skin_weights, 'is_sparse') and skin_weights.is_sparse:
            skin_weights = skin_weights.to_dense()
        skin_weights = skin_weights.cpu().numpy()

        rigging = np.zeros(self.num_markers, dtype=np.int32)
        for k, vidx in enumerate(self.vertex_indices):
            rigging[k] = np.argmax(skin_weights[vidx])

        self.bone_rigging = rigging
        return rigging

    def compute_triangle_offsets(
        self,
        betas: torch.Tensor,
        poses: torch.Tensor,
        trans: torch.Tensor,
        frame_idx: int = 0,
    ) -> Dict:
        """
        Compute triangle-frame relative offsets for pose-invariant marker reconstruction.

        Uses the approach from SMPL2AddBiomechanics/mapping.py:
        - For each marker, find the closest triangle on the skin mesh
        - Express the marker-to-triangle offset in the triangle's local frame
        - This offset is pose-invariant and can reconstruct markers for any pose

        Args:
            betas: Shape parameters [10] or [B, 10].
            poses: Pose parameters [T, 46].
            trans: Translations [T, 3].
            frame_idx: Frame to use for computing the mapping (default: 0).

        Returns:
            Dictionary with 'tri_indices' and 'rel_offsets'.
        """
        with torch.no_grad():
            if betas.dim() == 1:
                betas_exp = betas.unsqueeze(0).expand(poses.shape[0], -1)
            else:
                betas_exp = betas
            verts, _, _ = self.skel.forward(betas_exp, poses, trans)
            verts_np = verts[frame_idx].cpu().numpy()

        # Get marker positions for the reference frame
        marker_pos = verts_np[self.vertex_indices]  # [K, 3]

        # Build trimesh from skin mesh
        faces = self.skel.faces
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces, process=False)

        # Compute triangle-frame relative offsets (from mapping.py)
        # Use direct face lookup via vertex indices (no rtree needed)
        tri_indices = self._find_triangles_for_vertices(faces)

        triangles = mesh.triangles[tri_indices]
        v0 = triangles[:, 0]
        v1 = triangles[:, 1]
        v2 = triangles[:, 2]

        m_vect_abs = marker_pos - v0

        u = (v1 - v0) / np.linalg.norm(v1 - v0, axis=1)[:, np.newaxis]
        v = (v2 - v0) / np.linalg.norm(v2 - v0, axis=1)[:, np.newaxis]
        w = np.cross(u, v) / np.linalg.norm(np.cross(u, v), axis=1)[:, np.newaxis]

        A = np.dstack((u, v, w))  # [K, 3, 3]

        rel_offsets = np.zeros_like(m_vect_abs)
        for i in range(m_vect_abs.shape[0]):
            rel_offsets[i] = np.matmul(A[i].T, m_vect_abs[i])

        self.triangle_offsets = {
            'tri_indices': tri_indices,
            'rel_offsets': rel_offsets,
        }
        return self.triangle_offsets

    def _find_triangles_for_vertices(self, faces: np.ndarray) -> np.ndarray:
        """
        Find the first triangle containing each marker vertex.

        Since BSM markers are defined by vertex indices, each marker lies
        exactly on a vertex. We find one face containing that vertex.

        Args:
            faces: Mesh face indices [F, 3].

        Returns:
            tri_indices: [K] face index for each marker.
        """
        # Build vertex → face lookup
        tri_indices = np.zeros(self.num_markers, dtype=np.int64)
        for k, vidx in enumerate(self.vertex_indices):
            # Find first face containing this vertex
            face_mask = np.any(faces == vidx, axis=1)
            matching = np.where(face_mask)[0]
            if len(matching) > 0:
                tri_indices[k] = matching[0]
            else:
                tri_indices[k] = 0  # Fallback
        return tri_indices

    def reconstruct_from_offsets(
        self,
        betas: torch.Tensor,
        poses: torch.Tensor,
        trans: torch.Tensor,
    ) -> np.ndarray:
        """
        Reconstruct marker positions from triangle-frame offsets for new poses.

        Args:
            betas, poses, trans: SKEL parameters.

        Returns:
            Reconstructed marker positions [T, K, 3].
        """
        if self.triangle_offsets is None:
            raise RuntimeError("Must call compute_triangle_offsets() first")

        tri_indices = self.triangle_offsets['tri_indices']
        rel_offsets = self.triangle_offsets['rel_offsets']

        T = poses.shape[0]
        reconstructed = np.zeros((T, self.num_markers, 3), dtype=np.float32)

        with torch.no_grad():
            if betas.dim() == 1:
                betas_exp = betas.unsqueeze(0).expand(T, -1)
            else:
                betas_exp = betas
            verts, _, _ = self.skel.forward(betas_exp, poses, trans)

        faces = self.skel.faces
        for t in range(T):
            verts_np = verts[t].cpu().numpy()
            mesh = trimesh.Trimesh(vertices=verts_np, faces=faces, process=False)

            triangles = mesh.triangles[tri_indices]
            v0 = triangles[:, 0]
            v1 = triangles[:, 1]
            v2 = triangles[:, 2]

            u = (v1 - v0) / np.linalg.norm(v1 - v0, axis=1)[:, np.newaxis]
            v = (v2 - v0) / np.linalg.norm(v2 - v0, axis=1)[:, np.newaxis]
            w = np.cross(u, v) / np.linalg.norm(np.cross(u, v), axis=1)[:, np.newaxis]
            A = np.dstack((u, v, w))

            m_vect_abs = np.zeros_like(rel_offsets)
            for i in range(rel_offsets.shape[0]):
                m_vect_abs[i] = np.matmul(np.linalg.inv(A[i].T), rel_offsets[i])

            reconstructed[t] = v0 + m_vect_abs

        return reconstructed

    def save(
        self,
        output_dir: str,
        save_pkl: bool = True,
        save_npy: bool = True,
        save_trc: bool = True,
        fps: float = 100.0,
    ):
        """
        Save virtual markers in multiple formats.

        Args:
            output_dir: Output directory.
            save_pkl: Save .pkl with vertex_idx, rigging, offsets.
            save_npy: Save .npy with marker trajectories [T, K, 3].
            save_trc: Save .trc (AddBiomechanics-compatible marker file).
            fps: Frame rate for .trc file.
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.marker_trajectories is None:
            raise RuntimeError("Must call generate_markers() first")

        # .pkl: full marker definition
        if save_pkl:
            pkl_data = {
                'marker_names': self.marker_names,
                'vertex_indices': self.vertex_indices,
                'bone_rigging': self.bone_rigging,
                'triangle_offsets': self.triangle_offsets,
                'num_markers': self.num_markers,
            }
            pkl_path = os.path.join(output_dir, 'virtual_markers.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(pkl_data, f)

        # .npy: marker trajectories
        if save_npy:
            npy_path = os.path.join(output_dir, 'markers.npy')
            np.save(npy_path, self.marker_trajectories)

        # .trc: AddBiomechanics-compatible marker file
        if save_trc:
            self._save_trc(output_dir, fps)

    def _save_trc(self, output_dir: str, fps: float):
        """
        Save markers in .trc format using nimblephysics.

        Falls back to manual TRC writing if nimblephysics is not available.
        """
        trc_path = os.path.join(output_dir, 'markers.trc')
        markers = self.marker_trajectories  # [T, K, 3]
        T, K, _ = markers.shape

        try:
            import nimblephysics as nimble

            # Build marker list per frame as expected by nimble
            timestamps = []
            marker_observations = []
            for t in range(T):
                timestamps.append(t / fps)
                frame_markers = {}
                for k, name in enumerate(self.marker_names):
                    pos = markers[t, k]
                    frame_markers[name] = pos.tolist()
                marker_observations.append(frame_markers)

            nimble.biomechanics.OpenSimParser.saveTRC(
                trc_path, timestamps, marker_observations
            )
        except (ImportError, AttributeError):
            # Manual TRC writing
            self._write_trc_manual(trc_path, markers, fps)

    def _write_trc_manual(self, path: str, markers: np.ndarray, fps: float):
        """Manual TRC file writer as fallback."""
        T, K, _ = markers.shape

        with open(path, 'w') as f:
            # Header
            f.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(path))
            f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            f.write(f"{fps}\t{fps}\t{T}\t{K}\tm\t{fps}\t1\t{T}\n")

            # Marker names header
            header = "Frame#\tTime"
            for name in self.marker_names:
                header += f"\t{name}\t\t"
            f.write(header.rstrip() + "\n")

            # Sub-header (X/Y/Z per marker)
            sub = "\t"
            for k in range(K):
                sub += f"\tX{k+1}\tY{k+1}\tZ{k+1}"
            f.write(sub + "\n")

            # Data
            for t in range(T):
                line = f"{t+1}\t{t/fps:.6f}"
                for k in range(K):
                    line += f"\t{markers[t,k,0]:.6f}\t{markers[t,k,1]:.6f}\t{markers[t,k,2]:.6f}"
                f.write(line + "\n")

    def get_marker_quality(
        self,
        observed_markers: Optional[Dict[str, np.ndarray]] = None,
        marker_observations: Optional[List[Dict[str, np.ndarray]]] = None,
        flip_xz: bool = True,
    ) -> Dict[str, float]:
        """
        Compare virtual markers against observed AddB markers.

        Virtual markers are in SKEL coordinate system. Observed markers from
        AddBiomechanics are in AddB coordinate system. When flip_xz=True,
        observed markers are converted to SKEL coords (negate X and Z) before
        comparison.

        Args:
            observed_markers: Single-frame dict of {marker_name: position [3]}.
            marker_observations: Per-frame list of dicts.
            flip_xz: Convert observed markers from AddB to SKEL coords.

        Returns:
            Per-marker mean error in mm.
        """
        if self.marker_trajectories is None:
            return {}

        quality = {}

        if marker_observations is not None:
            T = min(len(marker_observations), self.marker_trajectories.shape[0])
            for k, name in enumerate(self.marker_names):
                errors = []
                for t in range(T):
                    if name in marker_observations[t]:
                        obs = np.array(marker_observations[t][name], dtype=np.float64)
                        if flip_xz:
                            obs[0] = -obs[0]
                            obs[2] = -obs[2]
                        pred = self.marker_trajectories[t, k]
                        err = np.linalg.norm(pred - obs) * 1000  # mm
                        errors.append(err)
                if errors:
                    quality[name] = float(np.mean(errors))

        elif observed_markers is not None:
            for k, name in enumerate(self.marker_names):
                if name in observed_markers:
                    obs = np.array(observed_markers[name], dtype=np.float64)
                    if flip_xz:
                        obs[0] = -obs[0]
                        obs[2] = -obs[2]
                    pred = self.marker_trajectories[0, k]
                    quality[name] = float(np.linalg.norm(pred - obs) * 1000)

        return quality
