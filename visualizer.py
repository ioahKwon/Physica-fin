"""
Main visualizer class for SKEL force visualization.
"""

import os
import json
import glob
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple

from .colormaps import torque_to_color_plasma, torque_to_color_dark_purple, AXIS_COLORS

# GRF arrow colors (lime shaft, red head)
GRF_SHAFT_COLOR = (0.5, 1.0, 0.3)  # Lime/neon green
GRF_HEAD_COLOR = (1.0, 0.2, 0.2)   # Red endpoint
from .mesh_utils import (
    create_line_mesh,
    create_sphere_mesh,
    create_arrow_mesh,
    read_obj_mesh,
    write_obj_with_color,
)
from .lbs_utils import (
    load_skel_lbs_weights,
    load_skel_template_vertices,
    get_dominant_joints_for_mesh,
    get_rotation_axis_for_joint,
    get_axis_index,
    SKEL_JOINT_NAMES,
    ADDB_TO_SKEL_JOINT_MAP,
    ADDB_TO_SKEL_ROTATION_MAP,
    ADDB_PARENT_FALLBACK,
    JOINT_ROTATION_AXES,
)


class SKELForceVisualizer:
    """
    Visualize joint torques on SKEL skeleton mesh with PhysPT-style rendering.

    Features:
    - Skeleton mesh colored by joint torque magnitude (plasma colormap)
    - 3-axis lines showing X, Y, Z torque components with endpoint spheres
    - Separate body mesh output

    Example:
        >>> vis = SKELForceVisualizer(
        ...     input_base="/path/to/frames",
        ...     output_dir="/path/to/output"
        ... )
        >>> vis.process_all_frames()
    """

    def __init__(
        self,
        input_base: str,
        output_dir: str,
        colormap: Callable = torque_to_color_plasma,
        max_torque: float = 300.0,
        line_scale: float = 0.002,
        line_radius: float = 0.004,
        sphere_radius: float = 0.012,
        torque_threshold: float = 0.5,
        axis_colors: Optional[Dict[str, tuple]] = None,
        skel_model_path: Optional[str] = None,
        gender: str = 'male',
        use_lbs_coloring: bool = True,
        show_grf: bool = True,
        grf_scale: float = 0.0005,
        grf_radius: float = 0.005,
        grf_shaft_color: tuple = GRF_SHAFT_COLOR,
        grf_head_color: tuple = GRF_HEAD_COLOR,
        unit_arrow_length: float = 0.05,
        mesh_override_dir: Optional[str] = None,
        coloring_mode: str = 'lbs_blend',
        smooth_sigma: float = 0.02,
        distance_falloff: float = 0.1,
        joints_ori_data: Optional[Dict] = None,
    ):
        """
        Initialize the visualizer.

        Args:
            input_base: Base directory containing frame_XXXX folders
            output_dir: Output directory for generated OBJ files
            colormap: Function to convert torque magnitude to RGB color
            max_torque: Maximum torque for colormap normalization (Nm)
            line_scale: Scale factor for axis line length (m per Nm) - used when unit_arrow_length=0
            line_radius: Radius of axis line cylinders (m)
            sphere_radius: Radius of endpoint spheres (m)
            torque_threshold: Minimum torque to show axis line (Nm)
            axis_colors: Dict with 'x', 'y', 'z' keys for axis colors
            skel_model_path: Path to SKEL model directory for LBS weights
            gender: 'male' or 'female' for SKEL model
            use_lbs_coloring: If True, use LBS weights for proper bone segmentation
            show_grf: If True, visualize Ground Reaction Force arrows
            grf_scale: Scale factor for GRF arrow length (m per N)
            grf_radius: Radius of GRF arrow cylinder (m)
            grf_shaft_color: RGB color for GRF arrow shaft (cylinder)
            grf_head_color: RGB color for GRF arrow head (sphere)
            unit_arrow_length: Fixed arrow length for torque vectors (m). If >0, arrows are unit vectors.
            mesh_override_dir: Optional directory containing corrected skeleton/body meshes
            coloring_mode: Vertex coloring method:
                - 'lbs_blend': LBS weight-based color blending (default)
                - 'gaussian': Spatial Gaussian smoothing of colors
                - 'distance': Distance-based gradient from joint positions
            smooth_sigma: Sigma for Gaussian smoothing (meters), used when coloring_mode='gaussian'
            distance_falloff: Falloff distance for distance-based gradient (meters), used when coloring_mode='distance'
            joints_ori_data: Dict with 'joints_ori' (T, 24, 3, 3) and 'frame_indices' for local→global torque transform
        """
        self.input_base = input_base
        self.mesh_override_dir = mesh_override_dir
        self.output_dir = output_dir
        self.colormap = colormap
        self.max_torque = max_torque
        self.torque_min = 0.0  # Will be updated by _scan_torque_range
        self.torque_max = max_torque  # Will be updated by _scan_torque_range
        self.torque_percentiles = None  # Will be updated by _scan_torque_range
        self.line_scale = line_scale
        self.line_radius = line_radius
        self.sphere_radius = sphere_radius
        self.torque_threshold = torque_threshold
        self.axis_colors = axis_colors or AXIS_COLORS
        self.use_lbs_coloring = use_lbs_coloring
        self.skel_model_path = skel_model_path
        self.gender = gender
        self.show_grf = show_grf
        self.grf_scale = grf_scale
        self.grf_radius = grf_radius
        self.grf_shaft_color = grf_shaft_color
        self.grf_head_color = grf_head_color
        self.unit_arrow_length = unit_arrow_length
        self.coloring_mode = coloring_mode
        self.smooth_sigma = smooth_sigma
        self.distance_falloff = distance_falloff

        # Store joints_ori data for local→global torque transformation
        self.joints_ori_data = joints_ori_data
        self._frame_to_ori_idx = {}  # Maps frame number to joints_ori index
        if joints_ori_data is not None:
            frame_indices = joints_ori_data.get('frame_indices', [])
            for i, frame_idx in enumerate(frame_indices):
                self._frame_to_ori_idx[frame_idx] = i

        # Load LBS weights for proper bone segmentation
        self.template_dominant_joints = None
        self.template_vertices = None
        self.lbs_weights = None
        self._mesh_dominant_joints_cache = {}  # Cache for computed mappings

        if skel_model_path and use_lbs_coloring:
            try:
                self.template_dominant_joints, self.lbs_weights = load_skel_lbs_weights(
                    skel_model_path, gender, mesh_type='skeleton'
                )
                self.template_vertices = load_skel_template_vertices(skel_model_path, gender)
                print(f"Loaded LBS weights: {self.lbs_weights.shape} for skeleton mesh")
                print(f"Loaded template vertices: {self.template_vertices.shape}")
            except Exception as e:
                print(f"Warning: Could not load LBS weights: {e}")
                print("Falling back to nearest-joint coloring")

        os.makedirs(output_dir, exist_ok=True)

    def process_frame(self, frame_dir: str, skeleton_mesh_override: Optional[str] = None, body_mesh_override: Optional[str] = None) -> Dict:
        """
        Process a single frame directory.

        Args:
            frame_dir: Path to frame directory
            skeleton_mesh_override: Optional path to use instead of default skeleton mesh
            body_mesh_override: Optional path to use instead of default body mesh

        Returns:
            Dict with processing statistics
        """
        frame_name = os.path.basename(frame_dir)

        # Check mesh override directory first
        if self.mesh_override_dir:
            override_frame_dir = os.path.join(self.mesh_override_dir, frame_name)
            skeleton_mesh_override = skeleton_mesh_override or os.path.join(override_frame_dir, "skeleton.obj")
            body_mesh_override = body_mesh_override or os.path.join(override_frame_dir, "body.obj")

        body_mesh_path = body_mesh_override or os.path.join(frame_dir, "skel_mesh.obj")
        skeleton_mesh_path = skeleton_mesh_override or os.path.join(frame_dir, "skel_skeleton.obj")
        force_path = os.path.join(frame_dir, "force_data.json")

        if not os.path.exists(skeleton_mesh_path) or not os.path.exists(force_path):
            return {"frame": frame_name, "status": "skipped", "reason": "missing files"}

        # Read force data
        with open(force_path, 'r') as f:
            force_data = json.load(f)

        # Read SKEL skeleton mesh
        skeleton_vertices, skeleton_faces = read_obj_mesh(skeleton_mesh_path)

        # Build joint dictionaries
        joint_positions = {}
        joint_torques = {}
        joint_tau_vectors = {}

        for jt in force_data.get('joint_torques', []):
            joint_name = jt['joint']
            joint_positions[joint_name] = np.array(jt['position'])
            joint_torques[joint_name] = jt['magnitude']
            joint_tau_vectors[joint_name] = np.array(jt['tau'])

        # Assign colors to skeleton mesh vertices based on joint torque
        # Pass faces for accurate face-based vertex mapping (handles deduplicated meshes)
        skeleton_vertex_colors = self._compute_vertex_colors(
            skeleton_vertices, joint_positions, joint_torques, faces=skeleton_faces
        )

        # Get world rotations for this frame (for local→global torque transformation)
        joint_world_rotations = None
        if self.joints_ori_data is not None:
            # Extract frame number from frame_name (e.g., "frame_0001" -> 1)
            try:
                frame_num = int(frame_name.split('_')[-1])
                ori_idx = self._frame_to_ori_idx.get(frame_num)
                if ori_idx is not None:
                    joints_ori = self.joints_ori_data['joints_ori']  # [T, 24, 3, 3]
                    frame_ori = joints_ori[ori_idx]  # [24, 3, 3]
                    # Build dict: AddB joint name -> world rotation matrix
                    joint_world_rotations = {}
                    for addb_joint_name, skel_joint_idx in ADDB_TO_SKEL_ROTATION_MAP.items():
                        joint_world_rotations[addb_joint_name] = frame_ori[skel_joint_idx]
            except (ValueError, KeyError, IndexError):
                pass  # Fallback: no transformation

        # Create 3-axis lines with endpoint spheres
        lines_data = self._create_axis_lines(
            joint_positions, joint_tau_vectors, joint_world_rotations
        )

        # Create GRF arrows if enabled
        grf_data = []
        if self.show_grf:
            grf_data = self._create_grf_arrows(force_data.get('grf', []))

        # Write combined skeleton + lines + GRF OBJ
        combined_path = os.path.join(self.output_dir, f"{frame_name}_skeleton_axes.obj")
        self._write_combined_obj(
            combined_path, skeleton_vertices, skeleton_faces,
            skeleton_vertex_colors, lines_data, grf_data
        )

        # Write combined PLY (better vertex color support in viewers)
        ply_path = os.path.join(self.output_dir, f"{frame_name}_skeleton_axes.ply")
        self._write_combined_ply(
            ply_path, skeleton_vertices, skeleton_faces,
            skeleton_vertex_colors, lines_data, grf_data
        )

        # Write body mesh if exists
        if os.path.exists(body_mesh_path):
            body_vertices, body_faces = read_obj_mesh(body_mesh_path)
            body_path = os.path.join(self.output_dir, f"{frame_name}_body.obj")
            write_obj_with_color(body_path, body_vertices, body_faces, (0.7, 0.7, 0.7))

        # Write force summary
        txt_path = os.path.join(self.output_dir, f"{frame_name}_force.txt")
        self._write_force_summary(txt_path, frame_name, joint_torques, joint_tau_vectors)

        return {
            "frame": frame_name,
            "status": "success",
            "joints": len(joint_torques),
            "axis_lines": len(lines_data),
            "grf_arrows": len(grf_data),
        }

    def process_all_frames(self, verbose: bool = True) -> List[Dict]:
        """
        Process all frames in the input directory.

        Args:
            verbose: Print progress messages

        Returns:
            List of processing results for each frame
        """
        frame_dirs = sorted(glob.glob(os.path.join(self.input_base, "frame_*")))

        if verbose:
            print(f"Found {len(frame_dirs)} frames")

        # First pass: scan all frames to find global torque range
        self._scan_torque_range(frame_dirs, verbose)

        results = []
        for frame_dir in frame_dirs:
            result = self.process_frame(frame_dir)
            results.append(result)

            if verbose and result["status"] == "success":
                grf_str = f", {result['grf_arrows']} GRF" if result.get('grf_arrows', 0) > 0 else ""
                print(f"  {result['frame']}: {result['joints']} joints, {result['axis_lines']} axes{grf_str}")

        if verbose:
            print(f"\nDone! Output: {self.output_dir}")

        return results

    def _scan_torque_range(self, frame_dirs: List[str], verbose: bool = True):
        """
        Scan all frames to find global min/max torque for colormap normalization.
        This ensures consistent coloring across the entire sequence.
        """
        all_torques = []

        for frame_dir in frame_dirs:
            force_file = os.path.join(frame_dir, "force_data.json")
            if os.path.exists(force_file):
                try:
                    with open(force_file, 'r') as f:
                        force_data = json.load(f)
                    # Extract magnitudes from joint_torques array
                    for joint_data in force_data.get("joint_torques", []):
                        magnitude = joint_data.get("magnitude", 0)
                        if magnitude > 0.05:  # Skip only near-zero torques (matches color threshold)
                            all_torques.append(magnitude)
                except:
                    continue

        if len(all_torques) > 0:
            self.torque_min = np.min(all_torques)
            self.torque_max = np.max(all_torques)
            # Store LOG-TRANSFORMED sorted torques for percentile-based color mapping
            # This ensures equal color spread across orders of magnitude (0.05-0.5, 0.5-5, 5-50, etc.)
            log_torques = np.log10(np.array(all_torques))
            self.torque_percentiles = np.sort(log_torques)
            if verbose:
                print(f"Global torque range: {self.torque_min:.1f} - {self.torque_max:.1f} Nm")
                # Show percentile distribution for color mapping reference
                p25 = np.percentile(all_torques, 25)
                p50 = np.percentile(all_torques, 50)
                p75 = np.percentile(all_torques, 75)
                p90 = np.percentile(all_torques, 90)
                print(f"  Percentiles: 25%={p25:.1f}, 50%={p50:.1f}, 75%={p75:.1f}, 90%={p90:.1f} Nm")
                print(f"  (Color mapping uses log scale for dynamic variation)")
        else:
            self.torque_min = 0.0
            self.torque_max = 300.0  # Default fallback
            self.torque_percentiles = None
            if verbose:
                print("No torque data found, using default range 0-300 Nm")

    def _compute_vertex_colors(
        self,
        vertices: np.ndarray,
        joint_positions: Dict[str, np.ndarray],
        joint_torques: Dict[str, float],
        faces: np.ndarray = None
    ) -> List[tuple]:
        """
        Compute vertex colors based on joint torque.

        Supports multiple coloring modes:
        - 'lbs_blend': LBS weight-based color blending
        - 'gaussian': Spatial Gaussian smoothing
        - 'distance': Distance-based gradient from joints
        - 'hotspot': Joint-only hotspot visualization (like pain relief ads)
        """
        if len(joint_positions) == 0:
            return [(0.8, 0.8, 0.8)] * len(vertices)

        # Choose coloring method based on mode
        if self.coloring_mode == 'gaussian':
            return self._compute_vertex_colors_gaussian(vertices, joint_positions, joint_torques, faces)
        elif self.coloring_mode == 'distance':
            return self._compute_vertex_colors_distance(vertices, joint_positions, joint_torques)
        elif self.coloring_mode == 'hotspot':
            return self._compute_vertex_colors_hotspot(vertices, joint_positions, joint_torques)
        elif self.template_dominant_joints is not None and self.use_lbs_coloring:
            return self._compute_vertex_colors_lbs(vertices, joint_torques, faces=faces)

        # Fallback: nearest joint coloring
        return self._compute_vertex_colors_nearest(vertices, joint_positions, joint_torques)

    def _compute_vertex_colors_lbs(
        self,
        vertices: np.ndarray,
        joint_torques: Dict[str, float],
        faces: np.ndarray = None
    ) -> List[tuple]:
        """
        Compute vertex colors using LBS weights with weighted blending.

        Each vertex color is computed as a weighted blend of all joint colors,
        where weights come from the LBS skinning weights. This creates smooth
        gradients across the mesh similar to PhysPT visualization.

        Formula: color_v = Σ_j (w_vj * colormap(torque_j))

        Handles mesh with deduplicated vertices by using face-based mapping.
        """
        num_vertices = len(vertices)
        num_template_vertices = len(self.template_vertices) if self.template_vertices is not None else 0

        # Build mapping from SKEL joint index to torque value
        skel_idx_to_torque = np.zeros(24)  # 24 SKEL joints
        for addb_name, torque in joint_torques.items():
            skel_indices = ADDB_TO_SKEL_JOINT_MAP.get(addb_name)
            if skel_indices is not None:
                for skel_idx in skel_indices:
                    if torque > skel_idx_to_torque[skel_idx]:
                        skel_idx_to_torque[skel_idx] = torque

        # Apply parent joint fallback for joints without torque data
        for addb_name, parent_addb_name in ADDB_PARENT_FALLBACK.items():
            skel_indices = ADDB_TO_SKEL_JOINT_MAP.get(addb_name)
            if skel_indices is None:
                continue
            for skel_idx in skel_indices:
                if skel_idx_to_torque[skel_idx] == 0:
                    parent_name = parent_addb_name
                    parent_torque = joint_torques.get(parent_name, 0)
                    while parent_torque == 0 and parent_name in ADDB_PARENT_FALLBACK:
                        parent_name = ADDB_PARENT_FALLBACK[parent_name]
                        parent_torque = joint_torques.get(parent_name, 0)
                    if parent_torque > 0:
                        skel_idx_to_torque[skel_idx] = parent_torque

        # Precompute colors for each joint (using global min/max range)
        joint_colors = np.array([
            self.colormap(t, self.torque_max, self.torque_min) for t in skel_idx_to_torque
        ])

        # Get LBS weights for vertices
        if self.lbs_weights is not None and num_vertices == num_template_vertices:
            # Direct correspondence - use template LBS weights
            lbs_weights = self.lbs_weights
        elif self.lbs_weights is not None:
            # Different vertex count - need to map weights
            cache_key = (num_vertices, len(faces) if faces is not None else 0, "weights")
            if cache_key in self._mesh_dominant_joints_cache:
                lbs_weights = self._mesh_dominant_joints_cache[cache_key]
            else:
                print(f"Computing LBS weight mapping for {num_vertices} vertices...")
                lbs_weights = self._map_lbs_weights_to_mesh(vertices, faces)
                self._mesh_dominant_joints_cache[cache_key] = lbs_weights
                print(f"LBS weight mapping computed and cached.")
        else:
            # No LBS weights - fallback to dominant joint only
            return self._compute_vertex_colors_lbs_dominant(vertices, joint_torques, faces)

        # Compute blended colors using LBS weights
        # color_v = Σ_j (w_vj * color_j) for all joints j
        colors = []
        for v_idx in range(num_vertices):
            if v_idx < len(lbs_weights):
                weights = lbs_weights[v_idx]  # Shape: (24,)
                # Weighted blend of all joint colors
                blended_color = np.zeros(3)
                weight_sum = 0.0
                for j in range(24):
                    if weights[j] > 0.01:  # Skip negligible weights
                        blended_color += weights[j] * np.array(joint_colors[j])
                        weight_sum += weights[j]
                if weight_sum > 0:
                    blended_color /= weight_sum
                else:
                    blended_color = np.array([0.8, 0.8, 0.8])
                colors.append(tuple(blended_color))
            else:
                colors.append((0.8, 0.8, 0.8))

        return colors

    def _compute_vertex_colors_lbs_dominant(
        self,
        vertices: np.ndarray,
        joint_torques: Dict[str, float],
        faces: np.ndarray = None
    ) -> List[tuple]:
        """
        Fallback: Compute vertex colors using dominant joint only (no blending).
        Used when LBS weights are not available.
        """
        num_vertices = len(vertices)

        # Build mapping from SKEL joint index to torque value
        skel_idx_to_torque = {}
        for addb_name, torque in joint_torques.items():
            skel_indices = ADDB_TO_SKEL_JOINT_MAP.get(addb_name)
            if skel_indices is not None:
                for skel_idx in skel_indices:
                    if skel_idx not in skel_idx_to_torque or torque > skel_idx_to_torque[skel_idx]:
                        skel_idx_to_torque[skel_idx] = torque

        # Get dominant joints
        cache_key = (num_vertices, len(faces) if faces is not None else 0)
        if cache_key in self._mesh_dominant_joints_cache:
            dominant_joints = self._mesh_dominant_joints_cache[cache_key]
        else:
            dominant_joints = get_dominant_joints_for_mesh(
                vertices, self.skel_model_path, self.gender, mesh_faces=faces
            )
            self._mesh_dominant_joints_cache[cache_key] = dominant_joints

        colors = []
        for v_idx in range(num_vertices):
            if v_idx < len(dominant_joints):
                joint_idx = dominant_joints[v_idx]
                torque = skel_idx_to_torque.get(joint_idx, 0)
            else:
                torque = 0
            color = self.colormap(torque, self.max_torque)
            colors.append(color)

        return colors

    def _map_lbs_weights_to_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """
        Map LBS weights from template mesh to a mesh with different vertex count.
        Uses nearest neighbor matching in template space.
        """
        from scipy.spatial import cKDTree

        if self.template_vertices is None or self.lbs_weights is None:
            return None

        # Build KD-tree from template vertices
        tree = cKDTree(self.template_vertices)

        # Find nearest template vertex for each mesh vertex
        _, indices = tree.query(vertices)

        # Map weights
        mapped_weights = self.lbs_weights[indices]

        return mapped_weights

    def _compute_vertex_colors_nearest(
        self,
        vertices: np.ndarray,
        joint_positions: Dict[str, np.ndarray],
        joint_torques: Dict[str, float]
    ) -> List[tuple]:
        """Compute vertex colors based on nearest joint torque (legacy method)."""
        joint_names = list(joint_positions.keys())
        joint_pos_array = np.array([joint_positions[name] for name in joint_names])

        colors = []
        for vertex in vertices:
            distances = np.linalg.norm(joint_pos_array - vertex, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_joint = joint_names[nearest_idx]
            torque_mag = joint_torques.get(nearest_joint, 0)
            color = self.colormap(torque_mag, self.max_torque)
            colors.append(color)

        return colors

    def _compute_vertex_colors_gaussian(
        self,
        vertices: np.ndarray,
        joint_positions: Dict[str, np.ndarray],
        joint_torques: Dict[str, float],
        faces: np.ndarray = None
    ) -> List[tuple]:
        """
        Compute vertex colors with spatial Gaussian smoothing.

        First computes base colors using LBS weights, then applies Gaussian
        smoothing based on vertex distances to create smooth gradients.
        """
        from scipy.spatial import cKDTree

        # First compute base colors using LBS
        if self.template_dominant_joints is not None and self.use_lbs_coloring:
            base_colors = self._compute_vertex_colors_lbs(vertices, joint_torques, faces=faces)
        else:
            base_colors = self._compute_vertex_colors_nearest(vertices, joint_positions, joint_torques)

        base_colors = np.array(base_colors)
        num_vertices = len(vertices)

        # Build KD-tree for spatial queries
        tree = cKDTree(vertices)

        # Gaussian smoothing
        sigma = self.smooth_sigma
        smoothed_colors = np.zeros_like(base_colors)

        # For each vertex, find neighbors within 3*sigma and compute weighted average
        search_radius = 3 * sigma
        for i in range(num_vertices):
            neighbor_indices = tree.query_ball_point(vertices[i], search_radius)
            if len(neighbor_indices) == 0:
                smoothed_colors[i] = base_colors[i]
                continue

            # Compute Gaussian weights
            neighbor_verts = vertices[neighbor_indices]
            distances = np.linalg.norm(neighbor_verts - vertices[i], axis=1)
            weights = np.exp(-distances**2 / (2 * sigma**2))
            weights /= weights.sum()

            # Weighted average of colors
            neighbor_colors = base_colors[neighbor_indices]
            smoothed_colors[i] = np.sum(weights[:, np.newaxis] * neighbor_colors, axis=0)

        return [tuple(c) for c in smoothed_colors]

    def _compute_vertex_colors_distance(
        self,
        vertices: np.ndarray,
        joint_positions: Dict[str, np.ndarray],
        joint_torques: Dict[str, float]
    ) -> List[tuple]:
        """
        Compute vertex colors using distance-based gradient from joint positions.

        Each vertex color is a weighted blend of all joint colors, where weights
        are inversely proportional to distance (with exponential falloff).
        This creates smooth gradients radiating from joint positions.

        Formula: color_v = Σ_j (w_vj * colormap(torque_j)) / Σ_j (w_vj)
        where w_vj = exp(-dist(v, j) / falloff)
        """
        joint_names = list(joint_positions.keys())
        if len(joint_names) == 0:
            return [(0.8, 0.8, 0.8)] * len(vertices)

        joint_pos_array = np.array([joint_positions[name] for name in joint_names])
        joint_torque_array = np.array([joint_torques.get(name, 0) for name in joint_names])

        # Precompute colors for each joint
        joint_colors = np.array([self.colormap(t, self.max_torque) for t in joint_torque_array])

        falloff = self.distance_falloff
        colors = []

        for vertex in vertices:
            # Compute distance to each joint
            distances = np.linalg.norm(joint_pos_array - vertex, axis=1)

            # Exponential falloff weights
            weights = np.exp(-distances / falloff)

            # Normalize weights
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights /= weight_sum
            else:
                colors.append((0.8, 0.8, 0.8))
                continue

            # Weighted blend of joint colors
            blended_color = np.sum(weights[:, np.newaxis] * joint_colors, axis=0)
            colors.append(tuple(blended_color))

        return colors

    def _compute_vertex_colors_hotspot(
        self,
        vertices: np.ndarray,
        joint_positions: Dict[str, np.ndarray],
        joint_torques: Dict[str, float]
    ) -> List[tuple]:
        """
        Compute vertex colors with hotspot visualization at joint locations.

        Background is bone color, joints show flat circles with torque-weighted colors.
        When multiple joint circles overlap, colors are blended by torque-weighted average.
        """
        joint_names = list(joint_positions.keys())
        if len(joint_names) == 0:
            return [(0.85, 0.82, 0.75)] * len(vertices)  # Bone color

        joint_pos_array = np.array([joint_positions[name] for name in joint_names])
        # Use list to preserve None values (None = no data, 0 = data exists but zero)
        joint_torque_list = [joint_torques.get(name, None) for name in joint_names]

        # Background color (slightly darker bone for better contrast)
        bg_color = np.array([0.75, 0.72, 0.65])

        # Hotspot colors (dark red -> bright yellow, with high contrast)
        def torque_to_hotspot_color(torque, torque_percentiles):
            # torque=None: no data → return None (use background color)
            if torque is None:
                return None

            # Use ABSOLUTE VALUE - negative torque has same magnitude as positive
            torque_magnitude = abs(torque)

            # torque magnitude ≈ 0: black
            MIN_TORQUE = 0.05  # Nm - threshold for "zero" torque
            if torque_magnitude < MIN_TORQUE:
                return np.array([0.0, 0.0, 0.0])  # Black for zero/near-zero torque

            # LOG-PERCENTILE hybrid: percentile on log-transformed values
            # This gives equal color spread for 0.05-0.5, 0.5-5, 5-50, 50-500 etc.
            log_torque = np.log10(torque_magnitude)

            if torque_percentiles is not None and len(torque_percentiles) > 0:
                # torque_percentiles should be log-transformed values
                # Find percentile rank using binary search on log scale
                idx = np.searchsorted(torque_percentiles, log_torque)
                t = idx / len(torque_percentiles)
            else:
                # Fallback: fixed log scale 0.05 ~ 300 Nm
                log_min = np.log10(MIN_TORQUE)  # -1.3
                log_max = np.log10(300.0)       # 2.48
                t = (log_torque - log_min) / (log_max - log_min)

            t = np.clip(t, 0, 1)

            # High-contrast gradient with more color stops for dynamic variation
            # Dark red -> Red -> Orange -> Yellow-Orange -> Bright Yellow
            colors = [
                np.array([0.3, 0.0, 0.0]),    # 0.00 - very dark red (lowest)
                np.array([0.6, 0.0, 0.0]),    # 0.20 - dark red
                np.array([0.9, 0.1, 0.0]),    # 0.40 - red
                np.array([1.0, 0.4, 0.0]),    # 0.60 - red-orange
                np.array([1.0, 0.7, 0.1]),    # 0.80 - orange
                np.array([1.0, 1.0, 0.4]),    # 1.00 - bright yellow (highest)
            ]
            n_segments = len(colors) - 1
            idx = t * n_segments
            i = int(idx)
            if i >= n_segments:
                return colors[n_segments]
            frac = idx - i
            return colors[i] * (1 - frac) + colors[i + 1] * frac

        # Joint-specific hotspot radii (to avoid bleeding into nearby bones)
        joint_radii = {
            # Upper body - smaller radii to avoid rib cage bleeding
            'left_shoulder': 0.08,
            'right_shoulder': 0.08,
            'left_elbow': 0.08,
            'right_elbow': 0.08,
            'left_wrist': 0.06,
            'right_wrist': 0.06,
            # Lower body - larger radii
            'left_hip': 0.12,
            'right_hip': 0.12,
            'left_knee': 0.12,
            'right_knee': 0.12,
            'left_ankle': 0.10,
            'right_ankle': 0.10,
            # Spine - medium radii
            'lumbar': 0.10,
            'thorax': 0.10,
            'head': 0.08,
        }
        default_radius = 0.10  # 10cm default for unlisted joints

        # Build radius array matching joint order
        joint_radii_array = np.array([
            joint_radii.get(name, default_radius) for name in joint_names
        ])

        colors = []
        for vertex in vertices:
            # Compute distance to each joint
            distances = np.linalg.norm(joint_pos_array - vertex, axis=1)

            # Collect all joints within their respective radii
            # Apply soft edge falloff at boundary (80-100% of radius)
            contributing_joints = []
            edge_falloff_start = 0.8  # Start fading at 80% of radius

            for j, (dist, torque, radius) in enumerate(zip(distances, joint_torque_list, joint_radii_array)):
                if dist < radius and torque is not None:  # Only process if data exists
                    hotspot_color = torque_to_hotspot_color(torque, self.torque_percentiles)
                    if hotspot_color is not None:
                        # Compute edge falloff: 1.0 at center, smoothly fades to 0 at edge
                        dist_ratio = dist / radius
                        if dist_ratio < edge_falloff_start:
                            edge_alpha = 1.0  # Full intensity inside 80%
                        else:
                            # Smooth falloff from 80% to 100% of radius
                            edge_alpha = 1.0 - (dist_ratio - edge_falloff_start) / (1.0 - edge_falloff_start)
                            edge_alpha = edge_alpha ** 2  # Quadratic for smoother transition

                        # Use max(torque, 0.1) for weight so zero-torque joints still contribute
                        weight = max(torque, 0.1) * edge_alpha
                        contributing_joints.append((hotspot_color, weight, edge_alpha))

            if len(contributing_joints) == 0:
                # No joints affect this vertex - use background
                colors.append(tuple(bg_color))
            else:
                # Torque-weighted average of all contributing joint colors, with edge blending
                total_weight = sum(w for _, w, _ in contributing_joints)
                max_alpha = max(a for _, _, a in contributing_joints)

                weighted_color = np.zeros(3)
                for color, weight, _ in contributing_joints:
                    weighted_color += color * (weight / total_weight)

                # Blend with background at edges
                final_color = weighted_color * max_alpha + bg_color * (1.0 - max_alpha)
                colors.append(tuple(np.clip(final_color, 0, 1)))

        return colors

    def _create_axis_lines(
        self,
        joint_positions: Dict[str, np.ndarray],
        joint_tau_vectors: Dict[str, np.ndarray],
        joint_world_rotations: Optional[Dict[str, np.ndarray]] = None
    ) -> List[tuple]:
        """
        Create 3-axis arrow visualization for each joint (like hand joint visualization).

        Features:
        - Shows ALL 3 axes (X, Y, Z) at each joint position
        - Each axis is a unit vector arrow in global coordinates
        - X = Red, Y = Blue, Z = Yellow
        - Arrow direction shows the transformed local axis in world frame

        Args:
            joint_positions: Dict mapping joint name to 3D position
            joint_tau_vectors: Dict mapping joint name to torque vector (local coordinates)
            joint_world_rotations: Dict mapping joint name to 3x3 world rotation matrix
                                   If provided, transforms local axes to global coordinates
        """
        lines_data = []

        # Arrow parameters
        arrow_length = self.unit_arrow_length if self.unit_arrow_length > 0 else 0.08
        shaft_radius = self.line_radius * 1.0
        head_radius = shaft_radius * 2.5
        head_length_ratio = 0.25

        # Local axis definitions
        local_axes = [
            np.array([1.0, 0.0, 0.0]),  # X axis (Red)
            np.array([0.0, 1.0, 0.0]),  # Y axis (Blue)
            np.array([0.0, 0.0, 1.0]),  # Z axis (Yellow)
        ]

        # Colors for each axis
        axes_colors = [
            (0.9, 0.2, 0.2),   # X = Red
            (0.2, 0.4, 0.9),   # Y = Blue
            (0.9, 0.9, 0.2),   # Z = Yellow
        ]

        for joint_name, pos in joint_positions.items():
            tau_vec = joint_tau_vectors.get(joint_name, np.array([]))

            if len(tau_vec) < 1:
                continue

            # Get world rotation for this joint (if available)
            world_rot = None
            if joint_world_rotations is not None:
                world_rot = joint_world_rotations.get(joint_name)

            # Determine number of DOFs and build tau_3d
            if len(tau_vec) == 1:
                # 1-DOF joint: place torque on the correct axis
                local_axis = get_rotation_axis_for_joint(joint_name)
                if local_axis is not None:
                    axis_idx = get_axis_index(local_axis)
                    tau_3d = np.zeros(3)
                    tau_3d[axis_idx] = tau_vec[0]
                    active_axes = [axis_idx]  # Only this axis is active
                else:
                    tau_3d = np.array([tau_vec[0], 0, 0])
                    active_axes = [0]
            elif len(tau_vec) == 2:
                tau_3d = np.array([tau_vec[0], tau_vec[1], 0])
                active_axes = [0, 1]  # X and Y are active
            elif len(tau_vec) >= 6:
                # 6-DOF (ground_pelvis): use rotation components (indices 3, 4, 5)
                tau_3d = np.array([tau_vec[3], tau_vec[4], tau_vec[5]])
                active_axes = [0, 1, 2]  # All 3 rotation axes
            else:
                tau_3d = tau_vec[:3]
                active_axes = [0, 1, 2]  # All 3 axes

            # Show ALL active axes for this joint
            for axis_idx in active_axes:
                local_axis = local_axes[axis_idx]
                torque_component = tau_3d[axis_idx] if axis_idx < len(tau_3d) else 0.0

                # Transform local axis to global coordinates
                if world_rot is not None:
                    global_axis = world_rot @ local_axis
                    global_axis = global_axis / (np.linalg.norm(global_axis) + 1e-8)
                else:
                    global_axis = local_axis

                # Direction based on torque sign (positive = axis direction)
                # Color is ALWAYS the axis color - intensity based on |torque|
                # Only truly zero torque gets gray
                torque_magnitude = abs(torque_component)

                if torque_magnitude < 0.01:
                    # Truly zero torque - show gray
                    direction = global_axis  # Default to positive direction
                    color = (0.5, 0.5, 0.5)  # Gray for zero torque
                else:
                    # Non-zero torque (positive OR negative)
                    # Direction flips based on sign, but COLOR stays the same
                    direction = global_axis if torque_component > 0 else -global_axis
                    color = axes_colors[axis_idx]  # Same color for + and - torque

                end_pos = pos + direction * arrow_length

                # Create arrow
                shaft_v, shaft_f, head_v, head_f = create_arrow_mesh(
                    start=pos, end=end_pos,
                    shaft_radius=shaft_radius,
                    head_radius=head_radius,
                    head_length_ratio=head_length_ratio,
                    segments=8
                )

                if len(shaft_v) > 0:
                    lines_data.append((shaft_v, shaft_f, color))
                if len(head_v) > 0:
                    lines_data.append((head_v, head_f, color))

        return lines_data

    def _create_grf_arrows(self, grf_list: List[Dict]) -> List[tuple]:
        """
        Create GRF (Ground Reaction Force) arrows.

        Each GRF is visualized as a cylinder arrow from CoP pointing in force direction.
        """
        grf_data = []

        for grf_entry in grf_list:
            if grf_entry.get('contact', 0) == 0:
                continue

            cop = np.array(grf_entry.get('cop', [0, 0, 0]))
            force_vec = np.array(grf_entry.get('grf', [0, 0, 0]))
            magnitude = grf_entry.get('magnitude', np.linalg.norm(force_vec))

            if magnitude < 1.0:  # Skip very small forces
                continue

            # Normalize and scale the force direction
            force_dir = force_vec / (np.linalg.norm(force_vec) + 1e-8)
            arrow_length = magnitude * self.grf_scale
            end_pos = cop + force_dir * arrow_length

            # Create arrow shaft (cylinder) - lime/neon green
            shaft_verts, shaft_faces = create_line_mesh(
                start=cop,
                end=end_pos,
                radius=self.grf_radius,
                segments=8
            )

            if len(shaft_verts) > 0:
                grf_data.append((shaft_verts, shaft_faces, self.grf_shaft_color))

            # Create arrow head (larger sphere at the end) - red
            head_verts, head_faces = create_sphere_mesh(
                center=end_pos,
                radius=self.grf_radius * 2.0,
                segments=10,
                rings=8
            )

            if len(head_verts) > 0:
                grf_data.append((head_verts, head_faces, self.grf_head_color))

        return grf_data

    def _write_combined_obj(
        self,
        filepath: str,
        skeleton_vertices: np.ndarray,
        skeleton_faces: np.ndarray,
        skeleton_colors: List[tuple],
        lines_data: List[tuple],
        grf_data: List[tuple] = None
    ) -> None:
        """Write combined skeleton + axes + GRF OBJ file."""
        with open(filepath, 'w') as f:
            f.write("# SKEL skeleton mesh + 3-axis torque lines + GRF arrows\n")
            f.write("# Skeleton: dark purple gradient by joint torque magnitude\n")
            f.write("# Axes: X=Red, Y=Blue, Z=Yellow (with endpoint balls)\n")
            f.write("# GRF: Lime shaft + Red head arrows from CoP\n")

            vertex_offset = 0

            # Write skeleton vertices
            for i, v in enumerate(skeleton_vertices):
                c = skeleton_colors[i]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")

            # Write skeleton faces
            for face in skeleton_faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

            vertex_offset = len(skeleton_vertices)

            # Write axis lines and spheres
            for line_verts, line_faces, line_color in lines_data:
                if len(line_verts) == 0:
                    continue

                for v in line_verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {line_color[0]:.4f} {line_color[1]:.4f} {line_color[2]:.4f}\n")

                for face in line_faces:
                    f.write(f"f {face[0]+1+vertex_offset} {face[1]+1+vertex_offset} {face[2]+1+vertex_offset}\n")

                vertex_offset += len(line_verts)

            # Write GRF arrows
            if grf_data:
                for grf_verts, grf_faces, grf_color in grf_data:
                    if len(grf_verts) == 0:
                        continue

                    for v in grf_verts:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {grf_color[0]:.4f} {grf_color[1]:.4f} {grf_color[2]:.4f}\n")

                    for face in grf_faces:
                        f.write(f"f {face[0]+1+vertex_offset} {face[1]+1+vertex_offset} {face[2]+1+vertex_offset}\n")

                    vertex_offset += len(grf_verts)

    def _write_combined_ply(
        self,
        filepath: str,
        skeleton_vertices: np.ndarray,
        skeleton_faces: np.ndarray,
        skeleton_colors: List[tuple],
        lines_data: List[tuple],
        grf_data: List[tuple] = None
    ) -> None:
        """Write combined skeleton + axes + GRF PLY file with vertex colors."""
        # Collect all vertices, faces, and colors
        all_vertices = []
        all_faces = []
        all_colors = []
        vertex_offset = 0

        # Skeleton mesh
        for i, v in enumerate(skeleton_vertices):
            all_vertices.append(v)
            all_colors.append(skeleton_colors[i])
        for face in skeleton_faces:
            all_faces.append([face[0], face[1], face[2]])
        vertex_offset = len(skeleton_vertices)

        # Axis lines and spheres
        for line_verts, line_faces, line_color in lines_data:
            if len(line_verts) == 0:
                continue
            for v in line_verts:
                all_vertices.append(v)
                all_colors.append(line_color)
            for face in line_faces:
                all_faces.append([face[0] + vertex_offset, face[1] + vertex_offset, face[2] + vertex_offset])
            vertex_offset += len(line_verts)

        # GRF arrows
        if grf_data:
            for grf_verts, grf_faces, grf_color in grf_data:
                if len(grf_verts) == 0:
                    continue
                for v in grf_verts:
                    all_vertices.append(v)
                    all_colors.append(grf_color)
                for face in grf_faces:
                    all_faces.append([face[0] + vertex_offset, face[1] + vertex_offset, face[2] + vertex_offset])
                vertex_offset += len(grf_verts)

        # Write PLY file
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {len(all_faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Write vertices with colors (0-255 range)
            for v, c in zip(all_vertices, all_colors):
                r = int(c[0] * 255)
                g = int(c[1] * 255)
                b = int(c[2] * 255)
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r} {g} {b}\n")

            # Write faces
            for face in all_faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def _write_force_summary(
        self,
        filepath: str,
        frame_name: str,
        joint_torques: Dict[str, float],
        joint_tau_vectors: Dict[str, np.ndarray]
    ) -> None:
        """Write force summary text file."""
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"{frame_name} - Joint Torque Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write("Skeleton: dark purple gradient by joint torque magnitude\n")
            f.write("Axes: X=Red, Y=Blue, Z=Yellow (with endpoint balls)\n\n")

            sorted_joints = sorted(joint_torques.items(), key=lambda x: x[1], reverse=True)
            for joint_name, torque in sorted_joints:
                color = self.colormap(torque, self.max_torque)
                tau_vec = joint_tau_vectors.get(joint_name, np.array([]))
                tau_str = ", ".join([f"{t:.2f}" for t in tau_vec])
                f.write(f"  {joint_name:20s}: {torque:8.2f} Nm  tau=[{tau_str}]  RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})\n")
