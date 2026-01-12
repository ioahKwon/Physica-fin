"""
Main visualizer class for SKEL force visualization.
"""

import os
import json
import glob
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple

from .colormaps import torque_to_color_plasma, AXIS_COLORS

# GRF arrow color (bright red)
GRF_COLOR = (1.0, 0.2, 0.2)
from .mesh_utils import (
    create_line_mesh,
    create_sphere_mesh,
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
        grf_scale: float = 0.001,
        grf_radius: float = 0.008,
        grf_color: tuple = GRF_COLOR,
        unit_arrow_length: float = 0.05,
        mesh_override_dir: Optional[str] = None,
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
            grf_color: RGB color for GRF arrows
            unit_arrow_length: Fixed arrow length for torque vectors (m). If >0, arrows are unit vectors.
            mesh_override_dir: Optional directory containing corrected skeleton/body meshes
        """
        self.input_base = input_base
        self.mesh_override_dir = mesh_override_dir
        self.output_dir = output_dir
        self.colormap = colormap
        self.max_torque = max_torque
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
        self.grf_color = grf_color
        self.unit_arrow_length = unit_arrow_length

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

        # Create 3-axis lines with endpoint spheres
        lines_data = self._create_axis_lines(
            joint_positions, joint_tau_vectors
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

    def _compute_vertex_colors(
        self,
        vertices: np.ndarray,
        joint_positions: Dict[str, np.ndarray],
        joint_torques: Dict[str, float],
        faces: np.ndarray = None
    ) -> List[tuple]:
        """
        Compute vertex colors based on joint torque.

        If LBS weights are loaded, uses proper bone segmentation (each bone colored
        by its controlling joint's torque). Otherwise falls back to nearest-joint.
        """
        if len(joint_positions) == 0:
            return [(0.8, 0.8, 0.8)] * len(vertices)

        # Use LBS-based coloring if available
        if self.template_dominant_joints is not None and self.use_lbs_coloring:
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
        Compute vertex colors using LBS weights for proper bone segmentation.

        Each vertex is colored based on its dominant joint (from skinning weights).
        This ensures each bone segment is uniformly colored by its controlling joint.

        Handles mesh with deduplicated vertices by using face-based mapping.
        """
        num_vertices = len(vertices)
        num_template_vertices = len(self.template_dominant_joints)

        # Get dominant joints for this mesh
        if num_vertices == num_template_vertices:
            # Direct correspondence - use template dominant joints
            dominant_joints = self.template_dominant_joints
        else:
            # Mesh has different vertex count - need to compute mapping
            # Use face count as cache key (more specific than vertex count)
            cache_key = (num_vertices, len(faces) if faces is not None else 0)
            if cache_key in self._mesh_dominant_joints_cache:
                dominant_joints = self._mesh_dominant_joints_cache[cache_key]
            else:
                print(f"Computing vertex mapping for {num_vertices} vertices...")
                dominant_joints = get_dominant_joints_for_mesh(
                    vertices, self.skel_model_path, self.gender, mesh_faces=faces
                )
                self._mesh_dominant_joints_cache[cache_key] = dominant_joints
                print(f"Vertex mapping computed and cached.")

        # Build mapping from SKEL joint index to torque value
        # joint_torques uses AddBiomechanics names, need to map to SKEL indices
        # Note: Some AddB joints map to multiple SKEL joints (e.g., 'back' -> lumbar, thorax, head)
        skel_idx_to_torque = {}
        for addb_name, torque in joint_torques.items():
            skel_indices = ADDB_TO_SKEL_JOINT_MAP.get(addb_name)
            if skel_indices is not None:
                for skel_idx in skel_indices:
                    # If multiple AddB joints map to same SKEL joint, take max torque
                    if skel_idx not in skel_idx_to_torque or torque > skel_idx_to_torque[skel_idx]:
                        skel_idx_to_torque[skel_idx] = torque

        # Apply parent joint fallback for joints without torque data
        # This handles cases where AddBiomechanics doesn't provide certain joint torques
        for addb_name, parent_addb_name in ADDB_PARENT_FALLBACK.items():
            skel_indices = ADDB_TO_SKEL_JOINT_MAP.get(addb_name)
            if skel_indices is None:
                continue
            # Check if any of these SKEL joints are missing torque data
            for skel_idx in skel_indices:
                if skel_idx not in skel_idx_to_torque or skel_idx_to_torque[skel_idx] == 0:
                    # Try to get parent's torque (with recursive fallback)
                    parent_name = parent_addb_name
                    parent_torque = joint_torques.get(parent_name, 0)
                    # Follow fallback chain if parent also missing
                    while parent_torque == 0 and parent_name in ADDB_PARENT_FALLBACK:
                        parent_name = ADDB_PARENT_FALLBACK[parent_name]
                        parent_torque = joint_torques.get(parent_name, 0)
                    if parent_torque > 0:
                        skel_idx_to_torque[skel_idx] = parent_torque

        colors = []
        for v_idx in range(num_vertices):
            if v_idx < len(dominant_joints):
                joint_idx = dominant_joints[v_idx]
                torque = skel_idx_to_torque.get(joint_idx, 0)
            else:
                # Fallback for vertices beyond LBS weight coverage
                torque = 0

            color = self.colormap(torque, self.max_torque)
            colors.append(color)

        return colors

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

    def _create_axis_lines(
        self,
        joint_positions: Dict[str, np.ndarray],
        joint_tau_vectors: Dict[str, np.ndarray]
    ) -> List[tuple]:
        """
        Create 3-axis lines with endpoint spheres for each joint.

        For 1-DOF joints (knee, ankle, elbow), uses the correct rotation axis
        from JOINT_ROTATION_AXES instead of defaulting to Y-axis.

        Arrow length is fixed (unit vector) when unit_arrow_length > 0,
        otherwise scales with magnitude.
        """
        lines_data = []

        axes_colors = [
            self.axis_colors['x'],
            self.axis_colors['y'],
            self.axis_colors['z'],
        ]

        axes_directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        for joint_name, pos in joint_positions.items():
            tau_vec = joint_tau_vectors.get(joint_name, np.array([]))

            if len(tau_vec) < 1:
                continue

            # Handle 1-DOF joints with correct rotation axis
            if len(tau_vec) == 1:
                rotation_axis = get_rotation_axis_for_joint(joint_name)
                if rotation_axis is not None:
                    # Use correct rotation axis from OpenSim model definition
                    axis_idx = get_axis_index(rotation_axis)
                    magnitude = abs(tau_vec[0])

                    if magnitude < self.torque_threshold:
                        continue

                    # Determine direction based on torque sign
                    direction = rotation_axis if tau_vec[0] > 0 else -rotation_axis

                    # Use unit arrow length or scale with magnitude
                    if self.unit_arrow_length > 0:
                        line_length = self.unit_arrow_length
                    else:
                        line_length = magnitude * self.line_scale

                    end_pos = pos + direction * line_length

                    # Create line cylinder
                    line_verts, line_faces = create_line_mesh(
                        start=pos,
                        end=end_pos,
                        radius=self.line_radius,
                        segments=6
                    )

                    if len(line_verts) > 0:
                        lines_data.append((line_verts, line_faces, axes_colors[axis_idx]))

                    # Create endpoint sphere
                    sphere_verts, sphere_faces = create_sphere_mesh(
                        center=end_pos,
                        radius=self.sphere_radius,
                        segments=8,
                        rings=6
                    )

                    if len(sphere_verts) > 0:
                        lines_data.append((sphere_verts, sphere_faces, axes_colors[axis_idx]))

                    continue  # Done with this joint

                # Fallback: unknown 1-DOF joint, default to X-axis
                tau_3d = np.array([tau_vec[0], 0, 0])
            elif len(tau_vec) == 2:
                tau_3d = np.array([tau_vec[0], tau_vec[1], 0])
            else:
                tau_3d = tau_vec[:3]

            # Handle multi-DOF joints (3-DOF: hip, shoulder, back, etc.)
            for axis_idx in range(3):
                component_magnitude = abs(tau_3d[axis_idx])

                if component_magnitude < self.torque_threshold:
                    continue

                # Use unit arrow length or scale with magnitude
                if self.unit_arrow_length > 0:
                    line_length = self.unit_arrow_length
                else:
                    line_length = component_magnitude * self.line_scale

                direction = axes_directions[axis_idx] if tau_3d[axis_idx] > 0 else -axes_directions[axis_idx]
                end_pos = pos + direction * line_length

                # Create line cylinder
                line_verts, line_faces = create_line_mesh(
                    start=pos,
                    end=end_pos,
                    radius=self.line_radius,
                    segments=6
                )

                if len(line_verts) > 0:
                    lines_data.append((line_verts, line_faces, axes_colors[axis_idx]))

                # Create endpoint sphere
                sphere_verts, sphere_faces = create_sphere_mesh(
                    center=end_pos,
                    radius=self.sphere_radius,
                    segments=8,
                    rings=6
                )

                if len(sphere_verts) > 0:
                    lines_data.append((sphere_verts, sphere_faces, axes_colors[axis_idx]))

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

            # Create arrow shaft (cylinder)
            shaft_verts, shaft_faces = create_line_mesh(
                start=cop,
                end=end_pos,
                radius=self.grf_radius,
                segments=8
            )

            if len(shaft_verts) > 0:
                grf_data.append((shaft_verts, shaft_faces, self.grf_color))

            # Create arrow head (larger sphere at the end)
            head_verts, head_faces = create_sphere_mesh(
                center=end_pos,
                radius=self.grf_radius * 2.0,
                segments=10,
                rings=8
            )

            if len(head_verts) > 0:
                grf_data.append((head_verts, head_faces, self.grf_color))

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
            f.write("# Skeleton: colored by joint torque magnitude\n")
            f.write("# Axes: X=Pink, Y=Neon Green, Z=Cyan (with endpoint balls)\n")
            f.write("# GRF: Red arrows from CoP in force direction\n")

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
            f.write("Skeleton: colored by joint torque magnitude\n")
            f.write("Axes: X=Pink, Y=Neon Green, Z=Cyan (with endpoint balls)\n\n")

            sorted_joints = sorted(joint_torques.items(), key=lambda x: x[1], reverse=True)
            for joint_name, torque in sorted_joints:
                color = self.colormap(torque, self.max_torque)
                tau_vec = joint_tau_vectors.get(joint_name, np.array([]))
                tau_str = ", ".join([f"{t:.2f}" for t in tau_vec])
                f.write(f"  {joint_name:20s}: {torque:8.2f} Nm  tau=[{tau_str}]  RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})\n")
