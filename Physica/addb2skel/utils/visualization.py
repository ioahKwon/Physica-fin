"""
Visualization utilities for creating OBJ meshes from joints and skeletons.

Includes:
- Geometric primitives (sphere, cylinder) for OBJ export
- Combined mesh + marker OBJ for MeshLab/Blender viewing
- Multi-view PNG rendering via matplotlib
- pyrender-based frame rendering for video export
"""

import os
from typing import List, Tuple, Optional, Dict
import numpy as np


def create_sphere(
    center: np.ndarray,
    radius: float = 0.02,
    n_segments: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sphere vertices and faces.

    Args:
        center: Sphere center [3].
        radius: Sphere radius.
        n_segments: Number of segments (resolution).

    Returns:
        vertices: [V, 3]
        faces: [F, 3] (0-indexed)
    """
    vertices = []
    for i in range(n_segments + 1):
        lat = np.pi * i / n_segments - np.pi / 2
        for j in range(n_segments):
            lon = 2 * np.pi * j / n_segments
            x = radius * np.cos(lat) * np.cos(lon) + center[0]
            y = radius * np.cos(lat) * np.sin(lon) + center[1]
            z = radius * np.sin(lat) + center[2]
            vertices.append([x, y, z])

    faces = []
    for i in range(n_segments):
        for j in range(n_segments):
            p1 = i * n_segments + j
            p2 = i * n_segments + (j + 1) % n_segments
            p3 = (i + 1) * n_segments + (j + 1) % n_segments
            p4 = (i + 1) * n_segments + j
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    return np.array(vertices), np.array(faces)


def create_cylinder(
    start: np.ndarray,
    end: np.ndarray,
    radius: float = 0.008,
    n_segments: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create cylinder between two points.

    Args:
        start: Start point [3].
        end: End point [3].
        radius: Cylinder radius.
        n_segments: Number of segments (resolution).

    Returns:
        vertices: [V, 3]
        faces: [F, 3] (0-indexed)
    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    direction = direction / length

    # Find perpendicular vectors
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(direction, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    # Create vertices
    vertices = []
    for t, center in enumerate([start, end]):
        for i in range(n_segments):
            angle = 2 * np.pi * i / n_segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(center + offset)

    # Create side faces
    faces = []
    for i in range(n_segments):
        p1, p2 = i, (i + 1) % n_segments
        p3, p4 = n_segments + (i + 1) % n_segments, n_segments + i
        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])

    # Create caps
    start_center_idx = len(vertices)
    vertices.append(start)
    for i in range(n_segments):
        faces.append([start_center_idx, (i + 1) % n_segments, i])

    end_center_idx = len(vertices)
    vertices.append(end)
    for i in range(n_segments):
        faces.append([end_center_idx, n_segments + i, n_segments + (i + 1) % n_segments])

    return np.array(vertices), np.array(faces)


def create_joint_spheres(
    joints: np.ndarray,
    radius: float = 0.015,
    n_segments: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spheres for all joints.

    Args:
        joints: Joint positions [J, 3].
        radius: Sphere radius.
        n_segments: Resolution.

    Returns:
        vertices: Combined vertices [V, 3]
        faces: Combined faces [F, 3]
    """
    all_verts = []
    all_faces = []
    offset = 0

    for j in joints:
        v, f = create_sphere(j, radius, n_segments)
        all_verts.append(v)
        all_faces.append(f + offset)
        offset += len(v)

    if all_verts:
        return np.vstack(all_verts), np.vstack(all_faces)
    return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)


def create_skeleton_bones(
    joints: np.ndarray,
    parents: List[int],
    radius: float = 0.008,
    n_segments: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create bone cylinders for skeleton.

    Args:
        joints: Joint positions [J, 3].
        parents: Parent index for each joint (-1 for root).
        radius: Cylinder radius.
        n_segments: Resolution.

    Returns:
        vertices: Combined vertices [V, 3]
        faces: Combined faces [F, 3]
    """
    all_verts = []
    all_faces = []
    offset = 0

    for i, p in enumerate(parents):
        if p >= 0:
            v, f = create_cylinder(joints[p], joints[i], radius, n_segments)
            if len(v) > 0:
                all_verts.append(v)
                all_faces.append(f + offset)
                offset += len(v)

    if all_verts:
        return np.vstack(all_verts), np.vstack(all_faces)
    return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)


def joints_to_obj(
    joints: np.ndarray,
    output_path: str,
    parents: Optional[List[int]] = None,
    joint_radius: float = 0.015,
    bone_radius: float = 0.008,
    include_bones: bool = True,
):
    """
    Save joints (and optionally bones) as OBJ file.

    Args:
        joints: Joint positions [J, 3].
        output_path: Output OBJ path.
        parents: Parent indices for bone connections.
        joint_radius: Radius for joint spheres.
        bone_radius: Radius for bone cylinders.
        include_bones: Whether to include bone connections.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create joint spheres
    joint_verts, joint_faces = create_joint_spheres(joints, joint_radius)

    # Create bones if requested
    if include_bones and parents is not None:
        bone_verts, bone_faces = create_skeleton_bones(joints, parents, bone_radius)
        if len(bone_verts) > 0:
            # Offset bone faces
            bone_faces = bone_faces + len(joint_verts)
            all_verts = np.vstack([joint_verts, bone_verts])
            all_faces = np.vstack([joint_faces, bone_faces])
        else:
            all_verts = joint_verts
            all_faces = joint_faces
    else:
        all_verts = joint_verts
        all_faces = joint_faces

    # Write OBJ
    with open(output_path, 'w') as f:
        for v in all_verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in all_faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def save_comparison_obj(
    addb_joints: np.ndarray,
    skel_joints: np.ndarray,
    output_dir: str,
    addb_parents: Optional[List[int]] = None,
    skel_parents: Optional[List[int]] = None,
):
    """
    Save AddB and SKEL joints as separate OBJ files for comparison.

    Args:
        addb_joints: AddB joint positions [J_addb, 3].
        skel_joints: SKEL joint positions [J_skel, 3].
        output_dir: Output directory.
        addb_parents: AddB parent indices.
        skel_parents: SKEL parent indices.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    joints_to_obj(
        addb_joints,
        os.path.join(output_dir, 'addb_skeleton.obj'),
        parents=addb_parents,
        joint_radius=0.02,
        bone_radius=0.01,
    )

    joints_to_obj(
        skel_joints,
        os.path.join(output_dir, 'skel_skeleton.obj'),
        parents=skel_parents,
        joint_radius=0.015,
        bone_radius=0.008,
    )


def create_combined_mesh_markers_obj(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    virtual_markers: np.ndarray,
    output_path: str,
    marker_names: Optional[List[str]] = None,
    observed_markers: Optional[Dict[str, np.ndarray]] = None,
    marker_radius: float = 0.008,
    mesh_color: Tuple[float, float, float] = (0.85, 0.75, 0.65),
    virtual_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    observed_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    n_segments: int = 8,
):
    """
    Create a single OBJ with body mesh + virtual marker spheres + optional observed markers.

    Args:
        mesh_vertices: Body mesh vertices [V, 3].
        mesh_faces: Body mesh faces [F, 3] (0-indexed).
        virtual_markers: Virtual marker positions [K, 3].
        output_path: Output OBJ path.
        marker_names: Optional marker names for comments.
        observed_markers: Optional dict of observed marker positions {name: [3]}.
        marker_radius: Radius for marker spheres.
        mesh_color: RGB color for mesh vertices.
        virtual_color: RGB color for virtual marker spheres.
        observed_color: RGB color for observed marker spheres.
        n_segments: Sphere resolution.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Also write separate OBJ files for each component
    base_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    # --- Separate OBJ: body mesh ---
    mesh_obj_path = os.path.join(base_dir, f'{base_name}_mesh.obj')
    with open(mesh_obj_path, 'w') as f:
        f.write("# Body mesh\n")
        f.write(f"# {len(mesh_vertices)} vertices, {len(mesh_faces)} faces\n")
        for v in mesh_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                    f"{mesh_color[0]:.3f} {mesh_color[1]:.3f} {mesh_color[2]:.3f}\n")
        for face in mesh_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # --- Separate OBJ: virtual markers (red) ---
    vm_obj_path = os.path.join(base_dir, f'{base_name}_virtual_markers.obj')
    with open(vm_obj_path, 'w') as f:
        f.write("# Virtual markers (SKEL fitted)\n")
        f.write(f"# {len(virtual_markers)} markers\n")
        v_offset = 0
        for k, pos in enumerate(virtual_markers):
            if marker_names and k < len(marker_names):
                f.write(f"# Marker {k}: {marker_names[k]}\n")
            sphere_v, sphere_f = create_sphere(pos, marker_radius, n_segments)
            for sv in sphere_v:
                f.write(f"v {sv[0]:.6f} {sv[1]:.6f} {sv[2]:.6f} "
                        f"{virtual_color[0]:.3f} {virtual_color[1]:.3f} {virtual_color[2]:.3f}\n")
            for sf in sphere_f:
                f.write(f"f {sf[0]+v_offset+1} {sf[1]+v_offset+1} {sf[2]+v_offset+1}\n")
            v_offset += len(sphere_v)

    # --- Separate OBJ: observed markers (green) ---
    if observed_markers and marker_names:
        obs_obj_path = os.path.join(base_dir, f'{base_name}_observed_markers.obj')
        with open(obs_obj_path, 'w') as f:
            f.write("# Observed GT markers\n")
            f.write(f"# {len(observed_markers)} markers\n")
            v_offset = 0
            for name, obs_pos in observed_markers.items():
                obs_pos = np.array(obs_pos, dtype=np.float64)
                f.write(f"# {name}\n")
                sphere_v, sphere_f = create_sphere(obs_pos, marker_radius, n_segments)
                for sv in sphere_v:
                    f.write(f"v {sv[0]:.6f} {sv[1]:.6f} {sv[2]:.6f} "
                            f"{observed_color[0]:.3f} {observed_color[1]:.3f} {observed_color[2]:.3f}\n")
                for sf in sphere_f:
                    f.write(f"f {sf[0]+v_offset+1} {sf[1]+v_offset+1} {sf[2]+v_offset+1}\n")
                v_offset += len(sphere_v)
        print(f"Saved separate OBJs: {base_name}_mesh.obj, _virtual_markers.obj, _observed_markers.obj")
    else:
        print(f"Saved separate OBJs: {base_name}_mesh.obj, _virtual_markers.obj")

    # --- Combined OBJ with groups ---
    with open(output_path, 'w') as f:
        f.write("# Combined mesh + virtual markers + observed markers\n")
        f.write(f"# Mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces\n")
        f.write(f"# Virtual markers: {len(virtual_markers)} (red spheres)\n")
        if observed_markers:
            f.write(f"# Observed markers: {len(observed_markers)} (green spheres)\n")

        # 1. Body mesh group
        f.write(f"\ng body_mesh\n")
        for v in mesh_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                    f"{mesh_color[0]:.3f} {mesh_color[1]:.3f} {mesh_color[2]:.3f}\n")
        for face in mesh_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        v_offset = len(mesh_vertices)

        # 2. Virtual marker spheres (red) group
        f.write(f"\ng virtual_markers\n")
        for k, pos in enumerate(virtual_markers):
            if marker_names and k < len(marker_names):
                f.write(f"# Marker {k}: {marker_names[k]}\n")
            sphere_v, sphere_f = create_sphere(pos, marker_radius, n_segments)
            for sv in sphere_v:
                f.write(f"v {sv[0]:.6f} {sv[1]:.6f} {sv[2]:.6f} "
                        f"{virtual_color[0]:.3f} {virtual_color[1]:.3f} {virtual_color[2]:.3f}\n")
            for sf in sphere_f:
                f.write(f"f {sf[0]+v_offset+1} {sf[1]+v_offset+1} {sf[2]+v_offset+1}\n")
            v_offset += len(sphere_v)

        # 3. Observed marker spheres (green) group — ALL GT markers
        if observed_markers:
            f.write(f"\ng observed_markers\n")
            vm_center_indices = []
            obs_center_indices = []

            # Build name→index lookup for connecting lines
            vm_name_to_idx = {}
            if marker_names:
                vm_name_to_idx = {n: i for i, n in enumerate(marker_names)}

            for name, obs_pos in observed_markers.items():
                obs_pos = np.array(obs_pos, dtype=np.float64)
                sphere_v, sphere_f = create_sphere(obs_pos, marker_radius, n_segments)
                f.write(f"# Observed: {name}\n")
                for sv in sphere_v:
                    f.write(f"v {sv[0]:.6f} {sv[1]:.6f} {sv[2]:.6f} "
                            f"{observed_color[0]:.3f} {observed_color[1]:.3f} {observed_color[2]:.3f}\n")
                for sf in sphere_f:
                    f.write(f"f {sf[0]+v_offset+1} {sf[1]+v_offset+1} {sf[2]+v_offset+1}\n")
                v_offset += len(sphere_v)

                # Connecting line for matched markers
                if name in vm_name_to_idx:
                    vm_center_indices.append(virtual_markers[vm_name_to_idx[name]])
                    obs_center_indices.append(obs_pos)

            # Write connecting lines as edges
            if vm_center_indices:
                f.write(f"\ng connecting_lines\n")
                for vm_pos, obs_pos in zip(vm_center_indices, obs_center_indices):
                    f.write(f"v {vm_pos[0]:.6f} {vm_pos[1]:.6f} {vm_pos[2]:.6f} "
                            f"0.0 0.0 1.0\n")
                    f.write(f"v {obs_pos[0]:.6f} {obs_pos[1]:.6f} {obs_pos[2]:.6f} "
                            f"0.0 0.0 1.0\n")
                    f.write(f"l {v_offset+1} {v_offset+2}\n")
                    v_offset += 2

    print(f"Saved combined OBJ: {output_path}")


def render_multiview_png(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    virtual_markers: np.ndarray,
    output_path: str,
    marker_names: Optional[List[str]] = None,
    observed_markers: Optional[Dict[str, np.ndarray]] = None,
    marker_errors_mm: Optional[Dict[str, float]] = None,
    worst_n: int = 5,
    face_subsample: int = 4,
    title: str = "",
):
    """
    Render 3-view (front/side/top) PNG of mesh with markers using matplotlib.

    Args:
        mesh_vertices: Body mesh vertices [V, 3].
        mesh_faces: Body mesh faces [F, 3] (0-indexed).
        virtual_markers: Virtual marker positions [K, 3].
        output_path: Output PNG path.
        marker_names: Optional marker names.
        observed_markers: Optional dict of observed marker positions.
        marker_errors_mm: Optional per-marker errors for annotation.
        worst_n: Number of worst-error markers to annotate.
        face_subsample: Subsample faces by this factor (for faster rendering).
        title: Plot title.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Subsample faces for faster rendering
    sub_faces = mesh_faces[::face_subsample]

    # Compute mesh center and bounds
    center = mesh_vertices.mean(axis=0)
    extent = mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)
    max_extent = extent.max() * 0.6

    # View configurations: (elevation, azimuth, label)
    views = [
        (10, -90, 'Front'),
        (10, 0, 'Side'),
        (90, -90, 'Top'),
    ]

    fig = plt.figure(figsize=(20, 7), facecolor='white')
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    for vi, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(1, 3, vi + 1, projection='3d',
                             facecolor='#F5F5F5')

        # Draw mesh triangles — more opaque, warm skin color, visible edges
        triangles = mesh_vertices[sub_faces]
        poly = Poly3DCollection(
            triangles, alpha=0.35, facecolor='#E8C4A8',
            edgecolor='#C0A080', linewidth=0.15,
        )
        ax.add_collection3d(poly)

        # Draw virtual markers (red) — larger, outlined
        ax.scatter(
            virtual_markers[:, 0], virtual_markers[:, 1], virtual_markers[:, 2],
            c='#E8530E', s=25, alpha=0.9, edgecolors='#AA3000',
            linewidths=0.3, label='SKEL Virtual', zorder=5,
        )

        # Draw observed markers (green) if available
        if observed_markers and marker_names:
            obs_positions = []
            for name in marker_names:
                if name in observed_markers:
                    obs_positions.append(observed_markers[name])
            # Also add observed markers NOT in marker_names (all GT markers)
            for name, pos in observed_markers.items():
                if name not in (marker_names or []):
                    obs_positions.append(np.array(pos, dtype=np.float64))
            if obs_positions:
                obs_arr = np.array(obs_positions)
                ax.scatter(
                    obs_arr[:, 0], obs_arr[:, 1], obs_arr[:, 2],
                    c='#1B7FBD', s=25, alpha=0.9, edgecolors='#0D5A8A',
                    linewidths=0.3, label='GT Observed', zorder=6,
                )

        # Annotate worst-N markers
        if marker_errors_mm and marker_names and worst_n > 0:
            sorted_errors = sorted(marker_errors_mm.items(),
                                   key=lambda x: x[1], reverse=True)
            for name, err in sorted_errors[:worst_n]:
                if name in (marker_names if marker_names else []):
                    idx = marker_names.index(name)
                    pos = virtual_markers[idx]
                    ax.text(pos[0], pos[1], pos[2],
                            f"  {name}\n  {err:.0f}mm",
                            fontsize=6, color='darkred', zorder=10)

        ax.set_xlim(center[0] - max_extent, center[0] + max_extent)
        ax.set_ylim(center[1] - max_extent, center[1] + max_extent)
        ax.set_zlim(center[2] - max_extent, center[2] + max_extent)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X', fontsize=8, color='#666')
        ax.set_ylabel('Y', fontsize=8, color='#666')
        ax.set_zlabel('Z', fontsize=8, color='#666')
        ax.tick_params(labelsize=6, colors='#888')
        ax.set_title(label, fontsize=12, fontweight='bold', pad=8)

        if vi == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
                      edgecolor='#CCC')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved multiview PNG: {output_path}")


# ---------------------------------------------------------------------------
#  pyrender-based frame rendering (for video export)
# ---------------------------------------------------------------------------

def _create_raymond_lights():
    """
    Create 3 directional Raymond lights for even illumination.
    Pattern from HSMR/lib/utils/vis/py_renderer/utils.py.
    """
    import pyrender

    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=0.75),
            matrix=matrix,
        ))
    return nodes


def render_frame_pyrender(
    vertices: np.ndarray,
    faces: np.ndarray,
    markers: Optional[np.ndarray] = None,
    observed_markers: Optional[np.ndarray] = None,
    resolution: Tuple[int, int] = (1280, 720),
    mesh_color: Tuple[float, ...] = (1.0, 0.7, 0.7, 0.6),
    marker_color: Tuple[float, ...] = (0.0, 0.8, 0.0, 1.0),
    observed_marker_color: Tuple[float, ...] = (0.2, 0.4, 1.0, 1.0),
    marker_radius: float = 0.008,
    camera_distance: float = 3.0,
    camera_elevation: float = 0.2,
    camera_azimuth: float = 0.0,
    bg_color: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Render a single frame with body mesh + marker spheres using pyrender.

    Args:
        vertices: Body mesh vertices [V, 3].
        faces: Body mesh faces [F, 3] (0-indexed).
        markers: Virtual marker positions [K, 3] (green by default).
        observed_markers: Observed marker positions [M, 3] (blue by default).
        resolution: (width, height).
        mesh_color: RGBA for body mesh (semi-transparent pink).
        marker_color: RGBA for virtual marker spheres.
        observed_marker_color: RGBA for observed marker spheres.
        marker_radius: Radius for marker spheres.
        camera_distance: Distance from mesh center.
        camera_elevation: Elevation angle in radians.
        camera_azimuth: Azimuth angle in radians (0=front).
        bg_color: Background color RGBA.

    Returns:
        RGB image as uint8 array [H, W, 3].
    """
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import trimesh
    import pyrender

    W, H = resolution
    scene = pyrender.Scene(
        bg_color=bg_color,
        ambient_light=(0.3, 0.3, 0.3),
    )

    # --- Body mesh (semi-transparent) ---
    body_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.7,
        alphaMode='BLEND',
        baseColorFactor=mesh_color,
    )
    body_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=body_material)
    scene.add(body_mesh, name='body')

    # --- Marker spheres ---
    def _add_marker_spheres(positions, color, name_prefix):
        if positions is None or len(positions) == 0:
            return
        mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.9,
            alphaMode='OPAQUE',
            baseColorFactor=color,
        )
        # Merge all marker spheres into a single mesh for performance
        all_verts = []
        all_faces = []
        offset = 0
        for pos in positions:
            s = trimesh.creation.uv_sphere(radius=marker_radius, count=[8, 8])
            s.apply_translation(pos)
            all_verts.append(s.vertices)
            all_faces.append(s.faces + offset)
            offset += len(s.vertices)
        merged = trimesh.Trimesh(
            vertices=np.vstack(all_verts),
            faces=np.vstack(all_faces),
            process=False,
        )
        scene.add(pyrender.Mesh.from_trimesh(merged, material=mat), name=name_prefix)

    _add_marker_spheres(markers, marker_color, 'virtual_markers')
    _add_marker_spheres(observed_markers, observed_marker_color, 'observed_markers')

    # --- Camera ---
    center = vertices.mean(axis=0)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 6.0, aspectRatio=W / H)

    # Camera position: orbit around center
    cx = center[0] + camera_distance * np.sin(camera_azimuth) * np.cos(camera_elevation)
    cy = center[1] + camera_distance * np.sin(camera_elevation)
    cz = center[2] + camera_distance * np.cos(camera_azimuth) * np.cos(camera_elevation)
    cam_pos = np.array([cx, cy, cz])

    # Look-at matrix
    forward = center - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = cam_pos
    scene.add(cam, pose=cam_pose)

    # --- Lights ---
    for node in _create_raymond_lights():
        scene.add_node(node)

    # --- Render ---
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    return color[:, :, :3]  # RGB uint8


def render_video_frames(
    all_vertices: np.ndarray,
    faces: np.ndarray,
    all_markers: Optional[np.ndarray] = None,
    all_observed: Optional[np.ndarray] = None,
    resolution: Tuple[int, int] = (1280, 720),
    mesh_color: Tuple[float, ...] = (1.0, 0.7, 0.7, 0.6),
    marker_color: Tuple[float, ...] = (0.0, 0.8, 0.0, 1.0),
    observed_marker_color: Tuple[float, ...] = (0.2, 0.4, 1.0, 1.0),
    marker_radius: float = 0.008,
    camera_distance: float = 3.0,
    camera_elevation: float = 0.2,
    camera_azimuth: float = 0.0,
    frame_texts: Optional[List[str]] = None,
    font_size: int = 24,
) -> List[np.ndarray]:
    """
    Render multiple frames for video.

    Args:
        all_vertices: [F, V, 3] body mesh vertices per frame.
        faces: [Nf, 3] shared face topology.
        all_markers: [F, K, 3] virtual markers per frame (optional).
        all_observed: [F, M, 3] observed markers per frame (optional).
        resolution: (width, height).
        frame_texts: Optional text overlay per frame.
        font_size: Font size for text overlay.

    Returns:
        List of RGB images [H, W, 3] uint8.
    """
    num_frames = len(all_vertices)
    frames = []

    for f_idx in range(num_frames):
        markers = all_markers[f_idx] if all_markers is not None else None
        observed = all_observed[f_idx] if all_observed is not None else None

        img = render_frame_pyrender(
            vertices=all_vertices[f_idx],
            faces=faces,
            markers=markers,
            observed_markers=observed,
            resolution=resolution,
            mesh_color=mesh_color,
            marker_color=marker_color,
            observed_marker_color=observed_marker_color,
            marker_radius=marker_radius,
            camera_distance=camera_distance,
            camera_elevation=camera_elevation,
            camera_azimuth=camera_azimuth,
        )

        # Add text overlay if provided
        if frame_texts and f_idx < len(frame_texts):
            img = add_text_overlay(img, frame_texts[f_idx], font_size=font_size)

        frames.append(img)
        if (f_idx + 1) % 10 == 0 or f_idx == num_frames - 1:
            print(f"  Rendered frame {f_idx + 1}/{num_frames}")

    return frames


def add_text_overlay(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 10),
    font_size: int = 24,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Add text overlay to image using PIL.

    Args:
        image: RGB image [H, W, 3] uint8.
        text: Text to overlay.
        position: (x, y) top-left position.
        font_size: Font size.
        color: RGB text color.

    Returns:
        Image with text overlay [H, W, 3] uint8.
    """
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw with shadow for readability
    shadow_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
    draw.text((position[0] + 1, position[1] + 1), text, fill=shadow_color, font=font)
    draw.text(position, text, fill=color, font=font)

    return np.array(pil_img)


def save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
):
    """
    Save frames as MP4 video using imageio + ffmpeg.

    Args:
        frames: List of RGB images [H, W, 3] uint8.
        output_path: Output MP4 path.
        fps: Frames per second.
    """
    import imageio

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p',
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"Saved video: {output_path} ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
#  Comparison OBJ generation (for MeshLab visualization)
# ---------------------------------------------------------------------------

def _flip_xz(arr: np.ndarray) -> np.ndarray:
    """SKEL internal coords → AddB original coords (or vice versa)."""
    out = arr.copy()
    out[..., 0] *= -1
    out[..., 2] *= -1
    return out


def _write_obj_color(path: str, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    """Write OBJ with per-vertex RGB color."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for i, v in enumerate(verts):
            c = colors[i]
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def _write_obj_with_mtl(path: str, mtl_name: str, verts: np.ndarray, faces: np.ndarray,
                        alpha: float = 0.3, color: Tuple[float, ...] = (0.75, 0.75, 0.75)):
    """Write OBJ + MTL for semi-transparent mesh."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mtl_path = os.path.splitext(path)[0] + '.mtl'
    with open(mtl_path, 'w') as f:
        f.write(f'newmtl {mtl_name}\n')
        f.write(f'Ka 0.15 0.15 0.15\n')
        f.write(f'Kd {color[0]:.2f} {color[1]:.2f} {color[2]:.2f}\n')
        f.write(f'Ks 0.1 0.1 0.1\nNs 10.0\nd {alpha}\nillum 2\n')
    with open(path, 'w') as f:
        f.write(f'mtllib {os.path.basename(mtl_path)}\nusemtl {mtl_name}\n')
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def _make_colored_spheres(positions: np.ndarray, color: List[float],
                          radius: float = 0.008) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create colored sphere meshes for a set of positions."""
    all_v, all_f, all_c = [], [], []
    offset = 0
    for pos in positions:
        if np.any(np.isnan(pos)):
            continue
        v, f = create_sphere(pos, radius=radius, n_segments=6)
        all_v.append(v)
        all_f.append(f + offset)
        all_c.append(np.tile(color, (len(v), 1)))
        offset += len(v)
    if not all_v:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), np.zeros((0, 3))
    return np.vstack(all_v), np.vstack(all_f), np.vstack(all_c)


def save_comparison_objs(
    output_dir: str,
    body_verts: np.ndarray,
    body_faces: np.ndarray,
    skel_joints: np.ndarray,
    parents: List[int],
    gt_joints: Optional[np.ndarray] = None,
    gt_markers: Optional[np.ndarray] = None,
    virtual_markers: Optional[np.ndarray] = None,
    foot_marker_offsets: Optional[Dict[str, np.ndarray]] = None,
    skel_model=None,
    skel_betas=None,
    skel_poses_frame=None,
    skel_trans_frame=None,
):
    """
    Save 7 comparison OBJ files for MeshLab/Blender visualization.

    All outputs are in AddB original coordinate system.

    Standard output (7 files):
      1_mesh.obj              — Body skin mesh (semi-transparent, MTL)
      2_gt_markers.obj        — GT observed markers (orange spheres)
      3_gt_pose.obj           — GT joints + bones (orange spheres + gray sticks)
      4_skel_joints.obj       — SKEL joint spheres only (light-blue)
      5_skel_skeleton.obj     — SKEL joints + bones (light-blue spheres + gray sticks)
      6_skel_virtual_markers.obj — SKEL virtual markers (blue spheres)
      7_skel_skeleton_mesh.obj   — SKEL bone mesh (semi-transparent, MTL)

    Args:
        output_dir: Output directory.
        body_verts: Body mesh vertices [V, 3] in AddB coords.
        body_faces: Body mesh faces [F, 3] (0-indexed).
        skel_joints: SKEL joint positions [J, 3] in AddB coords.
        parents: Parent index list for skeleton.
        gt_joints: GT joint positions [J_gt, 3] in AddB coords.
        gt_markers: GT marker positions [M, 3] in AddB coords.
        virtual_markers: Virtual marker positions [K, 3] in SKEL internal coords.
        skel_model: SKEL model instance (for bone mesh).
        skel_betas: SKEL betas tensor.
        skel_poses_frame: SKEL poses for this frame.
        skel_trans_frame: SKEL trans for this frame (AddB coords).
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)

    GRAY      = [0.75, 0.75, 0.75]
    ORANGE    = [1.0,  0.65, 0.0]   # GT markers/joints
    BLUE      = [0.2,  0.5,  1.0]   # SKEL virtual markers
    JOINT_CLR = [0.4,  0.7,  0.9]   # SKEL joints (light-blue)

    # 1) Body mesh (semi-transparent)
    _write_obj_with_mtl(
        os.path.join(output_dir, '1_mesh.obj'),
        'body_skin', body_verts, body_faces, alpha=0.3,
    )

    # 2) GT markers (orange)
    if gt_markers is not None and len(gt_markers) > 0:
        gtm_v, gtm_f, gtm_c = _make_colored_spheres(gt_markers, ORANGE, radius=0.008)
        _write_obj_color(os.path.join(output_dir, '2_gt_markers.obj'), gtm_v, gtm_f, gtm_c)

    # 3) GT pose — joints + bones (orange)
    if gt_joints is not None and len(gt_joints) > 0:
        gtj_v, gtj_f, gtj_c = _make_colored_spheres(gt_joints, ORANGE, radius=0.012)
        _write_obj_color(os.path.join(output_dir, '3_gt_pose.obj'), gtj_v, gtj_f, gtj_c)

    # 4) SKEL joints only (light-blue spheres)
    jv, jf, jc = _make_colored_spheres(skel_joints, JOINT_CLR, radius=0.010)
    _write_obj_color(os.path.join(output_dir, '4_skel_joints.obj'), jv, jf, jc)

    # 5) SKEL skeleton — joints + bones (light-blue spheres + gray sticks)
    bv, bf = create_skeleton_bones(skel_joints, parents, radius=0.005)
    if len(bv) > 0:
        bc = np.tile(GRAY, (len(bv), 1))
        bf_off = bf + len(jv)
        stick_v = np.vstack([jv, bv])
        stick_f = np.vstack([jf, bf_off])
        stick_c = np.vstack([jc, bc])
    else:
        stick_v, stick_f, stick_c = jv, jf, jc
    _write_obj_color(os.path.join(output_dir, '5_skel_skeleton.obj'), stick_v, stick_f, stick_c)

    # 6) SKEL virtual markers (blue) — flip_xz to AddB coords
    if virtual_markers is not None and len(virtual_markers) > 0:
        vm_addb = _flip_xz(virtual_markers)
        vm_v, vm_f, vm_c = _make_colored_spheres(vm_addb, BLUE, radius=0.008)
        _write_obj_color(os.path.join(output_dir, '6_skel_virtual_markers.obj'), vm_v, vm_f, vm_c)

    # 7) SKEL skeleton mesh (semi-transparent bone mesh)
    if skel_model is not None and skel_betas is not None and skel_poses_frame is not None:
        t_np = skel_trans_frame.cpu().numpy()
        t_skel = t_np.copy()
        t_skel[0] *= -1
        t_skel[2] *= -1
        t_skel_t = torch.tensor(t_skel, dtype=skel_trans_frame.dtype,
                                device=skel_trans_frame.device)
        with torch.no_grad():
            b = skel_betas.unsqueeze(0) if skel_betas.dim() == 1 else skel_betas
            p = skel_poses_frame.unsqueeze(0) if skel_poses_frame.dim() == 1 else skel_poses_frame
            t = t_skel_t.unsqueeze(0) if t_skel_t.dim() == 1 else t_skel_t
            out = skel_model(betas=b, poses=p, trans=t)
            bone_verts = _flip_xz(out.skel_verts[0].cpu().numpy())
            bone_faces = skel_model.skel_f.cpu().numpy()
        _write_obj_with_mtl(
            os.path.join(output_dir, '7_skel_skeleton_mesh.obj'),
            'bone_mesh', bone_verts, bone_faces, alpha=0.5,
            color=(0.95, 0.92, 0.85),
        )
