"""
Mesh utility functions for creating and reading OBJ meshes.
"""

import numpy as np
from typing import Tuple, List


def create_line_mesh(
    start: np.ndarray,
    end: np.ndarray,
    radius: float = 0.004,
    segments: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a thin cylinder mesh representing a line.

    Args:
        start: Start point (3D)
        end: End point (3D)
        radius: Cylinder radius
        segments: Number of segments around cylinder

    Returns:
        Tuple of (vertices, faces) as numpy arrays
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return np.array([]), np.array([])

    direction = direction / length

    # Find perpendicular vectors
    if abs(direction[1]) < 0.9:
        perp1 = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    else:
        perp1 = np.cross(direction, np.array([1.0, 0.0, 0.0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    vertices = []

    # Bottom circle
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(start + offset)

    # Top circle
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(end + offset)

    # Center points for caps
    vertices.append(start)
    vertices.append(end)

    vertices = np.array(vertices)

    # Generate faces
    faces = []

    # Side faces
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([i, i_next, segments + i_next])
        faces.append([i, segments + i_next, segments + i])

    # Bottom cap
    bottom_center = 2 * segments
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([bottom_center, i_next, i])

    # Top cap
    top_center = 2 * segments + 1
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([top_center, segments + i, segments + i_next])

    faces = np.array(faces)

    return vertices, faces


def create_sphere_mesh(
    center: np.ndarray,
    radius: float = 0.012,
    segments: int = 8,
    rings: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a sphere mesh at the given center point.

    Args:
        center: Center point (3D)
        radius: Sphere radius
        segments: Number of longitudinal segments
        rings: Number of latitudinal rings

    Returns:
        Tuple of (vertices, faces) as numpy arrays
    """
    center = np.array(center, dtype=float)
    vertices = []

    for i in range(rings + 1):
        phi = np.pi * i / rings
        for j in range(segments):
            theta = 2 * np.pi * j / segments
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)
            vertices.append(center + np.array([x, y, z]))

    vertices = np.array(vertices)

    faces = []
    for i in range(rings):
        for j in range(segments):
            j_next = (j + 1) % segments
            v1 = i * segments + j
            v2 = i * segments + j_next
            v3 = (i + 1) * segments + j_next
            v4 = (i + 1) * segments + j
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    faces = np.array(faces)

    return vertices, faces


def create_cone_mesh(
    base_center: np.ndarray,
    tip: np.ndarray,
    radius: float = 0.015,
    segments: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a cone mesh (arrowhead) pointing from base to tip.

    Args:
        base_center: Center of the cone base (3D)
        tip: Tip point of the cone (3D)
        radius: Radius of the cone base
        segments: Number of segments around the cone

    Returns:
        Tuple of (vertices, faces) as numpy arrays
    """
    base_center = np.array(base_center, dtype=float)
    tip = np.array(tip, dtype=float)

    direction = tip - base_center
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return np.array([]), np.array([])

    direction = direction / length

    # Find perpendicular vectors
    if abs(direction[1]) < 0.9:
        perp1 = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    else:
        perp1 = np.cross(direction, np.array([1.0, 0.0, 0.0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    vertices = []

    # Base circle vertices
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(base_center + offset)

    # Tip vertex
    vertices.append(tip)
    # Base center vertex (for bottom cap)
    vertices.append(base_center)

    vertices = np.array(vertices)

    faces = []

    # Side faces (triangles from base to tip)
    tip_idx = segments
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([i, i_next, tip_idx])

    # Bottom cap
    base_center_idx = segments + 1
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([base_center_idx, i_next, i])

    faces = np.array(faces)

    return vertices, faces


def create_arrow_mesh(
    start: np.ndarray,
    end: np.ndarray,
    shaft_radius: float = 0.004,
    head_radius: float = 0.012,
    head_length_ratio: float = 0.25,
    segments: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a complete arrow mesh with shaft (cylinder) and head (cone).

    Args:
        start: Start point of the arrow (3D)
        end: End point (tip) of the arrow (3D)
        shaft_radius: Radius of the arrow shaft
        head_radius: Radius of the arrowhead base
        head_length_ratio: Ratio of head length to total arrow length
        segments: Number of segments around the arrow

    Returns:
        Tuple of (shaft_verts, shaft_faces, head_verts, head_faces)
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    direction = end - start
    total_length = np.linalg.norm(direction)

    if total_length < 1e-6:
        return np.array([]), np.array([]), np.array([]), np.array([])

    direction_normalized = direction / total_length

    # Calculate shaft end and head start
    head_length = total_length * head_length_ratio
    shaft_end = end - direction_normalized * head_length

    # Create shaft (cylinder)
    shaft_verts, shaft_faces = create_line_mesh(
        start, shaft_end, radius=shaft_radius, segments=segments
    )

    # Create head (cone)
    head_verts, head_faces = create_cone_mesh(
        shaft_end, end, radius=head_radius, segments=segments
    )

    return shaft_verts, shaft_faces, head_verts, head_faces


def read_obj_mesh(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read OBJ file and return vertices and faces.

    Args:
        filepath: Path to OBJ file

    Returns:
        Tuple of (vertices, faces) as numpy arrays
    """
    vertices = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1
                    face.append(idx)
                if len(face) >= 3:
                    faces.append(face[:3])

    return np.array(vertices), np.array(faces) if faces else np.array([])


def write_obj_with_color(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple
) -> None:
    """
    Write OBJ file with uniform vertex colors.

    Args:
        filepath: Output path
        vertices: Vertex array (Nx3)
        faces: Face array (Mx3)
        color: RGB color tuple (0-1 range)
    """
    with open(filepath, 'w') as f:
        f.write(f"# OBJ with vertex colors\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {color[0]:.4f} {color[1]:.4f} {color[2]:.4f}\n")

        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def write_obj_with_vertex_colors(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: List[tuple]
) -> None:
    """
    Write OBJ file with per-vertex colors.

    Args:
        filepath: Output path
        vertices: Vertex array (Nx3)
        faces: Face array (Mx3)
        colors: List of RGB color tuples (0-1 range), one per vertex
    """
    with open(filepath, 'w') as f:
        f.write(f"# OBJ with per-vertex colors\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")

        for i, v in enumerate(vertices):
            c = colors[i]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")

        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
