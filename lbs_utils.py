"""
LBS (Linear Blend Skinning) utilities for SKEL model.

Provides functions to load skinning weights and compute bone-to-vertex mappings
for proper segmentation-based coloring of skeleton mesh.
"""

import os
import pickle
import numpy as np
from typing import Dict, Tuple, Optional


# SKEL 24 joint names (matches skel_weights column order)
SKEL_JOINT_NAMES = [
    'pelvis',      # 0
    'femur_r',     # 1
    'tibia_r',     # 2
    'talus_r',     # 3
    'calcn_r',     # 4
    'toes_r',      # 5
    'femur_l',     # 6
    'tibia_l',     # 7
    'talus_l',     # 8
    'calcn_l',     # 9
    'toes_l',      # 10
    'lumbar',      # 11
    'thorax',      # 12
    'head',        # 13
    'scapula_r',   # 14
    'humerus_r',   # 15
    'ulna_r',      # 16
    'radius_r',    # 17
    'hand_r',      # 18
    'scapula_l',   # 19
    'humerus_l',   # 20
    'ulna_l',      # 21
    'radius_l',    # 22
    'hand_l',      # 23
]

# Mapping from AddBiomechanics joint names to SKEL joint indices
# Some AddB joints map to multiple SKEL joints (use list for those)
ADDB_TO_SKEL_JOINT_MAP = {
    'ground_pelvis': [0],          # pelvis
    'hip_r': [1],                  # femur_r
    'walker_knee_r': [2],          # tibia_r
    'ankle_r': [3],                # talus_r
    'subtalar_r': [4],             # calcn_r
    'mtp_r': [5],                  # toes_r
    'hip_l': [6],                  # femur_l
    'walker_knee_l': [7],          # tibia_l
    'ankle_l': [8],                # talus_l
    'subtalar_l': [9],             # calcn_l
    'mtp_l': [10],                 # toes_l
    'back': [11, 12, 13],          # lumbar + thorax + head (spine torque affects whole upper body)
    'acromial_r': [14, 15],        # scapula_r + humerus_r (shoulder complex)
    'elbow_r': [16],               # ulna_r
    'radioulnar_r': [17],          # radius_r
    'radius_hand_r': [18],         # hand_r
    'acromial_l': [19, 20],        # scapula_l + humerus_l (shoulder complex)
    'elbow_l': [21],               # ulna_l
    'radioulnar_l': [22],          # radius_l
    'radius_hand_l': [23],         # hand_l
}

# AddBiomechanics/OpenSim joint rotation axes
# Maps joint name to rotation axis vector for 1-DOF joints
# Based on OpenSim Rajagopal2015 model definitions
JOINT_ROTATION_AXES = {
    # 1-DOF knee: X-axis rotation (flexion/extension)
    # OpenSim: <axis>1 0 0</axis>, range [0, 120°]
    # tau > 0 = flexion, tau < 0 = extension
    'walker_knee_r': np.array([1.0, 0.0, 0.0]),
    'walker_knee_l': np.array([1.0, 0.0, 0.0]),
    'knee_r': np.array([1.0, 0.0, 0.0]),  # alias
    'knee_l': np.array([1.0, 0.0, 0.0]),

    # 1-DOF ankle: X-axis rotation (dorsiflexion/plantarflexion)
    # OpenSim: <axis>-0.1 -0.0 0.99</axis> (approx X-axis in local frame)
    # tau > 0 = dorsiflexion, tau < 0 = plantarflexion
    'ankle_r': np.array([1.0, 0.0, 0.0]),
    'ankle_l': np.array([1.0, 0.0, 0.0]),

    # 1-DOF subtalar: oblique axis (inversion/eversion)
    # OpenSim: <axis>0.79 0.12 0.6</axis>
    # Simplified to Z-axis for visualization
    'subtalar_r': np.array([0.0, 0.0, 1.0]),
    'subtalar_l': np.array([0.0, 0.0, 1.0]),

    # 1-DOF elbow: X-axis rotation (flexion/extension)
    # OpenSim: <axis>0.23 0.97 -0.08</axis> (approx Y-axis in local frame)
    # Using X for sagittal plane visualization
    'elbow_r': np.array([1.0, 0.0, 0.0]),
    'elbow_l': np.array([1.0, 0.0, 0.0]),
}

# SKEL joint names (24 joints)
SKEL_JOINT_NAMES = [
    'pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r',
    'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l',
    'lumbar_body', 'thorax', 'head',
    'scapula_r', 'humerus_r', 'ulna_r', 'radius_r', 'hand_r',
    'scapula_l', 'humerus_l', 'ulna_l', 'radius_l', 'hand_l',
]

# AddBiomechanics joint name -> SKEL joint index for world rotation lookup
# Used to transform local torques to global coordinates
ADDB_TO_SKEL_ROTATION_MAP = {
    'ground_pelvis': 0,    # pelvis
    'hip_r': 1,            # femur_r
    'walker_knee_r': 2,    # tibia_r
    'ankle_r': 3,          # talus_r
    'subtalar_r': 4,       # calcn_r
    'mtp_r': 5,            # toes_r
    'hip_l': 6,            # femur_l
    'walker_knee_l': 7,    # tibia_l
    'ankle_l': 8,          # talus_l
    'subtalar_l': 9,       # calcn_l
    'mtp_l': 10,           # toes_l
    'back': 11,            # lumbar_body
    'acromial_r': 15,      # humerus_r (shoulder)
    'elbow_r': 16,         # ulna_r
    'radioulnar_r': 17,    # radius_r
    'radius_hand_r': 18,   # hand_r
    'acromial_l': 20,      # humerus_l (shoulder)
    'elbow_l': 21,         # ulna_l
    'radioulnar_l': 22,    # radius_l
    'radius_hand_l': 23,   # hand_l
}


def get_rotation_axis_for_joint(joint_name: str) -> Optional[np.ndarray]:
    """
    Get the rotation axis for a 1-DOF joint.

    Args:
        joint_name: AddBiomechanics joint name

    Returns:
        3D unit vector of rotation axis, or None if not a 1-DOF joint
    """
    return JOINT_ROTATION_AXES.get(joint_name)


def get_axis_index(axis_vector: np.ndarray) -> int:
    """
    Get the primary axis index (0=X, 1=Y, 2=Z) from an axis vector.

    Args:
        axis_vector: 3D unit vector

    Returns:
        Index of the dominant axis component (0, 1, or 2)
    """
    return int(np.argmax(np.abs(axis_vector)))


# Parent joint fallback for torque inheritance
# When a joint has no torque data, inherit from parent joint
# Format: AddB joint name -> parent AddB joint name
ADDB_PARENT_FALLBACK = {
    'mtp_r': 'subtalar_r',         # toes -> ankle
    'mtp_l': 'subtalar_l',         # toes -> ankle
    'radioulnar_r': 'elbow_r',     # forearm rotation -> elbow
    'radioulnar_l': 'elbow_l',     # forearm rotation -> elbow
    'radius_hand_r': 'elbow_r',    # wrist -> elbow
    'radius_hand_l': 'elbow_l',    # wrist -> elbow
    'subtalar_r': 'ankle_r',       # heel -> ankle (if ankle missing)
    'subtalar_l': 'ankle_l',       # heel -> ankle (if ankle missing)
    'ankle_r': 'walker_knee_r',    # ankle -> knee (if ankle missing)
    'ankle_l': 'walker_knee_l',    # ankle -> knee (if ankle missing)
}


def load_skel_lbs_weights(
    model_path: str,
    gender: str = 'male',
    mesh_type: str = 'skeleton'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SKEL LBS weights and compute dominant joint per vertex.

    Args:
        model_path: Path to SKEL model directory (e.g., skel_models_v1.1/)
        gender: 'male' or 'female'
        mesh_type: 'skeleton' for skeleton mesh (247252 verts) or 'skin' for skin mesh (6890 verts)

    Returns:
        dominant_joints: [V] array of dominant joint index per vertex
        weights: [V, 24] full weight matrix
    """
    pkl_path = os.path.join(model_path, f'skel_{gender}.pkl')

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"SKEL model not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Select weight matrix based on mesh type
    if mesh_type == 'skeleton':
        weights_key = 'skel_weights'  # [247252, 24]
    else:
        weights_key = 'skin_weights'  # [6890, 24]

    if weights_key not in data:
        raise KeyError(f"'{weights_key}' not found in SKEL model. Available keys: {list(data.keys())}")

    weights_sparse = data[weights_key]
    weights = weights_sparse.toarray()  # Convert sparse to dense

    # Compute dominant joint for each vertex
    dominant_joints = np.argmax(weights, axis=1)

    return dominant_joints, weights


def get_joint_to_vertices_mapping(dominant_joints: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Get vertex indices for each joint.

    Args:
        dominant_joints: [V] array of dominant joint index per vertex

    Returns:
        Dict mapping joint index -> array of vertex indices
    """
    mapping = {}
    for j in range(24):
        mapping[j] = np.where(dominant_joints == j)[0]
    return mapping


def get_joint_name(joint_idx: int) -> str:
    """Get SKEL joint name from index."""
    if 0 <= joint_idx < len(SKEL_JOINT_NAMES):
        return SKEL_JOINT_NAMES[joint_idx]
    return f"joint_{joint_idx}"


def get_joint_idx(joint_name: str) -> Optional[int]:
    """Get SKEL joint index from name."""
    if joint_name in SKEL_JOINT_NAMES:
        return SKEL_JOINT_NAMES.index(joint_name)
    return None


def addb_joint_to_skel_idx(addb_joint_name: str) -> Optional[int]:
    """Map AddBiomechanics joint name to SKEL joint index."""
    return ADDB_TO_SKEL_JOINT_MAP.get(addb_joint_name)


def create_joint_color_map(num_joints: int = 24) -> Dict[int, Tuple[float, float, float]]:
    """
    Create distinct colors for each joint (for debugging segmentation).

    Args:
        num_joints: Number of joints

    Returns:
        Dict mapping joint index -> RGB color tuple (0-1 range)
    """
    import colorsys

    colors = {}
    for i in range(num_joints):
        # Use HSV colorspace for distinct colors
        hue = i / num_joints
        saturation = 0.8
        value = 0.9
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[i] = (r, g, b)

    return colors


def load_skel_template_vertices(
    model_path: str,
    gender: str = 'male'
) -> np.ndarray:
    """
    Load SKEL skeleton template vertices (T-pose).

    Args:
        model_path: Path to SKEL model directory
        gender: 'male' or 'female'

    Returns:
        template_vertices: [247252, 3] array of template vertex positions
    """
    pkl_path = os.path.join(model_path, f'skel_{gender}.pkl')

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"SKEL model not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data['skel_template_v']


def compute_vertex_to_template_mapping(
    mesh_vertices: np.ndarray,
    template_vertices: np.ndarray,
    batch_size: int = 10000
) -> np.ndarray:
    """
    Compute mapping from mesh vertices to nearest template vertices.

    This is needed when the saved mesh has deduplicated vertices (115K)
    but we need to use LBS weights from the template (247K).

    Uses KDTree for efficient nearest neighbor search.

    Args:
        mesh_vertices: [N, 3] vertices from loaded OBJ file
        template_vertices: [M, 3] vertices from SKEL template (247252)
        batch_size: Not used (kept for API compatibility)

    Returns:
        mapping: [N] array where mapping[i] is the template vertex index
                 closest to mesh_vertices[i]
    """
    try:
        from scipy.spatial import cKDTree
        # Build KDTree from template vertices (fast)
        tree = cKDTree(template_vertices)
        # Query nearest neighbor for all mesh vertices at once
        _, mapping = tree.query(mesh_vertices, k=1)
        return mapping.astype(np.int32)
    except ImportError:
        # Fallback to slow method if scipy not available
        print("Warning: scipy not available, using slow nearest neighbor search")
        num_mesh_verts = len(mesh_vertices)
        mapping = np.zeros(num_mesh_verts, dtype=np.int32)

        for i, vert in enumerate(mesh_vertices):
            if i % 10000 == 0:
                print(f"  Processing vertex {i}/{num_mesh_verts}...")
            distances = np.linalg.norm(template_vertices - vert, axis=1)
            mapping[i] = np.argmin(distances)

        return mapping


def load_skel_template_faces(
    model_path: str,
    gender: str = 'male'
) -> np.ndarray:
    """
    Load SKEL skeleton template faces.

    Args:
        model_path: Path to SKEL model directory
        gender: 'male' or 'female'

    Returns:
        template_faces: [126665, 3] array of face indices (0-247251)
    """
    pkl_path = os.path.join(model_path, f'skel_{gender}.pkl')

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"SKEL model not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data['skel_template_f']


def compute_face_based_vertex_mapping(
    mesh_faces: np.ndarray,
    model_path: str,
    gender: str = 'male'
) -> np.ndarray:
    """
    Compute dominant joint for each mesh vertex using face correspondence.

    When skeleton mesh is saved, duplicate vertices may be merged, resulting in
    fewer vertices (e.g., 115K instead of 247K). However, face count remains the
    same (126665). By comparing face indices, we can recover which original
    vertices each saved vertex corresponds to.

    Args:
        mesh_faces: [F, 3] face indices from saved OBJ (references 0 to N-1)
        model_path: Path to SKEL model directory
        gender: 'male' or 'female'

    Returns:
        dominant_joints: [N] array of dominant joint index per mesh vertex
    """
    # Load original faces and LBS weights
    original_faces = load_skel_template_faces(model_path, gender)
    template_dominant_joints, _ = load_skel_lbs_weights(model_path, gender, 'skeleton')

    if mesh_faces.shape != original_faces.shape:
        raise ValueError(f"Face count mismatch: mesh has {mesh_faces.shape[0]}, "
                        f"template has {original_faces.shape[0]}")

    num_mesh_verts = mesh_faces.max() + 1

    # Build mapping from saved vertex to set of original vertices
    saved_to_original_sets = {}
    for f_idx in range(len(mesh_faces)):
        for local_idx in range(3):
            saved_v_idx = mesh_faces[f_idx, local_idx]
            orig_v_idx = original_faces[f_idx, local_idx]

            if saved_v_idx not in saved_to_original_sets:
                saved_to_original_sets[saved_v_idx] = set()
            saved_to_original_sets[saved_v_idx].add(orig_v_idx)

    # Assign dominant joint to each saved vertex
    # Use majority vote among original vertices (handles boundary cases)
    dominant_joints = np.zeros(num_mesh_verts, dtype=np.int32)

    for saved_v, orig_set in saved_to_original_sets.items():
        orig_joints = template_dominant_joints[list(orig_set)]
        # Use most common joint (mode)
        values, counts = np.unique(orig_joints, return_counts=True)
        dominant_joints[saved_v] = values[np.argmax(counts)]

    return dominant_joints


def get_dominant_joints_for_mesh(
    mesh_vertices: np.ndarray,
    model_path: str,
    gender: str = 'male',
    mesh_faces: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Get dominant joint index for each vertex of a (possibly deduplicated) mesh.

    This function handles the case where the mesh has fewer vertices than
    the SKEL template. If mesh_faces is provided, uses face-based mapping
    (more accurate). Otherwise falls back to nearest-neighbor mapping.

    Args:
        mesh_vertices: [N, 3] vertices from loaded mesh
        model_path: Path to SKEL model directory
        gender: 'male' or 'female'
        mesh_faces: [F, 3] face indices (optional, enables accurate mapping)

    Returns:
        dominant_joints: [N] array of dominant joint index per mesh vertex
    """
    # Load template data
    template_dominant_joints, _ = load_skel_lbs_weights(model_path, gender, 'skeleton')
    num_template_verts = len(template_dominant_joints)
    num_mesh_verts = len(mesh_vertices)

    # If vertex counts match, assume direct correspondence
    if num_mesh_verts == num_template_verts:
        return template_dominant_joints

    print(f"Mesh has {num_mesh_verts} vertices, template has {num_template_verts}")

    # Prefer face-based mapping if faces are provided
    if mesh_faces is not None:
        print("Using face-based vertex mapping (accurate)...")
        return compute_face_based_vertex_mapping(mesh_faces, model_path, gender)

    # Fallback to nearest-neighbor (less accurate for posed meshes)
    print("Computing nearest-neighbor vertex mapping (fallback)...")
    template_vertices = load_skel_template_vertices(model_path, gender)
    mapping = compute_vertex_to_template_mapping(mesh_vertices, template_vertices)
    return template_dominant_joints[mapping]


if __name__ == '__main__':
    # Test loading
    model_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'

    print("Loading SKEL LBS weights...")
    dominant_joints, weights = load_skel_lbs_weights(model_path, 'male', 'skeleton')

    print(f"Weights shape: {weights.shape}")
    print(f"Dominant joints shape: {dominant_joints.shape}")

    # Print vertex count per joint
    joint_to_verts = get_joint_to_vertices_mapping(dominant_joints)
    print("\nVertices per joint:")
    for j in range(24):
        count = len(joint_to_verts[j])
        print(f"  {j:2d} {SKEL_JOINT_NAMES[j]:12s}: {count:6d} vertices")
