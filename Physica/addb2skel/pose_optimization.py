"""
Per-frame pose optimization module.

Stage 2 of the 2-stage optimization pipeline:
1. Scale estimation (done in scale_estimation.py)
2. Pose optimization per frame with temporal smoothing

Implements IK-style gradient descent optimization following:
- HSMR SKELify approach for 2D keypoint fitting
- AddBiomechanics kinematics pass for marker fitting

Improvements (Phase 1 - Quick Wins):
- Cosine annealing learning rate schedule
- Huber loss for robustness to outliers
- Early stopping to prevent overfitting
"""

from typing import Optional, Tuple, Dict, List
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import (
    OptimizationConfig,
    SKEL_NUM_POSE_DOF,
    SPINE_DOF_INDICES,
    ARM_DOF_INDICES,
    SKEL_JOINT_WEIGHTS,
    SKEL_JOINT_TO_IDX,
)
from .skel_interface import SKELInterface
from .scapula_handler import ScapulaHandler, compute_shoulder_losses
from .pose_limits import compute_soft_limit_penalty
from .joint_definitions import (
    build_direct_joint_mapping,
    get_bone_indices,
    ADDB_BONE_PAIRS,
    SKEL_BONE_PAIRS,
    ADDB_JOINT_TO_IDX,
    SKEL_JOINT_TO_IDX,
    ADDB_ACROMIAL_R_IDX,
    ADDB_ACROMIAL_L_IDX,
    ADDB_SUBTALAR_R_IDX,
    ADDB_SUBTALAR_L_IDX,
)
from .utils.geometry import (
    compute_bone_lengths,
    compute_bone_directions,
    cosine_similarity_loss,
)
from .pose_limits import get_pose_bounds_tensor, clamp_poses
from scipy.spatial.transform import Rotation as ScipyRotation


# =============================================================================
# IK-based Pose Initialization (from working compare_smpl_skel.py)
# =============================================================================

# SKEL bone pairs for IK initialization (parent_joint_idx → child_joint_idx)
# Maps to SKEL DOF indices for rotation
#
# SKEL DOF structure for arms:
# Right arm (DOF 26-35):
#   26-28: scapula_r (abduction, elevation, upward_rot)
#   29-31: humerus_r / shoulder_r (flexion, adduction, rotation)
#   32: elbow_r (ulna flexion)
#   33: radioulnar_r (forearm pronation/supination)
#   34-35: wrist_r (flexion, deviation)
# Left arm (DOF 36-45):
#   36-38: scapula_l
#   39-41: humerus_l / shoulder_l
#   42: elbow_l
#   43: radioulnar_l
#   44-45: wrist_l
#
SKEL_IK_BONE_MAP = {
    # Legs
    (0, 1): [3, 4, 5],      # pelvis → femur_r: hip_r DOFs
    (1, 2): [6],            # femur_r → tibia_r: knee_r DOF
    (2, 3): [7],            # tibia_r → talus_r: ankle_r DOF
    (3, 4): [8],            # talus_r → calcn_r: subtalar_r DOF
    (4, 5): [9],            # calcn_r → toes_r: mtp_r DOF
    (0, 6): [10, 11, 12],   # pelvis → femur_l: hip_l DOFs
    (6, 7): [13],           # femur_l → tibia_l: knee_l DOF
    (7, 8): [14],           # tibia_l → talus_l: ankle_l DOF
    (8, 9): [15],           # talus_l → calcn_l: subtalar_l DOF
    (9, 10): [16],          # calcn_l → toes_l: mtp_l DOF
    # Spine
    (0, 11): [17, 18, 19],  # pelvis → lumbar: lumbar DOFs
    (11, 12): [20, 21, 22], # lumbar → thorax: thorax DOFs
    # Right arm (full hierarchy)
    (12, 15): [29, 30, 31], # thorax → humerus_r: shoulder_r DOFs (NOT scapula)
    (15, 16): [32],         # humerus_r → ulna_r: elbow_r DOF
    (16, 17): [33],         # ulna_r → radius_r: radioulnar_r DOF
    (17, 18): [34, 35],     # radius_r → hand_r: wrist_r DOFs
    # Left arm (full hierarchy)
    (12, 20): [39, 40, 41], # thorax → humerus_l: shoulder_l DOFs
    (20, 21): [42],         # humerus_l → ulna_l: elbow_l DOF
    (21, 22): [43],         # ulna_l → radius_l: radioulnar_l DOF
    (22, 23): [44, 45],     # radius_l → hand_l: wrist_l DOFs
}


def estimate_initial_poses_ik(
    target_joints: torch.Tensor,
    addb_indices: List[int],
    skel_indices: List[int],
    skel_interface: 'SKELInterface',
    device: torch.device,
) -> torch.Tensor:
    """
    Simple IK-based pose initialization for SKEL.

    Uses bone direction vectors to estimate initial rotations,
    giving optimization a better starting point than zero pose.

    Args:
        target_joints: [T, N_addb, 3] target joint positions
        addb_indices: AddB joint indices for mapped joints
        skel_indices: SKEL joint indices for mapped joints
        skel_interface: SKEL model interface
        device: torch device

    Returns:
        Initial poses [T, 46] in Euler angle representation
    """
    T = target_joints.shape[0]
    initial_poses = torch.zeros(T, SKEL_NUM_POSE_DOF, device=device)

    # Build mapping: skel_joint_idx → addb_joint_idx
    skel_to_addb = {}
    for addb_idx, skel_idx in zip(addb_indices, skel_indices):
        skel_to_addb[skel_idx] = addb_idx

    # Compute default bone directions from T-pose
    with torch.no_grad():
        betas_zero = torch.zeros(10, device=device)
        poses_zero = torch.zeros(1, SKEL_NUM_POSE_DOF, device=device)
        trans_zero = torch.zeros(1, 3, device=device)
        _, tpose_joints, _ = skel_interface.forward(
            betas_zero.unsqueeze(0), poses_zero, trans_zero
        )
        tpose_joints = tpose_joints[0]  # [24, 3]

    # Process each frame
    for t in range(T):
        target_frame = target_joints[t]

        # Skip frames with all NaN
        if torch.isnan(target_frame).all():
            continue

        # Process each bone in IK map
        for (parent_skel_idx, child_skel_idx), dof_indices in SKEL_IK_BONE_MAP.items():
            # Check if both joints are mapped
            if parent_skel_idx not in skel_to_addb or child_skel_idx not in skel_to_addb:
                continue

            parent_addb_idx = skel_to_addb[parent_skel_idx]
            child_addb_idx = skel_to_addb[child_skel_idx]

            # Get positions
            parent_pos = target_frame[parent_addb_idx]
            child_pos = target_frame[child_addb_idx]

            # Skip NaN
            if torch.isnan(parent_pos).any() or torch.isnan(child_pos).any():
                continue

            # Compute observed bone direction
            obs_dir = child_pos - parent_pos
            obs_dir_norm = torch.norm(obs_dir)
            if obs_dir_norm < 1e-6:
                continue
            obs_dir = obs_dir / obs_dir_norm

            # Get default bone direction from T-pose
            default_dir = tpose_joints[child_skel_idx] - tpose_joints[parent_skel_idx]
            default_dir_norm = torch.norm(default_dir)
            if default_dir_norm < 1e-6:
                continue
            default_dir = default_dir / default_dir_norm

            # Compute rotation from default to observed
            cross = torch.cross(default_dir, obs_dir)
            dot = torch.dot(default_dir, obs_dir).clamp(-1.0, 1.0)

            cross_norm = torch.norm(cross)
            if cross_norm < 1e-6:
                continue

            # Rotation axis and angle
            axis = cross / cross_norm
            angle = torch.acos(dot) * 0.3  # Scale down for stability

            # Distribute rotation to DOFs
            for i, dof_idx in enumerate(dof_indices):
                if i < 3:  # x, y, z components
                    initial_poses[t, dof_idx] = axis[i] * angle

    return initial_poses


# =============================================================================
# AddB DOF → SKEL DOF Direct Mapping Table
# =============================================================================
#
# Derived from correlation analysis (addb_skel_dof_correlation.py) on
# Carter2023 P002 (100 frames, with_arm data).
#
# Format: addb_name → (addb_idx, skel_idx, scale)
#   scale = linear regression slope (addb → skel, both in radians)
#   Negative scale means sign flip required.
#
# Only DOFs with |corr| > 0.5 are included.
# DOFs not listed: keep as 0 (T-pose) or rely on IK init.
#
_ADDB_TO_SKEL_DOF_MAP = [
    # (addb_idx, skel_idx, scale)
    # Scales updated using multi-subject regression (Falisse+Carter).
    # Most lower-limb DOFs are ~1:1 between AddB and SKEL.
    (0,  0,  0.91),   # pelvis_tilt
    (1,  1,  0.83),   # pelvis_list
    (2,  2,  0.50),   # pelvis_rotation (conservative)
    (6,  3,  1.0),    # hip_flexion_r  (1:1, robust across datasets)
    (7,  4,  1.0),    # hip_adduction_r (1:1 — regression varies by dataset)
    (8,  5,  1.0),    # hip_rotation_r (1:1 — sign varies by dataset, auto-corrected)
    (9,  6,  1.0),    # knee_angle_r   (1:1, regression slope≈1.0, R²≈1.0)
    (10, 7,  1.0),    # ankle_angle_r  (1:1 — regression varies ~1.0-1.4)
    (11, 8,  1.0),    # subtalar_angle_r (1:1 fallback)
    (12, 9,  1.0),    # mtp_angle_r (1:1 fallback)
    (13, 10,  1.0),   # hip_flexion_l  (1:1, robust across datasets)
    (14, 11,  1.0),   # hip_adduction_l (1:1 — regression varies by dataset)
    (15, 12,  1.0),   # hip_rotation_l (1:1 — sign varies by dataset, auto-corrected)
    (16, 13,  1.0),   # knee_angle_l   (1:1, regression slope≈1.0, R²≈1.0)
    (17, 14,  1.0),   # ankle_angle_l  (1:1 — regression varies ~0.8-1.2)
    (18, 15,  1.0),   # subtalar_angle_l (1:1 fallback)
    (19, 16,  1.0),   # mtp_angle_l (1:1 fallback)
    (20, 18,  0.90),  # lumbar_extension
    (21, 17,  0.54),  # lumbar_bending
    (22, 19,  0.65),  # lumbar_rotation (lumbar_twist)
    (23, 29,  0.11),  # arm_flex_r → shoulder_r_x
    (24, 30, -0.32),  # arm_add_r → shoulder_r_y
    (25, 31,  0.50),  # arm_rot_r → shoulder_r_z
    (26, 32,  1.12),  # elbow_flex_r
    # Shoulder L: AddB auto-negates adduction/rotation for left side,
    # but SKEL humerus_l has NO flip (unlike hip_l which has flip=[1,-1,-1]).
    # Signs adjusted to account for this convention mismatch.
    (30, 39,  0.95),  # arm_flex_l → shoulder_l_x (was -0.95: AddB negates, SKEL doesn't → positive)
    (31, 40,  0.32),  # arm_add_l → shoulder_l_y (was -0.32: same reason)
    (32, 41, -0.32),  # arm_rot_l → shoulder_l_z (was +0.32: inverted)
    (33, 42,  1.12),  # elbow_flex_l
]

# Clamp scale to avoid extreme extrapolation
_ADDB_DOF_SCALE_CLAMP = 2.0

# =============================================================================
# Rotation-matrix based AddB → SKEL DOF mapping
# =============================================================================
# Default: AddB faces +X, SKEL faces +Z → 90° Y rotation
_R_90_Y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
# Alternative: 180° Y rotation = X,Z flip (matches pipeline's joint coordinate conversion)
_R_FLIP_XZ = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)


def compute_addb_to_skel_rotation(addb_skel, addb_poses_arr) -> np.ndarray:
    """Compute subject-adaptive rotation from AddB to SKEL coordinate frame.

    Instead of hardcoded R_90_Y (which assumes all subjects face +X in AddB),
    this detects the actual pelvis forward direction from AddB skeleton FK and
    computes a Y-axis rotation to align with SKEL's +Z forward convention.

    Method:
    1. Set AddB skeleton to frame 0 pose, get pelvis global rotation R_pelvis
    2. AddB forward = R_pelvis @ [1, 0, 0] (local +X is forward in OpenSim)
    3. Project to XZ plane, compute angle from +Z
    4. R_align = Ry(-angle) to rotate AddB forward → SKEL +Z

    Args:
        addb_skel: nimblephysics Skeleton with DOFs.
        addb_poses_arr: [T, D] AddB pose array.

    Returns:
        R_align: [3, 3] rotation matrix from AddB to SKEL frame.
    """
    # Use middle frame for robustness (avoid edge artifacts)
    T = len(addb_poses_arr)
    mid = T // 2
    addb_skel.setPositions(addb_poses_arr[mid])
    T_pelvis = np.array(addb_skel.getBodyNode(0).getWorldTransform().matrix())
    R_pelvis = T_pelvis[:3, :3]

    # AddB pelvis local +X is forward direction
    fwd = R_pelvis @ np.array([1.0, 0.0, 0.0])
    fwd[1] = 0.0  # project to XZ plane
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-6:
        return _R_90_Y  # fallback

    fwd /= fwd_norm

    # Angle of forward direction from +Z axis
    # R_align = Ry(-theta) rotates AddB forward to +Z (SKEL forward)
    theta = np.arctan2(fwd[0], fwd[2])
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    R_align = np.array([
        [cos_t, 0, sin_t],
        [0,     1,     0],
        [-sin_t, 0, cos_t],
    ], dtype=np.float64)

    return R_align

# SKEL kinematic tree parents (from skel_interface.py line 92)
# pelvis(0)→-1, femur_r(1)→0, ..., thorax(12)→11, head(13)→12,
# scapula_r(14)→12(thorax), humerus_r(15)→14, ..., scapula_l(19)→12(thorax), ...
_SKEL_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]

# AddB body → SKEL joint mapping (for rotation matrix extraction)
# AddB has 20 body nodes, SKEL has 24 joints.
# AddB torso(11) maps to SKEL lumbar(11) — SKEL thorax(12)/head(13) have no AddB counterpart.
# AddB has no scapula — humerus(12,16) maps to SKEL humerus(15,20), skipping scapula(14,19).
#
# 15 DIRECT joint matches (used in joint position loss):
#   AddB[ 0] pelvis       → SKEL[ 0] pelvis
#   AddB[ 1] femur_r      → SKEL[ 1] femur_r
#   AddB[ 2] tibia_r      → SKEL[ 2] tibia_r
#   AddB[ 3] talus_r      → SKEL[ 3] talus_r
#   AddB[ 5] toes_r       → SKEL[ 5] toes_r
#   AddB[ 6] femur_l      → SKEL[ 6] femur_l
#   AddB[ 7] tibia_l      → SKEL[ 7] tibia_l
#   AddB[ 8] talus_l      → SKEL[ 8] talus_l
#   AddB[10] toes_l       → SKEL[10] toes_l
#   AddB[13] ulna_r       → SKEL[16] ulna_r
#   AddB[14] radius_r     → SKEL[17] radius_r
#   AddB[15] hand_r       → SKEL[18] hand_r
#   AddB[17] ulna_l       → SKEL[21] ulna_l
#   AddB[18] radius_l     → SKEL[22] radius_l
#   AddB[19] hand_l       → SKEL[23] hand_l
# Non-DIRECT: subtalar(4,9)=APPROXIMATE, back(11)=APPROXIMATE, acromial(12,16)=PROXY
_ADDB_BODY_TO_SKEL_JOINT = {
    # Lower body only — upper body rotation extraction is unreliable because:
    #   - AddB torso(11) = single body, SKEL splits into lumbar/thorax/head
    #   - AddB has no scapula → SKEL parent chain mismatch for humerus
    #   - Upper body q_reference caused overfitting (shoulder/spine distortion)
    # Upper body is left to position loss + scale-based initialization.
    0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10,
}

# SKEL joint Euler conventions: (axes, flips, n_dofs, dof_start)
# From skel_model.py: CustomJoint applies R = R_axes[-1] @ ... @ R_axes[0] (left-multiply)
# Decomposition: scipy intrinsic = reverse of axes string
_SKEL_JOINT_EULER_CONV = {
    0:  ('ZXY', [1,1,1], 3, 0),      # pelvis (CustomJoint axes=[Z,X,Y])
    1:  ('ZXY', [1,1,1], 3, 3),      # femur_r (CustomJoint axes=[Z,X,Y])
    2:  ('Z',   [-1], 1, 6),          # tibia_r (WalkerKnee: -Z)
    3:  ('Z',   [1], 1, 7),           # talus_r (PinJoint: ~Z axis)
    4:  ('Z',   [1], 1, 8),           # calcn_r (PinJoint: ~Z axis)
    5:  ('Z',   [1], 1, 9),           # toes_r (PinJoint: ~Z axis)
    6:  ('ZXY', [1,-1,-1], 3, 10),    # femur_l (CustomJoint axes=[Z,X,Y], L/R flip)
    7:  ('Z',   [-1], 1, 13),         # tibia_l (WalkerKnee: -Z)
    8:  ('Z',   [1], 1, 14),          # talus_l (PinJoint: ~Z axis)
    9:  ('Z',   [1], 1, 15),          # calcn_l (PinJoint: ~Z axis)
    10: ('Z',   [1], 1, 16),          # toes_l (PinJoint: ~Z axis)
}

# NOTE: Upper body conventions documented for future reference:
#   11: ('XZY', [1,1,1], 3, 17)     # lumbar (ConstantCurvature)
#   15: ('XYZ', [1,1,1], 3, 29)     # humerus_r (CustomJoint)
#   20: ('XYZ', [1,1,1], 3, 39)     # humerus_l (CustomJoint)
# Not used in q_reference because parent chain mismatch causes distortion.

# Actual rotation axes for PinJoint and CustomJoint1D joints
# Computed from: Ra = euler_to_matrix(parent_frame_ori, 'XYZ'); axis = Ra @ [0,0,1]
# Or directly from skel_model.py axis definition for CustomJoint1D
_SKEL_1DOF_AXES = {
    # PinJoint axes (from parent_frame_ori → rotated Z)
    3:  np.array([-0.10503, -0.17402, 0.97913]),   # talus_r (ankle_r)
    4:  np.array([0.78724, 0.60473, -0.12092]),     # calcn_r (subtalar_r)
    5:  np.array([0.58100, 0.00000, -0.81394]),     # toes_r (mtp_r)
    8:  np.array([-0.10503, -0.17402, 0.97913]),    # talus_l (ankle_l)
    9:  np.array([-0.78724, -0.60473, -0.12092]),   # calcn_l (subtalar_l)
    10: np.array([-0.58100, 0.00000, -0.81394]),    # toes_l (mtp_l)
    # CustomJoint1D axes (directly from skel_model.py, normalized)
    16: np.array([0.04940, 0.03660, 0.99811]),      # ulna_r (elbow)
    17: np.array([-0.01716, 0.99267, -0.11967]),    # radius_r (pronation/supination)
    21: np.array([-0.04940, -0.03660, 0.99811]),    # ulna_l (elbow)
    22: np.array([0.01716, -0.99267, -0.11967]),    # radius_l (pronation/supination)
}


def compute_gt_based_r_align(addb_joints):
    """Compute R_align from GT joint positions (Method B).

    Uses cross(hip_L - hip_R, up) to determine forward direction from GT joints.
    This is robust because GT joints are always accurate regardless of pelvis FK.

    Args:
        addb_joints: [T, J, 3] GT joint positions.

    Returns:
        R_align: [3, 3] rotation matrix.
    """
    # hip_R = joint 1, hip_L = joint 6
    hip_r = addb_joints[:, 1, :].mean(axis=0)  # average over frames
    hip_l = addb_joints[:, 6, :].mean(axis=0)
    lr = hip_l - hip_r  # left-right vector
    up = np.array([0.0, 1.0, 0.0])
    fwd = np.cross(lr, up)  # forward = L×up
    fwd[1] = 0.0  # project to XZ plane
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-6:
        return _R_90_Y  # fallback
    fwd /= fwd_norm

    # R_align rotates AddB forward → SKEL +Z
    theta = np.arctan2(fwd[0], fwd[2])
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    return np.array([
        [cos_t, 0, sin_t],
        [0,     1,     0],
        [-sin_t, 0, cos_t],
    ], dtype=np.float64)


def compute_skel_q_reference_from_addb(addb_skel, addb_poses_arr, T,
                                       addb_joints=None, use_gt_align=False,
                                       use_r_flip_xz=False):
    """Compute SKEL DOF reference from AddB GT using rotation matrix mapping.

    Args:
        addb_skel: nimblephysics Skeleton with DOFs set
        addb_poses_arr: [T, N_addb_dofs] numpy array of AddB DOF values
        T: number of frames to process
        addb_joints: [T, J, 3] GT joint positions (needed if use_gt_align=True)
        use_gt_align: If True, use Method B (GT-based R_align) instead of R_90_Y
        use_r_flip_xz: If True, use 180° Y (X,Z flip) instead of 90° Y

    Returns:
        skel_q_ref: [T, 46] numpy array of SKEL DOF values
    """
    if use_gt_align and addb_joints is not None:
        R_align = compute_gt_based_r_align(addb_joints)
    elif use_r_flip_xz:
        R_align = _R_FLIP_XZ
    else:
        R_align = _R_90_Y

    nd = addb_skel.getNumDofs()
    skel_dofs = np.zeros((T, SKEL_NUM_POSE_DOF), dtype=np.float64)

    for t in range(T):
        # Get per-body global rotation matrices from AddB
        addb_skel.setPositions(addb_poses_arr[t])
        nb = addb_skel.getNumBodyNodes()
        R_addb = np.zeros((nb, 3, 3))
        for i in range(nb):
            T_mat = np.array(addb_skel.getBodyNode(i).getWorldTransform().matrix())
            R_addb[i] = T_mat[:3, :3]

        # Transform to SKEL frame using subject-adaptive rotation
        R_skel = np.array([R_align @ R_addb[i] for i in range(nb)])

        # For each mapped joint: extract local rotation → decompose to SKEL Euler
        for addb_bi, skel_ji in _ADDB_BODY_TO_SKEL_JOINT.items():
            if addb_bi >= nb or skel_ji not in _SKEL_JOINT_EULER_CONV:
                continue

            axes, flips, n_dofs, dof_start = _SKEL_JOINT_EULER_CONV[skel_ji]

            # Global rotation of this body in SKEL frame
            R_global = R_skel[addb_bi]

            # Parent's global rotation
            parent_ji = _SKEL_PARENTS[skel_ji]
            R_parent = np.eye(3)
            if parent_ji >= 0:
                for ak, sv in _ADDB_BODY_TO_SKEL_JOINT.items():
                    if sv == parent_ji and ak < nb:
                        R_parent = R_skel[ak]
                        break

            # Local rotation: R_local = R_parent^T @ R_global
            R_local = R_parent.T @ R_global

            if n_dofs >= 3 and axes not in ('AXIS', 'XnZ', 'nXnZ'):
                # 3-DOF Euler decomposition using scipy
                # Reverse axis string for scipy intrinsic convention:
                # SKEL left-multiply [A,B,C] → R = R_C @ R_B @ R_A → intrinsic CBA
                scipy_seq = axes[:3][::-1]  # ZXY → YXZ, XZY → YZX, XYZ → ZYX
                try:
                    angles = ScipyRotation.from_matrix(R_local).as_euler(scipy_seq, degrees=False)[::-1]
                except:
                    angles = np.zeros(3)
                # Apply flip inverse
                for i in range(min(3, len(flips))):
                    if flips[i] != 0:
                        angles[i] /= flips[i]
                skel_dofs[t, dof_start:dof_start + 3] = angles[:3]

            elif n_dofs == 2 and axes in ('XnZ', 'nXnZ'):
                # 2-DOF joints (hand_r: [X, -Z], hand_l: [-X, -Z])
                # R = R_{-Z}(q1) @ R_X(q0) for hand_r (left-multiply)
                # Extract via scipy: approximate as 2 of 3 Euler angles
                # For XnZ: use XZY decomposition, take X and Z components
                try:
                    if axes == 'XnZ':
                        # hand_r: axes=[X, -Z], flips=[1,1]
                        # R = R_{-Z}(q1) @ R_X(q0) ≈ Euler XZ with negated Z
                        ang = ScipyRotation.from_matrix(R_local).as_euler('YZX', degrees=False)
                        # YZX intrinsic → extrinsic XZY: [Y, Z, X]
                        # We want X and -Z components
                        skel_dofs[t, dof_start] = ang[2]     # X component
                        skel_dofs[t, dof_start + 1] = -ang[1]  # -Z component
                    else:
                        # hand_l: axes=[-X, -Z], flips=[1,1]
                        ang = ScipyRotation.from_matrix(R_local).as_euler('YZX', degrees=False)
                        skel_dofs[t, dof_start] = -ang[2]    # -X component
                        skel_dofs[t, dof_start + 1] = -ang[1]  # -Z component
                except:
                    pass

            elif n_dofs == 1:
                if axes == 'AXIS' and skel_ji in _SKEL_1DOF_AXES:
                    # Custom axis: extract rotation angle around specific axis
                    # Using axis-angle decomposition from rotation matrix
                    axis = _SKEL_1DOF_AXES[skel_ji]
                    axis = axis / np.linalg.norm(axis)
                    # R_local = I + sin(θ)[n]× + (1-cos(θ))[n]×²
                    # trace(R) = 1 + 2cos(θ) → cos(θ) = (trace(R)-1)/2
                    # sin(θ) = (R - R^T) · n / 2
                    cos_th = (np.trace(R_local) - 1.0) / 2.0
                    cos_th = np.clip(cos_th, -1.0, 1.0)
                    # Skew-symmetric part: (R - R^T)/2
                    skew = (R_local - R_local.T) / 2.0
                    sin_th = skew[2,1]*axis[0] + skew[0,2]*axis[1] + skew[1,0]*axis[2]
                    angle = np.arctan2(sin_th, cos_th)
                    if flips[0] != 0:
                        angle /= flips[0]
                    skel_dofs[t, dof_start] = angle

                elif axes[0] == 'Z':
                    angle = np.arctan2(R_local[1, 0], R_local[0, 0])
                    if flips[0] != 0:
                        angle /= flips[0]
                    skel_dofs[t, dof_start] = angle

                elif axes[0] == 'X':
                    angle = np.arctan2(R_local[2, 1], R_local[1, 1])
                    if flips[0] != 0:
                        angle /= flips[0]
                    skel_dofs[t, dof_start] = angle

                elif axes[0] == 'Y':
                    angle = np.arctan2(R_local[0, 2], R_local[2, 2])
                    if flips[0] != 0:
                        angle /= flips[0]
                    skel_dofs[t, dof_start] = angle

    return skel_dofs


def initialize_poses_from_addb_dofs(
    addb_poses: np.ndarray,
    device: torch.device,
    pose_lower: torch.Tensor,
    pose_upper: torch.Tensor,
    skip_clamping: bool = False,
) -> torch.Tensor:
    """
    Initialize SKEL poses directly from AddB DOF values.

    Uses a precomputed linear mapping (scale) derived from correlation analysis.
    Only maps DOFs with |corr| > 0.5; others remain at 0 (T-pose).

    Includes automatic scale sign correction: if mapping puts >50% of frames
    out-of-bounds, the sign is flipped. This handles subject-dependent DOF
    convention differences (e.g., hip_flexion_r sign varies across datasets).

    Args:
        addb_poses: AddB pp.pos per frame [T, N_addb_dofs] in radians.
        device: torch device.
        pose_lower: Lower pose bounds [46].
        pose_upper: Upper pose bounds [46].
        skip_clamping: If True, skip clamping to pose bounds (for Direction-First Phase A).

    Returns:
        Initial SKEL poses [T, 46] in radians, clamped to pose bounds (unless skip_clamping).
    """
    T = addb_poses.shape[0]
    n_addb = addb_poses.shape[1]

    addb_t = torch.from_numpy(addb_poses).float().to(device)  # [T, N_addb]
    skel_poses = torch.zeros(T, SKEL_NUM_POSE_DOF, device=device)

    lower_np = pose_lower.cpu().numpy()
    upper_np = pose_upper.cpu().numpy()

    for (ai, si, scale) in _ADDB_TO_SKEL_DOF_MAP:
        if ai >= n_addb:
            continue  # safety: addb_poses doesn't have this DOF
        scale_clamped = float(np.clip(scale, -_ADDB_DOF_SCALE_CLAMP, _ADDB_DOF_SCALE_CLAMP))
        mapped = addb_t[:, ai] * scale_clamped

        # Auto-correct scale sign: if >50% of frames are OOB, flip sign
        with torch.no_grad():
            oob = ((mapped < pose_lower[si]) | (mapped > pose_upper[si])).sum().item()
            if oob > T * 0.5:
                # Try flipped sign
                mapped_flip = addb_t[:, ai] * (-scale_clamped)
                oob_flip = ((mapped_flip < pose_lower[si]) | (mapped_flip > pose_upper[si])).sum().item()
                if oob_flip < oob:
                    mapped = mapped_flip

        skel_poses[:, si] = mapped

    if not skip_clamping:
        # Clamp to physiological bounds
        skel_poses = torch.clamp(skel_poses, pose_lower, pose_upper)

    # Nudge knee DOFs away from lower bound (0) to avoid gradient=0 at boundary
    with torch.no_grad():
        KNEE_R_DOF, KNEE_L_DOF = 6, 13
        skel_poses[:, KNEE_R_DOF] = skel_poses[:, KNEE_R_DOF].clamp(min=0.1)
        skel_poses[:, KNEE_L_DOF] = skel_poses[:, KNEE_L_DOF].clamp(min=0.1)

    return skel_poses


class PoseOptimizer:
    """
    Optimizes SKEL pose parameters to match AddB joint positions.

    Uses a multi-term loss function:
    - Joint position loss (primary)
    - Bone direction loss
    - Bone length loss
    - Virtual acromial loss (for shoulder)
    - Shoulder width loss
    - Pose regularization
    - Temporal smoothness (optional)
    """

    def __init__(
        self,
        skel_interface: SKELInterface,
        config: Optional[OptimizationConfig] = None,
        marker_handler=None,
    ):
        """
        Initialize pose optimizer.

        Args:
            skel_interface: SKEL model interface.
            config: Optimization configuration.
            marker_handler: Optional MarkerHandler for marker-based loss.
        """
        self.skel = skel_interface
        self.config = config or OptimizationConfig()
        self.device = self.config.get_device()
        self.marker_handler = marker_handler

        # Build joint mapping (include ALL joints, including acromial mapped to humerus)
        self.addb_indices, self.skel_indices = build_direct_joint_mapping()

        # NOTE: Now we include acromial in joint loss by mapping to humerus
        # This is critical for good shoulder fitting (per compare_smpl_skel.py)
        # The separate virtual acromial loss is optional/additional

        # Build bone pair indices
        self.addb_bone_indices = get_bone_indices(ADDB_BONE_PAIRS, ADDB_JOINT_TO_IDX)
        self.skel_bone_indices = get_bone_indices(SKEL_BONE_PAIRS, SKEL_JOINT_TO_IDX)

        # Build per-bone weights for bone length loss (shin priority)
        self.bone_weights = self._build_bone_weights()

        # Build per-joint weights (critical for good results!)
        self.joint_weights = self._build_joint_weights()

        # Scapula handler
        self.scapula_handler = ScapulaHandler(skel_interface, config)

        # Pose bounds for clamping (comprehensive limits from literature)
        self.pose_lower, self.pose_upper = get_pose_bounds_tensor(self.device)

        # Fix F: Remove pelvis rotation limit (SKEL original has no limit for pelvis DOFs)
        if getattr(config, 'pelvis_rotation_unlimited', False):
            self.pose_lower[2] = -torch.pi  # pelvis_rotation: full range
            self.pose_upper[2] = torch.pi

        # q-matching reference (set in optimize_sequence if addb_poses provided)
        self._addb_q_reference = None
        self._addb_q_mask = None
        self._addb_q_confidence = None

        # dJ joint offset corrections (set in optimize_sequence from Stage 1)
        self._dJ = None

    def _get_cosine_lr(self, base_lr: float, current_iter: int, total_iters: int) -> float:
        """
        Compute learning rate with cosine annealing schedule.

        LR follows: lr = base_lr * (1 + cos(π * t / T)) / 2
        This smoothly decays from base_lr to 0.

        Args:
            base_lr: Initial learning rate
            current_iter: Current iteration (0-indexed)
            total_iters: Total number of iterations

        Returns:
            Current learning rate
        """
        return base_lr * (1 + math.cos(math.pi * current_iter / total_iters)) / 2

    def _update_optimizer_lr(self, optimizer: torch.optim.Optimizer, lr_multiplier: float):
        """
        Update learning rate for all parameter groups in optimizer.

        Args:
            optimizer: PyTorch optimizer
            lr_multiplier: Multiplier to apply to base learning rates
        """
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
            param_group['lr'] = param_group['initial_lr'] * lr_multiplier

    def _compute_soft_constraint_penalty(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Compute soft penalty for pose values outside bounds.

        Instead of hard clamping, adds a smooth penalty:
        penalty = sum(relu(pose - upper)^2 + relu(lower - pose)^2)

        This allows gradients to flow even when near bounds.

        Args:
            poses: Pose parameters [T, 46]

        Returns:
            Scalar penalty value
        """
        # Penalty for exceeding upper bounds
        upper_violation = F.relu(poses - self.pose_upper)
        # Penalty for going below lower bounds
        lower_violation = F.relu(self.pose_lower - poses)

        penalty = (upper_violation ** 2).sum() + (lower_violation ** 2).sum()
        return penalty

    def _update_dynamic_weights(
        self,
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
    ) -> None:
        """
        Update joint weights dynamically based on per-joint errors.

        High-error joints get increased weight to encourage the optimizer
        to focus on reducing their errors.

        Args:
            pred_joints: Predicted joint positions [T, num_joints, 3]
            target_joints: Target joint positions [T, num_joints, 3]
        """
        with torch.no_grad():
            # Compute per-joint errors
            per_joint_error = torch.norm(pred_joints - target_joints, dim=-1).mean(dim=0)  # [num_joints]

            # Compute scale factors (higher error → higher weight)
            mean_error = per_joint_error.mean()
            if mean_error > 1e-8:
                scale = per_joint_error / mean_error  # Normalize by mean
                # Apply scaling: weight *= (1 + scale_factor * (scale - 1))
                # This increases weight for above-average errors, decreases for below-average
                dynamic_weights = self.joint_weights * (1 + self.config.dynamic_weight_scale * (scale - 1))
                # Normalize to sum to original sum (preserve overall weight magnitude)
                dynamic_weights = dynamic_weights * (self.joint_weights.sum() / dynamic_weights.sum())
                self.joint_weights = dynamic_weights

    def _build_bone_weights(self) -> torch.Tensor:
        """
        Build per-bone weight tensor for bone length loss.

        Uses scale_bone_weights from config. Shin bones get higher weight
        because SKEL shape space couples thigh/shin and short shin causes
        tip-toe appearance.

        Returns:
            weights: [num_bone_pairs] weight tensor
        """
        weights = torch.ones(len(ADDB_BONE_PAIRS), device=self.device)
        if self.config.scale_bone_weights:
            for i, (a1, a2) in enumerate(ADDB_BONE_PAIRS):
                key = f"{a1}\u2192{a2}"
                if key in self.config.scale_bone_weights:
                    weights[i] = self.config.scale_bone_weights[key]
        return weights

    def _build_joint_weights(self) -> torch.Tensor:
        """
        Build per-joint weight tensor based on SKEL_JOINT_WEIGHTS config.

        Critical for good results! Shoulders get 10x, spine 5x, pelvis/femurs 2x.

        Returns:
            weights: [num_mapped_joints] weight tensor
        """
        from .joint_definitions import SKEL_JOINTS

        # Use runtime override if provided, otherwise default
        weight_dict = self.config.joint_weights_override if self.config.joint_weights_override is not None else SKEL_JOINT_WEIGHTS

        weights = []
        for skel_idx in self.skel_indices:
            joint_name = SKEL_JOINTS[skel_idx]
            # Map SKEL joint names to config names (handle naming variations)
            if joint_name == 'lumbar_body':
                config_name = 'lumbar'
            else:
                config_name = joint_name
            weight = weight_dict.get(config_name, 1.0)
            weights.append(weight)

        return torch.tensor(weights, device=self.device, dtype=torch.float32)

    def _compute_scapulohumeral_coupling_loss(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Compute scapulohumeral rhythm coupling constraint.

        Anatomically, the scapula rotates approximately 1/3 of the humerus
        elevation (2:1 glenohumeral to scapulothoracic ratio).

        This constraint encourages the scapula upward rotation to be
        approximately humerus_elevation / ratio.

        SKEL DOF indices:
        - Right scapula upward_rot: DOF 28
        - Right humerus elevation (shoulder_y): DOF 30
        - Left scapula upward_rot: DOF 38
        - Left humerus elevation (shoulder_y): DOF 40

        Args:
            poses: Pose parameters [T, 46]

        Returns:
            Coupling constraint loss (scalar)
        """
        # DOF indices (from config.py SCAPULA_DOF_INDICES and HUMERUS_DOF_INDICES)
        # Right arm
        scapula_upward_r = poses[:, 28]  # scapula upward rotation
        humerus_elev_r = poses[:, 30]    # humerus elevation (shoulder_y)

        # Left arm
        scapula_upward_l = poses[:, 38]
        humerus_elev_l = poses[:, 40]

        # Expected scapula rotation based on scapulohumeral rhythm
        ratio = self.config.scapulohumeral_ratio  # default 3.0 (i.e., scapula = humerus/3)
        expected_scapula_r = humerus_elev_r / ratio
        expected_scapula_l = humerus_elev_l / ratio

        # Coupling loss: penalize deviation from expected ratio
        coupling_loss_r = (scapula_upward_r - expected_scapula_r) ** 2
        coupling_loss_l = (scapula_upward_l - expected_scapula_l) ** 2

        return (coupling_loss_r.mean() + coupling_loss_l.mean()) / 2

    def _compute_weighted_joint_loss(
        self,
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted joint position loss.

        Supports both MSE and Huber (smooth L1) loss based on config.
        Huber loss is more robust to outliers (noisy joint detections).

        Args:
            pred_joints: Predicted joint positions [T, num_joints, 3]
            target_joints: Target joint positions [T, num_joints, 3]

        Returns:
            Weighted loss scalar
        """
        # Apply per-joint weights: [num_joints] -> [1, num_joints, 1]
        weights = self.joint_weights.view(1, -1, 1)

        if self.config.use_huber_loss:
            # Huber loss (smooth L1) - robust to outliers
            # F.smooth_l1_loss with beta parameter acts as Huber loss
            # For |x| < beta: loss = 0.5 * x^2 / beta
            # For |x| >= beta: loss = |x| - 0.5 * beta
            diff = pred_joints - target_joints
            # Per-element Huber loss
            huber = F.smooth_l1_loss(
                pred_joints, target_joints,
                beta=self.config.huber_delta,
                reduction='none'
            )
            weighted_loss = huber * weights
            return weighted_loss.mean()
        else:
            # Standard MSE loss
            sq_diff = (pred_joints - target_joints) ** 2
            weighted_sq_diff = sq_diff * weights
            return weighted_sq_diff.mean()

    def optimize_single_frame(
        self,
        addb_joints: torch.Tensor,
        betas: torch.Tensor,
        initial_poses: Optional[torch.Tensor] = None,
        initial_trans: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Optimize pose for a single frame.

        Args:
            addb_joints: AddB joint positions [20, 3] in meters.
            betas: SKEL shape parameters [10].
            initial_poses: Initial pose [46]. Default: zeros.
            initial_trans: Initial translation [3]. Default: pelvis position.
            verbose: Print progress.

        Returns:
            poses: Optimized pose parameters [46].
            trans: Optimized translation [3].
            stats: Optimization statistics.
        """
        # Ensure batch dimension
        if addb_joints.dim() == 2:
            addb_joints = addb_joints.unsqueeze(0)

        # Initialize pose
        if initial_poses is None:
            poses = torch.zeros(1, SKEL_NUM_POSE_DOF, device=self.device)
        else:
            poses = initial_poses.clone().unsqueeze(0) if initial_poses.dim() == 1 else initial_poses.clone()

        # Initialize translation from pelvis
        if initial_trans is None:
            trans = addb_joints[0, 0, :].clone().unsqueeze(0)
        else:
            trans = initial_trans.clone().unsqueeze(0) if initial_trans.dim() == 1 else initial_trans.clone()

        poses.requires_grad_(True)
        trans.requires_grad_(True)

        # Optimizer
        optimizer = torch.optim.Adam([poses, trans], lr=self.config.pose_lr)

        # Target bone lengths and directions
        target_bone_lengths = compute_bone_lengths(addb_joints, self.addb_bone_indices)
        target_bone_dirs = compute_bone_directions(addb_joints, self.addb_bone_indices)

        # Target shoulder width
        target_width = torch.norm(
            addb_joints[:, ADDB_ACROMIAL_R_IDX, :] - addb_joints[:, ADDB_ACROMIAL_L_IDX, :],
            dim=-1
        )

        best_loss = float('inf')
        best_poses = poses.clone()
        best_trans = trans.clone()

        for it in range(self.config.pose_iters):
            optimizer.zero_grad()

            # Forward through SKEL
            skel_verts, skel_joints, _ = self.skel.forward(
                betas.unsqueeze(0), poses, trans, return_skeleton=False
            )

            # --- Loss computation ---

            # 1. Joint position loss (weighted, includes acromial→humerus)
            pred_joints_mapped = skel_joints[:, self.skel_indices, :]
            target_joints = addb_joints[:, self.addb_indices, :]
            joint_loss = self._compute_weighted_joint_loss(pred_joints_mapped, target_joints)

            # 2. Bone direction loss
            pred_bone_dirs = compute_bone_directions(skel_joints, self.skel_bone_indices)
            bone_dir_loss = cosine_similarity_loss(pred_bone_dirs, target_bone_dirs)

            # 3. Bone length loss
            pred_bone_lengths = compute_bone_lengths(skel_joints, self.skel_bone_indices)
            bone_len_loss = F.mse_loss(pred_bone_lengths, target_bone_lengths)

            # 4. Shoulder width loss
            pred_width = self.skel.get_shoulder_width(skel_joints)
            width_loss = F.mse_loss(pred_width, target_width)

            # 5. Shoulder/scapula losses
            shoulder_losses = compute_shoulder_losses(
                skel_verts, skel_joints, poses, addb_joints,
                self.scapula_handler, self.config
            )

            # 6. Pose regularization
            pose_reg = self.config.weight_pose_reg * (poses ** 2).mean()

            # 7. Spine regularization
            spine_dofs = poses[:, SPINE_DOF_INDICES]
            spine_reg = self.config.weight_spine_reg * (spine_dofs ** 2).mean()

            # 7a. Arm regularization (elbow/wrist — prevents over-flexion)
            arm_dofs = poses[:, ARM_DOF_INDICES]
            arm_reg = self.config.weight_arm_reg * (arm_dofs ** 2).mean()

            # Combine losses
            loss = (
                self.config.weight_joint * joint_loss +
                self.config.weight_bone_dir * bone_dir_loss +
                self.config.weight_bone_len * bone_len_loss +
                self.config.weight_width * width_loss +
                shoulder_losses['acromial'] +
                shoulder_losses['humerus_align'] +
                shoulder_losses['humerus_on_line'] +
                shoulder_losses['scapula_reg'] +
                shoulder_losses['humerus_reg'] +
                pose_reg +
                spine_reg +
                arm_reg
            )

            loss.backward()
            optimizer.step()

            # Clamp all pose DOFs to physiological limits
            with torch.no_grad():
                poses.data = clamp_poses(poses.data, self.pose_lower, self.pose_upper)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_poses = poses.clone().detach()
                best_trans = trans.clone().detach()

            if verbose and ((it + 1) % 50 == 0 or it == 0):
                with torch.no_grad():
                    mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
                print(f"  Iter {it+1}/{self.config.pose_iters}: "
                      f"Loss={loss.item():.4f}, MPJPE={mpjpe:.1f}mm")

        # Final statistics
        with torch.no_grad():
            skel_verts, skel_joints, _ = self.skel.forward(
                betas.unsqueeze(0), best_poses, best_trans, dJ=self._dJ
            )
            mpjpe = self._compute_mpjpe(skel_joints, addb_joints)

        stats = {
            'final_loss': best_loss,
            'mpjpe_mm': mpjpe,
            'scapula_dofs': self.scapula_handler.get_scapula_dof_values(best_poses),
        }

        return best_poses.squeeze(0), best_trans.squeeze(0), stats

    def optimize_sequence(
        self,
        addb_joints: torch.Tensor,
        betas: torch.Tensor,
        use_temporal: bool = True,
        verbose: bool = False,
        addb_poses: Optional[np.ndarray] = None,
        dJ: Optional[torch.Tensor] = None,
        skel_q_reference: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Optimize poses for a sequence of frames using 2-stage optimization.

        Stage 1: Pose + Trans optimization (betas fixed) - first half of iterations
        Stage 2: All parameters (betas + poses + trans) - second half of iterations

        Args:
            addb_joints: AddB joint positions [T, 20, 3] in meters.
            betas: SKEL shape parameters [10].
            use_temporal: Use temporal smoothness regularization.
            verbose: Print progress.
            addb_poses: AddB DOF values [T, N_addb_dofs] in radians (pp.pos).
                        If provided, uses direct DOF mapping for pose initialization
                        instead of IK-based estimation.
            dJ: Joint offset corrections [1, 24, 3] from Stage 1.
                Fixed during pose optimization (not optimized further).

        Returns:
            poses: Optimized pose parameters [T, 46].
            trans: Optimized translations [T, 3].
            stats: Optimization statistics.
        """
        T = addb_joints.shape[0]

        # Store dJ for use in forward calls (fixed, not optimized)
        self._dJ = dJ.detach() if dJ is not None else None

        if verbose:
            print(f"Optimizing {T} frames...")

        # Initialize translation to align SKEL pelvis with AddB pelvis
        # SKEL has an inherent pelvis offset from the root translation
        # We need to compute: trans = addb_pelvis - skel_pelvis_offset
        with torch.no_grad():
            # Get SKEL pelvis offset in zero pose
            betas_zero = torch.zeros(10, device=self.device)
            poses_zero = torch.zeros(1, SKEL_NUM_POSE_DOF, device=self.device)
            trans_zero = torch.zeros(1, 3, device=self.device)
            _, joints_zero, _ = self.skel.forward(
                betas_zero.unsqueeze(0), poses_zero, trans_zero
            )
            pelvis_offset = joints_zero[0, 0]  # SKEL pelvis offset from origin

            # Initialize trans so that SKEL pelvis aligns with AddB pelvis
            trans = addb_joints[:, 0, :].clone() - pelvis_offset.unsqueeze(0)

        # Pose initialization: use AddB DOF direct mapping if available,
        # otherwise fall back to IK-based estimation.
        if addb_poses is not None and len(addb_poses) >= T:
            if verbose:
                print("  Using AddB DOF direct mapping for pose initialization...")
            with torch.no_grad():
                addb_poses_arr = np.array(addb_poses[:T])
                initial_poses = initialize_poses_from_addb_dofs(
                    addb_poses_arr, self.device, self.pose_lower, self.pose_upper
                )
        else:
            if verbose:
                print("  Computing IK-based pose initialization...")
            with torch.no_grad():
                initial_poses = estimate_initial_poses_ik(
                    addb_joints, self.addb_indices, self.skel_indices,
                    self.skel, self.device
                )
            # Knee lower bound is 0 — if IK gives 0, gradient is zero at the boundary.
            # Nudge knee DOFs to a small positive value so optimizer can explore.
            with torch.no_grad():
                KNEE_R_DOF, KNEE_L_DOF = 6, 13
                initial_poses[:, KNEE_R_DOF] = initial_poses[:, KNEE_R_DOF].clamp(min=0.1)
                initial_poses[:, KNEE_L_DOF] = initial_poses[:, KNEE_L_DOF].clamp(min=0.1)

        # If skel_q_reference is available, use it for lower body initialization
        # (more accurate than scale-based mapping for rotation order + coordinate diff)
        # Only lower body DOFs 0-16: upper body excluded (parent chain mismatch causes distortion)
        if skel_q_reference is not None and len(skel_q_reference) >= T:
            skel_init = torch.from_numpy(np.array(skel_q_reference[:T])).float().to(self.device)
            with torch.no_grad():
                initial_poses[:, :17] = skel_init[:, :17]
                initial_poses = clamp_poses(initial_poses, self.pose_lower, self.pose_upper)
            if verbose:
                print(f"  Pose init: rotmat-based for lower body DOFs 0-16")

        poses = initial_poses.clone()
        if verbose:
            print(f"  Pose init: norm = {torch.norm(poses).item():.4f}")

        # Store DOF reference for q-matching loss (Phase 3)
        self._addb_q_reference = None
        self._addb_q_mask = None
        self._addb_q_confidence = None

        if skel_q_reference is not None and len(skel_q_reference) >= T:
            # Direct SKEL DOF reference (from rotation matrix mapping, lower body only)
            q_ref_arr = np.array(skel_q_reference[:T])
            q_ref = torch.from_numpy(q_ref_arr).float().to(self.device)
            q_mask = torch.ones(SKEL_NUM_POSE_DOF, device=self.device)
            q_conf = torch.ones(SKEL_NUM_POSE_DOF, device=self.device)
            self._addb_q_reference = q_ref
            self._addb_q_mask = q_mask
            self._addb_q_confidence = q_conf
            if verbose:
                print(f"  q-match reference: direct SKEL DOF (all {SKEL_NUM_POSE_DOF} DOFs)")
        elif addb_poses is not None and len(addb_poses) >= T:
            # Legacy: AddB DOF reference mapped via scale factors
            addb_arr = np.array(addb_poses[:T])
            addb_t = torch.from_numpy(addb_arr).float().to(self.device)
            q_ref = torch.zeros(T, SKEL_NUM_POSE_DOF, device=self.device)
            q_mask = torch.zeros(SKEL_NUM_POSE_DOF, device=self.device)
            q_conf = torch.zeros(SKEL_NUM_POSE_DOF, device=self.device)
            for (ai, si, scale) in _ADDB_TO_SKEL_DOF_MAP:
                if ai < addb_t.shape[1]:
                    q_ref[:, si] = addb_t[:, ai] * scale
                    q_mask[si] = 1.0
                    q_conf[si] = abs(scale)
            self._addb_q_reference = q_ref
            self._addb_q_mask = q_mask
            self._addb_q_confidence = q_conf
            if verbose:
                n_mapped = int(q_mask.sum().item())
                print(f"  q-match reference: {n_mapped} mapped DOFs (scale-based)")

        # Fix E: Tighten DOF limits around q_reference
        if getattr(self.config, 'use_tight_dof_limits', False) and self._addb_q_reference is not None:
            margin = self.config.tight_dof_margin_rad
            q_ref_median = self._addb_q_reference.median(dim=0).values  # [46]
            tight_lower = q_ref_median - margin
            tight_upper = q_ref_median + margin
            # Only tighten (intersection with original limits), never widen
            self.pose_lower = torch.max(self.pose_lower, tight_lower)
            self.pose_upper = torch.min(self.pose_upper, tight_upper)
            # Re-clamp initial poses
            initial_poses = clamp_poses(initial_poses, self.pose_lower, self.pose_upper)
            if verbose:
                n_tightened = ((tight_lower > self.pose_lower).sum() + (tight_upper < self.pose_upper).sum()).item()
                print(f"  Tight DOF limits: margin=±{np.degrees(margin):.0f}° (tightened DOFs)")

        # Fix H: Adaptive pelvis rotation limit (only DOF 2)
        # Set pelvis_rot limit to [median - margin, median + margin]
        # Auto mode: only apply when q_ref pelvis_rot median is outside default ±90° limit
        apply_adaptive = getattr(self.config, 'use_adaptive_pelvis_limit', False)
        if getattr(self.config, 'auto_adaptive_pelvis', False) and self._addb_q_reference is not None:
            med = self._addb_q_reference[:, 2].median().item()
            # Apply adaptive only when median is outside the default ±45° zone
            # (i.e., subject walks at an angle, prone to oscillation)
            if abs(med) > 0.785:  # > 45°
                apply_adaptive = True
        if apply_adaptive and self._addb_q_reference is not None:
            margin = getattr(self.config, 'adaptive_pelvis_margin_rad', 0.785)  # default ±45°
            pelvis_rot_median = self._addb_q_reference[:, 2].median().item()
            self.pose_lower[2] = pelvis_rot_median - margin
            self.pose_upper[2] = pelvis_rot_median + margin
            initial_poses = clamp_poses(initial_poses, self.pose_lower, self.pose_upper)
            if verbose:
                print(f"  Adaptive pelvis limit: [{np.degrees(pelvis_rot_median - margin):.1f}°, "
                      f"{np.degrees(pelvis_rot_median + margin):.1f}°] "
                      f"(median={np.degrees(pelvis_rot_median):.1f}°, margin=±{np.degrees(margin):.0f}°)")

        # Make betas a parameter for Stage 2
        betas = betas.clone()

        # 2-Stage Optimization Setup (following working compare_smpl_skel.py)
        total_iters = self.config.pose_iters
        stage1_iters = total_iters // 2  # First half: pose only (betas fixed)
        stage2_iters = total_iters - stage1_iters  # Second half: all params

        if verbose:
            print(f"  2-Stage Optimization: Stage1={stage1_iters} iters (pose), Stage2={stage2_iters} iters (all)")

        # Stage 1: Pose + Trans only
        betas.requires_grad_(False)
        poses.requires_grad_(True)
        trans.requires_grad_(True)

        optimizer_stage1 = torch.optim.Adam([
            {'params': [poses], 'lr': 0.02},
            {'params': [trans], 'lr': 0.01}
        ])

        # Precompute targets
        target_bone_lengths = compute_bone_lengths(addb_joints, self.addb_bone_indices)
        target_bone_dirs = compute_bone_directions(addb_joints, self.addb_bone_indices)
        target_width = torch.norm(
            addb_joints[:, ADDB_ACROMIAL_R_IDX, :] - addb_joints[:, ADDB_ACROMIAL_L_IDX, :],
            dim=-1
        )

        best_loss = float('inf')
        best_poses = poses.clone()
        best_trans = trans.clone()
        best_betas = betas.clone()

        # =====================================================================
        # STAGE 0 (Optional): Multi-Resolution - Key Joints Only
        # =====================================================================
        if self.config.use_multi_resolution:
            if verbose:
                print(f"\n  === Stage 0: Multi-Resolution (key joints only) ===")
                print(f"    Key joint SKEL indices: {self.config.multi_res_key_joints}")

            # Build mask for key joints in the mapped joint list
            key_joint_mask = torch.zeros(len(self.skel_indices), dtype=torch.bool, device=self.device)
            for i, skel_idx in enumerate(self.skel_indices):
                if skel_idx in self.config.multi_res_key_joints:
                    key_joint_mask[i] = True

            if verbose:
                num_key = key_joint_mask.sum().item()
                print(f"    Using {num_key}/{len(self.skel_indices)} mapped joints in Stage 0")

            # Store original joint weights
            original_weights = self.joint_weights.clone()

            # Zero out weights for non-key joints (effectively ignoring them)
            self.joint_weights = self.joint_weights * key_joint_mask.float()
            # Renormalize so total weight is preserved
            if self.joint_weights.sum() > 0:
                self.joint_weights = self.joint_weights * (original_weights.sum() / self.joint_weights.sum())

            optimizer_stage0 = torch.optim.Adam([
                {'params': [poses], 'lr': 0.03},  # Higher LR for coarse alignment
                {'params': [trans], 'lr': 0.02}
            ])

            pbar = tqdm(range(self.config.multi_res_stage1_iters), disable=not verbose, desc="Stage0")
            for it in pbar:
                if self.config.use_cosine_lr:
                    lr_mult = (1 + math.cos(math.pi * it / self.config.multi_res_stage1_iters)) / 2
                    self._update_optimizer_lr(optimizer_stage0, lr_mult)

                optimizer_stage0.zero_grad()

                skel_verts, skel_joints, _ = self.skel.forward(
                    betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=self._dJ
                )

                # Simplified loss for key joints only
                pred_joints_mapped = skel_joints[:, self.skel_indices, :]
                target_joints_stage0 = addb_joints[:, self.addb_indices, :]
                loss = self._compute_weighted_joint_loss(pred_joints_mapped, target_joints_stage0)

                # Add pose regularization
                loss = loss + self.config.weight_pose_reg * (poses ** 2).mean()

                loss.backward()
                optimizer_stage0.step()

                # Clamp poses (skip if direction_first — Phase A handles its own clamping)
                if not self.config.use_soft_constraints and not self.config.use_direction_first:
                    with torch.no_grad():
                        poses.data = clamp_poses(poses.data, self.pose_lower, self.pose_upper)

                if verbose:
                    with torch.no_grad():
                        mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mpjpe': f'{mpjpe:.1f}mm'})

            # Restore original weights for subsequent stages
            self.joint_weights = original_weights

            if verbose:
                with torch.no_grad():
                    skel_verts, skel_joints, _ = self.skel.forward(
                        betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=self._dJ
                    )
                    mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
                print(f"    Stage 0 complete: MPJPE = {mpjpe:.1f} mm")

        # =====================================================================
        # STAGE 1: Pose + Trans optimization (betas fixed)
        # =====================================================================
        if self.config.use_direction_first:
            # =============================================================
            # Direction-First: Phase A (direction convergence, no DOF limits)
            #                + Phase B (DOF limits gradually applied)
            # =============================================================
            phaseA_iters = self.config.phaseA_iters
            phaseB_iters = self.config.phaseB_iters

            if verbose:
                print(f"\n  === Stage 1: Direction-First Optimization ===")
                print(f"    Phase A: {phaseA_iters} iters (direction, NO DOF limits)")
                print(f"    Phase B: {phaseB_iters} iters (soft→hard DOF limits)")

            # --- Phase A: Direction convergence (no clamping) ---
            optimizer_phaseA = torch.optim.Adam([
                {'params': [poses], 'lr': self.config.phaseA_lr},
                {'params': [trans], 'lr': 0.01}
            ])

            # Save original config weights to restore after Phase A
            orig_weight_bone_dir = self.config.weight_bone_dir
            orig_weight_joint = self.config.weight_joint
            orig_weight_bone_len = self.config.weight_bone_len
            orig_weight_pose_reg = self.config.weight_pose_reg
            orig_weight_marker = self.config.weight_marker

            # Override weights for Phase A (direction-focused)
            self.config.weight_bone_dir = self.config.phaseA_weight_bone_dir
            self.config.weight_joint = self.config.phaseA_weight_joint
            self.config.weight_bone_len = self.config.phaseA_weight_bone_len
            self.config.weight_pose_reg = self.config.phaseA_weight_pose_reg
            self.config.weight_marker = 0.0  # No marker pos loss in Phase A

            pbar = tqdm(range(phaseA_iters), disable=not verbose, desc="PhaseA")
            for it in pbar:
                if self.config.use_cosine_lr:
                    lr_mult = (1 + math.cos(math.pi * it / phaseA_iters)) / 2
                    self._update_optimizer_lr(optimizer_phaseA, lr_mult)

                optimizer_phaseA.zero_grad()

                skel_verts, skel_joints, _ = self.skel.forward(
                    betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=self._dJ
                )

                # Direction-focused loss (marker_dir_loss added in _compute_optimization_loss)
                loss = self._compute_optimization_loss(
                    skel_verts, skel_joints, poses, addb_joints,
                    target_bone_lengths, target_bone_dirs, target_width,
                    use_temporal, T, include_betas_reg=False,
                    include_marker_loss=False,  # No marker position loss
                    iteration=it, total_iters=phaseA_iters,
                    phase='A', trans=trans,
                )

                loss.backward()

                # Fix B: zero out gradients for pelvis DOFs before step
                if getattr(self.config, 'freeze_pelvis_in_phaseA', False):
                    poses.grad.data[:, :3] = 0.0

                optimizer_phaseA.step()

                # Enforce anatomical limits even in Phase A to prevent wrapping
                with torch.no_grad():
                    poses.data = clamp_poses(poses.data, self.pose_lower, self.pose_upper)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_poses = poses.clone().detach()
                    best_trans = trans.clone().detach()

                if verbose:
                    with torch.no_grad():
                        mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mpjpe': f'{mpjpe:.1f}mm'})

            # Log Phase A → Phase B transition
            if verbose:
                with torch.no_grad():
                    oob_count = ((poses.data < self.pose_lower) | (poses.data > self.pose_upper)).sum().item()
                    total_dofs = poses.data.numel()
                    print(f"\n    Phase A complete: {oob_count}/{total_dofs} DOF values out-of-bounds")

            # Restore original weights for Phase B
            self.config.weight_bone_dir = orig_weight_bone_dir
            self.config.weight_joint = orig_weight_joint
            self.config.weight_bone_len = orig_weight_bone_len
            self.config.weight_pose_reg = orig_weight_pose_reg
            self.config.weight_marker = orig_weight_marker

            # --- Phase B: DOF limits gradually applied ---
            optimizer_phaseB = torch.optim.Adam([
                {'params': [poses], 'lr': self.config.phaseB_lr},
                {'params': [trans], 'lr': 0.005}
            ])

            no_improve_count = 0
            hard_clamp_start_iter = int(phaseB_iters * self.config.phaseB_hard_clamp_start)

            pbar = tqdm(range(phaseB_iters), disable=not verbose, desc="PhaseB")
            for it in pbar:
                if self.config.use_cosine_lr:
                    lr_mult = (1 + math.cos(math.pi * it / phaseB_iters)) / 2
                    self._update_optimizer_lr(optimizer_phaseB, lr_mult)

                optimizer_phaseB.zero_grad()

                skel_verts, skel_joints, _ = self.skel.forward(
                    betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=self._dJ
                )

                loss = self._compute_optimization_loss(
                    skel_verts, skel_joints, poses, addb_joints,
                    target_bone_lengths, target_bone_dirs, target_width,
                    use_temporal, T, include_betas_reg=False,
                    include_marker_loss=self.config.marker_in_stage1,
                    iteration=it, total_iters=phaseB_iters,
                    phase='B', trans=trans,
                )

                # Soft constraint: ramp weight from phaseB_soft_constraint_weight to 1.0
                soft_w = self.config.phaseB_soft_constraint_weight + \
                    (1.0 - self.config.phaseB_soft_constraint_weight) * (it / max(phaseB_iters - 1, 1))
                soft_penalty = compute_soft_limit_penalty(poses, self.pose_lower, self.pose_upper)
                loss = loss + soft_w * soft_penalty

                loss.backward()
                optimizer_phaseB.step()

                # Hard clamp only in second half of Phase B
                if it >= hard_clamp_start_iter:
                    with torch.no_grad():
                        poses.data = clamp_poses(poses.data, self.pose_lower, self.pose_upper)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_poses = poses.clone().detach()
                    best_trans = trans.clone().detach()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if self.config.use_early_stopping and no_improve_count >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"\n    Phase B early stop at iter {it+1}")
                    break

                if verbose:
                    with torch.no_grad():
                        mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mpjpe': f'{mpjpe:.1f}mm'})

        else:
            # =============================================================
            # Original Stage 1 (fallback when direction_first=False)
            # =============================================================
            if verbose:
                print(f"\n  === Stage 1: Pose Optimization (betas fixed) ===")
                if self.config.use_cosine_lr:
                    print(f"    Using cosine annealing LR schedule")
                if self.config.use_huber_loss:
                    print(f"    Using Huber loss (delta={self.config.huber_delta})")
                if self.config.use_early_stopping:
                    print(f"    Early stopping patience: {self.config.early_stopping_patience}")

            no_improve_count = 0

            pbar = tqdm(range(stage1_iters), disable=not verbose, desc="Stage1")
            for it in pbar:
                if self.config.use_cosine_lr:
                    lr_mult = (1 + math.cos(math.pi * it / stage1_iters)) / 2
                    self._update_optimizer_lr(optimizer_stage1, lr_mult)

                optimizer_stage1.zero_grad()

                skel_verts, skel_joints, _ = self.skel.forward(
                    betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=self._dJ
                )

                loss = self._compute_optimization_loss(
                    skel_verts, skel_joints, poses, addb_joints,
                    target_bone_lengths, target_bone_dirs, target_width,
                    use_temporal, T, include_betas_reg=False,
                    include_marker_loss=self.config.marker_in_stage1,
                    iteration=it, total_iters=stage1_iters,
                    trans=trans,
                )

                if self.config.use_soft_constraints:
                    constraint_penalty = self._compute_soft_constraint_penalty(poses)
                    loss = loss + self.config.soft_constraint_weight * constraint_penalty

                loss.backward()
                optimizer_stage1.step()

                if not self.config.use_soft_constraints:
                    with torch.no_grad():
                        poses.data = clamp_poses(poses.data, self.pose_lower, self.pose_upper)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_poses = poses.clone().detach()
                    best_trans = trans.clone().detach()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if self.config.use_early_stopping and no_improve_count >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"\n    Early stopping at iteration {it+1} (no improvement for {no_improve_count} iters)")
                    break

                if verbose:
                    with torch.no_grad():
                        mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
                    current_lr = optimizer_stage1.param_groups[0]['lr']
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mpjpe': f'{mpjpe:.1f}mm', 'lr': f'{current_lr:.4f}'})

        # =====================================================================
        # Between Stages: Update dynamic joint weights if enabled
        # =====================================================================
        if self.config.use_dynamic_weights:
            with torch.no_grad():
                # Compute current predictions with best poses
                skel_verts, skel_joints, _ = self.skel.forward(
                    betas.unsqueeze(0).expand(T, -1), best_poses, best_trans, dJ=self._dJ
                )
                pred_joints_mapped = skel_joints[:, self.skel_indices, :]
                target_joints_mapped = addb_joints[:, self.addb_indices, :]
                self._update_dynamic_weights(pred_joints_mapped, target_joints_mapped)
                if verbose:
                    print(f"\n  Dynamic weights updated based on per-joint errors")

        # =====================================================================
        # STAGE 2: All parameters optimization (betas + poses + trans)
        # =====================================================================
        if verbose:
            print(f"\n  === Stage 2: Joint Optimization (all params) ===")

        # Enable gradients for all parameters
        betas.requires_grad_(True)
        poses.requires_grad_(True)
        trans.requires_grad_(True)

        # Lower learning rates for fine-tuning
        # Beta LR much lower to preserve Stage 1 bone lengths (especially shin)
        beta_lr = getattr(self.config, 'stage2_beta_lr', 0.002)
        optimizer_stage2 = torch.optim.Adam([
            {'params': [betas], 'lr': beta_lr},
            {'params': [poses], 'lr': 0.01},
            {'params': [trans], 'lr': 0.005}
        ])

        # Reset early stopping counter for Stage 2
        no_improve_count = 0
        stage2_best_loss = best_loss  # Track Stage 2 specific improvement

        pbar = tqdm(range(stage2_iters), disable=not verbose, desc="Stage2")
        for it in pbar:
            # Cosine annealing learning rate
            if self.config.use_cosine_lr:
                lr_mult = (1 + math.cos(math.pi * it / stage2_iters)) / 2
                self._update_optimizer_lr(optimizer_stage2, lr_mult)

            optimizer_stage2.zero_grad()

            # Forward through SKEL
            skel_verts, skel_joints, _ = self.skel.forward(
                betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=self._dJ
            )

            # Loss computation (include betas regularization)
            # Note: marker_in_stage2 is used here, but in 2-pass mode
            # marker_handler is None for Pass 1 (set by pipeline.py)
            loss = self._compute_optimization_loss(
                skel_verts, skel_joints, poses, addb_joints,
                target_bone_lengths, target_bone_dirs, target_width,
                use_temporal, T, include_betas_reg=True, betas=betas,
                include_marker_loss=self.config.marker_in_stage2,
                iteration=it, total_iters=stage2_iters,
                trans=trans,
            )

            # Soft constraint penalty: use direction_first soft penalty OR legacy soft constraints
            if self.config.use_direction_first:
                soft_penalty = compute_soft_limit_penalty(poses, self.pose_lower, self.pose_upper)
                loss = loss + 1.0 * soft_penalty
            elif self.config.use_soft_constraints:
                constraint_penalty = self._compute_soft_constraint_penalty(poses)
                loss = loss + self.config.soft_constraint_weight * constraint_penalty

            loss.backward()
            optimizer_stage2.step()

            # Hard clamp: always apply in Stage 2
            with torch.no_grad():
                poses.data = clamp_poses(poses.data, self.pose_lower, self.pose_upper)

            # Track best result (overall best)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_poses = poses.clone().detach()
                best_trans = trans.clone().detach()
                best_betas = betas.clone().detach()

            # Track Stage 2 specific improvement for early stopping
            if loss.item() < stage2_best_loss:
                stage2_best_loss = loss.item()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping
            if self.config.use_early_stopping and no_improve_count >= self.config.early_stopping_patience:
                if verbose:
                    print(f"\n    Early stopping at iteration {it+1} (no improvement for {no_improve_count} iters)")
                break

            if verbose:
                with torch.no_grad():
                    mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
                current_lr = optimizer_stage2.param_groups[0]['lr']
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mpjpe': f'{mpjpe:.1f}mm', 'lr': f'{current_lr:.4f}'})

        # =====================================================================
        # POST-OPTIMIZATION: Twist Detection & Correction
        # NOTE: Disabled for now — threshold too sensitive (flags 100% of frames)
        # TODO: Re-enable with higher threshold (5+ DOFs at 98%) after baseline validation
        # =====================================================================
        twist_stats = {'n_suspicious_frames': 0, 'total_frames': T, 'suspicious_frame_indices': []}
        # best_poses, twist_stats = self._detect_and_fix_twists(
        #     best_poses, best_betas, best_trans, addb_joints,
        #     target_bone_lengths, target_bone_dirs, target_width,
        #     use_temporal, T, verbose
        # )

        # Final statistics
        with torch.no_grad():
            skel_verts, skel_joints, _ = self.skel.forward(
                best_betas.unsqueeze(0).expand(T, -1), best_poses, best_trans, dJ=self._dJ
            )
            mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
            per_joint_error = self._compute_per_joint_error(skel_joints, addb_joints)

        # Marker diagnostics
        marker_stats = {}
        if self.marker_handler is not None and self.config.use_marker_loss:
            marker_stats = self.marker_handler.get_marker_statistics(skel_joints, skel_verts)

        stats = {
            'final_loss': best_loss,
            'mpjpe_mm': mpjpe,
            'per_joint_error_mm': per_joint_error,
            'scapula_dofs': self.scapula_handler.get_scapula_dof_values(best_poses.mean(dim=0)),
            'final_betas': best_betas,  # Return optimized betas
            'twist_stats': twist_stats,
            'marker_stats': marker_stats,
        }

        return best_poses, best_trans, stats

    def _detect_and_fix_twists(
        self,
        poses: torch.Tensor,
        betas: torch.Tensor,
        trans: torch.Tensor,
        addb_joints: torch.Tensor,
        target_bone_lengths: torch.Tensor,
        target_bone_dirs: torch.Tensor,
        target_width: torch.Tensor,
        use_temporal: bool,
        T: int,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Detect frames where DOFs hit pose limits (twisted/abnormal poses)
        and re-optimize those frames with stronger regularization.

        A DOF at >95% of its range is likely stuck in a local minimum.

        Args:
            poses: Optimized pose parameters [T, 46]
            betas, trans, addb_joints: For re-optimization
            target_*: Precomputed optimization targets
            use_temporal, T: Optimization settings
            verbose: Print diagnostics

        Returns:
            Fixed poses [T, 46] and twist statistics dict
        """
        twist_threshold = 0.95  # 95% of range = suspicious
        reopt_iters = 50

        with torch.no_grad():
            pose_range = self.pose_upper - self.pose_lower
            # Normalized position within range: 0=lower, 1=upper
            normalized = (poses - self.pose_lower) / (pose_range + 1e-8)
            # Frames where ANY non-root DOF is at >95% of range
            # Skip root DOFs (0-2: global rotation) which can legitimately be large
            at_limit = (normalized[:, 3:] > twist_threshold) | (normalized[:, 3:] < (1 - twist_threshold))
            # Count how many DOFs are at limit per frame
            dofs_at_limit = at_limit.sum(dim=1)  # [T]
            # Flag frames with 3+ DOFs at limit as suspicious
            suspicious_frames = (dofs_at_limit >= 3).nonzero(as_tuple=True)[0]

        n_suspicious = len(suspicious_frames)
        twist_stats = {
            'n_suspicious_frames': n_suspicious,
            'total_frames': T,
            'suspicious_frame_indices': suspicious_frames.cpu().tolist() if n_suspicious > 0 else [],
        }

        if n_suspicious == 0:
            if verbose:
                print(f"\n  Twist detection: no suspicious frames found")
            return poses, twist_stats

        if verbose:
            print(f"\n  Twist detection: {n_suspicious}/{T} suspicious frames, re-optimizing with stronger regularization...")

        # Re-optimize suspicious frames with 10x pose regularization
        fixed_poses = poses.clone().detach()
        original_pose_reg = self.config.weight_pose_reg

        for frame_idx in suspicious_frames:
            fi = frame_idx.item()
            frame_pose = fixed_poses[fi:fi+1].clone().requires_grad_(True)
            frame_trans = trans[fi:fi+1].clone().requires_grad_(True)
            frame_target = addb_joints[fi:fi+1]

            # Use neighboring frame as initialization if available (temporal prior)
            if fi > 0:
                frame_pose = fixed_poses[fi-1:fi].clone().requires_grad_(True)

            optimizer_fix = torch.optim.Adam([frame_pose, frame_trans], lr=0.005)

            # Temporarily increase regularization
            self.config.weight_pose_reg = original_pose_reg * 10.0

            for _ in range(reopt_iters):
                optimizer_fix.zero_grad()
                skel_verts, skel_joints, _ = self.skel.forward(
                    betas.unsqueeze(0), frame_pose, frame_trans, dJ=self._dJ
                )
                loss = self._compute_optimization_loss(
                    skel_verts, skel_joints, frame_pose, frame_target,
                    target_bone_lengths[fi:fi+1], target_bone_dirs[fi:fi+1],
                    target_width[fi:fi+1], False, 1,
                    include_betas_reg=False
                )
                loss.backward()
                optimizer_fix.step()
                with torch.no_grad():
                    frame_pose.data = clamp_poses(frame_pose.data, self.pose_lower, self.pose_upper)

            fixed_poses[fi] = frame_pose.detach().squeeze(0)

        # Restore original regularization weight
        self.config.weight_pose_reg = original_pose_reg

        if verbose:
            print(f"    Re-optimized {n_suspicious} frames")

        return fixed_poses, twist_stats

    def _compute_optimization_loss(
        self,
        skel_verts: torch.Tensor,
        skel_joints: torch.Tensor,
        poses: torch.Tensor,
        addb_joints: torch.Tensor,
        target_bone_lengths: torch.Tensor,
        target_bone_dirs: torch.Tensor,
        target_width: torch.Tensor,
        use_temporal: bool,
        T: int,
        include_betas_reg: bool = False,
        betas: Optional[torch.Tensor] = None,
        include_marker_loss: bool = False,
        iteration: int = -1,
        total_iters: int = -1,
        phase: Optional[str] = None,
        trans: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined optimization loss.

        Args:
            skel_verts: SKEL mesh vertices
            skel_joints: SKEL joint positions
            poses: Current pose parameters
            addb_joints: Target AddB joint positions
            target_bone_lengths: Precomputed target bone lengths
            target_bone_dirs: Precomputed target bone directions
            target_width: Precomputed target shoulder width
            use_temporal: Whether to include temporal smoothness
            T: Number of frames
            include_betas_reg: Whether to include betas regularization
            betas: Current betas (required if include_betas_reg=True)
            include_marker_loss: Whether to include marker position loss
            iteration: Current iteration (for warm-up schedule, -1 to disable)
            total_iters: Total iterations (for warm-up schedule)
            phase: 'A' for Phase A (direction-first), 'B' for Phase B, None for default

        Returns:
            Combined loss scalar
        """
        # 1. Joint position loss (weighted, includes acromial→humerus)
        pred_joints_mapped = skel_joints[:, self.skel_indices, :]
        target_joints = addb_joints[:, self.addb_indices, :]
        joint_loss = self._compute_weighted_joint_loss(pred_joints_mapped, target_joints)

        # 2. Bone direction loss
        pred_bone_dirs = compute_bone_directions(skel_joints, self.skel_bone_indices)
        bone_dir_loss = cosine_similarity_loss(pred_bone_dirs, target_bone_dirs)

        # 3. Bone length loss (weighted per-bone: shin priority)
        pred_bone_lengths = compute_bone_lengths(skel_joints, self.skel_bone_indices)
        bone_len_diff_sq = (pred_bone_lengths - target_bone_lengths) ** 2  # [T, num_bones]
        bone_len_loss = (self.bone_weights.unsqueeze(0) * bone_len_diff_sq).mean()

        # 4. Shoulder width loss
        pred_width = self.skel.get_shoulder_width(skel_joints)
        width_loss = F.mse_loss(pred_width, target_width)

        # 5. Shoulder/scapula losses
        shoulder_losses = compute_shoulder_losses(
            skel_verts, skel_joints, poses, addb_joints,
            self.scapula_handler, self.config
        )

        # 6. Pose regularization
        pose_reg = self.config.weight_pose_reg * (poses ** 2).mean()

        # 7. Spine regularization
        spine_dofs = poses[:, SPINE_DOF_INDICES]
        spine_reg = self.config.weight_spine_reg * (spine_dofs ** 2).mean()

        # 7a. Arm regularization (elbow/wrist — prevents over-flexion)
        arm_dofs = poses[:, ARM_DOF_INDICES]
        arm_reg = self.config.weight_arm_reg * (arm_dofs ** 2).mean()

        # 7b. Foot DOF regularization (subtalar + mtp → keep close to init/0)
        # AddB subtalar/mtp are typically 0; SKEL subtalar/mtp should not drift freely
        foot_reg = torch.tensor(0.0, device=self.device)
        foot_reg_weight = getattr(self.config, 'weight_foot_reg', 0.0)
        if foot_reg_weight > 0:
            # DOF indices: subtalar_r=8, mtp_r=9, subtalar_l=15, mtp_l=16
            foot_dof_indices = [8, 9, 15, 16]
            foot_dofs = poses[:, foot_dof_indices]
            foot_reg = foot_reg_weight * (foot_dofs ** 2).mean()

        # 7c. Foot height (Y-axis) loss — GRF critical
        # Match heel/toe Y position to GT, reducing vertical offset
        foot_height_loss = torch.tensor(0.0, device=self.device)
        foot_height_weight = getattr(self.config, 'weight_foot_height', 0.0)
        if foot_height_weight > 0:
            # SKEL indices: calcn_r=4, toes_r=5, calcn_l=9, toes_l=10
            # AddB indices: subtalar_r=4, mtp_r=5, subtalar_l=9, mtp_l=10
            skel_foot_idx = [4, 5, 9, 10]  # calcn_r, toes_r, calcn_l, toes_l
            addb_foot_idx = [4, 5, 9, 10]  # subtalar_r, mtp_r, subtalar_l, mtp_l
            pred_foot_y = skel_joints[:, skel_foot_idx, 1]  # [T, 4] Y only
            target_foot_y = addb_joints[:, addb_foot_idx, 1]  # [T, 4] Y only
            # Per-joint weight: calcn (idx 0, 2) = APPROXIMATE (subtalar-mapped), toes (idx 1, 3) = DIRECT
            approx_f = getattr(self.config, 'approximate_foot_weight_factor', 1.0)
            foot_w = torch.tensor([approx_f, 1.0, approx_f, 1.0], device=self.device)
            foot_height_loss = foot_height_weight * (foot_w.view(1, -1) * (pred_foot_y - target_foot_y) ** 2).mean()

        # 8. Temporal smoothness (velocity + acceleration)
        temporal_loss = torch.tensor(0.0, device=self.device)
        dt = self.config.dt if self.config.adaptive_temporal else 1.0
        if use_temporal and T > 1:
            # Pose velocity smoothness (dt-adaptive: normalize by dt → velocity units)
            pose_diff = poses[1:] - poses[:-1]
            temporal_loss = self.config.weight_temporal * ((pose_diff / dt) ** 2).mean()
            # Translation velocity smoothness
            if trans is not None:
                trans_diff = trans[1:] - trans[:-1]
                temporal_loss = temporal_loss + self.config.weight_temporal * ((trans_diff / dt) ** 2).mean()
        if use_temporal and T > 2 and self.config.weight_acceleration > 0:
            # Pose acceleration smoothness (dt-adaptive: normalize by dt² → acceleration units)
            pose_acc = poses[2:] - 2 * poses[1:-1] + poses[:-2]
            temporal_loss = temporal_loss + self.config.weight_acceleration * ((pose_acc / (dt ** 2)) ** 2).mean()
            # Translation acceleration smoothness
            if trans is not None:
                trans_acc = trans[2:] - 2 * trans[1:-1] + trans[:-2]
                temporal_loss = temporal_loss + self.config.weight_acceleration * ((trans_acc / (dt ** 2)) ** 2).mean()
        # 8a. Cartesian joint smoothness (operates on FK joint positions)
        cartesian_smooth_loss = torch.tensor(0.0, device=self.device)
        if use_temporal and T > 1 and self.config.weight_cartesian_temporal > 0:
            joint_vel = skel_joints[1:] - skel_joints[:-1]
            cartesian_smooth_loss = self.config.weight_cartesian_temporal * ((joint_vel / dt) ** 2).mean()
        if use_temporal and T > 2 and self.config.weight_cartesian_acceleration > 0:
            joint_acc = skel_joints[2:] - 2 * skel_joints[1:-1] + skel_joints[:-2]
            cartesian_smooth_loss = cartesian_smooth_loss + self.config.weight_cartesian_acceleration * ((joint_acc / (dt ** 2)) ** 2).mean()
        # 8b. DOF-space q-matching loss (Phase 3)
        # Two modes: rotation matrix comparison (HSMR-style, default) or Euler angle comparison (legacy)
        q_match_loss = torch.tensor(0.0, device=self.device)
        w_q = getattr(self.config, 'weight_q_match', 0.0)
        if w_q > 0 and self._addb_q_reference is not None:
            warmup = getattr(self.config, 'q_match_warmup_fraction', 0.3)
            progress = iteration / total_iters if total_iters > 0 else 1.0
            if progress >= warmup:
                ramp = (progress - warmup) / (1.0 - warmup)
                q_diff = (poses - self._addb_q_reference[:T]) * self._addb_q_mask
                if getattr(self.config, 'q_match_confidence_weighted', True):
                    q_diff = q_diff * self._addb_q_confidence
                q_match_loss = w_q * ramp * (q_diff ** 2).mean()

        # 8c. Chirality loss (Method A): penalize L/R swap
        # cross(hip_R - pelvis, hip_L - pelvis) should point UP (+Y)
        # If body is 180° flipped, cross product points DOWN → large penalty
        chirality_loss = torch.tensor(0.0, device=self.device)
        w_chiral = getattr(self.config, 'weight_chirality', 0.0)
        if w_chiral > 0:
            # SKEL joint indices: pelvis=0, femur_r=1, femur_l=6
            pelvis = skel_joints[:, 0, :]     # [T, 3]
            hip_r = skel_joints[:, 1, :]      # [T, 3]
            hip_l = skel_joints[:, 6, :]      # [T, 3]
            v_r = hip_r - pelvis              # [T, 3]
            v_l = hip_l - pelvis              # [T, 3]
            cross = torch.cross(v_r, v_l, dim=-1)  # [T, 3]
            # GT: same computation on target joints
            gt_pelvis = addb_joints[:, 0, :]
            gt_hip_r = addb_joints[:, 1, :]
            gt_hip_l = addb_joints[:, 6, :]
            gt_vr = gt_hip_r - gt_pelvis
            gt_vl = gt_hip_l - gt_pelvis
            gt_cross = torch.cross(gt_vr, gt_vl, dim=-1)
            # Penalize if cross product Y-component has different sign from GT
            # Use dot product: should be positive (same direction)
            dot_y = cross[:, 1] * gt_cross[:, 1]  # [T]
            # ReLU(-dot_y): penalty only when sign differs
            chirality_loss = w_chiral * torch.relu(-dot_y).mean()

        # 9. Betas regularization (only in Stage 2)
        betas_reg = torch.tensor(0.0, device=self.device)
        if include_betas_reg and betas is not None:
            betas_reg = 0.005 * (betas ** 2).mean()

        # 10. Scapulohumeral coupling constraint (Phase 3)
        scapulohumeral_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_scapulohumeral_coupling:
            scapulohumeral_loss = (
                self.config.scapulohumeral_weight *
                self._compute_scapulohumeral_coupling_loss(poses)
            )

        # 11. Marker position loss (with warm-up schedule + bony/soft weighting)
        marker_loss = torch.tensor(0.0, device=self.device)
        if (include_marker_loss
                and self.marker_handler is not None
                and self.config.use_marker_loss):
            # Warm-up schedule: 0→warmup_fraction = joints only,
            # warmup_fraction→1.0 = linear ramp to full marker weight
            marker_weight = self.config.weight_marker
            warmup_frac = self.config.marker_warmup_fraction
            if iteration >= 0 and total_iters > 0:
                progress = iteration / total_iters
                if progress < warmup_frac:
                    marker_weight = 0.0  # Pure joint optimization
                else:
                    # Linear ramp from 0 to full weight
                    ramp = (progress - warmup_frac) / (1.0 - warmup_frac)
                    marker_weight = self.config.weight_marker * ramp

            if marker_weight > 0:
                tier_wf = getattr(self.config, 'tier_weight_factors', None)
                use_adaptive = getattr(self.config, 'use_adaptive_marker_weight', False)
                adaptive_thr = getattr(self.config, 'adaptive_weight_threshold_mm', 50.0)

                # Bony/soft marker weighting (P2b)
                bony_soft_w = None
                if self.config.use_bony_soft_weighting:
                    bony_soft_w = {
                        'bony': self.config.bony_marker_weight,
                        'soft': self.config.soft_marker_weight,
                        'head_boost': getattr(self.config, 'head_marker_weight_boost', 1.0),
                    }

                marker_loss = marker_weight * self.marker_handler.compute_marker_loss(
                    skel_joints, skel_verts,
                    tier_weight_factors=tier_wf,
                    use_adaptive_weight=use_adaptive,
                    adaptive_threshold_mm=adaptive_thr,
                    bony_soft_weights=bony_soft_w,
                    upper_body_marker_boost=getattr(self.config, 'upper_body_marker_boost', 1.0),
                    head_marker_boost=getattr(self.config, 'head_marker_boost', 1.0),
                )

        # 12. Marker-pair direction loss (P2a)
        marker_dir_loss = torch.tensor(0.0, device=self.device)
        w_marker_dir = self.config.weight_marker_direction
        # In Phase A, use phaseA weight override
        if phase == 'A':
            w_marker_dir = self.config.phaseA_weight_marker_direction
        if w_marker_dir > 0 and self.marker_handler is not None and skel_verts is not None:
            if hasattr(self.marker_handler, 'compute_marker_direction_loss'):
                marker_dir_loss = w_marker_dir * self.marker_handler.compute_marker_direction_loss(
                    skel_verts,
                    bony_only=(phase == 'A' and self.config.phaseA_bony_only),
                )

        # Combine losses
        loss = (
            self.config.weight_joint * joint_loss +
            self.config.weight_bone_dir * bone_dir_loss +
            self.config.weight_bone_len * bone_len_loss +
            self.config.weight_width * width_loss +
            shoulder_losses['acromial'] +
            shoulder_losses['humerus_align'] +
            shoulder_losses['humerus_on_line'] +
            shoulder_losses['scapula_reg'] +
            shoulder_losses['humerus_reg'] +
            pose_reg +
            spine_reg +
            arm_reg +
            foot_reg +
            foot_height_loss +
            temporal_loss +
            cartesian_smooth_loss +
            q_match_loss +
            chirality_loss +
            betas_reg +
            scapulohumeral_loss +
            marker_loss +
            marker_dir_loss
        )

        return loss

    def _compute_mpjpe(
        self,
        skel_joints: torch.Tensor,
        addb_joints: torch.Tensor,
        exclude_acromial: bool = True,
    ) -> float:
        """
        Compute mean per-joint position error in mm.

        Uses MappingType from joint_definitions to exclude non-equivalent joints:
        - PROXY joints (acromial) are always excluded when exclude_acromial=True
        - APPROXIMATE joints (subtalar, back) are also excluded
        - Only DIRECT joints (15 anatomically equivalent pairs) contribute to MPJPE

        Args:
            skel_joints: SKEL joint positions [T, 24, 3]
            addb_joints: AddB joint positions [T, 20, 3]
            exclude_acromial: If True, exclude PROXY and APPROXIMATE joints.

        Returns:
            MPJPE in mm
        """
        from .joint_definitions import ADDB_JOINTS, ADDB_TO_SKEL_MAPPING, MappingType

        pred = skel_joints[:, self.skel_indices, :]
        target = addb_joints[:, self.addb_indices, :]

        if exclude_acromial:
            exclude_indices = []
            for i, addb_idx in enumerate(self.addb_indices):
                addb_name = ADDB_JOINTS[addb_idx]
                mapping = ADDB_TO_SKEL_MAPPING.get(addb_name)
                if mapping and mapping.mapping_type in (MappingType.PROXY, MappingType.APPROXIMATE):
                    exclude_indices.append(i)

            include_mask = torch.ones(len(self.addb_indices), dtype=torch.bool, device=pred.device)
            for idx in exclude_indices:
                include_mask[idx] = False

            pred = pred[:, include_mask, :]
            target = target[:, include_mask, :]

        error = torch.norm(pred - target, dim=-1)
        return error.mean().item() * 1000  # Convert to mm

    def _compute_per_joint_error(
        self,
        skel_joints: torch.Tensor,
        addb_joints: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute per-joint error in mm."""
        from .joint_definitions import ADDB_JOINTS

        errors = {}
        for i, (addb_idx, skel_idx) in enumerate(zip(self.addb_indices, self.skel_indices)):
            pred = skel_joints[:, skel_idx, :]
            target = addb_joints[:, addb_idx, :]
            error = torch.norm(pred - target, dim=-1).mean().item() * 1000
            errors[ADDB_JOINTS[addb_idx]] = error

        return errors

    def optimize_sequence_sequential(
        self,
        addb_joints: torch.Tensor,
        betas: torch.Tensor,
        verbose: bool = False,
        addb_poses: Optional[np.ndarray] = None,
        dJ: Optional[torch.Tensor] = None,
        skel_q_reference: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Frame-by-frame sequential optimization with warm start.

        Each frame is initialized from the previous frame's result,
        ensuring rotation continuity (prevents pelvis rotation oscillation).

        Args:
            addb_joints: [T, 20, 3] in meters.
            betas: [10].
            skel_q_reference: [T, 46] direct SKEL DOF reference.
        """
        T = addb_joints.shape[0]
        n_iters = self.config.sequential_iters

        self._dJ = dJ.detach() if dJ is not None else None

        # Override pose_iters for single-frame calls
        orig_pose_iters = self.config.pose_iters
        self.config.pose_iters = n_iters

        if verbose:
            print(f"Sequential optimization: {T} frames × {n_iters} iters/frame")

        # Initialize from q_ref or zeros
        if skel_q_reference is not None and len(skel_q_reference) >= T:
            q_ref = torch.from_numpy(np.array(skel_q_reference[:T])).float().to(self.device)
        else:
            q_ref = None

        # Compute pelvis offset for translation init
        with torch.no_grad():
            poses_zero = torch.zeros(1, SKEL_NUM_POSE_DOF, device=self.device)
            trans_zero = torch.zeros(1, 3, device=self.device)
            _, joints_zero, _ = self.skel.forward(
                betas.unsqueeze(0), poses_zero, trans_zero
            )
            pelvis_offset = joints_zero[0, 0]

        all_poses = torch.zeros(T, SKEL_NUM_POSE_DOF, device=self.device)
        all_trans = torch.zeros(T, 3, device=self.device)

        pbar = tqdm(range(T), disable=not verbose, desc="Sequential")
        for t in pbar:
            # Determine initial pose for this frame
            if t == 0:
                if q_ref is not None:
                    init_pose = q_ref[0].clone()
                else:
                    init_pose = torch.zeros(SKEL_NUM_POSE_DOF, device=self.device)
            else:
                init_pose = all_poses[t - 1].clone()

            # Initial translation: align pelvis
            init_trans = addb_joints[t, 0, :].clone() - pelvis_offset

            # Run single-frame optimization
            pose_t, trans_t, stats_t = self.optimize_single_frame(
                addb_joints[t:t + 1],
                betas,
                initial_poses=init_pose,
                initial_trans=init_trans,
                verbose=False,
            )

            all_poses[t] = pose_t
            all_trans[t] = trans_t

            if verbose and (t % 50 == 0 or t == T - 1):
                pbar.set_postfix({'mpjpe': f"{stats_t['mpjpe_mm']:.1f}mm"})

        # Restore original pose_iters
        self.config.pose_iters = orig_pose_iters

        # Final statistics
        with torch.no_grad():
            skel_verts, skel_joints, _ = self.skel.forward(
                betas.unsqueeze(0).expand(T, -1), all_poses, all_trans, dJ=self._dJ
            )
            mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
            per_joint_error = self._compute_per_joint_error(skel_joints, addb_joints)

        stats = {
            'final_loss': 0.0,
            'mpjpe_mm': mpjpe,
            'per_joint_error_mm': per_joint_error,
            'scapula_dofs': self.scapula_handler.get_scapula_dof_values(all_poses.mean(dim=0)),
            'final_betas': betas.clone(),
            'twist_stats': {'n_suspicious_frames': 0, 'total_frames': T, 'suspicious_frame_indices': []},
            'marker_stats': {},
        }

        if verbose:
            print(f"  Sequential complete: MPJPE = {mpjpe:.1f} mm")

        return all_poses, all_trans, stats

    def optimize_sequence_sliding(
        self,
        addb_joints: torch.Tensor,
        betas: torch.Tensor,
        verbose: bool = False,
        addb_poses: Optional[np.ndarray] = None,
        dJ: Optional[torch.Tensor] = None,
        skel_q_reference: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Sliding window optimization with temporal loss within windows
        and warm start between windows.

        Window n+1 is initialized from window n's overlapping results,
        ensuring rotation continuity while maintaining temporal smoothness.

        Args:
            addb_joints: [T, 20, 3] in meters.
            betas: [10].
            skel_q_reference: [T, 46] direct SKEL DOF reference.
        """
        T = addb_joints.shape[0]
        W = self.config.window_size
        S = self.config.window_stride
        blend = self.config.window_blend_frames

        self._dJ = dJ.detach() if dJ is not None else None

        if verbose:
            n_windows = max(1, (T - W) // S + 1) + (1 if (T - W) % S > 0 else 0)
            print(f"Sliding window optimization: {T} frames, W={W}, S={S}, blend={blend}")
            print(f"  ~{n_windows} windows")

        all_poses = torch.zeros(T, SKEL_NUM_POSE_DOF, device=self.device)
        all_trans = torch.zeros(T, 3, device=self.device)
        processed = torch.zeros(T, dtype=torch.bool, device=self.device)

        window_idx = 0
        start = 0
        while start < T:
            end = min(start + W, T)
            window_T = end - start

            # Prepare q_ref for this window
            window_q_ref = None
            if skel_q_reference is not None:
                window_q_ref = skel_q_reference[start:end]

            # Warm start: if we have results from previous window, use them
            if start > 0 and processed[start].item():
                # Override q_ref init with previous window's results for overlap zone
                if window_q_ref is not None:
                    window_q_ref = np.array(window_q_ref)  # ensure numpy
                    overlap = min(W - S, window_T)
                    prev_poses_np = all_poses[start:start + overlap].cpu().numpy()
                    window_q_ref[:overlap] = prev_poses_np

            # Prepare window-local addb_poses
            window_addb_poses = None
            if addb_poses is not None:
                window_addb_poses = addb_poses[start:end]

            # Create a fresh optimizer with same config for this window
            # (resets internal state like q_reference)
            window_optimizer = PoseOptimizer(
                self.skel, self.config, marker_handler=self.marker_handler
            )

            # Run batch optimization for this window
            window_poses, window_trans, window_stats = window_optimizer.optimize_sequence(
                addb_joints[start:end],
                betas,
                use_temporal=True,
                verbose=False,
                addb_poses=window_addb_poses,
                dJ=dJ,
                skel_q_reference=window_q_ref,
            )

            # Blend with previous results in overlap zone
            if start > 0 and processed[start].item():
                blend_zone = min(blend, end - start)
                if blend_zone > 0:
                    alpha = torch.linspace(0.0, 1.0, blend_zone, device=self.device).unsqueeze(1)
                    all_poses[start:start + blend_zone] = (
                        (1.0 - alpha) * all_poses[start:start + blend_zone] +
                        alpha * window_poses[:blend_zone]
                    )
                    all_trans[start:start + blend_zone] = (
                        (1.0 - alpha) * all_trans[start:start + blend_zone] +
                        alpha * window_trans[:blend_zone]
                    )
                    # Non-overlap part
                    if blend_zone < window_T:
                        all_poses[start + blend_zone:end] = window_poses[blend_zone:]
                        all_trans[start + blend_zone:end] = window_trans[blend_zone:]
                else:
                    all_poses[start:end] = window_poses
                    all_trans[start:end] = window_trans
            else:
                all_poses[start:end] = window_poses
                all_trans[start:end] = window_trans

            processed[start:end] = True

            if verbose:
                print(f"  Window {window_idx}: frames [{start},{end}) MPJPE={window_stats['mpjpe_mm']:.1f}mm")

            window_idx += 1
            start += S

            # Handle last partial window
            if start < T and start + S >= T and end < T:
                start = max(T - W, start)  # ensure we cover the tail

        # Update betas from last window
        final_betas = betas.clone()

        # Final statistics
        with torch.no_grad():
            skel_verts, skel_joints, _ = self.skel.forward(
                final_betas.unsqueeze(0).expand(T, -1), all_poses, all_trans, dJ=self._dJ
            )
            mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
            per_joint_error = self._compute_per_joint_error(skel_joints, addb_joints)

        stats = {
            'final_loss': 0.0,
            'mpjpe_mm': mpjpe,
            'per_joint_error_mm': per_joint_error,
            'scapula_dofs': self.scapula_handler.get_scapula_dof_values(all_poses.mean(dim=0)),
            'final_betas': final_betas,
            'twist_stats': {'n_suspicious_frames': 0, 'total_frames': T, 'suspicious_frame_indices': []},
            'marker_stats': {},
        }

        if verbose:
            print(f"  Sliding window complete: MPJPE = {mpjpe:.1f} mm ({window_idx} windows)")

        return all_poses, all_trans, stats


def optimize_poses(
    addb_joints: np.ndarray,
    betas: torch.Tensor,
    skel_interface: SKELInterface,
    config: Optional[OptimizationConfig] = None,
    verbose: bool = False,
    marker_handler=None,
    addb_poses: Optional[np.ndarray] = None,
    dJ: Optional[torch.Tensor] = None,
    skel_q_reference: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Convenience function to optimize poses for a sequence.

    Args:
        addb_joints: AddB joint positions [T, 20, 3] in meters.
        betas: SKEL shape parameters [10].
        skel_interface: SKEL model interface.
        config: Optimization configuration.
        verbose: Print progress.
        marker_handler: Optional MarkerHandler for marker-based loss.
        addb_poses: AddB DOF values [T, N_addb_dofs] in radians (pp.pos).
                    If provided, uses direct DOF mapping for pose initialization.
        dJ: Joint offset corrections [1, 24, 3] from Stage 1. Fixed during pose optimization.
        skel_q_reference: Direct SKEL DOF reference [T, 46] in radians.
                         If provided, used as q-match target (bypasses scale mapping).
                         Takes priority over addb_poses for q-match.

    Returns:
        poses: Optimized pose parameters [T, 46].
        trans: Optimized translations [T, 3].
        stats: Optimization statistics.
    """
    optimizer = PoseOptimizer(skel_interface, config, marker_handler=marker_handler)

    device = config.get_device() if config else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    addb_joints_t = torch.from_numpy(addb_joints).float().to(device)

    return optimizer.optimize_sequence(addb_joints_t, betas, verbose=verbose, addb_poses=addb_poses, dJ=dJ,
                                       skel_q_reference=skel_q_reference)


def finetune_with_markers(
    addb_joints: np.ndarray,
    betas: torch.Tensor,
    init_poses: torch.Tensor,
    init_trans: torch.Tensor,
    skel_interface: SKELInterface,
    config: OptimizationConfig,
    marker_handler,
    verbose: bool = False,
    n_iters: Optional[int] = None,
    dJ: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Fine-tune already-optimized poses with marker loss.

    2-pass approach:
    - Pass 1 (done before): optimize poses without marker loss
    - Pass 2 (this function): short fine-tuning with marker loss added

    Uses low learning rate and few iterations to avoid disturbing the
    already-good joint positions while pulling vertices toward markers.

    Args:
        addb_joints: AddB joint positions [T, 20, 3] in meters.
        betas: SKEL shape parameters [10].
        init_poses: Already optimized pose parameters [T, 46].
        init_trans: Already optimized translations [T, 3].
        skel_interface: SKEL model interface.
        config: Optimization configuration.
        marker_handler: VertexMarkerHandler with vertex_indices already set.
        verbose: Print progress.

    Returns:
        poses: Fine-tuned pose parameters [T, 46].
        trans: Fine-tuned translations [T, 3].
        stats: Optimization statistics.
    """
    device = config.get_device()
    T = addb_joints.shape[0]
    addb_joints_t = torch.from_numpy(addb_joints).float().to(device)

    # Clone initial parameters
    poses = init_poses.clone().detach().requires_grad_(True)
    trans = init_trans.clone().detach().requires_grad_(True)
    betas = betas.clone().detach().requires_grad_(True)

    # Create a PoseOptimizer just for loss computation (reuse loss function)
    optimizer_obj = PoseOptimizer(skel_interface, config, marker_handler=marker_handler)

    # Precompute targets (same as in optimize_sequence)
    addb_bone_indices = get_bone_indices(ADDB_BONE_PAIRS, ADDB_JOINT_TO_IDX)

    with torch.no_grad():
        target_bone_lengths = compute_bone_lengths(addb_joints_t, addb_bone_indices)
        target_bone_dirs = compute_bone_directions(addb_joints_t, addb_bone_indices)
        target_width = torch.norm(
            addb_joints_t[:, ADDB_ACROMIAL_R_IDX, :] - addb_joints_t[:, ADDB_ACROMIAL_L_IDX, :],
            dim=-1
        )

    # Fine-tuning hyperparameters: low LR, few iterations
    ft_iters = n_iters if n_iters is not None else config.marker_finetune_iters
    ft_lr = config.marker_finetune_lr

    opt_params = [
        {'params': [betas], 'lr': ft_lr * 0.5},
        {'params': [poses], 'lr': ft_lr},
        {'params': [trans], 'lr': ft_lr * 0.5},
    ]

    opt = torch.optim.Adam(opt_params)

    pose_lower, pose_upper = get_pose_bounds_tensor(device)

    if verbose:
        print(f"  Fine-tuning: {ft_iters} iters, lr={ft_lr}, weight_marker={config.weight_marker}")

    best_loss = float('inf')
    best_poses = poses.clone().detach()
    best_trans = trans.clone().detach()
    best_betas = betas.clone().detach()

    pbar = tqdm(range(ft_iters), disable=not verbose, desc="MarkerFT")
    for it in pbar:
        # Cosine annealing
        lr_mult = (1 + math.cos(math.pi * it / ft_iters)) / 2
        for pg in opt.param_groups:
            pg['lr'] = pg['lr'] * lr_mult / max(lr_mult, 1e-8)  # avoid division issues
        # Simpler: just set directly
        opt.param_groups[0]['lr'] = ft_lr * 0.5 * lr_mult
        opt.param_groups[1]['lr'] = ft_lr * lr_mult
        opt.param_groups[2]['lr'] = ft_lr * 0.5 * lr_mult

        opt.zero_grad()

        skel_verts, skel_joints, _ = skel_interface.forward(
            betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=dJ
        )

        # Joint loss (keep joints close to AddB targets)
        loss = optimizer_obj._compute_optimization_loss(
            skel_verts, skel_joints, poses, addb_joints_t,
            target_bone_lengths, target_bone_dirs, target_width,
            True, T, include_betas_reg=True, betas=betas,
            include_marker_loss=True, trans=trans,
        )

        loss.backward()
        opt.step()

        # Clamp poses
        with torch.no_grad():
            poses.data = clamp_poses(poses.data, pose_lower, pose_upper)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_poses = poses.clone().detach()
            best_trans = trans.clone().detach()
            best_betas = betas.clone().detach()

        if verbose:
            with torch.no_grad():
                pred = skel_joints[:, optimizer_obj.skel_indices, :]
                tgt = addb_joints_t[:, optimizer_obj.addb_indices, :]
                mpjpe = torch.norm(pred - tgt, dim=-1).mean().item() * 1000
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mpjpe': f'{mpjpe:.1f}mm'})

    # Final stats
    with torch.no_grad():
        skel_verts, skel_joints, _ = skel_interface.forward(
            best_betas.unsqueeze(0).expand(T, -1), best_poses, best_trans, dJ=dJ
        )
        mpjpe = optimizer_obj._compute_mpjpe(skel_joints, addb_joints_t)
        per_joint_error = optimizer_obj._compute_per_joint_error(skel_joints, addb_joints_t)

    marker_stats = {}
    if marker_handler is not None:
        marker_stats = marker_handler.get_marker_statistics(skel_joints, skel_verts)

    stats = {
        'final_loss': best_loss,
        'mpjpe_mm': mpjpe,
        'per_joint_error_mm': per_joint_error,
        'scapula_dofs': optimizer_obj.scapula_handler.get_scapula_dof_values(best_poses.mean(dim=0)),
        'final_betas': best_betas,
        'marker_stats': marker_stats,
    }

    return best_poses, best_trans, stats


def finetune_foot(
    addb_joints: np.ndarray,
    betas: torch.Tensor,
    init_poses: torch.Tensor,
    init_trans: torch.Tensor,
    skel_interface: SKELInterface,
    config: OptimizationConfig,
    verbose: bool = False,
    marker_observations: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Ultra-aggressive lower-body fine-tune: pelvis + hip + knee + foot DOFs + translation.

    After Stage 2, fine-tunes 17 lower-body DOFs (pelvis×3 + hip×3 + knee + ankle +
    subtalar + mtp × 2 sides) and translation to maximize foot position accuracy.
    Uses per-DOF regularization weights, XYZ-weighted foot position loss,
    and optional foot marker vertex loss (LTOE, RTOE, LHEE, RHEE, etc.).

    Args:
        addb_joints: AddB joint positions [T, 20, 3] in meters (SKEL coords).
        betas: SKEL shape parameters [10].
        init_poses: Already optimized pose parameters [T, 46].
        init_trans: Already optimized translations [T, 3].
        skel_interface: SKEL model interface.
        config: Optimization configuration.
        verbose: Print progress.
        marker_observations: Per-frame marker observations (AddB coords, NOT flipped).

    Returns:
        poses: Fine-tuned pose parameters [T, 46].
        trans: Fine-tuned translations [T, 3].
        stats: Optimization statistics.
    """
    device = config.get_device()
    T = addb_joints.shape[0]
    addb_joints_t = torch.from_numpy(addb_joints).float().to(device)

    # Lower-body + pelvis DOF indices (17 DOFs):
    # Pelvis: tilt=0, list=1, rotation=2
    # Right: hip_flex=3, hip_add=4, hip_rot=5, knee=6, ankle=7, subtalar=8, mtp=9
    # Left:  hip_flex=10, hip_add=11, hip_rot=12, knee=13, ankle=14, subtalar=15, mtp=16
    LOWER_DOF_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # Per-DOF regularization weights (pelvis loose → foot tight)
    REG_WEIGHTS = torch.tensor([
        0.001, 0.001, 0.001,           # pelvis (very loose — allow global adjustment)
        0.002, 0.002, 0.002, 0.005,    # hip_r × 3 + knee_r
        0.01, 0.01, 0.01,              # ankle_r, subtalar_r, mtp_r (moderately tight)
        0.002, 0.002, 0.002, 0.005,    # hip_l × 3 + knee_l
        0.01, 0.01, 0.01,              # ankle_l, subtalar_l, mtp_l
    ], device=device)

    # Extract lower-body DOFs as trainable parameters
    poses_full = init_poses.clone().detach()
    lower_dofs = poses_full[:, LOWER_DOF_INDICES].clone().detach().requires_grad_(True)
    trans = init_trans.clone().detach().requires_grad_(True)  # Also trainable!

    # Lower-body + pelvis joint indices (SKEL): pelvis, femur, tibia, talus, calcn, toes (×2)
    SKEL_LOWER_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ADDB_LOWER_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Foot-specific indices for XYZ-weighted foot loss
    SKEL_FOOT_JOINTS = [3, 4, 5, 8, 9, 10]  # talus, calcn, toes (×2)
    ADDB_FOOT_JOINTS = [3, 4, 5, 8, 9, 10]  # ankle, subtalar, mtp (×2)

    ft_iters = getattr(config, 'foot_finetune_iters', 200)
    ft_lr = getattr(config, 'foot_finetune_lr', 0.01)

    # XYZ foot position weights: X (direction), Y (height — GRF critical), Z (depth)
    foot_xyz_weights = torch.tensor(config.foot_xyz_weights, device=device)  # [3]

    # =========================================================================
    # Foot marker targets: directly match skin vertex → GT marker position
    # BSM vertex indices for foot markers (from bsm_markers.yaml)
    # =========================================================================
    # Foot markers for vertex-level fitting (BSM vertex index)
    FOOT_MARKER_BSM = {
        'LTOE': 3216, 'RTOE': 6618,   # toe tip
        'LHEE': 3387, 'RHEE': 6786,   # heel
        'LANK': 3328, 'RANK': 6727,   # ankle lateral
        'LMT5': 3303, 'RMT5': 6703,   # 5th metatarsal
        'LTOP': 3348, 'RTOP': 6749,   # foot top
        'LTOS': 3350, 'RTOS': 6750,   # toe second (1st metatarsal head area)
        'LFPI': 3223, 'RFPI': 6623,   # foot pinky (5th metatarsal head area)
        'LTIC': 3321, 'RTIC': 6721,   # tibia inner condyle (ankle medial)
    }
    # Also check aliases from AddB naming
    # NOTE: R1MTP→RTOS (not RTOE!), L5MTP→LFPI (not LMT5)
    FOOT_MARKER_ALIASES = {
        'LCAL': 'LHEE', 'RCAL': 'RHEE',       # calcaneus → heel
        'L1MTP': 'LTOS', 'R1MTP': 'RTOS',     # 1st metatarsal head → toe second (v3350/v6750)
        'L5MTP': 'LFPI', 'R5MTP': 'RFPI',     # 5th metatarsal head → foot pinky (v3223/v6623)
    }

    foot_marker_vidx = []  # vertex indices
    foot_marker_targets = []  # [T, K, 3] in SKEL coords
    foot_marker_names = []
    has_foot_markers = False

    if marker_observations is not None and len(marker_observations) > 0:
        # Find which foot markers exist in observations
        obs_keys_upper = {k.strip().upper() for k in marker_observations[0].keys()}
        for bsm_name, vidx in FOOT_MARKER_BSM.items():
            bsm_upper = bsm_name.upper()
            # Check direct name or alias
            found_name = None
            if bsm_upper in obs_keys_upper:
                # Find original-case key
                for k in marker_observations[0].keys():
                    if k.strip().upper() == bsm_upper:
                        found_name = k.strip()
                        break
            else:
                # Check aliases
                for alias, target in FOOT_MARKER_ALIASES.items():
                    if target.upper() == bsm_upper and alias.upper() in obs_keys_upper:
                        for k in marker_observations[0].keys():
                            if k.strip().upper() == alias.upper():
                                found_name = k.strip()
                                break
                        break

            if found_name is not None:
                # Build target positions for all frames
                positions = np.zeros((T, 3), dtype=np.float32)
                valid = np.zeros(T, dtype=bool)
                for t in range(min(T, len(marker_observations))):
                    if found_name in marker_observations[t]:
                        pos = marker_observations[t][found_name]
                        # Flip AddB → SKEL coords (negate X and Z)
                        positions[t] = [-float(pos[0]), float(pos[1]), -float(pos[2])]
                        valid[t] = True

                if valid.sum() > 0:
                    foot_marker_vidx.append(vidx)
                    foot_marker_targets.append(positions)
                    foot_marker_names.append(bsm_name)

        if foot_marker_vidx:
            has_foot_markers = True
            foot_marker_vidx_t = torch.tensor(foot_marker_vidx, dtype=torch.long, device=device)
            foot_marker_targets_t = torch.from_numpy(
                np.stack(foot_marker_targets, axis=1)  # [T, K_foot, 3]
            ).float().to(device)
            foot_marker_weight = 20.0  # Very strong — toe markers must match

            if verbose:
                print(f"  Foot markers: {len(foot_marker_vidx)} found: {foot_marker_names}")

    opt = torch.optim.Adam([
        {'params': [lower_dofs], 'lr': ft_lr},
        {'params': [trans], 'lr': ft_lr * 0.3},  # Lower LR for translation
    ])
    pose_lower, pose_upper = get_pose_bounds_tensor(device)

    if verbose:
        print(f"  Lower-body fine-tune: {ft_iters} iters, lr={ft_lr}, "
              f"{len(LOWER_DOF_INDICES)} DOFs + trans")

    best_loss = float('inf')
    best_lower_dofs = lower_dofs.clone().detach()
    best_trans = trans.clone().detach()

    pbar = tqdm(range(ft_iters), disable=not verbose, desc="LowerFT")
    for it in pbar:
        # Cosine annealing
        lr_mult = (1 + math.cos(math.pi * it / ft_iters)) / 2
        opt.param_groups[0]['lr'] = ft_lr * lr_mult
        opt.param_groups[1]['lr'] = ft_lr * 0.3 * lr_mult

        opt.zero_grad()

        # Assemble full poses
        poses_cur = poses_full.clone()
        poses_cur[:, LOWER_DOF_INDICES] = lower_dofs

        skel_verts, skel_joints, _ = skel_interface.forward(
            betas.unsqueeze(0).expand(T, -1), poses_cur, trans
        )

        # Loss 1: Lower-body joint position loss (3D)
        pred_lower = skel_joints[:, SKEL_LOWER_JOINTS, :]
        target_lower = addb_joints_t[:, ADDB_LOWER_JOINTS, :]
        # Apply approximate_foot_weight_factor to calcn (APPROXIMATE: subtalar→calcn)
        # SKEL_LOWER_JOINTS = [0,1,2,3,4,5,6,7,8,9,10]
        # calcn_r=idx4, calcn_l=idx9 in this array
        _approx_f = getattr(config, 'approximate_foot_weight_factor', 1.0)
        if config.foot_finetune_use_joint_weights:
            # Apply per-joint weights from SKEL_JOINT_WEIGHTS or override
            _lower_names = ['pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r',
                           'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l']
            _wdict = config.joint_weights_override if config.joint_weights_override is not None else SKEL_JOINT_WEIGHTS
            _lw = torch.tensor([_wdict.get(n, 1.0) for n in _lower_names], device=device)
            # Apply approximate factor to calcn joints (idx 4, 9)
            _lw[4] *= _approx_f
            _lw[9] *= _approx_f
            _lw = _lw / _lw.mean()  # normalize so total gradient magnitude is similar
            joint_loss = (_lw.view(1, -1, 1) * (pred_lower - target_lower) ** 2).mean()
        else:
            # Even without joint weights, apply approximate factor to calcn
            _lower_w = torch.ones(len(SKEL_LOWER_JOINTS), device=device)
            _lower_w[4] = _approx_f  # calcn_r
            _lower_w[9] = _approx_f  # calcn_l
            joint_loss = (_lower_w.view(1, -1, 1) * (pred_lower - target_lower) ** 2).mean()

        # Loss 2: XYZ-weighted foot position loss (replaces Y-only height_loss)
        pred_foot = skel_joints[:, SKEL_FOOT_JOINTS, :]       # [T, 6, 3]
        target_foot = addb_joints_t[:, ADDB_FOOT_JOINTS, :]   # [T, 6, 3]
        # SKEL_FOOT_JOINTS = [3, 4, 5, 8, 9, 10] = talus, calcn, toes (×2)
        # calcn is at idx 1 (calcn_r) and idx 4 (calcn_l) within this array
        _foot_jw = torch.ones(len(SKEL_FOOT_JOINTS), device=device)
        _foot_jw[1] = _approx_f  # calcn_r
        _foot_jw[4] = _approx_f  # calcn_l
        foot_xyz_loss = (_foot_jw.view(1, -1, 1) * foot_xyz_weights.view(1, 1, 3) *
                         (pred_foot - target_foot) ** 2).mean()

        # Loss 3: Per-DOF regularization (stay close to Stage 2, per-DOF weights)
        init_lower_dofs = init_poses[:, LOWER_DOF_INDICES].to(device)
        reg_loss = (REG_WEIGHTS.unsqueeze(0) *
                    (lower_dofs - init_lower_dofs) ** 2).mean()

        # Loss 3b: Translation regularization (small, just prevent drift)
        init_trans_val = init_trans.to(device)
        trans_reg = 0.005 * ((trans - init_trans_val) ** 2).mean()

        # Loss 4: Temporal smoothness (velocity + acceleration)
        temporal_loss = torch.tensor(0.0, device=device)
        if T > 1:
            dof_diff = lower_dofs[1:] - lower_dofs[:-1]
            temporal_loss = 0.01 * (dof_diff ** 2).mean()
        if T > 2:
            dof_acc = lower_dofs[2:] - 2 * lower_dofs[1:-1] + lower_dofs[:-2]
            temporal_loss = temporal_loss + 0.1 * (dof_acc ** 2).mean()

        loss = joint_loss + foot_xyz_loss + reg_loss + trans_reg + temporal_loss

        loss.backward()
        opt.step()

        # Clamp DOFs
        with torch.no_grad():
            for i, dof_idx in enumerate(LOWER_DOF_INDICES):
                lower_dofs.data[:, i] = torch.clamp(
                    lower_dofs.data[:, i], pose_lower[dof_idx], pose_upper[dof_idx]
                )

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_lower_dofs = lower_dofs.clone().detach()
            best_trans = trans.clone().detach()

        if verbose:
            with torch.no_grad():
                foot_err = torch.norm(pred_foot - target_foot, dim=-1).mean().item() * 1000
                heel_dy_r = (skel_joints[:, 4, 1] - addb_joints_t[:, 4, 1]).mean().item() * 1000
                toe_dy_r = (skel_joints[:, 5, 1] - addb_joints_t[:, 5, 1]).mean().item() * 1000
            pbar.set_postfix({
                'foot_err': f'{foot_err:.1f}mm',
                'heel_dY': f'{heel_dy_r:+.1f}',
                'toe_dY': f'{toe_dy_r:+.1f}',
            })

    # Apply best lower-body DOFs + translation
    final_poses = poses_full.clone()
    final_poses[:, LOWER_DOF_INDICES] = best_lower_dofs

    # =========================================================================
    # Phase 2: Compute per-subject foot marker offsets
    # SKEL vertex positions ≠ GT marker positions due to structural mismatch.
    # Compute constant offset (GT_marker - SKEL_vertex) per foot marker.
    # This offset is stored in stats and applied to virtual markers downstream.
    # =========================================================================
    foot_marker_offsets = {}
    if has_foot_markers:
        with torch.no_grad():
            v_final, _, _ = skel_interface.forward(
                betas.unsqueeze(0).expand(T, -1), final_poses, best_trans
            )
            pred_fm = v_final[:, foot_marker_vidx_t, :]  # [T, K, 3]
            # Mean offset over all frames: GT - pred (in SKEL coords)
            offsets = (foot_marker_targets_t - pred_fm).mean(dim=0)  # [K, 3]
            for k, name in enumerate(foot_marker_names):
                off = offsets[k].cpu().numpy()
                foot_marker_offsets[name] = off
                if verbose:
                    d_before = torch.norm(pred_fm[:, k, :] - foot_marker_targets_t[:, k, :], dim=-1).mean().item() * 1000
                    print(f"    {name}: offset=[{off[0]*1000:+.1f}, {off[1]*1000:+.1f}, {off[2]*1000:+.1f}]mm, before={d_before:.1f}mm")

    # Final statistics
    with torch.no_grad():
        skel_verts_final, skel_joints, _ = skel_interface.forward(
            betas.unsqueeze(0).expand(T, -1), final_poses, best_trans
        )
        optimizer_obj = PoseOptimizer(skel_interface, config)
        mpjpe = optimizer_obj._compute_mpjpe(skel_joints, addb_joints_t)
        per_joint_error = optimizer_obj._compute_per_joint_error(skel_joints, addb_joints_t)

        # Foot-specific metrics
        pred_foot = skel_joints[:, SKEL_FOOT_JOINTS, :]
        target_foot = addb_joints_t[:, ADDB_FOOT_JOINTS, :]
        foot_mpjpe = torch.norm(pred_foot - target_foot, dim=-1).mean().item() * 1000

        # Foot marker metrics
        foot_marker_errs = {}
        foot_marker_errs_y = {}
        if has_foot_markers:
            pred_fm = skel_verts_final[:, foot_marker_vidx_t, :]
            fm_diff = pred_fm - foot_marker_targets_t  # [T, K, 3]
            fm_dists = torch.norm(fm_diff, dim=-1)  # [T, K]
            for k, name in enumerate(foot_marker_names):
                foot_marker_errs[name] = fm_dists[:, k].mean().item() * 1000
                foot_marker_errs_y[name] = fm_diff[:, k, 1].mean().item() * 1000

    stats = {
        'mpjpe_mm': mpjpe,
        'foot_mpjpe_mm': foot_mpjpe,
        'per_joint_error_mm': per_joint_error,
        'scapula_dofs': optimizer_obj.scapula_handler.get_scapula_dof_values(final_poses.mean(dim=0)),
        'foot_marker_errors_mm': foot_marker_errs,
        'foot_marker_offsets': foot_marker_offsets,  # {name: [3] offset in SKEL coords}
    }

    if verbose:
        print(f"  Lower-body fine-tune done: MPJPE={mpjpe:.1f}mm, Foot={foot_mpjpe:.1f}mm")
        if foot_marker_errs:
            errs_str = ", ".join(f"{n}={e:.1f}" for n, e in foot_marker_errs.items())
            print(f"  Foot markers 3D (mm): {errs_str}")
            errs_y_str = ", ".join(f"{n}={e:+.1f}" for n, e in foot_marker_errs_y.items())
            print(f"  Foot markers ΔY (mm): {errs_y_str}")

    return final_poses, best_trans, stats
