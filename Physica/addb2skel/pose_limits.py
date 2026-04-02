"""
Comprehensive pose limits for SKEL model.

Combines:
1. SKEL official pose limits from kin_skel.py (32 parameters)
2. Literature-based limits for missing joints (14 parameters)

Sources:
- SKEL: /egr/research-zijunlab/kwonjoon/01_Code/SKEL/skel/kin_skel.py
- Pelvis: Physio-pedia, PMC5545133
- Hip: OrthoFixar, Physio-pedia, CDC
- Shoulder: OrthoFixar, ShoulderDoc, Meloq

References:
- https://www.physio-pedia.com/Pelvic_Tilt
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5545133/
- https://orthofixar.com/special-test/hip-range-of-motion-and-biomechanics/
- https://www.physio-pedia.com/Range_of_Motion_Normative_Values
- https://orthofixar.com/special-test/shoulder-range-of-motion/
- https://www.shoulderdoc.co.uk/article/913
"""

import math
from typing import Tuple
import torch

# SKEL pose parameter names (46 DOFs)
POSE_PARAM_NAMES = [
    'pelvis_tilt',        # 0
    'pelvis_list',        # 1
    'pelvis_rotation',    # 2
    'hip_flexion_r',      # 3
    'hip_adduction_r',    # 4
    'hip_rotation_r',     # 5
    'knee_angle_r',       # 6
    'ankle_angle_r',      # 7
    'subtalar_angle_r',   # 8
    'mtp_angle_r',        # 9
    'hip_flexion_l',      # 10
    'hip_adduction_l',    # 11
    'hip_rotation_l',     # 12
    'knee_angle_l',       # 13
    'ankle_angle_l',      # 14
    'subtalar_angle_l',   # 15
    'mtp_angle_l',        # 16
    'lumbar_bending',     # 17
    'lumbar_extension',   # 18
    'lumbar_twist',       # 19
    'thorax_bending',     # 20
    'thorax_extension',   # 21
    'thorax_twist',       # 22
    'head_bending',       # 23
    'head_extension',     # 24
    'head_twist',         # 25
    'scapula_abduction_r',   # 26
    'scapula_elevation_r',   # 27
    'scapula_upward_rot_r',  # 28
    'shoulder_r_x',       # 29
    'shoulder_r_y',       # 30
    'shoulder_r_z',       # 31
    'elbow_flexion_r',    # 32
    'pro_sup_r',          # 33
    'wrist_flexion_r',    # 34
    'wrist_deviation_r',  # 35
    'scapula_abduction_l',   # 36
    'scapula_elevation_l',   # 37
    'scapula_upward_rot_l',  # 38
    'shoulder_l_x',       # 39
    'shoulder_l_y',       # 40
    'shoulder_l_z',       # 41
    'elbow_flexion_l',    # 42
    'pro_sup_l',          # 43
    'wrist_flexion_l',    # 44
    'wrist_deviation_l',  # 45
]

# =============================================================================
# Comprehensive Pose Limits (46 DOFs)
# =============================================================================

COMPREHENSIVE_POSE_LIMITS = {
    # =========================================================================
    # PELVIS (DOF 0-2) - NOT in SKEL official pose_limits
    # 출처 미확인: Physio-pedia, PMC5545133에서 참고했으나 직접 확인하지 않음
    # pelvis_rotation ±90°는 oscillation 문제 원인 — 좁힐 필요 있음
    # =========================================================================
    'pelvis_tilt': [-0.35, 0.70],       # DOF 0: -20° to 40° — 출처 미확인
    'pelvis_list': [-0.52, 0.52],       # DOF 1: ±30° — 출처 미확인
    'pelvis_rotation': [-1.57, 1.57],   # DOF 2: ±90° — 출처 미확인 (oscillation 원인)

    # =========================================================================
    # HIP (DOF 3-5, 10-12) - NOT in SKEL official pose_limits
    # 출처 미확인: OrthoFixar, Physio-pedia, CDC에서 참고했으나 직접 확인하지 않음
    # =========================================================================
    'hip_flexion_r': [-0.52, 2.36],     # DOF 3: -30° to 135° — 출처 미확인
    'hip_adduction_r': [-0.87, 0.52],   # DOF 4: -50° to 30° — 출처 미확인
    'hip_rotation_r': [-1.05, 0.70],    # DOF 5: -60° to 40° — 출처 미확인

    'hip_flexion_l': [-0.52, 2.36],     # DOF 10: symmetric to right
    'hip_adduction_l': [-0.87, 0.52],   # DOF 11: symmetric to right
    'hip_rotation_l': [-1.05, 0.70],    # DOF 12: symmetric to right

    # =========================================================================
    # KNEE (DOF 6, 13) - SKEL Official (kin_skel.py)
    # =========================================================================
    'knee_angle_r': [0, 2.356],          # DOF 6: 0° to 135° (SKEL: 3/4*π)
    'knee_angle_l': [0, 2.356],          # DOF 13: 0° to 135° (SKEL: 3/4*π)

    # =========================================================================
    # ANKLE/FOOT (DOF 7-9, 14-16) - SKEL Official (kin_skel.py)
    # =========================================================================
    'ankle_angle_r': [-0.785, 0.785],     # DOF 7: ±45° (SKEL: π/4)
    'subtalar_angle_r': [-0.785, 0.785],  # DOF 8: ±45° (SKEL: π/4)
    'mtp_angle_r': [-0.785, 0.785],       # DOF 9: ±45° (SKEL: π/4)

    'ankle_angle_l': [-0.785, 0.785],     # DOF 14: ±45° (SKEL: π/4)
    'subtalar_angle_l': [-0.785, 0.785],  # DOF 15: ±45° (SKEL: π/4)
    'mtp_angle_l': [-0.785, 0.785],       # DOF 16: ±45° (SKEL: π/4)

    # =========================================================================
    # SPINE (DOF 17-22) - SKEL Official (kin_skel.py)
    # =========================================================================
    'lumbar_bending': [-0.524, 0.524],    # DOF 17: ±30° (SKEL: 2/3*π/4)
    'lumbar_extension': [-0.9, 0.785],    # DOF 18: -51.6° to +45° (PR#38: lower bound widened from -π/4)
    'lumbar_twist': [-0.785, 0.785],      # DOF 19: ±45° (SKEL: π/4)

    'thorax_bending': [-0.785, 0.785],    # DOF 20: ±45° (SKEL: π/4)
    'thorax_extension': [-0.785, 0.785],  # DOF 21: ±45° (SKEL: π/4)
    'thorax_twist': [-0.785, 0.785],      # DOF 22: ±45° (SKEL: π/4)

    # =========================================================================
    # HEAD (DOF 23-25) - SKEL Official (kin_skel.py)
    # =========================================================================
    'head_bending': [-0.785, 0.785],      # DOF 23: ±45° (SKEL: π/4)
    'head_extension': [-0.785, 0.785],    # DOF 24: ±45° (SKEL: π/4)
    'head_twist': [-0.785, 0.785],        # DOF 25: ±45° (SKEL: π/4)

    # =========================================================================
    # RIGHT SCAPULA (DOF 26-28) - PR#38 fix (github.com/MarilynKeller/SKEL/pull/38)
    # Original SKEL had elevation entirely negative [-0.4,-0.1] → 0° outside range
    # PR#38: widen to include 0° and match clinical ROM (Kenhub: Scapulothoracic joint)
    # =========================================================================
    'scapula_abduction_r': [-0.628, 0.628],    # DOF 26: ±36° (SKEL official, unchanged)
    'scapula_elevation_r': [-0.5, 0.6981],     # DOF 27: -28.6° to +40° (PR#38)
    'scapula_upward_rot_r': [-0.5236, 1.0472], # DOF 28: -30° to +60° (PR#38: -π/6 to π/3)

    # =========================================================================
    # RIGHT SHOULDER (DOF 29-31)
    # SKEL official defines only shoulder_r_y. Others from literature (출처 미확인).
    # =========================================================================
    'shoulder_r_x': [-1.05, 3.14],      # DOF 29: -60° to 180° — 출처 미확인 (literature)
    'shoulder_r_y': [-1.571, 1.571],    # DOF 30: ±90° (SKEL official: [-π/2, π/2])
    'shoulder_r_z': [-1.57, 1.57],      # DOF 31: ±90° — 출처 미확인 (literature)

    # =========================================================================
    # RIGHT ELBOW/WRIST (DOF 32-35) - SKEL Official (kin_skel.py)
    # =========================================================================
    'elbow_flexion_r': [0, 2.356],       # DOF 32: 0° to 135° (SKEL: 3/4*π)
    'pro_sup_r': [-1.178, 1.178],        # DOF 33: ±67.5° (SKEL: 3/4*π/2)
    'wrist_flexion_r': [-1.571, 1.571],  # DOF 34: ±90° (SKEL: π/2)
    'wrist_deviation_r': [-1.0, 0.785],  # DOF 35: -57° to +45° (PR#38: lower bound widened)

    # =========================================================================
    # LEFT SCAPULA (DOF 36-38) - PR#38 fix (github.com/MarilynKeller/SKEL/pull/38)
    # Original SKEL had [-0.1,-0.4] (min>max bug). PR#38 fixes and symmetrizes with right.
    # =========================================================================
    'scapula_abduction_l': [-0.628, 0.628],    # DOF 36: ±36° (SKEL official, unchanged)
    'scapula_elevation_l': [-0.5, 0.6981],     # DOF 37: -28.6° to +40° (PR#38, symmetric to right)
    'scapula_upward_rot_l': [-0.5236, 1.0472], # DOF 38: -30° to +60° (PR#38, symmetric to right)

    # =========================================================================
    # LEFT SHOULDER (DOF 39-41)
    # SKEL official has NO left shoulder limits. Using symmetric to right.
    # =========================================================================
    'shoulder_l_x': [-1.05, 3.14],      # DOF 39: -60° to 180° — 출처 미확인, symmetric to right
    'shoulder_l_y': [-1.571, 1.571],    # DOF 40: ±90° — symmetric to right (SKEL has shoulder_r_y only)
    'shoulder_l_z': [-1.57, 1.57],      # DOF 41: ±90° — 출처 미확인, symmetric to right

    # =========================================================================
    # LEFT ELBOW/WRIST (DOF 42-45) - SKEL Official (kin_skel.py)
    # =========================================================================
    'elbow_flexion_l': [0, 2.356],       # DOF 42: 0° to 135° (SKEL: 3/4*π)
    'pro_sup_l': [-1.571, 1.571],        # DOF 43: ±90° (SKEL: π/2)
    'wrist_flexion_l': [-1.571, 1.571],  # DOF 44: ±90° (SKEL: π/2)
    'wrist_deviation_l': [-1.0, 0.785],  # DOF 45: -57° to +45° (PR#38: lower bound widened)
}


def get_pose_bounds_tensor(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create min/max bound tensors for all 46 DOFs.

    Args:
        device: torch device (cuda or cpu)

    Returns:
        Tuple of (lower_bounds, upper_bounds), each shape [46]
    """
    lower = torch.zeros(46, device=device)
    upper = torch.zeros(46, device=device)

    for i, param_name in enumerate(POSE_PARAM_NAMES):
        if param_name in COMPREHENSIVE_POSE_LIMITS:
            limits = COMPREHENSIVE_POSE_LIMITS[param_name]
            lower[i] = limits[0]
            upper[i] = limits[1]
        else:
            # Fallback for any missing parameters (shouldn't happen)
            lower[i] = -math.pi
            upper[i] = math.pi

    return lower, upper


def clamp_poses(
    poses: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor
) -> torch.Tensor:
    """
    Clamp pose parameters to physiological bounds.

    Args:
        poses: Pose tensor [B, 46] or [46]
        lower: Lower bounds [46]
        upper: Upper bounds [46]

    Returns:
        Clamped poses with same shape as input
    """
    return torch.clamp(poses, lower, upper)


def compute_soft_limit_penalty(
    poses: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Soft DOF limit penalty: preserves gradients at boundaries (unlike hard clamp).

    Loss = mean( relu(pose - (upper+margin))² + relu((lower-margin) - pose)² )

    Args:
        poses: Pose tensor [B, 46] or [46]
        lower: Lower bounds [46]
        upper: Upper bounds [46]
        margin: Extra margin beyond bounds before penalty kicks in (radians)

    Returns:
        Scalar penalty loss
    """
    over = torch.relu(poses - (upper + margin))
    under = torch.relu((lower - margin) - poses)
    return (over ** 2 + under ** 2).mean()


def get_pose_limit_for_param(param_name: str) -> Tuple[float, float]:
    """
    Get pose limits for a specific parameter by name.

    Args:
        param_name: Name of the pose parameter (e.g., 'hip_flexion_r')

    Returns:
        Tuple of (min_value, max_value) in radians
    """
    if param_name in COMPREHENSIVE_POSE_LIMITS:
        return tuple(COMPREHENSIVE_POSE_LIMITS[param_name])
    else:
        return (-math.pi, math.pi)


def print_pose_limits_summary():
    """Print a human-readable summary of all pose limits."""
    print("=" * 70)
    print("SKEL Comprehensive Pose Limits")
    print("=" * 70)

    for i, param_name in enumerate(POSE_PARAM_NAMES):
        if param_name in COMPREHENSIVE_POSE_LIMITS:
            limits = COMPREHENSIVE_POSE_LIMITS[param_name]
            deg_min = math.degrees(limits[0])
            deg_max = math.degrees(limits[1])
            print(f"  {i:2d}: {param_name:25s} [{limits[0]:6.2f}, {limits[1]:6.2f}] rad "
                  f"= [{deg_min:7.1f}°, {deg_max:6.1f}°]")
        else:
            print(f"  {i:2d}: {param_name:25s} (NO LIMIT)")

    print("=" * 70)


if __name__ == "__main__":
    print_pose_limits_summary()
