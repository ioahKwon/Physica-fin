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
    # PELVIS (DOF 0-2) - Literature based
    # Sources: Physio-pedia, PMC5545133
    # =========================================================================
    'pelvis_tilt': [-0.35, 0.70],       # DOF 0: -20° to 40° (posterior to anterior)
    'pelvis_list': [-0.52, 0.52],       # DOF 1: ±30° (lateral drop during gait)
    'pelvis_rotation': [-0.79, 0.79],   # DOF 2: ±45° (transverse rotation)

    # =========================================================================
    # HIP (DOF 3-5, 10-12) - Literature based
    # Sources: OrthoFixar, Physio-pedia, CDC
    # Normal: Flex 120-135°, Ext 10-30°, Abd 45-50°, Add 20-30°, IR 30-40°, ER 40-60°
    # =========================================================================
    'hip_flexion_r': [-0.52, 2.36],     # DOF 3: -30° to 135° (conservative)
    'hip_adduction_r': [-0.87, 0.52],   # DOF 4: -50° (abduction) to 30° (adduction)
    'hip_rotation_r': [-1.05, 0.70],    # DOF 5: -60° (external) to 40° (internal)

    'hip_flexion_l': [-0.52, 2.36],     # DOF 10: same as right
    'hip_adduction_l': [-0.87, 0.52],   # DOF 11: same as right
    'hip_rotation_l': [-1.05, 0.70],    # DOF 12: same as right

    # =========================================================================
    # KNEE (DOF 6, 13) - SKEL Official
    # =========================================================================
    'knee_angle_r': [0, 2.36],          # DOF 6: 0° to 135°
    'knee_angle_l': [0, 2.36],          # DOF 13: 0° to 135°

    # =========================================================================
    # ANKLE/FOOT (DOF 7-9, 14-16) - SKEL Official
    # =========================================================================
    'ankle_angle_r': [-0.79, 0.79],     # DOF 7: ±45°
    'subtalar_angle_r': [-0.79, 0.79],  # DOF 8: ±45°
    'mtp_angle_r': [-0.79, 0.79],       # DOF 9: ±45°

    'ankle_angle_l': [-0.79, 0.79],     # DOF 14: ±45°
    'subtalar_angle_l': [-0.79, 0.79],  # DOF 15: ±45°
    'mtp_angle_l': [-0.79, 0.79],       # DOF 16: ±45°

    # =========================================================================
    # SPINE (DOF 17-22) - SKEL Official
    # =========================================================================
    'lumbar_bending': [-0.52, 0.52],    # DOF 17: ±30°
    'lumbar_extension': [-0.79, 0.79],  # DOF 18: ±45°
    'lumbar_twist': [-0.79, 0.79],      # DOF 19: ±45°

    'thorax_bending': [-0.79, 0.79],    # DOF 20: ±45°
    'thorax_extension': [-0.79, 0.79],  # DOF 21: ±45°
    'thorax_twist': [-0.79, 0.79],      # DOF 22: ±45°

    # =========================================================================
    # HEAD (DOF 23-25) - SKEL Official
    # =========================================================================
    'head_bending': [-0.79, 0.79],      # DOF 23: ±45°
    'head_extension': [-0.79, 0.79],    # DOF 24: ±45°
    'head_twist': [-0.79, 0.79],        # DOF 25: ±45°

    # =========================================================================
    # RIGHT SCAPULA (DOF 26-28) - SKEL Official
    # =========================================================================
    'scapula_abduction_r': [-0.63, 0.63],   # DOF 26: ±36°
    'scapula_elevation_r': [-0.4, -0.1],    # DOF 27: -23° to -6° (always negative)
    'scapula_upward_rot_r': [-0.19, 0.32],  # DOF 28: -11° to 18°

    # =========================================================================
    # RIGHT SHOULDER (DOF 29-31) - Mixed
    # Sources: OrthoFixar, ShoulderDoc, Meloq
    # Normal: Flex 160-180°, Ext 45-60°, IR 70-90°, ER 90-100°
    # =========================================================================
    'shoulder_r_x': [-1.05, 3.14],      # DOF 29: -60° to 180° (extension to flexion)
    'shoulder_r_y': [-1.57, 1.57],      # DOF 30: ±90° (SKEL Official)
    'shoulder_r_z': [-1.57, 1.57],      # DOF 31: ±90° (internal/external rotation)

    # =========================================================================
    # RIGHT ELBOW/WRIST (DOF 32-35) - SKEL Official
    # =========================================================================
    'elbow_flexion_r': [0, 2.36],       # DOF 32: 0° to 135°
    'pro_sup_r': [-1.18, 1.18],         # DOF 33: ±68°
    'wrist_flexion_r': [-1.57, 1.57],   # DOF 34: ±90°
    'wrist_deviation_r': [-0.79, 0.79], # DOF 35: ±45°

    # =========================================================================
    # LEFT SCAPULA (DOF 36-38) - SKEL Official
    # Note: scapula_elevation_l is flipped compared to right
    # =========================================================================
    'scapula_abduction_l': [-0.63, 0.63],   # DOF 36: ±36°
    'scapula_elevation_l': [-0.4, -0.1],    # DOF 37: -23° to -6° (negative range)
    'scapula_upward_rot_l': [-0.21, 0.22],  # DOF 38: -12° to 13°

    # =========================================================================
    # LEFT SHOULDER (DOF 39-41) - Mixed
    # Note: shoulder_l_y is NOT in SKEL official (only shoulder_r_y)
    # =========================================================================
    'shoulder_l_x': [-1.05, 3.14],      # DOF 39: -60° to 180° (same as right)
    'shoulder_l_y': [-1.57, 1.57],      # DOF 40: ±90° (added, not in SKEL official)
    'shoulder_l_z': [-1.57, 1.57],      # DOF 41: ±90° (same as right)

    # =========================================================================
    # LEFT ELBOW/WRIST (DOF 42-45) - SKEL Official
    # =========================================================================
    'elbow_flexion_l': [0, 2.36],       # DOF 42: 0° to 135°
    'pro_sup_l': [-1.57, 1.57],         # DOF 43: ±90°
    'wrist_flexion_l': [-1.57, 1.57],   # DOF 44: ±90°
    'wrist_deviation_l': [-0.79, 0.79], # DOF 45: ±45°
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
