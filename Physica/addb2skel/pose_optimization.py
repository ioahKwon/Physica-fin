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
    SKEL_JOINT_WEIGHTS,
    SKEL_JOINT_TO_IDX,
)
from .skel_interface import SKELInterface
from .scapula_handler import ScapulaHandler, compute_shoulder_losses
from .joint_definitions import (
    build_direct_joint_mapping,
    get_bone_indices,
    ADDB_BONE_PAIRS,
    SKEL_BONE_PAIRS,
    ADDB_JOINT_TO_IDX,
    SKEL_JOINT_TO_IDX,
    ADDB_ACROMIAL_R_IDX,
    ADDB_ACROMIAL_L_IDX,
)
from .utils.geometry import (
    compute_bone_lengths,
    compute_bone_directions,
    cosine_similarity_loss,
)
from .pose_limits import get_pose_bounds_tensor, clamp_poses


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
    (0, 6): [10, 11, 12],   # pelvis → femur_l: hip_l DOFs
    (6, 7): [13],           # femur_l → tibia_l: knee_l DOF
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
    ):
        """
        Initialize pose optimizer.

        Args:
            skel_interface: SKEL model interface.
            config: Optimization configuration.
        """
        self.skel = skel_interface
        self.config = config or OptimizationConfig()
        self.device = self.config.get_device()

        # Build joint mapping (include ALL joints, including acromial mapped to humerus)
        self.addb_indices, self.skel_indices = build_direct_joint_mapping()

        # NOTE: Now we include acromial in joint loss by mapping to humerus
        # This is critical for good shoulder fitting (per compare_smpl_skel.py)
        # The separate virtual acromial loss is optional/additional

        # Build bone pair indices
        self.addb_bone_indices = get_bone_indices(ADDB_BONE_PAIRS, ADDB_JOINT_TO_IDX)
        self.skel_bone_indices = get_bone_indices(SKEL_BONE_PAIRS, SKEL_JOINT_TO_IDX)

        # Build per-joint weights (critical for good results!)
        self.joint_weights = self._build_joint_weights()

        # Scapula handler
        self.scapula_handler = ScapulaHandler(skel_interface, config)

        # Pose bounds for clamping (comprehensive limits from literature)
        self.pose_lower, self.pose_upper = get_pose_bounds_tensor(self.device)

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

    def _build_joint_weights(self) -> torch.Tensor:
        """
        Build per-joint weight tensor based on SKEL_JOINT_WEIGHTS config.

        Critical for good results! Shoulders get 10x, spine 5x, pelvis/femurs 2x.

        Returns:
            weights: [num_mapped_joints] weight tensor
        """
        from .joint_definitions import SKEL_JOINTS

        weights = []
        for skel_idx in self.skel_indices:
            joint_name = SKEL_JOINTS[skel_idx]
            # Map SKEL joint names to config names (handle naming variations)
            if joint_name == 'lumbar_body':
                config_name = 'lumbar'
            else:
                config_name = joint_name
            weight = SKEL_JOINT_WEIGHTS.get(config_name, 1.0)
            weights.append(weight)

        return torch.tensor(weights, device=self.device, dtype=torch.float32)

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
                spine_reg
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
                betas.unsqueeze(0), best_poses, best_trans
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

        Returns:
            poses: Optimized pose parameters [T, 46].
            trans: Optimized translations [T, 3].
            stats: Optimization statistics.
        """
        T = addb_joints.shape[0]

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

        # IK-based pose initialization (better than zero pose!)
        if verbose:
            print("  Computing IK-based pose initialization...")
        with torch.no_grad():
            initial_poses = estimate_initial_poses_ik(
                addb_joints, self.addb_indices, self.skel_indices,
                self.skel, self.device
            )
        poses = initial_poses.clone()
        if verbose:
            print(f"  IK init: pose norm = {torch.norm(poses).item():.4f}")

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
        # STAGE 1: Pose + Trans optimization (betas fixed)
        # =====================================================================
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
            # Cosine annealing learning rate
            if self.config.use_cosine_lr:
                lr_mult = (1 + math.cos(math.pi * it / stage1_iters)) / 2
                self._update_optimizer_lr(optimizer_stage1, lr_mult)

            optimizer_stage1.zero_grad()

            # Forward through SKEL
            skel_verts, skel_joints, _ = self.skel.forward(
                betas.unsqueeze(0).expand(T, -1), poses, trans
            )

            # Loss computation
            loss = self._compute_optimization_loss(
                skel_verts, skel_joints, poses, addb_joints,
                target_bone_lengths, target_bone_dirs, target_width,
                use_temporal, T, include_betas_reg=False
            )

            # Add soft constraint penalty if enabled
            if self.config.use_soft_constraints:
                constraint_penalty = self._compute_soft_constraint_penalty(poses)
                loss = loss + self.config.soft_constraint_weight * constraint_penalty

            loss.backward()
            optimizer_stage1.step()

            # Clamp all pose DOFs to physiological limits (unless using soft constraints only)
            if not self.config.use_soft_constraints:
                with torch.no_grad():
                    poses.data = clamp_poses(poses.data, self.pose_lower, self.pose_upper)

            # Track best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_poses = poses.clone().detach()
                best_trans = trans.clone().detach()
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
                current_lr = optimizer_stage1.param_groups[0]['lr']
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mpjpe': f'{mpjpe:.1f}mm', 'lr': f'{current_lr:.4f}'})

        # =====================================================================
        # Between Stages: Update dynamic joint weights if enabled
        # =====================================================================
        if self.config.use_dynamic_weights:
            with torch.no_grad():
                # Compute current predictions with best poses
                skel_verts, skel_joints, _ = self.skel.forward(
                    betas.unsqueeze(0).expand(T, -1), best_poses, best_trans
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

        # Lower learning rates for fine-tuning (matching working code)
        optimizer_stage2 = torch.optim.Adam([
            {'params': [betas], 'lr': 0.02},
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
                betas.unsqueeze(0).expand(T, -1), poses, trans
            )

            # Loss computation (include betas regularization)
            loss = self._compute_optimization_loss(
                skel_verts, skel_joints, poses, addb_joints,
                target_bone_lengths, target_bone_dirs, target_width,
                use_temporal, T, include_betas_reg=True, betas=betas
            )

            # Add soft constraint penalty if enabled
            if self.config.use_soft_constraints:
                constraint_penalty = self._compute_soft_constraint_penalty(poses)
                loss = loss + self.config.soft_constraint_weight * constraint_penalty

            loss.backward()
            optimizer_stage2.step()

            # Clamp all pose DOFs to physiological limits (unless using soft constraints only)
            if not self.config.use_soft_constraints:
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

        # Final statistics
        with torch.no_grad():
            skel_verts, skel_joints, _ = self.skel.forward(
                best_betas.unsqueeze(0).expand(T, -1), best_poses, best_trans
            )
            mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
            per_joint_error = self._compute_per_joint_error(skel_joints, addb_joints)

        stats = {
            'final_loss': best_loss,
            'mpjpe_mm': mpjpe,
            'per_joint_error_mm': per_joint_error,
            'scapula_dofs': self.scapula_handler.get_scapula_dof_values(best_poses.mean(dim=0)),
            'final_betas': best_betas,  # Return optimized betas
        }

        return best_poses, best_trans, stats

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

        # 8. Temporal smoothness
        temporal_loss = torch.tensor(0.0, device=self.device)
        if use_temporal and T > 1:
            pose_diff = poses[1:] - poses[:-1]
            temporal_loss = self.config.weight_temporal * (pose_diff ** 2).mean()

        # 9. Betas regularization (only in Stage 2)
        betas_reg = torch.tensor(0.0, device=self.device)
        if include_betas_reg and betas is not None:
            betas_reg = 0.005 * (betas ** 2).mean()

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
            temporal_loss +
            betas_reg
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

        Args:
            skel_joints: SKEL joint positions [T, 24, 3]
            addb_joints: AddB joint positions [T, 20, 3]
            exclude_acromial: If True, exclude acromial from MPJPE calculation.
                              This is important because AddB acromial is a surface
                              landmark while SKEL humerus is the glenohumeral center.
                              Working code excludes acromial for fair comparison.

        Returns:
            MPJPE in mm
        """
        pred = skel_joints[:, self.skel_indices, :]
        target = addb_joints[:, self.addb_indices, :]

        if exclude_acromial:
            # Find indices to exclude (acromial_r, acromial_l)
            exclude_indices = []
            for i, addb_idx in enumerate(self.addb_indices):
                if addb_idx == ADDB_ACROMIAL_R_IDX or addb_idx == ADDB_ACROMIAL_L_IDX:
                    exclude_indices.append(i)

            # Create mask for included joints
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


def optimize_poses(
    addb_joints: np.ndarray,
    betas: torch.Tensor,
    skel_interface: SKELInterface,
    config: Optional[OptimizationConfig] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Convenience function to optimize poses for a sequence.

    Args:
        addb_joints: AddB joint positions [T, 20, 3] in meters.
        betas: SKEL shape parameters [10].
        skel_interface: SKEL model interface.
        config: Optimization configuration.
        verbose: Print progress.

    Returns:
        poses: Optimized pose parameters [T, 46].
        trans: Optimized translations [T, 3].
        stats: Optimization statistics.
    """
    optimizer = PoseOptimizer(skel_interface, config)

    device = config.get_device() if config else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    addb_joints_t = torch.from_numpy(addb_joints).float().to(device)

    return optimizer.optimize_sequence(addb_joints_t, betas, verbose=verbose)
