"""
Subject-specific scale estimation module.

Stage 1 of the 2-stage optimization pipeline (following AddBiomechanics approach):
1. Estimate subject scale/proportions first (global over all frames)
2. Then optimize pose per frame

This module estimates SKEL beta parameters to match AddB bone lengths.
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
import torch
import torch.nn.functional as F

from .config import OptimizationConfig, SKEL_NUM_BETAS, SKEL_NUM_POSE_DOF
from .skel_interface import SKELInterface
from .joint_definitions import (
    RELIABLE_BONE_PAIRS_ADDB,
    RELIABLE_BONE_PAIRS_SKEL,
    ADDB_JOINT_TO_IDX,
    SKEL_JOINT_TO_IDX,
    get_bone_indices,
)
import logging
logger = logging.getLogger(__name__)
from .utils.geometry import compute_bone_lengths


class ScaleEstimator:
    """
    Estimates SKEL shape parameters (beta) to match AddB subject proportions.

    Uses reliable bone lengths (legs, forearms) that are not affected by
    shoulder/scapula uncertainty.
    """

    def __init__(
        self,
        skel_interface: SKELInterface,
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize scale estimator.

        Args:
            skel_interface: SKEL model interface.
            config: Optimization configuration.
        """
        self.skel = skel_interface
        self.config = config or OptimizationConfig()
        self.device = self.config.get_device()

        # Build bone pair indices
        self.addb_bone_indices = get_bone_indices(
            RELIABLE_BONE_PAIRS_ADDB, ADDB_JOINT_TO_IDX
        )
        self.skel_bone_indices = get_bone_indices(
            RELIABLE_BONE_PAIRS_SKEL, SKEL_JOINT_TO_IDX
        )

    def estimate_from_bone_lengths(
        self,
        addb_joints: np.ndarray,
        initial_betas: Optional[torch.Tensor] = None,
        verbose: bool = False,
        marker_handler=None,
        target_mass_kg: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Estimate SKEL betas by matching reliable bone lengths.

        Args:
            addb_joints: AddB joint positions [T, 20, 3] in meters.
            initial_betas: Initial beta values [10]. Default: zeros.
            verbose: Print progress.
            target_mass_kg: Optional target body mass in kg (SKEL Section 6.5).

        Returns:
            betas: Estimated shape parameters [10].
            stats: Dictionary with estimation statistics.
        """
        # Compute target bone lengths from AddB (average over frames)
        addb_joints_t = torch.from_numpy(addb_joints).float().to(self.device)
        target_lengths = compute_bone_lengths(
            addb_joints_t, self.addb_bone_indices
        ).mean(dim=0)  # [num_bones]

        if verbose:
            print(f"Target bone lengths (mm): {target_lengths.cpu().numpy() * 1000}")

        # Build per-bone weights from config
        bone_weights = torch.ones(len(RELIABLE_BONE_PAIRS_ADDB), device=self.device)
        if self.config.scale_bone_weights:
            for i, (a1, a2) in enumerate(RELIABLE_BONE_PAIRS_ADDB):
                key = f"{a1}→{a2}"
                if key in self.config.scale_bone_weights:
                    bone_weights[i] = self.config.scale_bone_weights[key]
            if verbose:
                weighted = [(f"{a1}→{a2}", bone_weights[i].item())
                            for i, (a1, a2) in enumerate(RELIABLE_BONE_PAIRS_ADDB)
                            if bone_weights[i].item() != 1.0]
                if weighted:
                    print(f"  Per-bone weights: {weighted}")

        # Initialize betas
        if initial_betas is None:
            betas = torch.zeros(SKEL_NUM_BETAS, device=self.device)
        else:
            betas = initial_betas.clone().to(self.device)

        betas.requires_grad_(True)

        # Initialize dJ (joint offset corrections)
        from .config import SKEL_NUM_JOINTS
        use_dj = self.config.use_dj_optimization
        dJ = torch.zeros(1, SKEL_NUM_JOINTS, 3, device=self.device, requires_grad=use_dj)

        # Optimizer — include dJ if enabled (low LR: betas do main work, dJ is residual)
        opt_params = [{'params': [betas], 'lr': self.config.scale_lr}]
        if use_dj:
            dj_lr = self.config.scale_lr * getattr(self.config, 'dj_lr_factor', 0.1)
            opt_params.append({'params': [dJ], 'lr': dj_lr})
        optimizer = torch.optim.Adam(opt_params)

        # T-pose for scale estimation
        poses = torch.zeros(1, SKEL_NUM_POSE_DOF, device=self.device)
        trans = torch.zeros(1, 3, device=self.device)

        # Target shoulder width from AddB acromial (acromial_r=12, acromial_l=16)
        target_shoulder_width = torch.norm(
            addb_joints_t[:, 12, :] - addb_joints_t[:, 16, :], dim=-1
        ).mean()  # scalar, meters

        best_loss = float('inf')
        best_betas = betas.clone()
        best_dJ = dJ.clone().detach()

        # Precompute mean marker targets for T-pose comparison (if available)
        marker_mean_targets = None
        marker_mean_mask = None
        if marker_handler is not None and marker_handler.num_markers > 0:
            # Average observed marker positions across all frames (for T-pose matching)
            marker_mean_targets = marker_handler.marker_targets.mean(dim=0)  # [K, 3]
            marker_mean_mask = marker_handler.marker_mask.any(dim=0)  # [K] at least 1 frame observed
            if verbose:
                n_valid = marker_mean_mask.sum().item()
                print(f"  Marker-aware scale: {n_valid}/{marker_handler.num_markers} markers with observations")

        # Need vertices when using marker handler
        need_verts = (marker_handler is not None and marker_handler.num_markers > 0)

        for it in range(self.config.scale_iters):
            optimizer.zero_grad()

            # Forward through SKEL (with dJ offset)
            dJ_arg = dJ if use_dj else None
            if need_verts:
                verts, skel_joints, _ = self.skel.forward(
                    betas.unsqueeze(0), poses, trans, dJ=dJ_arg
                )
            else:
                _, skel_joints, _ = self.skel.forward(
                    betas.unsqueeze(0), poses, trans, dJ=dJ_arg
                )

            # Compute SKEL bone lengths
            skel_lengths = compute_bone_lengths(
                skel_joints, self.skel_bone_indices
            )[0]  # [num_bones]

            # Bone length loss (weighted per-bone)
            length_loss = (bone_weights * (skel_lengths - target_lengths) ** 2).mean()

            # Shoulder width loss: match SKEL scapula width to AddB acromial width
            # scapula_r=14, scapula_l=19 in SKEL; acromial_r=12, acromial_l=16 in AddB
            shoulder_width_loss = torch.tensor(0.0, device=self.device)
            w_sw = getattr(self.config, 'weight_shoulder_width_in_scale', 1.0)
            if w_sw > 0:
                skel_shoulder_w = torch.norm(skel_joints[0, 14, :] - skel_joints[0, 19, :])
                shoulder_width_loss = w_sw * (skel_shoulder_w - target_shoulder_width) ** 2

            # Marker loss in T-pose (joint + offset, rotation=identity in T-pose)
            marker_loss = torch.tensor(0.0, device=self.device)
            if (marker_handler is not None and marker_handler.num_markers > 0
                    and marker_mean_targets is not None):
                # Predict marker positions: joint_pos + local_offset
                pred_marker = (skel_joints[0, marker_handler.skel_joint_indices, :]
                               + marker_handler.local_offsets)  # [K, 3]
                diff = pred_marker - marker_mean_targets  # [K, 3]
                sq_error = (diff ** 2).sum(dim=-1)  # [K]
                masked = sq_error[marker_mean_mask]
                if masked.numel() > 0:
                    marker_loss = masked.mean()

            # Beta regularization (prefer smaller betas)
            reg_loss = 0.001 * (betas ** 2).mean()

            # dJ regularization (prevent large joint offsets)
            dj_reg_loss = torch.tensor(0.0, device=self.device)
            if use_dj:
                dj_reg_loss = self.config.weight_dj_reg * (dJ ** 2).mean()

            loss = (length_loss
                    + shoulder_width_loss
                    + self.config.weight_marker_scale * marker_loss
                    + reg_loss
                    + dj_reg_loss)

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_betas = betas.clone().detach()
                best_dJ = dJ.clone().detach()

            if verbose and (it + 1) % 50 == 0:
                with torch.no_grad():
                    length_err = (skel_lengths - target_lengths).abs().mean() * 1000
                marker_str = f", MarkerLoss={marker_loss.item():.6f}" if marker_loss.item() > 0 else ""
                dj_str = ""
                if use_dj:
                    dj_mag = dJ.abs().mean().item() * 1000  # mm
                    dj_str = f", dJ_mean={dj_mag:.2f}mm"
                print(f"  Iter {it+1}/{self.config.scale_iters}: "
                      f"Loss={loss.item():.6f}, LenErr={length_err:.2f}mm{marker_str}{dj_str}")

        # Final statistics
        dJ_arg_final = best_dJ if use_dj else None
        with torch.no_grad():
            verts_final, skel_joints, _ = self.skel.forward(
                best_betas.unsqueeze(0), poses, trans, dJ=dJ_arg_final
            )
            skel_lengths = compute_bone_lengths(
                skel_joints, self.skel_bone_indices
            )[0]
            length_errors = (skel_lengths - target_lengths).abs() * 1000

        stats = {
            'final_loss': best_loss,
            'bone_length_errors_mm': length_errors.cpu().numpy(),
            'mean_length_error_mm': length_errors.mean().item(),
            'target_lengths_mm': target_lengths.cpu().numpy() * 1000,
            'fitted_lengths_mm': skel_lengths.cpu().numpy() * 1000,
        }

        # Add dJ statistics
        if use_dj:
            stats['dJ'] = best_dJ
            dj_abs = best_dJ.abs()
            stats['dJ_mean_mm'] = dj_abs.mean().item() * 1000
            stats['dJ_max_mm'] = dj_abs.max().item() * 1000

        return best_betas, stats

    def estimate_from_height_width(
        self,
        height_m: float,
        shoulder_width_m: float,
        sex: str = 'male',
    ) -> torch.Tensor:
        """
        Estimate betas from height and shoulder width.

        This is a simpler initialization method based on anthropometric data.

        Args:
            height_m: Subject height in meters.
            shoulder_width_m: Shoulder width in meters.
            sex: 'male' or 'female'.

        Returns:
            betas: Estimated shape parameters [10].
        """
        # Baseline values for male SKEL model (from T-pose)
        if sex == 'male':
            baseline_height = 1.58  # meters
            baseline_shoulder = 0.35  # meters
        else:
            baseline_height = 1.52
            baseline_shoulder = 0.32

        # Beta[0] primarily affects height
        # Beta[1] affects shoulder width
        height_ratio = height_m / baseline_height
        shoulder_ratio = shoulder_width_m / baseline_shoulder

        # Empirical scaling (approximate)
        beta0 = -3.5 * (height_ratio - 1.0)  # Height adjustment
        beta1 = 2.0 * (shoulder_ratio - 1.0)  # Shoulder adjustment

        betas = torch.zeros(SKEL_NUM_BETAS, device=self.device)
        betas[0] = beta0
        betas[1] = beta1

        return betas


def estimate_subject_scale(
    addb_joints: np.ndarray,
    skel_interface: SKELInterface,
    config: Optional[OptimizationConfig] = None,
    height_m: Optional[float] = None,
    shoulder_width_m: Optional[float] = None,
    verbose: bool = False,
    marker_handler=None,
    target_mass_kg: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Convenience function to estimate subject scale.

    Args:
        addb_joints: AddB joint positions [T, 20, 3] in meters.
        skel_interface: SKEL model interface.
        config: Optimization configuration.
        height_m: Optional known height for initialization.
        shoulder_width_m: Optional known shoulder width for initialization.
        verbose: Print progress.
        marker_handler: Optional MarkerHandler for marker-aware scale estimation.
        target_mass_kg: Optional target body mass in kg (SKEL Section 6.5).

    Returns:
        betas: Estimated shape parameters [10].
        stats: Estimation statistics.
    """
    estimator = ScaleEstimator(skel_interface, config)

    # Get initial betas from height/width if available
    initial_betas = None
    if height_m is not None and shoulder_width_m is not None:
        initial_betas = estimator.estimate_from_height_width(
            height_m, shoulder_width_m
        )
        if verbose:
            print(f"Initial betas from height/width: {initial_betas[:3].cpu().numpy()}")

    # Refine using bone lengths (+ optional marker loss + optional mass constraint)
    betas, stats = estimator.estimate_from_bone_lengths(
        addb_joints, initial_betas, verbose,
        marker_handler=marker_handler,
        target_mass_kg=target_mass_kg,
    )

    # Extract dJ from stats (if optimized)
    dJ = stats.pop('dJ', None)
    stats['dJ'] = dJ  # Keep reference in stats for pipeline access

    return betas, stats
