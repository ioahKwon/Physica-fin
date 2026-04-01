"""
Evaluation metrics for addb2skel pipeline.

Metrics:
  - MPJPE (root-relative): mm
  - W-MPJPE (world): mm
  - PA-MPJPE (Procrustes-aligned): mm
  - ACCL (acceleration error): mm/frame²
  - VEL (velocity error): mm/frame
  - FS (foot sliding): mm
  - GP (ground penetration): mm

Reference: PhysPT (arXiv 2404.04430) Table 1
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


# =============================================================================
# Joint-based metrics (torch tensors, [T, J, 3] in meters)
# =============================================================================

def compute_mpjpe(pred: torch.Tensor, gt: torch.Tensor, root_idx: int = 0) -> float:
    """
    Root-relative MPJPE [mm].

    Subtracts root joint (pelvis) position before computing L2 error.
    This is the standard MPJPE used in Human3.6M benchmarks.

    Args:
        pred: [T, J, 3] predicted joint positions in meters
        gt: [T, J, 3] ground truth joint positions in meters
        root_idx: root joint index (default 0 = pelvis)

    Returns:
        MPJPE in mm
    """
    pred_rel = pred - pred[:, [root_idx], :]
    gt_rel = gt - gt[:, [root_idx], :]
    return (pred_rel - gt_rel).norm(dim=-1).mean().item() * 1000


def compute_w_mpjpe(pred: torch.Tensor, gt: torch.Tensor, root_idx: int = 0) -> float:
    """
    World-coordinate MPJPE [mm] with per-frame root alignment.

    Aligns pred root to GT root per frame, then computes L2 error.
    This measures pose accuracy in world scale without penalizing
    global translation offset between coordinate systems.

    Args:
        pred: [T, J, 3] predicted joint positions in meters
        gt: [T, J, 3] ground truth joint positions in meters
        root_idx: root joint index for alignment (default 0 = pelvis)

    Returns:
        W-MPJPE in mm
    """
    # Per-frame root alignment: shift pred so pred_root = gt_root
    offset = gt[:, [root_idx], :] - pred[:, [root_idx], :]
    pred_aligned = pred + offset
    return (pred_aligned - gt).norm(dim=-1).mean().item() * 1000


def _similarity_align_to(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Procrustes alignment: find s, R, t such that s*R*S1 + t ≈ S2.

    Per-frame independent alignment via SVD.
    Based on WHAM/HSMR implementation.

    Args:
        S1: [...B, N, 3] source points (prediction)
        S2: [...B, N, 3] target points (ground truth)

    Returns:
        S1_aligned: [...B, N, 3] aligned source points
    """
    original_shape = S1.shape
    N = original_shape[-2]
    S1 = S1.reshape(-1, N, 3)
    S2 = S2.reshape(-1, N, 3)
    B = S1.shape[0]
    device = S2.device
    S1 = S1.to(device)

    # Transpose to (B, 3, N)
    S1_t = S1.transpose(-1, -2)
    S2_t = S2.transpose(-1, -2)

    # 1. Remove mean
    mu1 = S1_t.mean(dim=-1, keepdim=True)  # (B, 3, 1)
    mu2 = S2_t.mean(dim=-1, keepdim=True)
    X1 = S1_t - mu1
    X2 = S2_t - mu2

    # 2. Variance of X1
    var1 = torch.einsum('BDN->B', X1 ** 2)  # (B,)

    # 3. Cross-covariance
    K = X1 @ X2.transpose(-1, -2)  # (B, 3, 3)

    # 4. SVD
    U, s, V = torch.svd(K)

    # 5. Rotation with reflection correction
    Z = torch.eye(3, device=device)[None].repeat(B, 1, 1)
    Z[:, -1, -1] *= (U @ V.transpose(-1, -2)).det().sign()
    R = V @ (Z @ U.transpose(-1, -2))

    # 6. Scale
    traces = torch.stack([torch.trace(x) for x in (R @ K)])
    scales = (traces / var1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

    # 7. Translation
    t = mu2 - (scales * (R @ mu1))  # (B, 3, 1)

    # 8. Apply transform
    S1_aligned = scales * (R @ S1_t) + t  # (B, 3, N)
    S1_aligned = S1_aligned.transpose(-1, -2).reshape(original_shape)
    return S1_aligned


def compute_pa_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Procrustes-Aligned MPJPE [mm].

    Per-frame similarity transform (scale + rotation + translation) alignment,
    then compute L2 error. Evaluates skeletal shape accuracy only.

    Args:
        pred: [T, J, 3] predicted joint positions in meters
        gt: [T, J, 3] ground truth joint positions in meters

    Returns:
        PA-MPJPE in mm
    """
    pred_aligned = _similarity_align_to(pred, gt)
    return (pred_aligned - gt).norm(dim=-1).mean().item() * 1000


def compute_accel_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Acceleration error [mm/frame²].

    2nd-order central finite difference. NOT divided by dt (paper convention).
    a_t = J_{t+1} - 2*J_t + J_{t-1}

    Args:
        pred: [T, J, 3] predicted joint positions in meters
        gt: [T, J, 3] ground truth joint positions in meters

    Returns:
        Acceleration error in mm/frame²
    """
    if pred.shape[0] < 3:
        return 0.0
    pred_acc = pred[2:] - 2 * pred[1:-1] + pred[:-2]  # [T-2, J, 3]
    gt_acc = gt[2:] - 2 * gt[1:-1] + gt[:-2]
    return (pred_acc - gt_acc).norm(dim=-1).mean().item() * 1000


def compute_vel_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Velocity error [mm/frame].

    1st-order forward finite difference. NOT divided by dt (paper convention).
    v_t = J_{t+1} - J_t

    Args:
        pred: [T, J, 3] predicted joint positions in meters
        gt: [T, J, 3] ground truth joint positions in meters

    Returns:
        Velocity error in mm/frame
    """
    if pred.shape[0] < 2:
        return 0.0
    pred_vel = pred[1:] - pred[:-1]  # [T-1, J, 3]
    gt_vel = gt[1:] - gt[:-1]
    return (pred_vel - gt_vel).norm(dim=-1).mean().item() * 1000


# =============================================================================
# Mesh-based metrics (numpy arrays, [T, V, 3] in meters)
# =============================================================================

def compute_foot_sliding(
    vertices: np.ndarray,
    dt: float,
    pos_thresh: float = 0.03,
    vel_thresh: float = 1.0,
) -> Tuple[float, Dict]:
    """
    Foot sliding error [mm].

    Contact detection: vertex Y < pos_thresh AND vertex velocity < vel_thresh.
    Sliding = displacement of contact vertices between consecutive contact frames.

    Threshold source: Yufei Zhang (PhysPT author), Slack 2026-03-12.
    Heuristic values determined by visual verification of contact detection.

    Args:
        vertices: [T, V, 3] mesh vertices in meters (Y-axis vertical)
        dt: timestep in seconds
        pos_thresh: height threshold for contact (meters), default 0.03
        vel_thresh: velocity threshold for contact (m/s), default 1.0

    Returns:
        Tuple of (foot_sliding_mm, diagnostics_dict)
    """
    T, V, _ = vertices.shape

    # Vertex Y positions and velocities
    vert_y = vertices[:-1, :, 1]  # [T-1, V]
    vert_vel = np.linalg.norm(
        (vertices[1:] - vertices[:-1]) / dt, axis=2
    )  # [T-1, V]

    # Contact: low Y AND low velocity
    contact = np.logical_and(vert_y < pos_thresh, vert_vel < vel_thresh)

    # Foot sliding per frame
    foot_sliding_errors = []
    for t in range(T - 2):
        displacement = vert_vel[t] * contact[t] * contact[t + 1]
        sliding_t = np.abs(np.sum(displacement)) * dt
        if sliding_t > 0:
            n_contact = (displacement > 0).sum()
            if n_contact > 0:
                sliding_t /= n_contact
        foot_sliding_errors.append(sliding_t)

    foot_sliding_mm = float(np.mean(foot_sliding_errors) * 1000) if foot_sliding_errors else 0.0

    contact_count = contact.sum(axis=1)
    diagnostics = {
        'contact_count_mean': float(contact_count.mean()),
        'contact_count_max': int(contact_count.max()) if len(contact_count) > 0 else 0,
    }

    return foot_sliding_mm, diagnostics


def compute_foot_sliding_gt_contact(
    vertices: np.ndarray,
    dt: float,
    gt_contact: np.ndarray,
    foot_height_thresh: float = 0.03,
) -> Tuple[float, Dict]:
    """
    Foot sliding error [mm] using GT contact labels from AddB b3d.

    Uses body-level GT contact (calcn_r/calcn_l) to determine *when* contact
    occurs, then measures vertex displacement of low-Y (foot) vertices during
    consecutive contact frames.

    Args:
        vertices: [T, V, 3] mesh vertices in meters (Y-axis vertical)
        dt: timestep in seconds
        gt_contact: [T, N_bodies] binary contact states from pp.contact.
                    N_bodies is typically 2: [calcn_r, calcn_l].
                    A frame is contact if ANY body is in contact.
        foot_height_thresh: height threshold to select foot vertices (meters).
                            This selects *which vertices* are on the foot sole,
                            NOT whether contact occurs (that comes from GT).

    Returns:
        Tuple of (foot_sliding_mm, diagnostics_dict)
    """
    T, V, _ = vertices.shape

    # Frame-level contact: any foot body in contact
    frame_contact = gt_contact.any(axis=1) if gt_contact.ndim > 1 else gt_contact.astype(bool)
    # frame_contact: [T] boolean

    # Vertex velocities (speed)
    vert_vel = np.linalg.norm(
        (vertices[1:] - vertices[:-1]) / dt, axis=2
    )  # [T-1, V]

    # For each frame pair (t, t+1), select foot vertices by height
    vert_y = vertices[:-1, :, 1]  # [T-1, V]

    foot_sliding_errors = []
    n_contact_frames = 0
    for t in range(T - 2):
        # Both frames must be GT contact
        if not (frame_contact[t] and frame_contact[t + 1]):
            foot_sliding_errors.append(0.0)
            continue

        n_contact_frames += 1

        # Select foot vertices: low Y in this frame
        foot_mask = vert_y[t] < foot_height_thresh  # [V] boolean
        if not foot_mask.any():
            foot_sliding_errors.append(0.0)
            continue

        # Displacement of foot vertices
        displacement = vert_vel[t] * foot_mask
        sliding_t = np.abs(np.sum(displacement)) * dt
        n_foot = foot_mask.sum()
        if n_foot > 0:
            sliding_t /= n_foot
        foot_sliding_errors.append(sliding_t)

    foot_sliding_mm = float(np.mean(foot_sliding_errors) * 1000) if foot_sliding_errors else 0.0

    diagnostics = {
        'gt_contact_frames': n_contact_frames,
        'gt_contact_ratio': n_contact_frames / max(T - 2, 1),
        'total_frames': T,
    }

    return foot_sliding_mm, diagnostics


def compute_ground_penetration(vertices: np.ndarray) -> float:
    """
    Ground penetration [mm].

    Average depth of mesh vertices below ground plane (Y=0).
    Only penetrating vertices contribute; frames with none contribute 0.

    Args:
        vertices: [T, V, 3] mesh vertices in meters (Y-axis vertical)

    Returns:
        Ground penetration in mm
    """
    vert_y = vertices[:, :, 1].copy()  # [T, V]
    vert_y[vert_y > 0] = 0  # keep only below-ground

    penetration_per_frame = np.abs(np.sum(vert_y, axis=1))  # [T]
    n_below = (vert_y < 0).sum(axis=1)  # [T]
    mask = n_below > 0
    penetration_per_frame[mask] /= n_below[mask]

    return float(np.mean(penetration_per_frame) * 1000)


# =============================================================================
# Combined evaluation
# =============================================================================

def compute_all_metrics(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    root_idx: int = 0,
    vertices: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all 7 evaluation metrics.

    Args:
        pred_joints: [T, J, 3] predicted joint positions in meters (DIRECT joints only)
        gt_joints: [T, J, 3] GT joint positions in meters (matched to pred)
        root_idx: root joint index for MPJPE (default 0 = pelvis)
        vertices: [T, V, 3] mesh vertices in meters (optional, for FS/GP)
        dt: timestep in seconds (required if vertices provided)

    Returns:
        Dict with keys: mpjpe_mm, w_mpjpe_mm, pa_mpjpe_mm,
                        accel_error, vel_error,
                        foot_sliding_mm, ground_penetration_mm
    """
    results = {
        'mpjpe_mm': compute_mpjpe(pred_joints, gt_joints, root_idx),
        'w_mpjpe_mm': compute_w_mpjpe(pred_joints, gt_joints),
        'pa_mpjpe_mm': compute_pa_mpjpe(pred_joints, gt_joints),
        'accel_error': compute_accel_error(pred_joints, gt_joints),
        'vel_error': compute_vel_error(pred_joints, gt_joints),
    }

    if vertices is not None and dt is not None:
        fs_mm, fs_diag = compute_foot_sliding(vertices, dt)
        results['foot_sliding_mm'] = fs_mm
        results['ground_penetration_mm'] = compute_ground_penetration(vertices)
        results['fs_diagnostics'] = fs_diag

    return results


def format_metrics_table(metrics: Dict[str, float], name: str = '') -> str:
    """Format metrics as a printable table."""
    lines = []
    if name:
        lines.append(f"\n{'=' * 50}")
        lines.append(f"  Evaluation Metrics: {name}")
        lines.append(f"{'=' * 50}")

    lines.append(f"  {'Metric':<25} {'Value':>10} {'Unit':>15}")
    lines.append(f"  {'-' * 50}")

    metric_info = [
        ('mpjpe_mm', 'MPJPE (root-rel)', 'mm'),
        ('w_mpjpe_mm', 'W-MPJPE (world)', 'mm'),
        ('pa_mpjpe_mm', 'PA-MPJPE', 'mm'),
        ('accel_error', 'ACCL', 'mm/frame²'),
        ('vel_error', 'VEL', 'mm/frame'),
        ('foot_sliding_mm', 'Foot Sliding', 'mm'),
        ('ground_penetration_mm', 'Ground Penetration', 'mm'),
    ]

    for key, label, unit in metric_info:
        if key in metrics:
            lines.append(f"  {label:<25} {metrics[key]:>10.3f} {unit:>15}")

    return '\n'.join(lines)
