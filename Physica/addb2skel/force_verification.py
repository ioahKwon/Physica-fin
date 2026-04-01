"""
Force IK Verification Module.

Verifies that SKEL-fitted joint positions are physically consistent with
AddBiomechanics dynamics data (GRF, joint torques).

Approach:
1. Load GT dynamics from b3d (pos, vel, acc, tau, GRF, CoP, contact torque)
2. Run corrected inverse dynamics on GT kinematics → tau_gt_verified (baseline)
3. Compare SKEL-fitted joint positions vs GT joint centers
4. Assess whether fitting errors would cause significant dynamics errors

Key insight: SKEL has a different skeleton (46 DOF) than OpenSim (30+ DOF),
so we can't directly run SKEL q through OpenSim ID. Instead, we verify
that the SKEL joint positions are close enough to GT that the same dynamics
could be reproduced.
"""

import os
from typing import Dict, Optional, List
import numpy as np


def verify_force_consistency(
    b3d_path: str,
    skel_joints: np.ndarray,
    num_frames: Optional[int] = None,
    processing_pass: int = 2,
    verbose: bool = True,
) -> Dict:
    """
    Verify force consistency between SKEL-fitted joints and AddB GT dynamics.

    Args:
        b3d_path: Path to AddB .b3d file.
        skel_joints: SKEL-fitted joint positions [T, 24, 3] in AddB coords.
        num_frames: Number of frames to verify. None = all available.
        processing_pass: b3d processing pass (2=DYNAMICS).
        verbose: Print progress.

    Returns:
        Dictionary with verification results:
        - tau_gt: [T, N_dof] GT joint torques
        - tau_id_corrected: [T, N_dof] corrected ID from GT kinematics
        - tau_rmse: float, RMSE between corrected ID and GT tau (Nm)
        - residual_force_mean: float, mean residual force magnitude (N)
        - joint_position_error: dict of per-joint errors (mm)
        - dof_names: list of DOF names
        - per_dof_tau_rmse: dict of per-DOF tau RMSE
    """
    import sys
    sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica')
    from perturbation_sensitivity.scripts.grf_utils import (
        load_grf_data_from_b3d,
        run_inverse_dynamics_with_full_correction,
    )

    # 1. Load GT dynamics data
    if verbose:
        print(f"Loading dynamics data from: {b3d_path}")

    gt_data = load_grf_data_from_b3d(
        b3d_path,
        processing_pass=processing_pass,
        start_frame=1,
        num_frames=num_frames or 1000,
    )

    T_gt = gt_data['num_frames']
    T_skel = skel_joints.shape[0]
    T = min(T_gt, T_skel)

    if verbose:
        print(f"  GT frames: {T_gt}, SKEL frames: {T_skel}, using: {T}")
        print(f"  DOFs: {gt_data['n_dof']}")
        print(f"  Ground force bodies: {gt_data['ground_force_body_names']}")

    skel = gt_data['skel']
    n_dof = gt_data['n_dof']

    # 2. Run corrected ID on GT kinematics (baseline verification)
    if verbose:
        print(f"\n  Running corrected inverse dynamics on GT kinematics...")

    tau_id_corrected = run_inverse_dynamics_with_full_correction(
        skel,
        gt_data['pos'][:T],
        gt_data['vel'][:T],
        gt_data['acc'][:T],
        gt_data['grf'][:T],
        gt_data['cop'][:T],
        contact_torque=gt_data['contact_torque'][:T],
        residual_wrench=gt_data['residual_wrench'][:T],
        ground_force_body_names=gt_data['ground_force_body_names'],
    )

    gt_tau = gt_data['tau'][:T]

    # 3. Compute tau RMSE (corrected ID vs GT tau)
    tau_diff = tau_id_corrected - gt_tau
    tau_rmse = float(np.sqrt(np.mean(tau_diff ** 2)))

    # Per-DOF RMSE (skip root 6 DOFs which are unconstrained)
    dof_names = _get_dof_names(skel)
    per_dof_tau_rmse = {}
    for d in range(n_dof):
        rmse = float(np.sqrt(np.mean(tau_diff[:, d] ** 2)))
        name = dof_names[d] if d < len(dof_names) else f"dof_{d}"
        per_dof_tau_rmse[name] = rmse

    if verbose:
        print(f"  Corrected ID vs GT tau RMSE: {tau_rmse:.3f} Nm")
        # Show top-5 worst DOFs (skip root)
        sorted_dofs = sorted(per_dof_tau_rmse.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top-5 worst DOF tau RMSE:")
        for name, rmse in sorted_dofs[:5]:
            print(f"    {name:20s}: {rmse:.3f} Nm")

    # 4. Compute residual force statistics
    residual = gt_data['residual_wrench'][:T]
    residual_force_mean = float(np.mean(np.linalg.norm(residual[:, :3], axis=1)))
    residual_moment_mean = float(np.mean(np.linalg.norm(residual[:, 3:], axis=1)))

    if verbose:
        print(f"\n  Residual wrench statistics:")
        print(f"    Mean residual force:  {residual_force_mean:.2f} N")
        print(f"    Mean residual moment: {residual_moment_mean:.2f} Nm")

    # 5. Compare SKEL joint positions vs GT joint centers
    gt_joint_centers = gt_data['joint_centers'][:T]  # [T, N_joints, 3]
    n_gt_joints = gt_joint_centers.shape[1]
    n_skel_joints = skel_joints.shape[1]  # 24

    # Build mapping: GT joint names → SKEL joint indices (closest match)
    joint_pos_errors = _compute_joint_position_errors(
        skel_joints[:T], gt_joint_centers, skel, verbose
    )

    if verbose:
        print(f"\n  SKEL vs GT joint position errors:")
        sorted_joints = sorted(joint_pos_errors.items(), key=lambda x: x[1], reverse=True)
        for name, err in sorted_joints[:10]:
            print(f"    {name:20s}: {err:.1f} mm")
        mean_err = float(np.mean(list(joint_pos_errors.values())))
        print(f"    Mean: {mean_err:.1f} mm")

    # 6. GRF statistics
    grf = gt_data['grf'][:T]
    total_grf_per_frame = np.linalg.norm(grf.sum(axis=1), axis=1)  # [T]
    grf_mean = float(np.mean(total_grf_per_frame))
    grf_max = float(np.max(total_grf_per_frame))

    if verbose:
        print(f"\n  GRF statistics:")
        print(f"    Mean total GRF: {grf_mean:.1f} N")
        print(f"    Max total GRF:  {grf_max:.1f} N")

    return {
        'tau_gt': gt_tau,
        'tau_id_corrected': tau_id_corrected,
        'tau_rmse': tau_rmse,
        'tau_rmse_no_root': float(np.sqrt(np.mean(tau_diff[:, 6:] ** 2))),
        'residual_force_mean': residual_force_mean,
        'residual_moment_mean': residual_moment_mean,
        'joint_position_error': joint_pos_errors,
        'dof_names': dof_names,
        'per_dof_tau_rmse': per_dof_tau_rmse,
        'grf_mean': grf_mean,
        'grf_max': grf_max,
        'num_frames': T,
        'n_dof': n_dof,
        'gt_data': gt_data,
    }


def _get_dof_names(skel) -> List[str]:
    """Extract DOF names from nimblephysics skeleton via joint iteration."""
    names = []
    for i in range(skel.getNumJoints()):
        joint = skel.getJoint(i)
        joint_name = joint.getName()
        for d in range(joint.getNumDofs()):
            try:
                dof_name = joint.getDofName(d)
                names.append(f"{joint_name}/{dof_name}")
            except Exception:
                names.append(f"{joint_name}/dof{d}")
    return names


def _compute_joint_position_errors(
    skel_joints: np.ndarray,
    gt_joint_centers: np.ndarray,
    opensim_skel,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute per-joint position errors between SKEL and GT joint centers.

    Uses nearest-joint matching since SKEL (24 joints) and OpenSim (20 joints)
    have different joint definitions.

    Args:
        skel_joints: [T, 24, 3] SKEL joint positions (AddB coords).
        gt_joint_centers: [T, N_gt, 3] GT joint centers (AddB coords).
        opensim_skel: nimblephysics skeleton for joint name lookup.

    Returns:
        Per-joint mean error in mm.
    """
    T = skel_joints.shape[0]
    n_gt = gt_joint_centers.shape[1]

    # Get GT joint names
    gt_names = []
    for i in range(opensim_skel.getNumJoints()):
        joint = opensim_skel.getJoint(i)
        gt_names.append(joint.getName())

    # Truncate to actual joint centers count
    gt_names = gt_names[:n_gt]

    # For each GT joint, find nearest SKEL joint (by mean position)
    skel_mean = skel_joints.mean(axis=0)  # [24, 3]
    gt_mean = gt_joint_centers.mean(axis=0)  # [N_gt, 3]

    errors = {}
    for j in range(n_gt):
        # Find nearest SKEL joint
        dists = np.linalg.norm(skel_mean - gt_mean[j], axis=1)
        best_skel_idx = np.argmin(dists)

        # Compute per-frame error for this pair
        err_per_frame = np.linalg.norm(
            skel_joints[:, best_skel_idx, :] - gt_joint_centers[:, j, :],
            axis=1
        )
        name = gt_names[j] if j < len(gt_names) else f"joint_{j}"
        errors[name] = float(np.mean(err_per_frame) * 1000)  # mm

    return errors


def verify_force_comprehensive(
    b3d_path: str,
    skel_joints: np.ndarray,
    num_frames: Optional[int] = None,
    processing_pass: int = 2,
    verbose: bool = True,
) -> Dict:
    """
    Comprehensive force verification with ALL ID/FD methods.

    Methods:
    1. Standard ID (no GRF):      tau = M*a + C*v + g
    2. GRF-corrected ID:          tau = Standard ID - J^T*F_grf
    3. Full-corrected ID:         tau = Standard ID - J^T*F_grf - J_ang^T*M_contact
    4. Single-step FD:            GT state + GT forces → FD acc vs GT acc

    Args:
        b3d_path: Path to AddB .b3d file.
        skel_joints: SKEL-fitted joint positions [T, 24, 3] in AddB coords.
        num_frames: Number of frames to verify.
        processing_pass: b3d processing pass (2=DYNAMICS).
        verbose: Print progress.

    Returns:
        Dictionary with per-method verification results.
    """
    import sys
    sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica')
    # run_forward_dynamics uses relative import of grf_utils, so add its directory
    _fd_dir = '/egr/research-zijunlab/kwonjoon/01_Code/Physica/perturbation_sensitivity/scripts'
    if _fd_dir not in sys.path:
        sys.path.insert(0, _fd_dir)
    from perturbation_sensitivity.scripts.grf_utils import (
        load_grf_data_from_b3d,
        run_inverse_dynamics_with_grf,
        run_inverse_dynamics_with_full_correction,
    )
    from perturbation_sensitivity.scripts.run_forward_dynamics import (
        compute_gt_net_torque,
        run_single_step_fd,
    )

    # 1. Load GT dynamics data
    if verbose:
        print(f"Loading dynamics data from: {b3d_path}")

    gt_data = load_grf_data_from_b3d(
        b3d_path,
        processing_pass=processing_pass,
        start_frame=1,
        num_frames=num_frames or 1000,
    )

    T_gt = gt_data['num_frames']
    T_skel = skel_joints.shape[0]
    T = min(T_gt, T_skel)

    skel = gt_data['skel']
    n_dof = gt_data['n_dof']
    gt_tau = gt_data['tau'][:T]
    dof_names = _get_dof_names(skel)

    if verbose:
        print(f"  GT frames: {T_gt}, SKEL frames: {T_skel}, using: {T}")
        print(f"  DOFs: {n_dof}")
        print(f"  Ground force bodies: {gt_data['ground_force_body_names']}")

    pos = gt_data['pos'][:T]
    vel = gt_data['vel'][:T]
    acc = gt_data['acc'][:T]
    grf = gt_data['grf'][:T]
    cop = gt_data['cop'][:T]

    # ---- Method 1: Standard ID (no GRF) ----
    if verbose:
        print(f"\n  [Method 1] Standard ID (no GRF)...")
    tau_standard = np.zeros((T, n_dof), dtype=np.float64)
    for t in range(T):
        skel.setPositions(pos[t])
        skel.setVelocities(vel[t])
        tau_standard[t] = np.array(skel.getInverseDynamics(acc[t]), dtype=np.float64)[:n_dof]

    rmse_standard = float(np.sqrt(np.mean((tau_standard - gt_tau) ** 2)))
    rmse_standard_no_root = float(np.sqrt(np.mean((tau_standard[:, 6:] - gt_tau[:, 6:]) ** 2)))

    if verbose:
        print(f"    RMSE: {rmse_standard:.3f} Nm (all), {rmse_standard_no_root:.3f} Nm (no root)")

    # ---- Method 2: GRF-corrected ID ----
    if verbose:
        print(f"\n  [Method 2] GRF-corrected ID...")
    tau_grf_corrected = run_inverse_dynamics_with_grf(
        skel, pos, vel, acc, grf, cop,
        ground_force_body_names=gt_data['ground_force_body_names'],
    )

    rmse_grf = float(np.sqrt(np.mean((tau_grf_corrected - gt_tau) ** 2)))
    rmse_grf_no_root = float(np.sqrt(np.mean((tau_grf_corrected[:, 6:] - gt_tau[:, 6:]) ** 2)))

    if verbose:
        print(f"    RMSE: {rmse_grf:.3f} Nm (all), {rmse_grf_no_root:.3f} Nm (no root)")

    # ---- Method 3: Full-corrected ID ----
    if verbose:
        print(f"\n  [Method 3] Full-corrected ID...")
    tau_full_corrected = run_inverse_dynamics_with_full_correction(
        skel, pos, vel, acc, grf, cop,
        contact_torque=gt_data['contact_torque'][:T],
        residual_wrench=gt_data['residual_wrench'][:T],
        ground_force_body_names=gt_data['ground_force_body_names'],
    )

    rmse_full = float(np.sqrt(np.mean((tau_full_corrected - gt_tau) ** 2)))
    rmse_full_no_root = float(np.sqrt(np.mean((tau_full_corrected[:, 6:] - gt_tau[:, 6:]) ** 2)))

    if verbose:
        print(f"    RMSE: {rmse_full:.3f} Nm (all), {rmse_full_no_root:.3f} Nm (no root)")

    # ---- Method 4: Single-step FD ----
    if verbose:
        print(f"\n  [Method 4] Single-step Forward Dynamics...")
    fd_result = run_single_step_fd(skel, gt_data)
    fd_acc = fd_result['fd_acc'][:T]
    gt_acc_ref = fd_result['gt_acc'][:T]

    fd_acc_rmse = float(np.sqrt(np.mean((fd_acc - gt_acc_ref) ** 2)))
    fd_acc_rmse_no_root = float(np.sqrt(np.mean((fd_acc[:, 6:] - gt_acc_ref[:, 6:]) ** 2)))

    if verbose:
        print(f"    Acc RMSE: {fd_acc_rmse:.4f} rad/s² (all), {fd_acc_rmse_no_root:.4f} (no root)")

    # ---- Joint position errors ----
    gt_joint_centers = gt_data['joint_centers'][:T]
    joint_pos_errors = _compute_joint_position_errors(
        skel_joints[:T], gt_joint_centers, skel, verbose=False,
    )
    mean_joint_err = float(np.mean(list(joint_pos_errors.values())))

    if verbose:
        print(f"\n  Joint position error (SKEL vs GT): {mean_joint_err:.1f} mm mean")

    # ---- GRF statistics ----
    total_grf = np.linalg.norm(grf.sum(axis=1), axis=1)
    grf_mean = float(np.mean(total_grf))

    # ---- Improvement percentages ----
    def _improvement(rmse_new, rmse_base):
        if rmse_base < 1e-9:
            return 0.0
        return float((1.0 - rmse_new / rmse_base) * 100.0)

    # ---- Summary ----
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Force Verification Summary")
        print(f"{'='*60}")
        print(f"  {'Method':<30s} {'RMSE (Nm)':>10s} {'vs Baseline':>12s}")
        print(f"  {'-'*52}")
        print(f"  {'Standard ID (no GRF)':<30s} {rmse_standard:>10.3f} {'baseline':>12s}")
        print(f"  {'GRF-corrected ID':<30s} {rmse_grf:>10.3f} {_improvement(rmse_grf, rmse_standard):>+11.1f}%")
        print(f"  {'Full-corrected ID':<30s} {rmse_full:>10.3f} {_improvement(rmse_full, rmse_standard):>+11.1f}%")
        print(f"  {'Single-step FD (acc)':<30s} {fd_acc_rmse:>10.4f} {'rad/s²':>12s}")
        print(f"  {'-'*52}")
        print(f"  {'Joint position error':<30s} {mean_joint_err:>10.1f} {'mm':>12s}")
        print(f"  {'Mean GRF':<30s} {grf_mean:>10.1f} {'N':>12s}")
        print(f"{'='*60}")

    return {
        'methods': {
            'standard_id': {
                'tau': tau_standard,
                'rmse': rmse_standard,
                'rmse_no_root': rmse_standard_no_root,
                'label': 'Standard ID (no GRF)',
            },
            'grf_corrected': {
                'tau': tau_grf_corrected,
                'rmse': rmse_grf,
                'rmse_no_root': rmse_grf_no_root,
                'improvement': _improvement(rmse_grf, rmse_standard),
                'label': 'GRF-corrected ID',
            },
            'full_corrected': {
                'tau': tau_full_corrected,
                'rmse': rmse_full,
                'rmse_no_root': rmse_full_no_root,
                'improvement': _improvement(rmse_full, rmse_standard),
                'label': 'Full-corrected ID',
            },
            'single_step_fd': {
                'fd_acc': fd_acc,
                'gt_acc': gt_acc_ref,
                'acc_rmse': fd_acc_rmse,
                'acc_rmse_no_root': fd_acc_rmse_no_root,
                'label': 'Single-step FD',
            },
        },
        'gt_tau': gt_tau,
        'dof_names': dof_names,
        'joint_position_error': joint_pos_errors,
        'mean_joint_error_mm': mean_joint_err,
        'grf_mean': grf_mean,
        'num_frames': T,
        'n_dof': n_dof,
        'gt_data': gt_data,
    }


def plot_tau_comparison(
    results: Dict,
    output_dir: str,
    dofs_to_plot: Optional[List[str]] = None,
):
    """
    Plot tau comparison between corrected ID and GT for selected DOFs.

    Args:
        results: Output from verify_force_consistency().
        output_dir: Directory to save plots.
        dofs_to_plot: DOF names to plot. Default: hip, knee, ankle DOFs.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    tau_gt = results['tau_gt']
    tau_id = results['tau_id_corrected']
    dof_names = results['dof_names']
    T = results['num_frames']
    n_dof = results['n_dof']

    # Default: plot hip, knee, ankle DOFs
    if dofs_to_plot is None:
        keywords = ['hip', 'knee', 'ankle', 'femur', 'tibia', 'talus']
        dofs_to_plot = [
            name for name in dof_names
            if any(kw in name.lower() for kw in keywords)
        ]
        # Limit to 12 DOFs max
        dofs_to_plot = dofs_to_plot[:12]

    if not dofs_to_plot:
        # Fallback: plot DOFs 6-17 (skip root)
        dofs_to_plot = dof_names[6:min(18, n_dof)]

    n_plots = len(dofs_to_plot)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    time = np.arange(T)

    for i, dof_name in enumerate(dofs_to_plot):
        if dof_name not in dof_names:
            continue
        dof_idx = dof_names.index(dof_name)
        ax = axes[i]

        ax.plot(time, tau_gt[:, dof_idx], 'b-', label='GT tau', linewidth=1.5)
        ax.plot(time, tau_id[:, dof_idx], 'r--', label='Corrected ID', linewidth=1.0)

        rmse = np.sqrt(np.mean((tau_gt[:, dof_idx] - tau_id[:, dof_idx]) ** 2))
        ax.set_title(f"{dof_name}\nRMSE={rmse:.2f} Nm", fontsize=9)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Torque (Nm)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Corrected Inverse Dynamics vs GT Torques', fontsize=12, y=1.02)
    plt.tight_layout()

    png_path = os.path.join(output_dir, 'tau_comparison.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if True:
        print(f"  Saved tau comparison plot: {png_path}")

    return png_path


def plot_tau_comparison_comprehensive(
    results: Dict,
    output_dir: str,
    dofs_to_plot: Optional[List[str]] = None,
):
    """
    Plot tau comparison across ALL methods for selected DOFs.

    Shows GT tau vs Standard ID vs GRF-corrected vs Full-corrected.

    Args:
        results: Output from verify_force_comprehensive().
        output_dir: Directory to save plots.
        dofs_to_plot: DOF names to plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    gt_tau = results['gt_tau']
    dof_names = results['dof_names']
    T = results['num_frames']
    n_dof = results['n_dof']
    methods = results['methods']

    # Select DOFs to plot
    if dofs_to_plot is None:
        keywords = ['hip', 'knee', 'ankle', 'femur', 'tibia', 'talus']
        dofs_to_plot = [
            name for name in dof_names
            if any(kw in name.lower() for kw in keywords)
        ]
        dofs_to_plot = dofs_to_plot[:12]

    if not dofs_to_plot:
        dofs_to_plot = dof_names[6:min(18, n_dof)]

    n_plots = len(dofs_to_plot)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    time = np.arange(T)

    # Method styles
    method_styles = {
        'standard_id':    {'color': 'gray',   'linestyle': '--', 'linewidth': 1.0, 'alpha': 0.6},
        'grf_corrected':  {'color': 'orange', 'linestyle': '--', 'linewidth': 1.0, 'alpha': 0.8},
        'full_corrected': {'color': 'red',    'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.9},
    }

    for i, dof_name in enumerate(dofs_to_plot):
        if dof_name not in dof_names:
            continue
        dof_idx = dof_names.index(dof_name)
        ax = axes[i]

        # GT tau
        ax.plot(time, gt_tau[:, dof_idx], 'b-', label='GT tau', linewidth=1.5)

        # Each method
        rmse_texts = []
        for method_key, style in method_styles.items():
            if method_key not in methods or 'tau' not in methods[method_key]:
                continue
            tau_m = methods[method_key]['tau']
            label = methods[method_key]['label']
            ax.plot(time, tau_m[:, dof_idx], label=label, **style)
            rmse = np.sqrt(np.mean((tau_m[:, dof_idx] - gt_tau[:, dof_idx]) ** 2))
            rmse_texts.append(f"{label}: {rmse:.2f}")

        # Short DOF name for title
        short_name = dof_name.split('/')[-1] if '/' in dof_name else dof_name
        ax.set_title(short_name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Frame', fontsize=8)
        ax.set_ylabel('Torque (Nm)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=6, loc='upper right')

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Multi-Method Inverse Dynamics Comparison', fontsize=13, y=1.02)
    plt.tight_layout()

    png_path = os.path.join(output_dir, 'tau_comparison_comprehensive.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved comprehensive tau comparison: {png_path}")

    # Also save summary bar chart
    _plot_method_rmse_bar(methods, output_dir)

    return png_path


def _plot_method_rmse_bar(methods: Dict, output_dir: str):
    """Plot bar chart comparing method RMSEs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = []
    rmse_all = []
    rmse_no_root = []
    colors = ['gray', 'orange', 'red']

    for key in ['standard_id', 'grf_corrected', 'full_corrected']:
        if key not in methods:
            continue
        m = methods[key]
        labels.append(m['label'])
        rmse_all.append(m['rmse'])
        rmse_no_root.append(m['rmse_no_root'])

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width / 2, rmse_all, width, label='All DOFs', color=colors[:len(labels)], alpha=0.7)
    bars2 = ax.bar(x + width / 2, rmse_no_root, width, label='No Root', color=colors[:len(labels)], alpha=0.4)

    ax.set_ylabel('RMSE (Nm)')
    ax.set_title('ID Method Comparison — Tau RMSE vs GT')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(output_dir, 'method_rmse_comparison.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved RMSE comparison bar chart: {png_path}")
