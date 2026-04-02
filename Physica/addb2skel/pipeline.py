"""
Main AddB → SKEL conversion pipeline.

Implements the full 3-stage conversion:
1. Scale/Beta Estimation: bone lengths + marker pairwise distances → betas
2. Pose Optimization: joints + marker-assisted warm-up → poses
3. Virtual Marker Generation: BSM markers from fitted skin_verts → .pkl/.trc/.npy

Following the approaches from:
- AddBiomechanics: https://nimblephysics.org/docs/working-with-addbiomechanics-data.html
- HSMR SKELify: https://github.com/IsshikiHugh/HSMR
- SMPL2AddBiomechanics: BSM marker definitions and triangle-frame mapping
"""

import os
from typing import Optional, Tuple, Dict, Union
import numpy as np
import torch

from .config import OptimizationConfig, ConversionResult, DEFAULT_CONFIG, SKEL_NUM_POSE_DOF
from .skel_interface import SKELInterface, create_skel_interface
from .scale_estimation import estimate_subject_scale
from .pose_optimization import optimize_poses
from .scapula_handler import ScapulaHandler
from .joint_definitions import build_direct_joint_mapping, ADDB_JOINTS, SKEL_JOINTS


def _estimate_sex(
    height_m: float = None,
    mass_kg: float = None,
) -> str:
    """Estimate biological sex from height and mass when B3D metadata is unavailable.

    Uses a simple threshold based on adult population averages:
    - CDC NHANES 2015-2018 (US adults 20+):
      Male:   height=175.4cm, weight=90.6kg
      Female: height=161.3cm, weight=77.5kg
      Midpoint: height=168.3cm, weight=84.1kg
    - Ref: Fryar et al. "Mean Body Weight, Height, Waist Circumference, and
      Body Mass Index Among Adults" NCHS Health Statistics Reports, No. 122, 2018
      https://www.cdc.gov/nchs/data/nhsr/nhsr122-508.pdf

    Decision: If both available, use height (more discriminative).
    If only mass, use mass threshold.
    If neither, default to 'male'.
    """
    if height_m is not None and height_m > 0:
        # CDC midpoint: (175.4 + 161.3) / 2 = 168.3 cm
        return 'female' if height_m < 1.683 else 'male'
    if mass_kg is not None and mass_kg > 0:
        # CDC midpoint: (90.6 + 77.5) / 2 = 84.1 kg
        return 'female' if mass_kg < 84.1 else 'male'
    return 'male'


def convert_addb_to_skel(
    addb_joints: np.ndarray,
    sex: str = 'auto',
    config: Optional[OptimizationConfig] = None,
    height_m: Optional[float] = None,
    shoulder_width_m: Optional[float] = None,
    mass_kg: Optional[float] = None,
    return_vertices: bool = False,
    verbose: bool = True,
    markers_map=None,
    marker_observations=None,
    output_dir: Optional[str] = None,
    addb_skel=None,
    addb_poses=None,
    skel_q_reference: Optional[np.ndarray] = None,
) -> ConversionResult:
    """
    Convert AddBiomechanics joint trajectories to SKEL format.

    This is the main entry point for the addb2skel pipeline.

    3-Stage Pipeline:
    1. Beta estimation: bone lengths + marker pairwise distances
    2. Pose optimization: joints (primary) + marker warm-up (auxiliary)
    3. Virtual marker generation: BSM markers from fitted body mesh

    Args:
        addb_joints: AddB joint positions [T, 20, 3] in meters.
        sex: Subject biological sex ('male', 'female', or 'auto').
             'auto' estimates from height/mass using population statistics.
        config: Optimization configuration. Default: DEFAULT_CONFIG.
        height_m: Optional subject height in meters (for initialization).
        shoulder_width_m: Optional shoulder width in meters.
        mass_kg: Optional subject mass in kg (SKEL Section 6.5 mass constraint).
        return_vertices: Whether to return mesh vertices.
        verbose: Print progress.
        markers_map: Optional osim.markersMap from nimblephysics.
        marker_observations: Optional list of per-frame marker dicts.
        output_dir: Optional directory for saving virtual markers.

    Returns:
        ConversionResult with joints, poses, betas, trans, virtual_markers.
    """
    config = config or DEFAULT_CONFIG
    device = config.get_device()

    # Auto-detect sex for SKEL model selection
    # If sex is not 'male'/'female', estimate from height and mass.
    if sex not in ('male', 'female'):
        sex = _estimate_sex(height_m, mass_kg)

    if verbose:
        print("=" * 60)
        print("AddB → SKEL Conversion Pipeline (Virtual Marker IK)")
        print("=" * 60)
        print(f"Input: {addb_joints.shape[0]} frames, {addb_joints.shape[1]} joints")
        print(f"Sex: {sex}")
        print(f"Device: {device}")

    # Validate input shape
    T, num_joints, _ = addb_joints.shape
    if num_joints != 20:
        raise ValueError(f"Expected 20 AddB joints, got {num_joints}")

    # Coordinate conversion: AddB → SKEL (flip X and Z)
    addb_joints_converted = addb_joints.copy()
    addb_joints_converted[:, :, 0] = -addb_joints_converted[:, :, 0]
    addb_joints_converted[:, :, 2] = -addb_joints_converted[:, :, 2]

    # =========================================================================
    # Prepare Marker Handler
    # =========================================================================
    marker_handler = None
    has_markers = markers_map is not None and marker_observations is not None

    if config.use_marker_loss and has_markers:
        if config.use_bsm_marker_handler:
            from .marker_handler import BSMVertexMarkerHandler
            marker_handler = BSMVertexMarkerHandler(
                device=device, bsm_markers_path=config.bsm_markers_path
            )
        elif config.use_vertex_marker_handler:
            from .marker_handler import VertexMarkerHandler
            marker_handler = VertexMarkerHandler(
                device=device, bsm_markers_path=config.bsm_markers_path
            )
        else:
            from .marker_handler import MarkerHandler
            marker_handler = MarkerHandler(device=device)

        flip_xz = getattr(config, 'marker_flip_xz', True)
        blacklist = list(getattr(config, 'marker_blacklist', []))
        # Un-blacklist head markers if use_head_markers is enabled
        if getattr(config, 'use_head_markers', False):
            _head_markers = {'LFHD', 'RFHD', 'LBHD', 'RBHD'}
            blacklist = [m for m in blacklist if m.upper() not in _head_markers]
        num_matched = marker_handler.prepare_from_markersmap(
            markers_map, marker_observations, flip_xz=flip_xz, blacklist=blacklist
        )
        if verbose:
            print(f"\n--- Marker Handler ---")
            print(f"  {marker_handler.get_summary()}")
        if num_matched == 0:
            marker_handler = None

    # Initialize SKEL model
    if verbose:
        print("\n--- Loading SKEL Model ---")
    skel = create_skel_interface(sex=sex, device=str(device))

    # =========================================================================
    # Stage 1: Beta Estimation (bone lengths + marker pairwise distances)
    # =========================================================================
    if verbose:
        print("\n--- Stage 1: Beta Estimation ---")

    # 1a. Bone-length-based estimation
    scale_marker_handler = marker_handler if config.marker_in_stage1 else None
    betas, scale_stats = estimate_subject_scale(
        addb_joints_converted,
        skel,
        config,
        height_m=height_m,
        shoulder_width_m=shoulder_width_m,
        verbose=verbose,
        marker_handler=scale_marker_handler,
        target_mass_kg=mass_kg,
    )

    # Extract dJ from scale estimation (if optimized)
    dJ = scale_stats.get('dJ')

    if verbose:
        print(f"  Bone length betas: {betas[:5].cpu().numpy()}")
        print(f"  Mean bone length error: {scale_stats['mean_length_error_mm']:.2f} mm")
        if dJ is not None:
            print(f"  dJ offsets: mean={scale_stats.get('dJ_mean_mm', 0):.2f}mm, "
                  f"max={scale_stats.get('dJ_max_mm', 0):.2f}mm")

    # 1b. Marker pairwise distance refinement (optional)
    marker_beta_stats = {}
    if config.use_marker_beta and has_markers:
        if verbose:
            print(f"\n  Marker-based beta refinement...")
        from .marker_beta_estimation import MarkerBetaEstimator
        marker_beta = MarkerBetaEstimator(
            skel, config.bsm_markers_path, config
        )
        betas_marker, marker_beta_stats = marker_beta.estimate_betas(
            marker_observations, initial_betas=betas,
            flip_xz=True, verbose=verbose,
        )
        # Weighted average of bone-length and marker-based betas
        alpha = 0.5
        betas = alpha * betas + (1 - alpha) * betas_marker
        if verbose:
            print(f"  Combined betas: {betas[:5].cpu().numpy()}")
            print(f"  Marker pairs used: {marker_beta_stats.get('n_pairs', 0)}")

    # =========================================================================
    # Stage 1c: Offset-Based Marker Matching (Tier 2.5)
    # Uses AddB body-local offsets + SKEL T-pose to auto-map all markers
    # Must run after Stage 1 (betas needed for subject-shaped T-pose)
    # =========================================================================
    if (marker_handler is not None and has_markers
            and config.use_offset_matching
            and hasattr(marker_handler, 'prepare_from_markersmap_with_offsets')):
        n_offset = marker_handler.prepare_from_markersmap_with_offsets(
            markers_map,
            marker_observations,
            skel_interface=skel,
            betas=betas,
            flip_xz=True,
            max_dist_mm=config.offset_matching_max_dist_mm,
        )
        if verbose and n_offset > 0:
            print(f"\n--- Tier 2.5 Offset Matching: +{n_offset} markers ---")
            print(f"  {marker_handler.get_summary()}")

    # =========================================================================
    # Stage 2: Pose Optimization (joint + marker warm-up)
    # =========================================================================
    if verbose:
        print("\n--- Stage 2: Pose Optimization ---")

    # Decide marker integration strategy
    pose_marker_handler = None
    if config.use_marker_in_pose and marker_handler is not None:
        pose_marker_handler = marker_handler
        if verbose:
            print(f"  Marker warm-up enabled (warmup_fraction={config.marker_warmup_fraction})")

    # Optimize poses (marker handler integrated with warm-up schedule)
    # addb_poses: pp.pos per frame [T, N_addb_dofs] — used for direct DOF initialization
    # skel_q_reference: direct SKEL DOF reference [T, 46] — bypasses scale mapping for q-match
    addb_poses_arr = np.array(addb_poses) if addb_poses else None

    # Auto-compute skel_q_reference from AddB poses using rotation matrix mapping
    # Uses subject-adaptive rotation (auto-detected from GT joints) for AddB→SKEL alignment
    if skel_q_reference is None and addb_poses_arr is not None and addb_skel is not None:
        if config.weight_q_match > 0:
            from .pose_optimization import compute_skel_q_reference_from_addb
            if verbose:
                print(f"  Auto-computing skel_q_reference (rotmat mapping, {T} frames)...")
            skel_q_reference = compute_skel_q_reference_from_addb(
                addb_skel, addb_poses_arr, T,
                addb_joints=addb_joints,
                use_gt_align=getattr(config, 'use_gt_align', False),
                use_r_flip_xz=getattr(config, 'use_r_flip_xz', False))
            if verbose:
                print(f"  Done.")

    # Auto-adaptive pelvis: enabled after four-pass if probe can't discriminate directions

    # Method C: Four-pass — try 0°/90°/180°/270° pelvis rotation, pick lower MPJPE
    # Why 4 directions: R_90_Y transform maps AddB pelvis to SKEL, but subjects
    # walking along +X or -X axis land pelvis_Y at ±90°, which is 90° away from
    # both 0° and 180° probes. Adding 90°/270° covers all cardinal directions.
    if getattr(config, 'use_two_pass', False) and skel_q_reference is not None:
        import copy
        two_pass_iters = getattr(config, 'two_pass_iters', 20)
        offsets = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        if verbose:
            print(f"  Four-pass probe: {two_pass_iters} iters × {len(offsets)} directions...")

        best_pass_mpjpe = float('inf')
        best_q_ref = skel_q_reference
        probe_results = []

        for pass_idx, pelvis_offset in enumerate(offsets):
            q_ref_trial = skel_q_reference.copy()
            q_ref_trial[:, 2] += pelvis_offset  # pelvis_rotation DOF

            config_probe = copy.deepcopy(config)
            config_probe.pose_iters = two_pass_iters
            config_probe.use_direction_first = False  # skip Phase A for speed

            _, _, probe_stats = optimize_poses(
                addb_joints_converted, betas, skel, config_probe,
                verbose=False,
                marker_handler=pose_marker_handler,
                addb_poses=addb_poses_arr, dJ=dJ,
                skel_q_reference=q_ref_trial,
            )
            probe_mpjpe = probe_stats.get('mpjpe_mm', float('inf'))
            probe_results.append(probe_mpjpe)
            if verbose:
                print(f"    Pass {pass_idx} (offset={np.degrees(pelvis_offset):.0f}°): MPJPE={probe_mpjpe:.1f}mm")

            if probe_mpjpe < best_pass_mpjpe:
                best_pass_mpjpe = probe_mpjpe
                best_q_ref = q_ref_trial

        skel_q_reference = best_q_ref
        if verbose:
            print(f"    Selected: offset with MPJPE={best_pass_mpjpe:.1f}mm")

        # Auto-adaptive: if probe can't discriminate (all MPJPEs similar),
        # the direction is ambiguous → enable adaptive pelvis limit
        if getattr(config, 'auto_adaptive_pelvis', False):
            spread = max(probe_results) - min(probe_results)
            if spread < 10.0:  # < 10mm spread = can't discriminate at all
                config.use_adaptive_pelvis_limit = True
                config.use_direction_first = False
                config.auto_adaptive_pelvis = False  # consumed, prevent double-apply
                if verbose:
                    print(f"    Auto-adaptive: probe spread={spread:.1f}mm (<10) → adaptive ON, Phase A OFF")
            else:
                config.use_adaptive_pelvis_limit = False  # explicitly OFF
                config.auto_adaptive_pelvis = False  # consumed
                if verbose:
                    print(f"    Auto-adaptive: probe spread={spread:.1f}mm (>=10) → baseline settings")

    if config.use_sequential or config.use_sliding_window:
        from .pose_optimization import PoseOptimizer
        device = config.get_device()
        addb_joints_t = torch.from_numpy(addb_joints_converted).float().to(device)
        opt = PoseOptimizer(skel, config, marker_handler=pose_marker_handler)
        if config.use_sequential:
            if verbose:
                print("\n--- Stage 2: Sequential Optimization ---")
            poses, trans, pose_stats = opt.optimize_sequence_sequential(
                addb_joints_t, betas, verbose=verbose,
                addb_poses=addb_poses_arr, dJ=dJ,
                skel_q_reference=skel_q_reference,
            )
        else:
            if verbose:
                print("\n--- Stage 2: Sliding Window Optimization ---")
            poses, trans, pose_stats = opt.optimize_sequence_sliding(
                addb_joints_t, betas, verbose=verbose,
                addb_poses=addb_poses_arr, dJ=dJ,
                skel_q_reference=skel_q_reference,
            )
    else:
        poses, trans, pose_stats = optimize_poses(
            addb_joints_converted,
            betas,
            skel,
            config,
            verbose=verbose,
            marker_handler=pose_marker_handler,
            addb_poses=addb_poses_arr,
            dJ=dJ,
            skel_q_reference=skel_q_reference,
        )

    # Use optimized betas from Stage 2
    if 'final_betas' in pose_stats:
        betas = pose_stats['final_betas']

    # Save Stage 2 checkpoint (best MPJPE reference for subsequent stages)
    best_mpjpe = pose_stats['mpjpe_mm']
    best_poses = poses.clone()
    best_trans = trans.clone()
    best_betas = betas.clone()
    best_stage = 'Stage 2'

    if verbose:
        print(f"\n  Stage 2 MPJPE: {pose_stats['mpjpe_mm']:.1f} mm")
        print(f"  Scapula DOFs (mean):")
        for k, v in pose_stats['scapula_dofs'].items():
            print(f"    {k}: {v:.3f} rad")
        # Per-marker error diagnostics
        marker_stats = pose_stats.get('marker_stats', {})
        if marker_stats:
            mean_err = sum(marker_stats.values()) / len(marker_stats)
            worst = sorted(marker_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Marker error (mean): {mean_err:.1f} mm")
            print(f"  Worst 5 markers: " + ", ".join(f"{n}={e:.1f}mm" for n, e in worst))

    # =========================================================================
    # Stage 2b: Marker Fine-tuning (if using 2-pass approach)
    # =========================================================================
    if marker_handler is not None and not config.use_marker_in_pose:
        # 2-pass approach: first pure joint optimization, then marker fine-tuning
        if verbose:
            print("\n--- Stage 2b: Marker Fine-tuning (2-pass) ---")

        # Find nearest vertices using fitted mesh
        with torch.no_grad():
            verts_fitted, _, _ = skel.forward(
                betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=dJ
            )
        if hasattr(marker_handler, 'find_nearest_vertices_from_mesh'):
            marker_handler.find_nearest_vertices_from_mesh(verts_fitted)
        if verbose:
            print(f"  Marker vertex matching: {marker_handler.get_summary()}")

        from .pose_optimization import finetune_with_markers
        poses, trans, ft_stats = finetune_with_markers(
            addb_joints_converted, betas, poses, trans,
            skel, config, marker_handler, verbose=verbose,
            dJ=dJ,
        )

        if 'final_betas' in ft_stats:
            betas = ft_stats['final_betas']

        if verbose:
            print(f"\n  Fine-tuned MPJPE: {ft_stats['mpjpe_mm']:.1f} mm "
                  f"(before: {pose_stats['mpjpe_mm']:.1f} mm)")
            delta = ft_stats['mpjpe_mm'] - pose_stats['mpjpe_mm']
            print(f"  MPJPE change: {delta:+.1f} mm")
            marker_stats_ft = ft_stats.get('marker_stats', {})
            if marker_stats_ft:
                mean_err = sum(marker_stats_ft.values()) / len(marker_stats_ft)
                worst = sorted(marker_stats_ft.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Marker error (mean): {mean_err:.1f} mm")
                print(f"  Worst 5 markers: " + ", ".join(f"{n}={e:.1f}mm" for n, e in worst))

        # Rollback if MPJPE degraded
        if ft_stats['mpjpe_mm'] < best_mpjpe:
            best_mpjpe = ft_stats['mpjpe_mm']
            best_poses = poses.clone()
            best_trans = trans.clone()
            best_betas = betas.clone()
            best_stage = 'Stage 2b'
        else:
            if verbose:
                print(f"  [Rollback] Stage 2b degraded MPJPE ({ft_stats['mpjpe_mm']:.1f} > {best_mpjpe:.1f}), keeping {best_stage}")
            poses = best_poses.clone()
            trans = best_trans.clone()
            betas = best_betas.clone()

        pose_stats = ft_stats

    # =========================================================================
    # Stage 2b: Tier 1/2 Pre-Fine-Tune (before Tier 3 markers are added)
    # =========================================================================
    # Gives Tier 1/2 anchors stronger convergence before Tier 3 cluster markers
    # are introduced. Only runs when use_marker_in_pose=True (Stage 2 already
    # included markers) or when 2-pass was used above.
    tier12_iters = getattr(config, 'tier12_finetune_iters', 0)
    if (marker_handler is not None and config.use_marker_loss and tier12_iters > 0):
        if verbose:
            print(f"\n--- Stage 2b: Tier 1/2 pre-fine-tune ({tier12_iters} iters) ---")
        from .pose_optimization import finetune_with_markers
        poses, trans, ft12_stats = finetune_with_markers(
            addb_joints_converted, betas, poses, trans,
            skel, config, marker_handler, verbose=verbose,
            n_iters=tier12_iters, dJ=dJ,
        )
        if 'final_betas' in ft12_stats:
            betas = ft12_stats['final_betas']
        if verbose and 'mpjpe_mm' in ft12_stats:
            print(f"  Stage 2b MPJPE: {ft12_stats['mpjpe_mm']:.2f}mm")

        # Rollback if MPJPE degraded
        if ft12_stats.get('mpjpe_mm', float('inf')) < best_mpjpe:
            best_mpjpe = ft12_stats['mpjpe_mm']
            best_poses = poses.clone()
            best_trans = trans.clone()
            best_betas = betas.clone()
            best_stage = 'Stage 2b-t12'
        else:
            if verbose:
                print(f"  [Rollback] Stage 2b-t12 degraded MPJPE ({ft12_stats.get('mpjpe_mm', 0):.1f} > {best_mpjpe:.1f}), keeping {best_stage}")
            poses = best_poses.clone()
            trans = best_trans.clone()
            betas = best_betas.clone()

    # =========================================================================
    # Stage 2c: Tier 3 Position-Based Marker Matching (if applicable)
    # =========================================================================
    if (marker_handler is not None and has_markers
            and config.use_position_based_fallback
            and hasattr(marker_handler, 'match_remaining_by_position')):
        with torch.no_grad():
            verts_for_t3, _, _ = skel.forward(
                betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=dJ
            )
        n_new = marker_handler.match_remaining_by_position(
            verts_for_t3,
            marker_observations=marker_observations,
            max_distance_mm=config.position_fallback_max_dist_mm,
            markers_map=markers_map,
            skel_interface=skel,
            addb_skel=addb_skel,
            addb_poses=addb_poses,
        )
        if verbose and n_new > 0:
            print(f"\n--- Tier 3 Position Matching: +{n_new} markers ---")
            print(f"  {marker_handler.get_summary()}")

        # Stage 2d: Fine-tune with Tier 3 markers (if any were added)
        if n_new > 0 and config.use_marker_loss and config.marker_finetune_iters > 0:
            if verbose:
                print(f"\n--- Stage 2d: Fine-tuning with Tier 3 markers (+{n_new}) ---")
            from .pose_optimization import finetune_with_markers
            poses, trans, ft3_stats = finetune_with_markers(
                addb_joints_converted, betas, poses, trans,
                skel, config, marker_handler, verbose=verbose,
                dJ=dJ,
            )
            if 'final_betas' in ft3_stats:
                betas = ft3_stats['final_betas']
            if verbose:
                print(f"  Stage 2d MPJPE: {ft3_stats.get('mpjpe_mm', 0):.1f}mm")

            # Rollback if MPJPE degraded
            if ft3_stats.get('mpjpe_mm', float('inf')) < best_mpjpe:
                best_mpjpe = ft3_stats['mpjpe_mm']
                best_poses = poses.clone()
                best_trans = trans.clone()
                best_betas = betas.clone()
                best_stage = 'Stage 2d'
            else:
                if verbose:
                    print(f"  [Rollback] Stage 2d degraded MPJPE ({ft3_stats.get('mpjpe_mm', 0):.1f} > {best_mpjpe:.1f}), keeping {best_stage}")
                poses = best_poses.clone()
                trans = best_trans.clone()
                betas = best_betas.clone()

            pose_stats = ft3_stats

    # =========================================================================
    # Stage 2e: Foot Fine-Tune (foot DOFs only — GRF accuracy)
    # =========================================================================
    foot_stats = {}
    ft_foot_iters = getattr(config, 'foot_finetune_iters', 0)
    if ft_foot_iters > 0:
        if verbose:
            print(f"\n--- Stage 2e: Foot Fine-Tune ({ft_foot_iters} iters, 6 DOFs) ---")
        from .pose_optimization import finetune_foot
        poses, trans, foot_stats = finetune_foot(
            addb_joints_converted, betas, poses, trans,
            skel, config, verbose=verbose,
            marker_observations=marker_observations,
            dJ=dJ,
        )
        if verbose:
            print(f"  Stage 2e MPJPE: {foot_stats.get('mpjpe_mm', 0):.1f}mm, "
                  f"Foot: {foot_stats.get('foot_mpjpe_mm', 0):.1f}mm")

        # Rollback if MPJPE degraded
        if foot_stats.get('mpjpe_mm', float('inf')) < best_mpjpe:
            best_mpjpe = foot_stats['mpjpe_mm']
            best_poses = poses.clone()
            best_trans = trans.clone()
            best_betas = betas.clone()
            best_stage = 'Stage 2e'
        else:
            if verbose:
                print(f"  [Rollback] Stage 2e degraded MPJPE ({foot_stats.get('mpjpe_mm', 0):.1f} > {best_mpjpe:.1f}), keeping {best_stage}")
            poses = best_poses.clone()
            trans = best_trans.clone()
            betas = best_betas.clone()

        # Update pose_stats with foot results (use best_mpjpe if rollback occurred)
        pose_stats.update({
            'mpjpe_mm': best_mpjpe,
            'per_joint_error_mm': foot_stats.get('per_joint_error_mm'),
            'scapula_dofs': foot_stats.get('scapula_dofs'),
        })

    # Extract foot marker offsets (computed in finetune_foot Phase 2)
    foot_marker_offsets = None
    if foot_stats.get('foot_marker_offsets'):
        foot_marker_offsets = foot_stats['foot_marker_offsets']
        if foot_marker_offsets and verbose:
            print(f"  Foot marker offsets: {len(foot_marker_offsets)} markers")
            for name, off in foot_marker_offsets.items():
                print(f"    {name}: [{off[0]*1000:+.1f}, {off[1]*1000:+.1f}, {off[2]*1000:+.1f}]mm")

    # =========================================================================
    # Stage 3: Virtual Marker Generation
    # =========================================================================
    virtual_markers = None
    virtual_marker_names = None
    marker_quality = None

    if config.save_virtual_markers and has_markers:
        if verbose:
            print("\n--- Stage 3: Virtual Marker Generation ---")

        from .virtual_marker_generator import VirtualMarkerGenerator
        vmg = VirtualMarkerGenerator(skel, config.bsm_markers_path, device)

        # Generate marker trajectories from fitted skin_verts
        virtual_markers = vmg.generate_markers(betas, poses, trans)
        virtual_marker_names = vmg.marker_names

        if verbose:
            print(f"  Generated {vmg.num_markers} virtual markers, "
                  f"shape: {virtual_markers.shape}")

        # Compute bone rigging
        bone_rigging = vmg.compute_bone_rigging()
        if verbose:
            unique_bones = len(set(bone_rigging.tolist()))
            print(f"  Bone rigging: markers assigned to {unique_bones} bones")

        # Compute triangle-frame offsets
        vmg.compute_triangle_offsets(betas, poses, trans)
        if verbose:
            print(f"  Triangle-frame offsets computed")

        # Apply per-subject foot marker offsets to virtual markers
        if foot_marker_offsets and virtual_markers is not None:
            n_adjusted = 0
            for k, name in enumerate(virtual_marker_names):
                if name in foot_marker_offsets:
                    virtual_markers[:, k, :] += foot_marker_offsets[name]
                    n_adjusted += 1
            if verbose and n_adjusted > 0:
                print(f"  Applied foot marker offsets to {n_adjusted} virtual markers")

        # Marker quality assessment (flip_xz converts AddB coords → SKEL coords)
        marker_quality = vmg.get_marker_quality(
            marker_observations=marker_observations, flip_xz=True
        )
        if marker_quality and verbose:
            errors = list(marker_quality.values())
            print(f"  Marker quality: mean={np.mean(errors):.1f}mm, "
                  f"median={np.median(errors):.1f}mm, "
                  f"max={np.max(errors):.1f}mm ({len(errors)} markers)")

        # Save virtual markers
        if output_dir is not None:
            marker_dir = os.path.join(output_dir, 'virtual_markers')
            vmg.save(
                marker_dir,
                save_pkl=True,
                save_npy=True,
                save_trc=config.save_trc,
            )
            if verbose:
                print(f"  Saved to {marker_dir}/")

    # =========================================================================
    # Generate Final Outputs
    # =========================================================================
    if verbose:
        print("\n--- Generating Final Outputs ---")

    # Forward pass to get joints (and optionally vertices)
    with torch.no_grad():
        if return_vertices or config.save_virtual_markers:
            vertices, joints, _ = skel.forward(
                betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=dJ, return_skeleton=False
            )
        else:
            joints = skel.forward_kinematics(
                betas.unsqueeze(0).expand(T, -1), poses, trans, dJ=dJ
            )
            vertices = None

    # Flip X and Z back to original AddB coordinate system
    joints[:, :, 0] = -joints[:, :, 0]
    joints[:, :, 2] = -joints[:, :, 2]
    trans[:, 0] = -trans[:, 0]
    trans[:, 2] = -trans[:, 2]
    if vertices is not None:
        vertices[:, :, 0] = -vertices[:, :, 0]
        vertices[:, :, 2] = -vertices[:, :, 2]

    # Ensure pose_stats reflects the best MPJPE after all rollback gates
    pose_stats['mpjpe_mm'] = best_mpjpe

    # Compute evaluation metrics (all 7)
    evaluation_metrics = None
    try:
        from .utils.metrics import compute_all_metrics
        from .joint_definitions import (
            ADDB_TO_SKEL_MAPPING, ADDB_JOINTS, ADDB_JOINT_TO_IDX,
            SKEL_JOINT_TO_IDX, MappingType,
        )

        # Build DIRECT-only joint indices
        direct_skel_idxs = []
        direct_addb_idxs = []
        for addb_name, mapping in ADDB_TO_SKEL_MAPPING.items():
            if mapping.mapping_type == MappingType.DIRECT and mapping.use_in_loss:
                direct_addb_idxs.append(ADDB_JOINT_TO_IDX[addb_name])
                direct_skel_idxs.append(SKEL_JOINT_TO_IDX[mapping.skel_joints[0]])

        # Both joints and addb_joints are now in original AddB coordinate system
        addb_joints_torch = torch.from_numpy(addb_joints).float().to(joints.device)
        pred_matched = joints[:, direct_skel_idxs, :]
        gt_matched = addb_joints_torch[:T, direct_addb_idxs, :]

        verts_np = vertices.cpu().numpy() if vertices is not None else None
        evaluation_metrics = compute_all_metrics(
            pred_matched, gt_matched,
            root_idx=0,
            vertices=verts_np,
            dt=getattr(config, 'dt', None),
        )
    except Exception as e:
        if verbose:
            print(f"  [Warning] Evaluation metrics computation failed: {e}")

    # Build result
    result = ConversionResult(
        skel_joints=joints,
        skel_poses=poses,
        skel_betas=betas,
        skel_trans=trans,
        skel_vertices=vertices if return_vertices else None,
        virtual_markers=virtual_markers,
        virtual_marker_names=virtual_marker_names,
        foot_marker_offsets=foot_marker_offsets,
        marker_quality=pose_stats.get('marker_stats') or marker_quality,
        marker_tier_counts=getattr(marker_handler, '_tier_counts', None) if marker_handler else None,
        mpjpe_mm=best_mpjpe,
        per_joint_error=pose_stats.get('per_joint_error_mm'),
        scapula_dofs=pose_stats['scapula_dofs'],
        evaluation_metrics=evaluation_metrics,
        num_frames=T,
        sex=sex,
        scale_stats=scale_stats,
        finetune_stats=pose_stats,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Conversion Complete!")
        print("=" * 60)
        print(f"Output shapes:")
        print(f"  skel_joints: {result.skel_joints.shape}")
        print(f"  skel_poses:  {result.skel_poses.shape}")
        print(f"  skel_betas:  {result.skel_betas.shape}")
        print(f"  skel_trans:  {result.skel_trans.shape}")
        if result.skel_vertices is not None:
            print(f"  skel_vertices: {result.skel_vertices.shape}")
        if result.virtual_markers is not None:
            print(f"  virtual_markers: {result.virtual_markers.shape}")
        if result.evaluation_metrics:
            from .utils.metrics import format_metrics_table
            print(format_metrics_table(result.evaluation_metrics))

    return result


def save_conversion_result(
    result: ConversionResult,
    output_dir: str,
    save_obj: bool = True,
    save_npz: bool = True,
    save_comparison_obj: bool = True,
    gt_joints: Optional[np.ndarray] = None,
    gt_markers: Optional[np.ndarray] = None,
    frame: int = 0,
):
    """
    Save conversion result to files.

    Args:
        result: ConversionResult from convert_addb_to_skel().
        output_dir: Output directory.
        save_obj: Save OBJ mesh files.
        save_npz: Save NPZ parameter files.
        save_comparison_obj: Save comparison OBJs (body+skeleton+markers+joints).
        gt_joints: GT joint positions [J, 3] in AddB coords (for comparison OBJ).
        gt_markers: GT marker positions [M, 3] in AddB coords (for comparison OBJ).
        frame: Frame index for comparison OBJ (default 0).
    """
    from .utils.io import save_obj as write_obj
    from .utils.visualization import joints_to_obj, save_comparison_objs

    os.makedirs(output_dir, exist_ok=True)

    # Save parameters as NPZ
    if save_npz:
        save_data = {
            'poses': result.skel_poses.cpu().numpy(),
            'betas': result.skel_betas.cpu().numpy(),
            'trans': result.skel_trans.cpu().numpy(),
            'joints': result.skel_joints.cpu().numpy(),
        }
        if result.virtual_markers is not None:
            save_data['virtual_markers'] = result.virtual_markers
        if result.virtual_marker_names is not None:
            save_data['virtual_marker_names'] = np.array(result.virtual_marker_names)

        np.savez(os.path.join(output_dir, 'skel_params.npz'), **save_data)
        print(f"Saved parameters to {output_dir}/skel_params.npz")

    skel = create_skel_interface(result.sex) if (save_obj or save_comparison_obj) else None

    # Save OBJ files
    if save_obj and skel is not None:
        if result.skel_vertices is not None:
            write_obj(
                result.skel_vertices[0].cpu().numpy(),
                skel.faces,
                os.path.join(output_dir, 'skel_mesh.obj'),
            )
            print(f"Saved mesh to {output_dir}/skel_mesh.obj")

        for t in range(min(result.num_frames, 10)):
            joints_to_obj(
                result.skel_joints[t].cpu().numpy(),
                os.path.join(output_dir, f'skel_joints_frame{t:04d}.obj'),
                parents=skel.parents,
            )

        print(f"Saved joint OBJs to {output_dir}/")

    # Save comparison OBJs (body + skeleton mesh + markers + GT)
    if save_comparison_obj and skel is not None and result.skel_vertices is not None:
        vm = result.virtual_markers[frame] if result.virtual_markers is not None else None
        save_comparison_objs(
            output_dir=os.path.join(output_dir, 'comparison'),
            body_verts=result.skel_vertices[frame].cpu().numpy(),
            body_faces=skel.faces,
            skel_joints=result.skel_joints[frame].cpu().numpy(),
            parents=skel.parents,
            gt_joints=gt_joints,
            gt_markers=gt_markers,
            virtual_markers=vm,
            foot_marker_offsets=result.foot_marker_offsets,
            skel_model=skel.model,
            skel_betas=result.skel_betas,
            skel_poses_frame=result.skel_poses[frame],
            skel_trans_frame=result.skel_trans[frame],
        )
        print(f"Saved comparison OBJs to {output_dir}/comparison/")


def quick_convert(
    b3d_path: str,
    output_dir: str,
    num_frames: Optional[int] = None,
    verbose: bool = True,
    use_virtual_markers: bool = True,
) -> ConversionResult:
    """
    Quick conversion from .b3d file to SKEL with virtual markers.

    Args:
        b3d_path: Path to AddB .b3d file.
        output_dir: Output directory for results.
        num_frames: Number of frames to process. None for all.
        verbose: Print progress.
        use_virtual_markers: Enable full virtual marker pipeline.

    Returns:
        ConversionResult.
    """
    from .utils.io import load_b3d

    if verbose:
        print(f"Loading {b3d_path}...")

    addb_joints, joint_names, metadata = load_b3d(
        b3d_path, num_frames=num_frames, load_markers=True
    )

    if verbose:
        print(f"  Subject: height={metadata['height_m']:.2f}m, "
              f"mass={metadata['mass_kg']:.1f}kg, sex={metadata['sex']}")

    sex = metadata.get('sex', 'male')
    if sex not in ('male', 'female'):
        sex = 'male'

    # Configure for virtual marker pipeline
    config = OptimizationConfig()
    if use_virtual_markers:
        config.use_marker_loss = True
        config.use_bsm_marker_handler = True
        config.use_marker_in_pose = True
        config.save_virtual_markers = True
        config.save_trc = True

    result = convert_addb_to_skel(
        addb_joints,
        sex=sex,
        config=config,
        height_m=metadata['height_m'],
        mass_kg=metadata.get('mass_kg'),
        return_vertices=True,
        verbose=verbose,
        markers_map=metadata.get('markers_map'),
        marker_observations=metadata.get('marker_observations'),
        output_dir=output_dir,
        addb_skel=metadata.get('addb_skel'),
        addb_poses=metadata.get('addb_poses'),
    )

    save_conversion_result(result, output_dir, save_obj=True, save_npz=True)

    return result
