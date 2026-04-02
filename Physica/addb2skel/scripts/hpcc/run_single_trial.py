#!/usr/bin/env python3
"""
Process a single trial from AddBiomechanics .b3d file.

Designed for HPCC SLURM job array: each job runs this script with a
different --task_id that maps to a specific (b3d_path, trial_idx) via
the trial manifest CSV.

Algorithm: Fix K (2026-04-02)
  - Phase A OFF + adaptive pelvis ±90°
  - Marker T1+T2 (T3 disabled)
  - PR#38 scapula limits
  - spine_reg=0.3, width=2.0
  - Full metrics + marker matching stats

Usage:
    python run_single_trial.py \
        --manifest trial_manifest.csv \
        --task_id 0 \
        --output_dir /mnt/scratch/kwonjoon/output/addb2skel/ \
        --device cuda

Output per trial:
    {output_dir}/{subject_id}/trial_{trial_idx}/
        result.npz      — poses, betas, trans, skel_joints, gt_joints
        metrics.json     — full evaluation metrics
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Add addb2skel package to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_ADDB2SKEL_DIR = _SCRIPT_DIR.parent.parent
_PHYSICA_DIR = _ADDB2SKEL_DIR.parent
sys.path.insert(0, str(_PHYSICA_DIR))

import numpy as np


def load_b3d_trial(b3d_path, trial_idx, target_fps=100.0, load_markers=True):
    """
    Load a specific trial from a .b3d file with optional resampling and markers.
    """
    import nimblephysics as nimble
    from scipy.signal import resample_poly
    from fractions import Fraction

    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    skel = subject.readSkel(0, ignoreGeometry=True)

    # Frame rate
    timestep = subject.getTrialTimestep(trial_idx)
    original_fps = 1.0 / timestep if timestep > 0 else 100.0

    # Resampling parameters
    needs_resample = False
    up, down = 1, 1
    actual_fps = original_fps
    if target_fps is not None:
        ratio = target_fps / original_fps
        if abs(ratio - 1.0) >= 0.05:
            frac = Fraction(round(target_fps), round(original_fps))
            up, down = frac.numerator, frac.denominator
            actual_fps = original_fps * up / down
            needs_resample = True

    # Read all frames
    trial_length = subject.getTrialLength(trial_idx)
    trial_frames = subject.readFrames(
        trial=trial_idx, startFrame=0,
        numFramesToRead=trial_length, stride=1,
    )
    T_raw = len(trial_frames)
    if T_raw == 0:
        raise ValueError(f"Trial {trial_idx} has 0 frames")

    # Processing pass index: use kinematics pass (0)
    pp_idx = 0

    # Joint names from first frame
    pp = trial_frames[0].processingPasses[pp_idx]
    pos = np.asarray(pp.pos, dtype=np.float32)
    if pos.size == skel.getNumDofs():
        skel.setPositions(pos)

    world_joints = []
    for i in range(skel.getNumJoints()):
        joint = skel.getJoint(i)
        world = joint.getChildBodyNode().getWorldTransform().translation()
        world_joints.append((joint.getName(), world))

    joint_centers_first = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
    num_joints = joint_centers_first.shape[0]

    joint_names = []
    for center in joint_centers_first:
        dists = [np.linalg.norm(center - w) for _, w in world_joints]
        joint_names.append(world_joints[int(np.argmin(dists))][0])

    # Extract joint positions + marker observations + addb_poses
    joints_raw = np.zeros((T_raw, num_joints, 3), dtype=np.float32)
    marker_obs_raw = [] if load_markers else None
    addb_poses_raw = [] if load_markers else None

    for t, frame in enumerate(trial_frames):
        pp = frame.processingPasses[pp_idx]
        joints_raw[t] = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)

        if load_markers:
            marker_dict = {}
            try:
                for mname, mpos in frame.markerObservations:
                    mname = mname.strip()
                    marker_dict[mname] = np.array(
                        [float(mpos[0]), float(mpos[1]), float(mpos[2])],
                        dtype=np.float32)
            except Exception:
                pass
            marker_obs_raw.append(marker_dict)
            addb_poses_raw.append(np.asarray(pp.pos, dtype=np.float32))

    # Load markers_map and addb_skel
    markers_map = None
    addb_skel = None
    if load_markers:
        try:
            # Use processing_pass=2 for dynamics pass (has markers)
            n_passes = len(trial_frames[0].processingPasses)
            osim_pass = min(2, n_passes - 1)
            osim = subject.readOpenSimFile(processingPass=osim_pass, ignoreGeometry=True)
            markers_map = osim.markersMap
            for _, val in markers_map.items():
                try:
                    body_node, _ = val
                    addb_skel = body_node.getSkeleton()
                    break
                except Exception:
                    pass
        except Exception:
            markers_map = None

    # GRF mask
    missing_grf_all = subject.getMissingGRF(trial_idx)
    notMissing = nimble.biomechanics.MissingGRFReason.notMissingGRF
    grf_mask_raw = np.zeros(T_raw, dtype=bool)
    for t in range(T_raw):
        if t < len(missing_grf_all):
            grf_mask_raw[t] = (missing_grf_all[t] == notMissing)

    # Resample
    if needs_resample:
        joints = resample_poly(joints_raw, up, down, axis=0).astype(np.float32)
        T = joints.shape[0]
        src = np.round(np.arange(T) * (T_raw - 1) / max(T - 1, 1)).astype(int)
        src = np.clip(src, 0, T_raw - 1)
        grf_valid_mask = grf_mask_raw[src]

        if load_markers and addb_poses_raw:
            poses_arr = np.stack(addb_poses_raw, axis=0)
            poses_resampled = resample_poly(poses_arr, up, down, axis=0).astype(np.float32)
            addb_poses = [poses_resampled[t] for t in range(T)]
        else:
            addb_poses = addb_poses_raw

        if load_markers and marker_obs_raw:
            marker_observations = [marker_obs_raw[i] for i in src]
        else:
            marker_observations = marker_obs_raw
    else:
        joints = joints_raw
        T = T_raw
        grf_valid_mask = grf_mask_raw
        marker_observations = marker_obs_raw
        addb_poses = addb_poses_raw

    # Subject metadata
    age = subject.getAgeYears() if hasattr(subject, 'getAgeYears') else None

    metadata = {
        'height_m': subject.getHeightM(),
        'mass_kg': subject.getMassKg(),
        'sex': subject.getBiologicalSex(),
        'age': age,
        'num_frames': T,
        'num_frames_raw': T_raw,
        'num_joints': num_joints,
        'original_fps': original_fps,
        'fps': actual_fps,
        'grf_valid_mask': grf_valid_mask,
        'grf_n_valid': int(grf_valid_mask.sum()),
        'grf_n_missing': T - int(grf_valid_mask.sum()),
        'trial_idx': trial_idx,
        'trial_length_native': trial_length,
    }

    if load_markers:
        metadata['markers_map'] = markers_map
        metadata['marker_observations'] = marker_observations
        metadata['addb_poses'] = addb_poses
        metadata['addb_skel'] = addb_skel

    return joints, joint_names, metadata


def main():
    parser = argparse.ArgumentParser(description='Process single trial for HPCC')
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--skel_model_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--target_fps', type=float, default=100.0)
    parser.add_argument('--save_viz', action='store_true',
                        help='Save quick_viz PNG + OBJ (requires more memory/time)')
    args = parser.parse_args()

    # Read manifest row
    with open(args.manifest, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['task_id']) == args.task_id:
                break
        else:
            print(f"ERROR: task_id {args.task_id} not found in manifest")
            sys.exit(1)

    b3d_path = row['b3d_path']
    trial_idx = int(row['trial_idx'])
    subject_id = row['subject_id']
    category = row['category']

    print(f"{'='*60}")
    print(f"Task ID:    {args.task_id}")
    print(f"Subject:    {subject_id} ({category})")
    print(f"Trial:      {trial_idx}")
    print(f"B3D:        {b3d_path}")
    print(f"Device:     {args.device}")
    print(f"Target FPS: {args.target_fps}")
    print(f"{'='*60}")

    t_start = time.time()

    # Output directory
    trial_output_dir = os.path.join(
        args.output_dir, subject_id, f"trial_{trial_idx:04d}")
    os.makedirs(trial_output_dir, exist_ok=True)

    # Skip if already completed
    metrics_path = os.path.join(trial_output_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        print(f"SKIP: Already completed (metrics.json exists)")
        sys.exit(0)

    # Override SKEL model path if specified
    if args.skel_model_path:
        import addb2skel.config as _cfg
        _cfg.SKEL_MODEL_PATH = args.skel_model_path

    from addb2skel.pipeline import convert_addb_to_skel
    from addb2skel.config import OptimizationConfig

    # Load trial data (with markers)
    print(f"\nLoading trial {trial_idx}...")
    t_load = time.time()
    addb_joints, joint_names, metadata = load_b3d_trial(
        b3d_path, trial_idx, target_fps=args.target_fps, load_markers=True)
    load_time = time.time() - t_load
    print(f"  Loaded {metadata['num_frames']} frames "
          f"(from {metadata['num_frames_raw']} @ {metadata['original_fps']:.0f}Hz → "
          f"{metadata['fps']:.0f}Hz) in {load_time:.1f}s")

    # Child filtering
    age = metadata.get('age')
    if age is not None and 0 < age < 18:
        print(f"SKIP: Child subject (age={age})")
        metrics = {
            'task_id': args.task_id, 'subject_id': subject_id,
            'trial_idx': trial_idx, 'category': category,
            'status': 'skipped_child', 'age': age,
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        sys.exit(0)

    sex = metadata.get('sex', 'auto')

    # Config: Fix K defaults (2026-04-02) — all settings in config.py defaults
    config = OptimizationConfig(device=args.device)

    # Override paths for HPCC (bsm_markers.yaml lives next to this script)
    bsm_path = os.path.join(_SCRIPT_DIR, 'bsm_markers.yaml')
    if os.path.exists(bsm_path):
        config.bsm_markers_path = bsm_path

    # dt from trial metadata (for temporal smoothness normalization)
    config.dt = 1.0 / metadata['fps']

    # Run conversion
    print(f"\nRunning addb2skel pipeline (sex={sex})...")
    t_convert = time.time()
    try:
        result = convert_addb_to_skel(
            addb_joints,
            sex=sex,
            config=config,
            height_m=metadata['height_m'],
            mass_kg=metadata.get('mass_kg'),
            return_vertices=args.save_viz,
            verbose=True,
            markers_map=metadata.get('markers_map'),
            marker_observations=metadata.get('marker_observations'),
            addb_skel=metadata.get('addb_skel'),
            addb_poses=metadata.get('addb_poses'),
        )
        convert_time = time.time() - t_convert
        status = 'success'

        # Extract metrics
        em = result.evaluation_metrics or {}
        mpjpe = em.get('mpjpe_mm', result.mpjpe_mm)
        print(f"\n  MPJPE: {mpjpe:.1f} mm ({convert_time:.1f}s)")

    except Exception as e:
        convert_time = time.time() - t_convert
        status = f'error: {str(e)[:200]}'
        mpjpe = None
        result = None
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()

    total_time = time.time() - t_start

    # Save results
    if status == 'success':
        # NPZ
        npz_path = os.path.join(trial_output_dir, 'result.npz')
        _np = lambda x: x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)
        np.savez_compressed(
            npz_path,
            poses=_np(result.skel_poses),
            betas=_np(result.skel_betas),
            trans=_np(result.skel_trans),
            skel_joints=_np(result.skel_joints),
            gt_joints=addb_joints,
        )
        print(f"  Saved: {npz_path}")

        # Quick viz + OBJ
        if args.save_viz and result.skel_vertices is not None:
            try:
                from addb2skel.viz.quick_viz import visualize_result
                from addb2skel.skel_interface import create_skel_interface
                resolved_sex = result.sex if hasattr(result, 'sex') else 'male'
                skel_if = create_skel_interface(sex=resolved_sex)
                visualize_result(
                    result, addb_joints, trial_output_dir,
                    name=subject_id, skel_interface=skel_if)
                print(f"  Saved viz + OBJ to {trial_output_dir}")
            except Exception as e:
                print(f"  Warning: viz failed: {e}")

    # Marker stats
    mk_mpjpe = float('nan')
    mk_matched = 0
    mk_t1 = 0
    mk_t2 = 0
    if status == 'success' and result.marker_quality:
        mk_errors = list(result.marker_quality.values())
        mk_mpjpe = float(np.mean(mk_errors)) if mk_errors else float('nan')
        mk_matched = len(mk_errors)
    if status == 'success' and result.marker_tier_counts:
        mk_t1 = result.marker_tier_counts.get(1, 0)
        mk_t2 = result.marker_tier_counts.get(2, 0)

    # Full metrics JSON
    metrics = {
        'task_id': args.task_id,
        'subject_id': subject_id,
        'trial_idx': trial_idx,
        'category': category,
        'b3d_path': b3d_path,
        'status': status,
        'sex': sex,
        'age': age,
        # Joint metrics
        'mpjpe_mm': float(mpjpe) if mpjpe is not None else None,
        'w_mpjpe_mm': float(em.get('w_mpjpe_mm', float('nan'))) if status == 'success' else None,
        'pa_mpjpe_mm': float(em.get('pa_mpjpe_mm', float('nan'))) if status == 'success' else None,
        'accel_error': float(em.get('accel_error', float('nan'))) if status == 'success' else None,
        'vel_error': float(em.get('vel_error', float('nan'))) if status == 'success' else None,
        # Mesh metrics
        'foot_sliding_mm': float(em.get('foot_sliding_mm', float('nan'))) if status == 'success' else None,
        'ground_penetration_mm': float(em.get('ground_penetration_mm', float('nan'))) if status == 'success' else None,
        # Marker metrics
        'marker_mpjpe_mm': float(mk_mpjpe),
        'marker_matched': mk_matched,
        'marker_t1': mk_t1,
        'marker_t2': mk_t2,
        # Stability
        'pelvis_std_deg': float(np.std(np.degrees(
            _np(result.skel_poses)[:, 2]))) if status == 'success' else None,
        # Per-joint error
        'per_joint_error': {k: float(v) for k, v in result.per_joint_error.items()} if (
            status == 'success' and result.per_joint_error) else None,
        # Metadata
        'num_frames': metadata['num_frames'],
        'num_frames_raw': metadata['num_frames_raw'],
        'original_fps': metadata['original_fps'],
        'target_fps': args.target_fps,
        'height_m': metadata['height_m'],
        'mass_kg': metadata.get('mass_kg'),
        'grf_n_valid': metadata['grf_n_valid'],
        'grf_n_missing': metadata['grf_n_missing'],
        # Timing
        'timing': {
            'load_s': round(load_time, 1),
            'convert_s': round(convert_time, 1),
            'total_s': round(total_time, 1),
        },
        'device': args.device,
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved: {metrics_path}")
    print(f"\nDone in {total_time:.1f}s")


if __name__ == '__main__':
    main()
