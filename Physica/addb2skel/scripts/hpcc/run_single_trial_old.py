#!/usr/bin/env python3
"""
Process a single trial from AddBiomechanics .b3d file.

Designed for HPCC SLURM job array: each job runs this script with a
different --task_id that maps to a specific (b3d_path, trial_idx) via
the trial manifest CSV.

Usage:
    python run_single_trial.py \
        --manifest trial_manifest.csv \
        --task_id 0 \
        --output_dir /mnt/scratch/kwonjoon/output/addb2skel/ \
        --device cuda

Output per trial:
    {output_dir}/{subject_id}/trial_{trial_idx}/
        result.npz      — poses, betas, trans, skel_joints, gt_joints
        metrics.json     — MPJPE, per-joint errors, timing
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Add addb2skel package to path (flat package under Physica/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_ADDB2SKEL_DIR = _SCRIPT_DIR.parent.parent  # scripts/hpcc/ -> scripts/ -> addb2skel/
_PHYSICA_DIR = _ADDB2SKEL_DIR.parent         # addb2skel/ -> Physica/
sys.path.insert(0, str(_PHYSICA_DIR))

import numpy as np


def load_b3d_trial(b3d_path, trial_idx, target_fps=100.0):
    """
    Load a specific trial from a .b3d file with optional resampling.

    This extends the standard load_b3d() which only loads trial 0.
    """
    import nimblephysics as nimble
    from scipy.signal import resample_poly
    from fractions import Fraction

    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    skel = subject.readSkel(0, ignoreGeometry=True)

    # Frame rate for this specific trial
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

    # Read all frames of this trial
    trial_length = subject.getTrialLength(trial_idx)
    trial_frames = subject.readFrames(
        trial=trial_idx,
        startFrame=0,
        numFramesToRead=trial_length,
        stride=1,
    )
    T_raw = len(trial_frames)

    if T_raw == 0:
        raise ValueError(f"Trial {trial_idx} has 0 frames")

    # Get joint names from first frame
    pp_idx = 0
    pp = trial_frames[0].processingPasses[pp_idx]
    pos = np.asarray(pp.pos, dtype=np.float32)
    if pos.size == skel.getNumDofs():
        skel.setPositions(pos)

    world_joints = []
    for i in range(skel.getNumJoints()):
        joint = skel.getJoint(i)
        world = joint.getChildBodyNode().getWorldTransform().translation()
        joint_name = joint.getName()
        world_joints.append((joint_name, world))

    joint_centers_first = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
    num_joints = joint_centers_first.shape[0]

    joint_names = []
    for center in joint_centers_first:
        dists = [np.linalg.norm(center - w) for _, w in world_joints]
        best = int(np.argmin(dists))
        joint_names.append(world_joints[best][0])

    # Extract joint positions for all frames
    joints_raw = np.zeros((T_raw, num_joints, 3), dtype=np.float32)
    for t, frame in enumerate(trial_frames):
        pp = frame.processingPasses[pp_idx]
        joints_raw[t] = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)

    # GRF validity mask
    missing_grf_all = subject.getMissingGRF(trial_idx)
    notMissing = nimble.biomechanics.MissingGRFReason.notMissingGRF
    grf_mask_raw = np.zeros(T_raw, dtype=bool)
    for t in range(T_raw):
        if t < len(missing_grf_all):
            grf_mask_raw[t] = (missing_grf_all[t] == notMissing)

    # Resample to target_fps
    if needs_resample:
        joints = resample_poly(joints_raw, up, down, axis=0).astype(np.float32)
        T = joints.shape[0]
        # Nearest-neighbor for GRF mask
        src = np.round(np.arange(T) * (T_raw - 1) / max(T - 1, 1)).astype(int)
        src = np.clip(src, 0, T_raw - 1)
        grf_valid_mask = grf_mask_raw[src]
    else:
        joints = joints_raw
        T = T_raw
        grf_valid_mask = grf_mask_raw

    metadata = {
        'height_m': subject.getHeightM(),
        'mass_kg': subject.getMassKg(),
        'sex': subject.getBiologicalSex(),
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

    return joints, joint_names, metadata


def main():
    parser = argparse.ArgumentParser(description='Process single trial for HPCC')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to trial_manifest.csv')
    parser.add_argument('--task_id', type=int, required=True,
                        help='SLURM_ARRAY_TASK_ID')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root output directory')
    parser.add_argument('--skel_model_path', type=str, default=None,
                        help='Override SKEL model path')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--target_fps', type=float, default=100.0,
                        help='Resample to this frame rate (Hz)')
    parser.add_argument('--save_obj', action='store_true',
                        help='Save OBJ mesh (large, disabled by default)')
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
        args.output_dir, subject_id, f"trial_{trial_idx:04d}"
    )
    os.makedirs(trial_output_dir, exist_ok=True)

    # Skip if already completed
    metrics_path = os.path.join(trial_output_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        print(f"SKIP: Already completed (metrics.json exists)")
        sys.exit(0)

    # Override SKEL model path if specified (must patch before importing pipeline)
    if args.skel_model_path:
        import addb2skel.config as _cfg
        _cfg.SKEL_MODEL_PATH = args.skel_model_path

    # Import pipeline
    from addb2skel.pipeline import convert_addb_to_skel, save_conversion_result
    from addb2skel.config import OptimizationConfig

    # Load trial data
    print(f"\nLoading trial {trial_idx}...")
    t_load = time.time()
    addb_joints, joint_names, metadata = load_b3d_trial(
        b3d_path, trial_idx, target_fps=args.target_fps
    )
    load_time = time.time() - t_load
    print(f"  Loaded {metadata['num_frames']} frames "
          f"(from {metadata['num_frames_raw']} @ {metadata['original_fps']:.0f}Hz → "
          f"{metadata['fps']:.0f}Hz) in {load_time:.1f}s")

    gender = 'male' if metadata['sex'] == 'male' else 'female'

    # Create config — combo_g+temporal settings
    # Source: run_combo_g_temporal_all.py (20260310_hyperparam_tuning)
    config = OptimizationConfig(device=args.device)
    if args.skel_model_path:
        config.skel_model_path = args.skel_model_path

    # --- BASE overrides ---
    config.weight_width = 0.0
    config.use_marker_loss = True
    config.tier_weight_factors = {1: 2.0, 2: 0.0, 3: 0.0}

    # --- combo_g overrides ---
    config.weight_shoulder = 0.0
    config.weight_scapula_reg = 0.0
    config.weight_bone_dir = 0.0
    config.weight_bone_len = 0.3
    config.weight_spine_reg = 0.02
    config.foot_xyz_weights = [10.0, 10.0, 10.0]
    config.weight_foot_height = 50.0
    config.foot_finetune_use_joint_weights = True
    config.approximate_foot_weight_factor = 0.0
    config.weight_humerus_align = 0.0
    config.weight_humerus_reg = 0.0
    config.use_head_markers = True

    # --- temporal overrides (combo_g → combo_g+temporal) ---
    config.weight_acceleration = 10.0
    config.weight_temporal = 10.0

    # --- Joint weights: tibia 50.0 (knee emphasis) ---
    from addb2skel.config import SKEL_JOINT_WEIGHTS
    config.joint_weights_override = {**SKEL_JOINT_WEIGHTS, 'tibia_r': 50.0, 'tibia_l': 50.0}

    # --- dt from trial metadata ---
    config.dt = 1.0 / metadata['fps']

    # Run conversion
    print(f"\nRunning addb2skel pipeline...")
    t_convert = time.time()
    try:
        result = convert_addb_to_skel(
            addb_joints,
            gender=gender,
            config=config,
            height_m=metadata['height_m'],
            return_vertices=False,  # Save memory on HPCC
            verbose=True,
        )
        convert_time = time.time() - t_convert
        status = 'success'
        mpjpe = result.mpjpe_mm
        print(f"\n  MPJPE: {mpjpe:.1f} mm ({convert_time:.1f}s)")
    except Exception as e:
        convert_time = time.time() - t_convert
        status = f'error: {str(e)[:200]}'
        mpjpe = None
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()

    total_time = time.time() - t_start

    # Save results
    if status == 'success':
        # Save NPZ
        npz_path = os.path.join(trial_output_dir, 'result.npz')
        np.savez_compressed(
            npz_path,
            poses=result.skel_poses.cpu().numpy() if hasattr(result.skel_poses, 'cpu') else result.skel_poses,
            betas=result.skel_betas.cpu().numpy() if hasattr(result.skel_betas, 'cpu') else result.skel_betas,
            trans=result.skel_trans.cpu().numpy() if hasattr(result.skel_trans, 'cpu') else result.skel_trans,
            skel_joints=result.skel_joints.cpu().numpy() if hasattr(result.skel_joints, 'cpu') else result.skel_joints,
            gt_joints=addb_joints,
        )
        print(f"  Saved: {npz_path}")

        # Save OBJ if requested
        if args.save_obj and result.skel_vertices is not None:
            save_conversion_result(result, trial_output_dir, save_obj=True)

    # Save metrics JSON (always, even on error)
    metrics = {
        'task_id': args.task_id,
        'subject_id': subject_id,
        'trial_idx': trial_idx,
        'category': category,
        'b3d_path': b3d_path,
        'status': status,
        'mpjpe_mm': mpjpe,
        'per_joint_error': result.per_joint_error if status == 'success' else None,
        'num_frames': metadata['num_frames'],
        'num_frames_raw': metadata['num_frames_raw'],
        'original_fps': metadata['original_fps'],
        'target_fps': args.target_fps,
        'gender': gender,
        'height_m': metadata['height_m'],
        'mass_kg': metadata['mass_kg'],
        'grf_n_valid': metadata['grf_n_valid'],
        'grf_n_missing': metadata['grf_n_missing'],
        'timing': {
            'load_s': round(load_time, 1),
            'convert_s': round(convert_time, 1),
            'total_s': round(total_time, 1),
        },
        'device': args.device,
        'evaluation_metrics': result.evaluation_metrics if status == 'success' else None,
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved: {metrics_path}")

    print(f"\nDone in {total_time:.1f}s")


if __name__ == '__main__':
    main()
