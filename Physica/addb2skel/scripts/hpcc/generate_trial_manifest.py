#!/usr/bin/env python3
"""
Generate trial manifest for HPCC SLURM job array.

Scans all .b3d files, enumerates every trial, and outputs a CSV manifest
with one row per trial. This manifest is used by run_single_trial.py to
map SLURM_ARRAY_TASK_ID → (b3d_path, trial_index).

Usage:
    python generate_trial_manifest.py \
        --input_dir /mnt/scratch/kwonjoon/AddB/train \
        --output trial_manifest.csv

Output CSV columns:
    task_id, b3d_path, trial_idx, num_frames_native, native_hz, num_frames_100hz,
    subject_id, category, sex, height_m, mass_kg
"""

import argparse
import csv
import glob
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='Generate trial manifest for HPCC batch processing')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing .b3d files')
    parser.add_argument('--output', type=str, default='trial_manifest.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    try:
        import nimblephysics as nimble
    except ImportError:
        print("ERROR: nimblephysics required. pip install nimblephysics")
        sys.exit(1)

    b3d_files = sorted(glob.glob(os.path.join(args.input_dir, '**/*.b3d'), recursive=True))
    print(f"Found {len(b3d_files)} .b3d files")

    # Filter With_Arm only
    b3d_files = [f for f in b3d_files if 'With_Arm' in f]
    print(f"With_Arm only: {len(b3d_files)} .b3d files")

    rows = []
    task_id = 0
    corrupted = 0
    skipped_children = 0
    total_frames_native = 0
    total_frames_100hz = 0

    for i, b3d_path in enumerate(b3d_files):
        if (i + 1) % 100 == 0:
            print(f"  Scanning {i+1}/{len(b3d_files)}... ({task_id} trials so far)")

        try:
            subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
        except Exception as e:
            print(f"  SKIP (corrupted): {b3d_path} — {e}")
            corrupted += 1
            continue

        n_trials = subject.getNumTrials()
        sex = subject.getBiologicalSex()
        height_m = subject.getHeightM()
        mass_kg = subject.getMassKg()
        age = subject.getAgeYears() if hasattr(subject, 'getAgeYears') else None

        # Child filtering: skip subjects with age > 0 and age < 18
        if age is not None and 0 < age < 18:
            skipped_children += 1
            print(f"  SKIP (child, age={age}): {b3d_path}")
            continue

        # Extract subject_id and category from path
        subject_id = os.path.basename(b3d_path).replace('.b3d', '')
        category = 'With_Arm'

        for t in range(n_trials):
            nf = subject.getTrialLength(t)
            dt = subject.getTrialTimestep(t)
            hz = round(1.0 / dt) if dt > 0 else 100

            # Estimate frames after 100Hz resample
            if hz > 0:
                nf_100hz = int(nf * 100.0 / hz)
            else:
                nf_100hz = nf

            total_frames_native += nf
            total_frames_100hz += nf_100hz

            rows.append({
                'task_id': task_id,
                'b3d_path': b3d_path,
                'trial_idx': t,
                'num_frames_native': nf,
                'native_hz': hz,
                'num_frames_100hz': nf_100hz,
                'subject_id': subject_id,
                'category': category,
                'sex': sex,
                'age': age if age is not None else '',
                'height_m': f'{height_m:.4f}',
                'mass_kg': f'{mass_kg:.2f}',
            })
            task_id += 1

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"MANIFEST GENERATED: {args.output}")
    print(f"{'='*60}")
    print(f"Total b3d files:        {len(b3d_files)} (With_Arm only)")
    print(f"Corrupted (skipped):    {corrupted}")
    print(f"Children (skipped):     {skipped_children}")
    print(f"Total trials:           {task_id}")
    print(f"Total frames (native):  {total_frames_native:,}")
    print(f"Total frames (100Hz):   {total_frames_100hz:,}")
    print(f"Hours at 100Hz:         {total_frames_100hz / 100 / 3600:.1f}")
    print(f"\nMax SLURM array index:  {task_id - 1}")
    print(f"Suggested batch split (10k limit):")
    for batch in range(0, task_id, 10000):
        end = min(batch + 9999, task_id - 1)
        print(f"  sbatch --array={batch}-{end}%200 run_hpcc_batch.sh")


if __name__ == '__main__':
    main()
