#!/usr/bin/env python3
"""
Collect and summarize results from HPCC batch run.

Scans all metrics.json files in the output directory and generates
a comprehensive summary CSV and statistics.

Usage:
    python collect_results.py \
        --input_dir /mnt/scratch/kwonjoon/output/addb2skel/ \
        --output results_summary.csv
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Collect HPCC batch results')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root output directory with subject/trial subdirs')
    parser.add_argument('--output', type=str, default='results_summary.csv',
                        help='Output summary CSV')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Original manifest to check completeness')
    args = parser.parse_args()

    # Find all metrics.json files
    metrics_files = sorted(glob.glob(
        os.path.join(args.input_dir, '**/metrics.json'), recursive=True
    ))
    print(f"Found {len(metrics_files)} metrics.json files")

    if not metrics_files:
        print("No results found!")
        sys.exit(1)

    # Load all metrics
    all_metrics = []
    for mf in metrics_files:
        with open(mf, 'r') as f:
            m = json.load(f)
        m['metrics_path'] = mf
        all_metrics.append(m)

    # Separate by status
    successful = [m for m in all_metrics if m['status'] == 'success']
    skipped_child = [m for m in all_metrics if m.get('status') == 'skipped_child']
    failed = [m for m in all_metrics if m['status'] not in ('success', 'skipped_child')]

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total trials processed: {len(all_metrics)}")
    print(f"Successful:             {len(successful)}")
    print(f"Skipped (child):        {len(skipped_child)}")
    print(f"Failed:                 {len(failed)}")

    if successful:
        mpjpes = [m['mpjpe_mm'] for m in successful if m['mpjpe_mm'] is not None]
        times = [m['timing']['total_s'] for m in successful]
        frames = [m['num_frames'] for m in successful]

        print(f"\n--- MPJPE Statistics ---")
        print(f"Mean:   {np.mean(mpjpes):.1f} mm")
        print(f"Median: {np.median(mpjpes):.1f} mm")
        print(f"Std:    {np.std(mpjpes):.1f} mm")
        print(f"Min:    {np.min(mpjpes):.1f} mm")
        print(f"Max:    {np.max(mpjpes):.1f} mm")
        print(f"P25:    {np.percentile(mpjpes, 25):.1f} mm")
        print(f"P75:    {np.percentile(mpjpes, 75):.1f} mm")
        print(f"P95:    {np.percentile(mpjpes, 95):.1f} mm")

        print(f"\n--- Timing ---")
        print(f"Mean time/trial:  {np.mean(times):.0f}s ({np.mean(times)/60:.1f}min)")
        print(f"Total GPU-hours:  {np.sum(times)/3600:.1f}h")
        print(f"Total frames:     {np.sum(frames):,}")

        # Per-category stats
        categories = set(m['category'] for m in successful)
        for cat in sorted(categories):
            cat_metrics = [m for m in successful if m['category'] == cat]
            cat_mpjpes = [m['mpjpe_mm'] for m in cat_metrics if m['mpjpe_mm'] is not None]
            if cat_mpjpes:
                print(f"\n--- {cat} ({len(cat_metrics)} trials) ---")
                print(f"  Mean MPJPE: {np.mean(cat_mpjpes):.1f} mm")
                print(f"  Median:     {np.median(cat_mpjpes):.1f} mm")

        # Marker matching stats
        mk_t1s = [m.get('marker_t1', 0) for m in successful]
        mk_t2s = [m.get('marker_t2', 0) for m in successful]
        mk_mpjpes = [m.get('marker_mpjpe_mm', float('nan')) for m in successful
                     if m.get('marker_mpjpe_mm') is not None and not np.isnan(m.get('marker_mpjpe_mm', float('nan')))]
        print(f"\n--- Marker Matching ---")
        print(f"  Mean T1 matched: {np.mean(mk_t1s):.1f}")
        print(f"  Mean T2 matched: {np.mean(mk_t2s):.1f}")
        if mk_mpjpes:
            print(f"  Mean Marker MPJPE: {np.mean(mk_mpjpes):.1f} mm")

        # Per-sex stats
        for s in ['male', 'female']:
            sex_metrics = [m for m in successful if m.get('sex') == s]
            if sex_metrics:
                sex_mpjpes = [m['mpjpe_mm'] for m in sex_metrics if m['mpjpe_mm'] is not None]
                if sex_mpjpes:
                    print(f"\n--- {s} ({len(sex_metrics)} trials) ---")
                    print(f"  Mean MPJPE: {np.mean(sex_mpjpes):.1f} mm")

        # Per-subject aggregate
        subject_mpjpes = {}
        for m in successful:
            sid = m['subject_id']
            if sid not in subject_mpjpes:
                subject_mpjpes[sid] = []
            if m['mpjpe_mm'] is not None:
                subject_mpjpes[sid].append(m['mpjpe_mm'])

        subject_means = {s: np.mean(v) for s, v in subject_mpjpes.items() if v}
        print(f"\n--- Per-Subject Aggregate ({len(subject_means)} subjects) ---")
        vals = list(subject_means.values())
        print(f"  Mean of subject means: {np.mean(vals):.1f} mm")
        print(f"  Worst 5 subjects:")
        for sid, v in sorted(subject_means.items(), key=lambda x: -x[1])[:5]:
            print(f"    {sid}: {v:.1f} mm")

    if failed:
        print(f"\n--- Failed Trials ({len(failed)}) ---")
        # Group by error type
        error_types = {}
        for m in failed:
            err = m['status'][:80]
            error_types[err] = error_types.get(err, 0) + 1
        for err, cnt in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  [{cnt}x] {err}")

    # Check completeness against manifest
    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest, 'r') as f:
            reader = csv.DictReader(f)
            manifest_ids = set(int(row['task_id']) for row in reader)
        completed_ids = set(m['task_id'] for m in all_metrics)
        missing = manifest_ids - completed_ids
        if missing:
            print(f"\n--- Missing Trials ({len(missing)}) ---")
            missing_sorted = sorted(missing)
            print(f"  Task IDs: {missing_sorted[:20]}{'...' if len(missing) > 20 else ''}")
        else:
            print(f"\n  All {len(manifest_ids)} trials completed!")

    # Write summary CSV
    fieldnames = [
        'task_id', 'subject_id', 'trial_idx', 'category', 'status',
        'sex', 'age', 'mpjpe_mm', 'pa_mpjpe_mm', 'accel_error',
        'marker_mpjpe_mm', 'marker_t1', 'marker_t2',
        'foot_sliding_mm', 'ground_penetration_mm', 'pelvis_std_deg',
        'num_frames', 'original_fps', 'height_m', 'mass_kg', 'time_s',
    ]
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            timing = m.get('timing', {})
            writer.writerow({
                'task_id': m.get('task_id'),
                'subject_id': m.get('subject_id'),
                'trial_idx': m.get('trial_idx'),
                'category': m.get('category'),
                'status': m.get('status'),
                'sex': m.get('sex'),
                'age': m.get('age'),
                'mpjpe_mm': m.get('mpjpe_mm'),
                'pa_mpjpe_mm': m.get('pa_mpjpe_mm'),
                'accel_error': m.get('accel_error'),
                'marker_mpjpe_mm': m.get('marker_mpjpe_mm'),
                'marker_t1': m.get('marker_t1'),
                'marker_t2': m.get('marker_t2'),
                'foot_sliding_mm': m.get('foot_sliding_mm'),
                'ground_penetration_mm': m.get('ground_penetration_mm'),
                'pelvis_std_deg': m.get('pelvis_std_deg'),
                'num_frames': m.get('num_frames'),
                'original_fps': m.get('original_fps'),
                'height_m': m.get('height_m'),
                'mass_kg': m.get('mass_kg'),
                'time_s': timing.get('total_s') if timing else None,
            })

    print(f"\nSummary CSV saved: {args.output}")


if __name__ == '__main__':
    main()
