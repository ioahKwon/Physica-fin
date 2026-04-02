#!/usr/bin/env python3
"""
Real-time monitoring of HPCC addb2skel batch jobs.

Usage (from domogpu):
    ssh hpcc "ssh dev-amd24-h200 'ml purge && ml Miniforge3 && conda activate physpt && \
        cd /mnt/research/domolab/kwonjoon/Code/Physica/addb2skel/scripts/hpcc && \
        python monitor_jobs.py --interval 0'"
"""

import argparse
import json
import os
import glob
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

TOTAL_WITH_ARM = 12154  # From with_arm_manifest.csv (Fix K, 2026-04-02)


def collect_metrics(output_dir):
    """Scan all metrics.json files and aggregate results."""
    metrics_files = glob.glob(os.path.join(output_dir, '*/trial_*/metrics.json'))

    results = {
        'success': [],
        'skipped_child': [],
        'error_other': [],
        'total_files': 0,
        'first_time': None,
        'last_time': None,
    }

    for mf in metrics_files:
        results['total_files'] += 1
        try:
            with open(mf) as f:
                m = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Track file modification times for ETA
        mtime = os.path.getmtime(mf)
        if results['first_time'] is None or mtime < results['first_time']:
            results['first_time'] = mtime
        if results['last_time'] is None or mtime > results['last_time']:
            results['last_time'] = mtime

        status = m.get('status', '')
        if status == 'success':
            results['success'].append(m)
        elif status == 'skipped_child':
            results['skipped_child'].append(m)
        else:
            results['error_other'].append(m)

    return results


def progress_bar(current, total, width=40):
    """Create a text progress bar."""
    if total == 0:
        return '[' + ' ' * width + ']'
    pct = current / total
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    return f'[{bar}] {pct*100:5.1f}%'


def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m"


def print_summary(results, clear=True):
    """Print formatted summary table."""
    if clear:
        os.system('clear' if os.name != 'nt' else 'cls')

    n_success = len(results['success'])
    n_skip = len(results['skipped_child'])
    n_err = len(results['error_other'])
    n_total = results['total_files']
    now = time.time()

    print("=" * 80)
    print(f"  addb2skel HPCC Batch Monitor (Fix K) — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Progress bar
    print(f"  Progress:  {progress_bar(n_total, TOTAL_WITH_ARM)}")
    print(f"             {n_success:,} success / {n_skip:,} child-skip / {n_err:,} error  ({n_total:,} / {TOTAL_WITH_ARM:,})")
    print()

    # ETA calculation
    if n_success > 0 and results['first_time'] and results['last_time']:
        elapsed = results['last_time'] - results['first_time']
        if elapsed > 0 and n_success > 1:
            rate = n_success / elapsed  # trials per second
            remaining = TOTAL_WITH_ARM - n_success
            eta_seconds = remaining / rate
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)

            print(f"  Elapsed:       {format_duration(elapsed)}")
            print(f"  Rate:          {rate*60:.1f} trials/min  ({1/rate:.1f}s per trial)")
            print(f"  Remaining:     {remaining:,} trials")
            print(f"  ETA:           {format_duration(eta_seconds)}  (finish ~{eta_time.strftime('%H:%M')})")
        elif n_success == 1:
            t = results['success'][0].get('timing', {}).get('convert_s', 230)
            eta_seconds = (TOTAL_WITH_ARM - 1) * t / 200  # assume 200 parallel
            print(f"  ETA (est):     ~{format_duration(eta_seconds)} (assuming 200 parallel jobs)")
    print()

    # Summary counts
    total_frames = sum(m.get('num_frames', 0) for m in results['success'])
    total_frames_expected = int(TOTAL_WITH_ARM * (total_frames / n_success)) if n_success > 0 else 0
    remaining_trials = TOTAL_WITH_ARM - n_total
    remaining_frames = total_frames_expected - total_frames if total_frames_expected > 0 else 0

    print(f"  Trials:    {n_success:>6,} success / {n_skip:>4,} child-skip / {n_err:>4,} error  =  {n_total:,} done, {remaining_trials:,} remaining")
    print(f"  Frames:    {total_frames:>12,} processed  /  ~{total_frames_expected:>12,} total  ({remaining_frames:,} remaining)")
    print()

    if n_success == 0:
        print("  No successful trials yet. Waiting for results...")
        print()
        if n_err > 0:
            print("  Recent errors:")
            for m in results['error_other'][-5:]:
                print(f"    {m.get('subject_id', '?')}/trial_{m.get('trial_idx', '?')}: "
                      f"{m.get('status', '?')[:80]}")
        return

    # Aggregate metrics (current format: flat keys in metrics.json)
    mpjpe_vals = [m['mpjpe_mm'] for m in results['success'] if m.get('mpjpe_mm') is not None]

    eval_keys = [
        ('mpjpe_mm', 'MPJPE', 'mm'),
        ('pa_mpjpe_mm', 'PA-MPJPE', 'mm'),
        ('accel_error', 'Accel Error', 'mm/f²'),
        ('vel_error', 'Vel Error', 'mm/f'),
        ('foot_sliding_mm', 'Foot Sliding', 'mm'),
        ('ground_penetration_mm', 'Ground Penetration', 'mm'),
        ('marker_mpjpe_mm', 'Marker MPJPE', 'mm'),
    ]
    eval_data = defaultdict(list)
    for m in results['success']:
        for key, label, unit in eval_keys:
            val = m.get(key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                eval_data[label].append(val)

    # Per-joint errors
    joint_errors = defaultdict(list)
    for m in results['success']:
        pje = m.get('per_joint_error', {})
        if pje:
            for jname, err in pje.items():
                if err is not None:
                    joint_errors[jname].append(err)

    # Timing
    times = [m.get('timing', {}).get('convert_s', 0) for m in results['success'] if m.get('timing')]
    total_frames = sum(m.get('num_frames', 0) for m in results['success'])

    # Print metrics table
    print("-" * 80)
    print(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Unit':>10}")
    print("-" * 80)

    for key, label, unit in eval_keys:
        if label in eval_data and len(eval_data[label]) > 0:
            a = np.array(eval_data[label])
            print(f"  {label:<30} {a.mean():>10.2f} {a.std():>10.2f} {a.min():>10.2f} {a.max():>10.2f} {unit:>10}")

    # Marker matching stats
    mk_t1 = [m.get('marker_t1', 0) for m in results['success']]
    mk_t2 = [m.get('marker_t2', 0) for m in results['success']]
    if mk_t1:
        print(f"  {'Markers T1 (mean)':<30} {np.mean(mk_t1):>10.1f}")
        print(f"  {'Markers T2 (mean)':<30} {np.mean(mk_t2):>10.1f}")

    print("-" * 80)
    print()

    # Timing
    if times:
        t = np.array(times)
        print(f"  Timing: {t.mean():.0f}s avg, {t.min():.0f}s min, {t.max():.0f}s max per trial")
        print(f"  Total frames processed: {total_frames:,}")
        fps = total_frames / t.sum() if t.sum() > 0 else 0
        print(f"  Throughput: {fps:.1f} frames/sec (across all trials)")
    print()

    # Per-joint error table
    if joint_errors:
        print("-" * 80)
        print(f"  {'Joint':<25} {'Mean (mm)':>10} {'Std':>10} {'Max':>10}")
        print("-" * 80)
        sorted_joints = sorted(joint_errors.items(), key=lambda x: np.mean(x[1]), reverse=True)
        for jname, errs in sorted_joints:
            a = np.array(errs)
            print(f"  {jname:<25} {a.mean():>10.2f} {a.std():>10.2f} {a.max():>10.2f}")
        print("-" * 80)
        print()

    # Per-category breakdown
    cat_mpjpe = defaultdict(list)
    for m in results['success']:
        cat = m.get('category', 'unknown')
        if m.get('mpjpe_mm') is not None:
            cat_mpjpe[cat].append(m['mpjpe_mm'])

    if len(cat_mpjpe) > 1:
        print(f"  {'Category':<25} {'N':>6} {'Mean MPJPE':>12} {'Std':>10}")
        print("-" * 60)
        for cat, vals in sorted(cat_mpjpe.items()):
            a = np.array(vals)
            print(f"  {cat:<25} {len(vals):>6} {a.mean():>12.2f} {a.std():>10.2f}")
        print()

    # Per-sex breakdown
    sex_mpjpe = defaultdict(list)
    for m in results['success']:
        s = m.get('sex', 'unknown')
        if m.get('mpjpe_mm') is not None:
            sex_mpjpe[s].append(m['mpjpe_mm'])

    if len(sex_mpjpe) > 1:
        print(f"  {'Sex':<25} {'N':>6} {'Mean MPJPE':>12} {'Std':>10}")
        print("-" * 60)
        for s, vals in sorted(sex_mpjpe.items()):
            a = np.array(vals)
            print(f"  {s:<25} {len(vals):>6} {a.mean():>12.2f} {a.std():>10.2f}")
        print()

    # Recent errors
    if n_err > 0:
        print(f"  Recent errors ({n_err} total):")
        for m in results['error_other'][-5:]:
            print(f"    {m.get('subject_id', '?')}/trial_{m.get('trial_idx', '?')}: "
                  f"{m.get('status', '?')[:80]}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Monitor HPCC addb2skel batch jobs')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/scratch/kwonjoon/output/addb2skel',
                        help='Output directory with results')
    parser.add_argument('--interval', type=int, default=30,
                        help='Refresh interval in seconds (0 = run once)')
    args = parser.parse_args()

    if args.interval == 0:
        results = collect_metrics(args.output_dir)
        print_summary(results, clear=False)
    else:
        print(f"Monitoring {args.output_dir} every {args.interval}s... (Ctrl+C to stop)")
        try:
            while True:
                results = collect_metrics(args.output_dir)
                print_summary(results)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == '__main__':
    main()
