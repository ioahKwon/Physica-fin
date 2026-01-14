#!/usr/bin/env python3
"""
Verify MPJPE statistics - double check all values
"""

import json
import numpy as np
from pathlib import Path

# 1. Load dataset info
dataset_info_path = "/egr/research-zijunlab/kwonjoon/03_Output/with_arm_dataset_info.json"
with open(dataset_info_path, 'r') as f:
    dataset_info = json.load(f)

print(f"Dataset info entries: {len(dataset_info)}")

# 2. Load MPJPE from comparison_metrics.json
result_dir = Path("/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/full_with_arm")
mpjpe_results = {}

for metrics_file in result_dir.rglob("comparison_metrics.json"):
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            if 'skel' in metrics and 'mpjpe_mm' in metrics['skel']:
                subject_dir = metrics_file.parent.name
                mpjpe_results[subject_dir] = metrics['skel']['mpjpe_mm']
    except:
        continue

print(f"MPJPE results loaded: {len(mpjpe_results)}")

# 3. Create lookup
info_lookup = {}
for entry in dataset_info:
    key = f"{entry['dataset']}_{entry['subject']}"
    info_lookup[key] = entry

# 4. Filter subjects
valid_mpjpe = []
excluded_reasons = {}

for subject_key, mpjpe in mpjpe_results.items():
    if subject_key not in info_lookup:
        excluded_reasons[subject_key] = 'no_metadata'
        continue

    entry = info_lookup[subject_key]

    # Check criteria
    age = entry.get('age', -1)
    sex = entry.get('sex', '').lower()
    height = entry.get('height_m', 0)
    mass = entry.get('mass_kg', 0)

    reason = None
    if age < 0:
        reason = 'age_negative'
    elif age == 0:
        reason = 'age_zero'
    elif age < 18:
        reason = 'age_child'
    elif sex not in ['male', 'female', 'm', 'f']:
        reason = 'sex_invalid'
    elif height < 1.4:
        reason = 'height_short'
    elif mass < 40:
        reason = 'mass_light'

    if reason:
        excluded_reasons[subject_key] = reason
    else:
        valid_mpjpe.append(mpjpe)

print(f"\n=== Filtering Results ===")
print(f"Valid subjects: {len(valid_mpjpe)}")
print(f"Excluded subjects: {len(excluded_reasons)}")

# Count by reason
from collections import Counter
reason_counts = Counter(excluded_reasons.values())
for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"  - {reason}: {count}")

# 5. Calculate statistics
mpjpe_array = np.array(valid_mpjpe)

print(f"\n=== MPJPE Statistics (N={len(mpjpe_array)}) ===")
print(f"Mean:   {np.mean(mpjpe_array):.2f} mm")
print(f"Median: {np.median(mpjpe_array):.2f} mm")
print(f"Std:    {np.std(mpjpe_array):.2f} mm")
print(f"Min:    {np.min(mpjpe_array):.2f} mm")
print(f"Max:    {np.max(mpjpe_array):.2f} mm")

print(f"\n=== Percentiles ===")
for p in [5, 10, 25, 50, 75, 90, 95]:
    val = np.percentile(mpjpe_array, p)
    print(f"  {p}th percentile: {val:.2f} mm")

# 6. Compare with saved summary
print(f"\n=== Comparing with saved summary ===")
with open("/egr/research-zijunlab/kwonjoon/skel_force_vis/output/filtered_analysis/filtered_summary.json", 'r') as f:
    saved = json.load(f)

saved_stats = saved['statistics']
print(f"{'Metric':<15} {'Calculated':<15} {'Saved':<15} {'Match'}")
print("-" * 55)

checks = [
    ('count', len(mpjpe_array), saved_stats['count']),
    ('mean', np.mean(mpjpe_array), saved_stats['mean']),
    ('median', np.median(mpjpe_array), saved_stats['median']),
    ('std', np.std(mpjpe_array), saved_stats['std']),
    ('min', np.min(mpjpe_array), saved_stats['min']),
    ('max', np.max(mpjpe_array), saved_stats['max']),
    ('p5', np.percentile(mpjpe_array, 5), saved_stats['percentile_5']),
    ('p10', np.percentile(mpjpe_array, 10), saved_stats['percentile_10']),
    ('p25', np.percentile(mpjpe_array, 25), saved_stats['percentile_25']),
    ('p50', np.percentile(mpjpe_array, 50), saved_stats['percentile_50']),
    ('p75', np.percentile(mpjpe_array, 75), saved_stats['percentile_75']),
    ('p90', np.percentile(mpjpe_array, 90), saved_stats['percentile_90']),
    ('p95', np.percentile(mpjpe_array, 95), saved_stats['percentile_95']),
]

all_match = True
for name, calc, saved_val in checks:
    match = abs(calc - saved_val) < 0.01
    all_match = all_match and match
    status = "✓" if match else "✗"
    print(f"{name:<15} {calc:<15.2f} {saved_val:<15.2f} {status}")

print(f"\n{'='*55}")
if all_match:
    print("✓ ALL VALUES VERIFIED - Statistics are correct!")
else:
    print("✗ MISMATCH DETECTED - Please check!")

# 7. Show some sample subjects to verify filtering
print(f"\n=== Sample Excluded Subjects ===")
for subject, reason in list(excluded_reasons.items())[:10]:
    if subject in info_lookup:
        entry = info_lookup[subject]
        print(f"  {subject}: {reason} (age={entry.get('age')}, sex={entry.get('sex')}, h={entry.get('height_m')}m)")
    else:
        print(f"  {subject}: {reason}")

# 8. Show worst performers
print(f"\n=== Worst 5 Performers (MPJPE > 50mm) ===")
subject_mpjpe = [(k, v) for k, v in mpjpe_results.items() if k not in excluded_reasons]
subject_mpjpe.sort(key=lambda x: -x[1])
for subj, mpjpe in subject_mpjpe[:5]:
    print(f"  {subj}: {mpjpe:.2f} mm")
