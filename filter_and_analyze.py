#!/usr/bin/env python3
"""
Filter abnormal subjects and recalculate MPJPE statistics.

Filtering criteria:
1. Age < 18 (children)
2. Age = 0 or Age < 0 (invalid data)
3. Sex not in ['male', 'female', 'm', 'f']
4. Height < 1.4m (abnormally short - likely children or data error)
5. Mass < 40kg (abnormally light - likely children or data error)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats


def load_dataset_info(path):
    """Load dataset info JSON"""
    with open(path, 'r') as f:
        return json.load(f)


def load_mpjpe_results(result_dir):
    """Load MPJPE from comparison_metrics.json files"""
    result_dir = Path(result_dir)
    results = {}

    for metrics_file in result_dir.rglob("comparison_metrics.json"):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                if 'skel' in metrics and 'mpjpe_mm' in metrics['skel']:
                    # Get subject name from parent directory
                    subject_dir = metrics_file.parent.name
                    results[subject_dir] = {
                        'mpjpe': metrics['skel']['mpjpe_mm'],
                        'frames': metrics.get('num_frames', 0)
                    }
        except:
            continue

    return results


def filter_subjects(dataset_info, mpjpe_results):
    """
    Filter subjects based on criteria.

    Returns:
        valid_subjects: list of valid subject entries
        excluded: dict of exclusion reasons with counts
        excluded_subjects: list of excluded subject names with reasons
    """
    valid_subjects = []
    excluded = defaultdict(int)
    excluded_subjects = []

    # Create lookup by subject name
    info_lookup = {}
    for entry in dataset_info:
        key = f"{entry['dataset']}_{entry['subject']}"
        info_lookup[key] = entry

    for subject_key, mpjpe_data in mpjpe_results.items():
        if subject_key not in info_lookup:
            excluded['no_metadata'] += 1
            excluded_subjects.append((subject_key, 'no_metadata', None))
            continue

        entry = info_lookup[subject_key]

        # Check exclusion criteria
        reason = None

        # 1. Age checks
        age = entry.get('age', -1)
        if age < 0:
            reason = 'age_negative'
        elif age == 0:
            reason = 'age_zero'
        elif age < 18:
            reason = 'age_child'

        # 2. Sex check
        if reason is None:
            sex = entry.get('sex', '').lower()
            if sex not in ['male', 'female', 'm', 'f']:
                reason = 'sex_invalid'

        # 3. Height check (< 1.4m)
        if reason is None:
            height = entry.get('height_m', 0)
            if height < 1.4:
                reason = 'height_short'

        # 4. Mass check (< 40kg)
        if reason is None:
            mass = entry.get('mass_kg', 0)
            if mass < 40:
                reason = 'mass_light'

        if reason:
            excluded[reason] += 1
            excluded_subjects.append((subject_key, reason, entry))
        else:
            valid_entry = entry.copy()
            valid_entry['mpjpe'] = mpjpe_data['mpjpe']
            valid_entry['subject_key'] = subject_key
            valid_subjects.append(valid_entry)

    return valid_subjects, dict(excluded), excluded_subjects


def compute_statistics(subjects):
    """Compute MPJPE statistics for valid subjects"""
    mpjpe_values = np.array([s['mpjpe'] for s in subjects])

    return {
        'count': len(mpjpe_values),
        'mean': float(np.mean(mpjpe_values)),
        'median': float(np.median(mpjpe_values)),
        'std': float(np.std(mpjpe_values)),
        'min': float(np.min(mpjpe_values)),
        'max': float(np.max(mpjpe_values)),
        'percentile_5': float(np.percentile(mpjpe_values, 5)),
        'percentile_10': float(np.percentile(mpjpe_values, 10)),
        'percentile_25': float(np.percentile(mpjpe_values, 25)),
        'percentile_50': float(np.percentile(mpjpe_values, 50)),
        'percentile_70': float(np.percentile(mpjpe_values, 70)),
        'percentile_75': float(np.percentile(mpjpe_values, 75)),
        'percentile_90': float(np.percentile(mpjpe_values, 90)),
        'percentile_95': float(np.percentile(mpjpe_values, 95)),
    }


def get_poor_performers(subjects, n=15):
    """Get worst n performers by MPJPE"""
    sorted_subjects = sorted(subjects, key=lambda x: x['mpjpe'], reverse=True)
    return sorted_subjects[:n]


def create_mpjpe_distribution_plot(subjects, mpjpe_stats, output_path):
    """Create MPJPE normal distribution plot with percentile thresholds"""
    from scipy.stats import norm

    mpjpe_values = np.array([s['mpjpe'] for s in subjects])

    fig, ax = plt.subplots(figsize=(16, 10))

    mu = mpjpe_stats['mean']
    sigma = mpjpe_stats['std']

    # Create normal distribution curve
    x = np.linspace(0, 80, 1000)
    y = norm.pdf(x, mu, sigma)

    ax.plot(x, y, 'b-', linewidth=3, label='Normal Distribution')
    ax.fill_between(x, y, alpha=0.3)

    # Mean line
    ax.axvline(mu, color='red', linestyle='-', linewidth=2.5, label=f'Mean = {mu:.1f} mm')

    # Percentile thresholds - stagger y positions to avoid overlap
    percentiles = [
        (5, 'Top 5%', 'darkgreen', mpjpe_stats['percentile_5'], -0.006),
        (10, 'Top 10%', 'forestgreen', mpjpe_stats['percentile_10'], -0.003),
        (50, 'Top 50%', 'darkorange', mpjpe_stats['percentile_50'], 0.005),
        (90, 'Top 90%', 'crimson', mpjpe_stats['percentile_90'], 0.003),
    ]

    for pct, label, color, value, y_offset in percentiles:
        ax.axvline(value, color=color, linestyle='--', linewidth=2, alpha=0.8)
        # Add point on curve
        y_val = norm.pdf(value, mu, sigma)
        ax.plot(value, y_val, 'o', color=color, markersize=10, zorder=5)

        # Z-score
        z = (value - mu) / sigma

        # Add label with offset to avoid overlap
        ax.annotate(f'{label}: {value:.1f}mm',
                   xy=(value, y_val),
                   xytext=(value + 2, y_val + y_offset),
                   ha='left', fontsize=12, fontweight='bold',
                   color=color,
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Add text box with percentile thresholds
    textstr = f'Distribution: N(μ={mu:.1f}, σ={sigma:.1f})\n'
    textstr += f'─────────────────────\n'
    textstr += f'Top 5%:   < {mpjpe_stats["percentile_5"]:.1f} mm\n'
    textstr += f'Top 10%:  < {mpjpe_stats["percentile_10"]:.1f} mm\n'
    textstr += f'Top 25%:  < {mpjpe_stats["percentile_25"]:.1f} mm\n'
    textstr += f'Top 50%:  < {mpjpe_stats["percentile_50"]:.1f} mm\n'
    textstr += f'Top 75%:  < {mpjpe_stats["percentile_75"]:.1f} mm\n'
    textstr += f'Top 90%:  < {mpjpe_stats["percentile_90"]:.1f} mm\n'
    textstr += f'Top 95%:  < {mpjpe_stats["percentile_95"]:.1f} mm'

    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

    ax.set_xlabel('MPJPE (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax.set_title(f'MPJPE Distribution (After Filtering, N={len(subjects)})',
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 80)
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def create_poor_performers_table(poor_performers, stats, output_path):
    """Create LaTeX table for poor performers"""

    mu = stats['mean']
    sigma = stats['std']

    latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage[margin=1in]{geometry}

\definecolor{lightred}{rgb}{1.0, 0.9, 0.9}

\begin{document}

\section*{Poor Performers Analysis (MPJPE > 50mm) - After Filtering}

\begin{table}[h]
\centering
\caption{Subjects with MPJPE > 50mm (Poor Performance) - Filtered Dataset}
\begin{tabular}{clcccc}
\toprule
\textbf{Rank} & \textbf{Subject} & \textbf{MPJPE} & \textbf{Height} & \textbf{Mass} & \textbf{Z-Score} \\
\midrule
"""

    for i, subj in enumerate(poor_performers):
        mpjpe = subj['mpjpe']
        height = subj.get('height_m', 0)
        mass = subj.get('mass_kg', 0)
        z_score = (mpjpe - mu) / sigma

        # Highlight if z-score > 3
        if z_score > 3:
            latex += r"\rowcolor{lightred}" + "\n"

        # Bold if height < 1.5m or mass < 50kg (potential data issues)
        height_str = f"{height:.2f} m"
        mass_str = f"{mass:.0f} kg"
        if height < 1.5:
            height_str = r"\textbf{" + height_str + "}"
        if mass < 50:
            mass_str = r"\textbf{" + mass_str + "}"

        subject_name = subj['subject_key'].replace('_', r'\_')
        latex += f"{i+1} & {subject_name} & {mpjpe:.2f} mm & {height_str} & {mass_str} & +{z_score:.2f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}

\textbf{Notes:}
\begin{itemize}
\item Highlighted rows: Z-score > 3 (extreme outliers)
\item Bold height/mass: Potentially unusual values
\item Dataset filtered: Excluded children (age<18), invalid sex/age, abnormal height (<1.4m) or mass (<40kg)
\end{itemize}

\end{document}
"""

    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {output_path}")


def main():
    # Paths
    dataset_info_path = "/egr/research-zijunlab/kwonjoon/03_Output/with_arm_dataset_info.json"
    result_dir = "/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/full_with_arm"
    output_dir = Path("/egr/research-zijunlab/kwonjoon/skel_force_vis/output/filtered_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MPJPE Analysis with Filtering")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    dataset_info = load_dataset_info(dataset_info_path)
    mpjpe_results = load_mpjpe_results(result_dir)

    print(f"  Dataset info entries: {len(dataset_info)}")
    print(f"  MPJPE results: {len(mpjpe_results)}")

    # Filter subjects
    print("\nFiltering subjects...")
    valid_subjects, excluded, excluded_subjects = filter_subjects(dataset_info, mpjpe_results)

    print(f"\n  Valid subjects: {len(valid_subjects)}")
    print(f"  Excluded subjects: {sum(excluded.values())}")
    print("\n  Exclusion breakdown:")
    for reason, count in sorted(excluded.items(), key=lambda x: -x[1]):
        print(f"    - {reason}: {count}")

    # Compute statistics
    print("\n" + "=" * 60)
    print("Statistics (After Filtering)")
    print("=" * 60)

    mpjpe_stats = compute_statistics(valid_subjects)

    print(f"\n  Total subjects: {mpjpe_stats['count']}")
    print(f"  Mean MPJPE: {mpjpe_stats['mean']:.2f} mm")
    print(f"  Median MPJPE: {mpjpe_stats['median']:.2f} mm")
    print(f"  Std Dev: {mpjpe_stats['std']:.2f} mm")
    print(f"  Min: {mpjpe_stats['min']:.2f} mm")
    print(f"  Max: {mpjpe_stats['max']:.2f} mm")
    print(f"\n  Percentiles:")
    print(f"    5th:  {mpjpe_stats['percentile_5']:.2f} mm")
    print(f"    10th: {mpjpe_stats['percentile_10']:.2f} mm")
    print(f"    25th: {mpjpe_stats['percentile_25']:.2f} mm")
    print(f"    50th: {mpjpe_stats['percentile_50']:.2f} mm")
    print(f"    75th: {mpjpe_stats['percentile_75']:.2f} mm")
    print(f"    90th: {mpjpe_stats['percentile_90']:.2f} mm")
    print(f"    95th: {mpjpe_stats['percentile_95']:.2f} mm")

    # Get poor performers (MPJPE > 50mm)
    print("\n" + "=" * 60)
    print("Poor Performers (MPJPE > 50mm)")
    print("=" * 60)

    poor_performers = [s for s in valid_subjects if s['mpjpe'] > 50]
    poor_performers = sorted(poor_performers, key=lambda x: x['mpjpe'], reverse=True)

    print(f"\n  Subjects with MPJPE > 50mm: {len(poor_performers)}")

    if len(poor_performers) > 0:
        print("\n  Top 15 worst performers:")
        for i, subj in enumerate(poor_performers[:15]):
            z = (subj['mpjpe'] - mpjpe_stats['mean']) / mpjpe_stats['std']
            print(f"    {i+1}. {subj['subject_key']}: {subj['mpjpe']:.2f}mm (z={z:+.2f})")

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    # Save summary JSON
    summary = {
        'filtering': {
            'total_before': len(mpjpe_results),
            'valid_after': len(valid_subjects),
            'excluded': excluded
        },
        'statistics': mpjpe_stats,
        'poor_performers_count': len(poor_performers)
    }

    with open(output_dir / "filtered_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {output_dir / 'filtered_summary.json'}")

    # Create plots
    create_mpjpe_distribution_plot(valid_subjects, mpjpe_stats, output_dir / "mpjpe_distribution_filtered.png")

    # Create poor performers table
    if len(poor_performers) > 0:
        create_poor_performers_table(poor_performers[:15], mpjpe_stats, output_dir / "poor_performers_filtered.tex")

    # Save excluded subjects list
    with open(output_dir / "excluded_subjects.txt", 'w') as f:
        f.write("Excluded Subjects\n")
        f.write("=" * 60 + "\n\n")
        for reason in sorted(set(e[1] for e in excluded_subjects)):
            f.write(f"\n{reason}:\n")
            for subj, r, entry in excluded_subjects:
                if r == reason:
                    if entry:
                        f.write(f"  - {subj}: age={entry.get('age', 'N/A')}, sex={entry.get('sex', 'N/A')}, height={entry.get('height_m', 'N/A')}m, mass={entry.get('mass_kg', 'N/A')}kg\n")
                    else:
                        f.write(f"  - {subj}\n")
    print(f"  Saved: {output_dir / 'excluded_subjects.txt'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
