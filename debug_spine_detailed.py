#!/usr/bin/env python3
"""
Detailed spine gap analysis - check all regions.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
from models.skel_model import SKELModelWrapper


def detailed_gap_analysis(verts, dominant_joints, label=""):
    """Detailed spine gap analysis."""
    x_center = (verts[:, 0].min() + verts[:, 0].max()) / 2
    z_center = (verts[:, 2].min() + verts[:, 2].max()) / 2

    print(f"\n=== {label} ===")
    print(f"Mesh center: X={x_center:.4f}, Z={z_center:.4f}")

    # Different X radii
    for x_radius in [0.03, 0.05, 0.08, 0.10]:
        spine_mask = np.abs(verts[:, 0] - x_center) < x_radius

        pelvis = spine_mask & (dominant_joints == 0)
        lumbar = spine_mask & (dominant_joints == 11)

        if pelvis.sum() == 0 or lumbar.sum() == 0:
            continue

        p_max = verts[pelvis, 1].max()
        l_min = verts[lumbar, 1].min()
        gap = l_min - p_max

        print(f"  X_radius={x_radius:.2f}: pelvis_verts={pelvis.sum()}, lumbar_verts={lumbar.sum()}, gap={gap*100:.2f}cm")

        # Check Z regions
        anterior = spine_mask & (verts[:, 2] < z_center)
        posterior = spine_mask & (verts[:, 2] > z_center)

        for name, region in [("  Anterior", anterior), ("  Posterior", posterior)]:
            p_r = region & (dominant_joints == 0)
            l_r = region & (dominant_joints == 11)
            if p_r.sum() > 0 and l_r.sum() > 0:
                p_max_r = verts[p_r, 1].max()
                l_min_r = verts[l_r, 1].min()
                print(f"    {name}: gap={(l_min_r - p_max_r)*100:.2f}cm")

    # Check which joints are near the gap
    print(f"\n  Joints near spine (|X - center| < 0.05):")
    spine_narrow = np.abs(verts[:, 0] - x_center) < 0.05
    unique_joints = np.unique(dominant_joints[spine_narrow])
    for j in unique_joints:
        j_mask = spine_narrow & (dominant_joints == j)
        y_range = (verts[j_mask, 1].min(), verts[j_mask, 1].max())
        print(f"    Joint {j}: Y range [{y_range[0]:.4f}, {y_range[1]:.4f}]")


def main():
    device = torch.device('cpu')
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device
    )

    skel_weights = skel.model.skel_weights.to_dense().cpu().numpy()
    dominant_joints = np.argmax(skel_weights, axis=1)

    # Load actual pose data
    data = np.load("/egr/research-zijunlab/kwonjoon/skel_force_vis/output/head8_thorax0/skel_params_fixed.npz")
    poses_data = data['poses']
    betas_data = data['betas']
    trans_data = data['trans']

    betas_t = torch.from_numpy(betas_data).float().unsqueeze(0)

    # T-pose
    print("\n" + "="*60)
    print("T-POSE ANALYSIS")
    print("="*60)
    poses_zero = torch.zeros(1, 46).float()
    trans_zero = torch.zeros(1, 3).float()

    with torch.no_grad():
        output = skel.model(betas=betas_t, poses=poses_zero, trans=trans_zero)
        verts_tpose = output.skel_verts[0].cpu().numpy()

    detailed_gap_analysis(verts_tpose, dominant_joints, "T-pose")

    # Frame 0
    print("\n" + "="*60)
    print("FRAME 0 POSED ANALYSIS")
    print("="*60)
    poses_t = torch.from_numpy(poses_data[0:1]).float()
    trans_t = torch.from_numpy(trans_data[0:1]).float()

    with torch.no_grad():
        output = skel.model(betas=betas_t, poses=poses_t, trans=trans_t)
        verts_posed = output.skel_verts[0].cpu().numpy()

    detailed_gap_analysis(verts_posed, dominant_joints, "Frame 0 posed")

    # Check all frames for maximum gap
    print("\n" + "="*60)
    print("SCAN ALL FRAMES FOR MAXIMUM GAP")
    print("="*60)

    max_gap = -1000
    max_gap_frame = 0

    for i in range(len(poses_data)):
        poses_t = torch.from_numpy(poses_data[i:i+1]).float()
        trans_t = torch.from_numpy(trans_data[i:i+1]).float()

        with torch.no_grad():
            output = skel.model(betas=betas_t, poses=poses_t, trans=trans_t)
            verts = output.skel_verts[0].cpu().numpy()

        x_center = (verts[:, 0].min() + verts[:, 0].max()) / 2
        spine_mask = np.abs(verts[:, 0] - x_center) < 0.10
        pelvis = spine_mask & (dominant_joints == 0)
        lumbar = spine_mask & (dominant_joints == 11)

        if pelvis.sum() > 0 and lumbar.sum() > 0:
            p_max = verts[pelvis, 1].max()
            l_min = verts[lumbar, 1].min()
            gap = l_min - p_max

            if gap > max_gap:
                max_gap = gap
                max_gap_frame = i

    print(f"Maximum gap: {max_gap*100:.2f}cm at frame {max_gap_frame}")

    if max_gap > 0:
        # Analyze worst frame
        poses_t = torch.from_numpy(poses_data[max_gap_frame:max_gap_frame+1]).float()
        trans_t = torch.from_numpy(trans_data[max_gap_frame:max_gap_frame+1]).float()

        with torch.no_grad():
            output = skel.model(betas=betas_t, poses=poses_t, trans=trans_t)
            verts_worst = output.skel_verts[0].cpu().numpy()

        detailed_gap_analysis(verts_worst, dominant_joints, f"Frame {max_gap_frame} (worst gap)")

        # Save worst frame
        output_dir = "/egr/research-zijunlab/kwonjoon/skel_force_vis/output/debug_gap"
        faces = skel.model.skel_f.cpu().numpy()
        with open(os.path.join(output_dir, f"frame{max_gap_frame}_worst_gap.obj"), 'w') as f:
            for v in verts_worst:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"\nSaved worst frame to: {output_dir}/frame{max_gap_frame}_worst_gap.obj")


if __name__ == "__main__":
    main()
