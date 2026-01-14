#!/usr/bin/env python3
"""
Generate comparison OBJs: T-pose, Frame 0 without fix, Frame 0 with fix.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
from models.skel_model import SKELModelWrapper
from spine_bridge import fill_gap_bidirectional


def main():
    output_dir = "/egr/research-zijunlab/kwonjoon/skel_force_vis/output/comparison"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cpu')
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device
    )

    skel_weights = skel.model.skel_weights.to_dense().cpu().numpy()
    dominant_joints = np.argmax(skel_weights, axis=1)
    faces = skel.model.skel_f.cpu().numpy()

    # Load actual pose data
    data = np.load("/egr/research-zijunlab/kwonjoon/skel_force_vis/output/head8_thorax0/skel_params_fixed.npz")
    poses_data = data['poses']
    betas_data = data['betas']
    trans_data = data['trans']

    betas_t = torch.from_numpy(betas_data).float().unsqueeze(0)

    def analyze_and_save(verts, name, path):
        x_center = (verts[:, 0].min() + verts[:, 0].max()) / 2
        spine_mask = np.abs(verts[:, 0] - x_center) < 0.05
        pelvis = spine_mask & (dominant_joints == 0)
        lumbar = spine_mask & (dominant_joints == 11)
        if pelvis.sum() > 0 and lumbar.sum() > 0:
            gap = (verts[lumbar, 1].min() - verts[pelvis, 1].max()) * 100
            print(f"{name}: gap = {gap:.2f}cm")
        with open(path, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"  Saved: {path}")

    # 1. T-pose (no pose applied)
    print("\n=== 1. T-pose ===")
    with torch.no_grad():
        output = skel.model(
            betas=betas_t,
            poses=torch.zeros(1, 46).float(),
            trans=torch.zeros(1, 3).float()
        )
        verts_tpose = output.skel_verts[0].cpu().numpy()
    analyze_and_save(verts_tpose, "T-pose", os.path.join(output_dir, "01_tpose.obj"))

    # 2. Frame 0 posed (no gap fix)
    print("\n=== 2. Frame 0 posed (no gap fix) ===")
    with torch.no_grad():
        output = skel.model(
            betas=betas_t,
            poses=torch.from_numpy(poses_data[0:1]).float(),
            trans=torch.from_numpy(trans_data[0:1]).float()
        )
        verts_posed = output.skel_verts[0].cpu().numpy()
    analyze_and_save(verts_posed, "Frame 0 (no fix)", os.path.join(output_dir, "02_frame0_no_fix.obj"))

    # 3. Frame 0 posed (with gap fix)
    print("\n=== 3. Frame 0 posed (with gap fix) ===")
    verts_fixed = fill_gap_bidirectional(verts_posed, dominant_joints, verbose=True)
    analyze_and_save(verts_fixed, "Frame 0 (fixed)", os.path.join(output_dir, "03_frame0_fixed.obj"))

    # 4. Frame 43 (worst gap frame) - no fix
    print("\n=== 4. Frame 43 (worst gap, no fix) ===")
    with torch.no_grad():
        output = skel.model(
            betas=betas_t,
            poses=torch.from_numpy(poses_data[43:44]).float(),
            trans=torch.from_numpy(trans_data[43:44]).float()
        )
        verts_worst = output.skel_verts[0].cpu().numpy()
    analyze_and_save(verts_worst, "Frame 43 (no fix)", os.path.join(output_dir, "04_frame43_no_fix.obj"))

    # 5. Frame 43 (worst gap frame) - with fix
    print("\n=== 5. Frame 43 (worst gap, with fix) ===")
    verts_worst_fixed = fill_gap_bidirectional(verts_worst, dominant_joints, verbose=True)
    analyze_and_save(verts_worst_fixed, "Frame 43 (fixed)", os.path.join(output_dir, "05_frame43_fixed.obj"))

    print(f"\nAll files saved to: {output_dir}")


if __name__ == "__main__":
    main()
