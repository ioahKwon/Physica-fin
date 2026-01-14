#!/usr/bin/env python3
"""
Debug spine gap with actual pose vs T-pose.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
from models.skel_model import SKELModelWrapper


def analyze_gap(verts, dominant_joints, label=""):
    """Analyze spine gap."""
    x_center = (verts[:, 0].min() + verts[:, 0].max()) / 2
    z_center = (verts[:, 2].min() + verts[:, 2].max()) / 2

    spine_mask = np.abs(verts[:, 0] - x_center) < 0.10
    pelvis_spine = spine_mask & (dominant_joints == 0)
    lumbar_spine = spine_mask & (dominant_joints == 11)

    if pelvis_spine.sum() == 0 or lumbar_spine.sum() == 0:
        print(f"[{label}] No spine vertices found")
        return

    pelvis_y_max = verts[pelvis_spine, 1].max()
    lumbar_y_min = verts[lumbar_spine, 1].min()
    gap = lumbar_y_min - pelvis_y_max

    print(f"[{label}] Pelvis Y max: {pelvis_y_max:.4f}, Lumbar Y min: {lumbar_y_min:.4f}, Gap: {gap*100:.2f}cm")


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

    output_dir = "/egr/research-zijunlab/kwonjoon/skel_force_vis/output/debug_gap"
    os.makedirs(output_dir, exist_ok=True)

    print("=== T-pose (pose=0) ===")
    poses_zero = torch.zeros(1, 46).float()
    betas_t = torch.from_numpy(betas_data).float().unsqueeze(0)
    trans_zero = torch.zeros(1, 3).float()

    with torch.no_grad():
        output = skel.model(betas=betas_t, poses=poses_zero, trans=trans_zero)
        verts_tpose = output.skel_verts[0].cpu().numpy()

    analyze_gap(verts_tpose, dominant_joints, "T-pose with subject betas")

    # Save T-pose
    faces = skel.model.skel_f.cpu().numpy()
    with open(os.path.join(output_dir, "tpose_with_betas.obj"), 'w') as f:
        for v in verts_tpose:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print("\n=== Frame 0 (actual pose, no gap fix) ===")
    poses_t = torch.from_numpy(poses_data[0:1]).float()
    trans_t = torch.from_numpy(trans_data[0:1]).float()

    with torch.no_grad():
        output = skel.model(betas=betas_t, poses=poses_t, trans=trans_t)
        verts_posed = output.skel_verts[0].cpu().numpy()

    analyze_gap(verts_posed, dominant_joints, "Frame 0 posed")

    # Save posed (no fix)
    with open(os.path.join(output_dir, "frame0_posed_no_fix.obj"), 'w') as f:
        for v in verts_posed:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print("\n=== Frame 0 (with gap fix) ===")
    from spine_bridge import fill_gap_bidirectional
    verts_fixed = fill_gap_bidirectional(verts_posed, dominant_joints)

    analyze_gap(verts_fixed, dominant_joints, "Frame 0 fixed")

    # Save fixed
    with open(os.path.join(output_dir, "frame0_posed_with_fix.obj"), 'w') as f:
        for v in verts_fixed:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # Print pose parameters for spine joints
    print("\n=== Spine-related pose parameters (Frame 0) ===")
    pose_names = {
        17: "lumbar_bending",
        18: "lumbar_extension",
        19: "lumbar_twist",
        20: "thorax_bending",
        21: "thorax_extension",
        22: "thorax_twist",
    }
    for idx, name in pose_names.items():
        print(f"  {name} (idx {idx}): {np.degrees(poses_data[0, idx]):.2f}°")

    print(f"\nSaved debug OBJs to: {output_dir}")
    print("  - tpose_with_betas.obj")
    print("  - frame0_posed_no_fix.obj")
    print("  - frame0_posed_with_fix.obj")


if __name__ == "__main__":
    main()
