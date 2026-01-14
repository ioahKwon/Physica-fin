#!/usr/bin/env python3
"""
Generate basic T-pose skeleton without any modifications.
For debugging spine gap issues.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
from models.skel_model import SKELModelWrapper


def generate_tpose(output_dir: str, gender: str = "male"):
    """Generate T-pose skeleton OBJ."""

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cpu')
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender=gender,
        device=device
    )

    # Zero pose = T-pose
    poses = torch.zeros(1, 46).float()
    betas = torch.zeros(1, 10).float()
    trans = torch.zeros(1, 3).float()

    print(f"Generating T-pose skeleton...")
    print(f"  poses: all zeros (T-pose)")
    print(f"  betas: all zeros (average shape)")
    print(f"  trans: all zeros (origin)")

    with torch.no_grad():
        output = skel.model(betas=betas, poses=poses, trans=trans)
        skel_verts = output.skel_verts[0].cpu().numpy()
        skel_faces = skel.model.skel_f.cpu().numpy()

    # Analyze spine gap
    skel_weights = skel.model.skel_weights.to_dense().cpu().numpy()
    dominant_joints = np.argmax(skel_weights, axis=1)

    x_center = (skel_verts[:, 0].min() + skel_verts[:, 0].max()) / 2
    z_center = (skel_verts[:, 2].min() + skel_verts[:, 2].max()) / 2

    spine_mask = np.abs(skel_verts[:, 0] - x_center) < 0.10
    pelvis_spine = spine_mask & (dominant_joints == 0)
    lumbar_spine = spine_mask & (dominant_joints == 11)

    if pelvis_spine.sum() > 0 and lumbar_spine.sum() > 0:
        pelvis_y_max = skel_verts[pelvis_spine, 1].max()
        lumbar_y_min = skel_verts[lumbar_spine, 1].min()
        gap = lumbar_y_min - pelvis_y_max

        print(f"\nSpine gap analysis (T-pose):")
        print(f"  Pelvis Y max: {pelvis_y_max:.4f}")
        print(f"  Lumbar Y min: {lumbar_y_min:.4f}")
        print(f"  Gap: {gap*100:.2f} cm")

        # Check anterior/posterior
        anterior = spine_mask & (skel_verts[:, 2] < z_center)
        posterior = spine_mask & (skel_verts[:, 2] > z_center)

        for name, mask in [("Anterior", anterior), ("Posterior", posterior)]:
            pelvis_region = mask & (dominant_joints == 0)
            lumbar_region = mask & (dominant_joints == 11)
            if pelvis_region.sum() > 0 and lumbar_region.sum() > 0:
                p_max = skel_verts[pelvis_region, 1].max()
                l_min = skel_verts[lumbar_region, 1].min()
                print(f"  {name}: pelvis_max={p_max:.4f}, lumbar_min={l_min:.4f}, gap={(l_min-p_max)*100:.2f}cm")

    # Save OBJ without any modifications
    obj_path = os.path.join(output_dir, "tpose_skeleton_raw.obj")
    with open(obj_path, 'w') as f:
        for v in skel_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in skel_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"\nSaved: {obj_path}")

    # Also save skin for reference
    skin_verts = output.skin_verts[0].cpu().numpy()
    skin_faces = skel.model.skin_f.cpu().numpy()
    skin_path = os.path.join(output_dir, "tpose_skin_raw.obj")
    with open(skin_path, 'w') as f:
        for v in skin_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in skin_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Saved: {skin_path}")

    return obj_path


if __name__ == "__main__":
    output_dir = "/egr/research-zijunlab/kwonjoon/skel_force_vis/output/tpose_raw"
    generate_tpose(output_dir)
