#!/usr/bin/env python3
"""
Fix Head/Neck Poses for SKEL Model

AddBiomechanics data doesn't include head/neck joints, so SKEL fitting
leaves head pose parameters (indices 23-25) at zero. This script applies
a heuristic fix based on thorax orientation.

Usage:
    python -m skel_force_vis.fix_head_pose \
        --input /path/to/skel_params.npz \
        --output /path/to/output_dir

    # Or with custom coupling factors
    python -m skel_force_vis.fix_head_pose \
        --input /path/to/skel_params.npz \
        --output /path/to/output_dir \
        --bend-factor 0.2 \
        --ext-factor 0.3 \
        --twist-factor 0.3
"""

import os
import sys
import argparse
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')


# SKEL pose parameter indices
THORAX_BENDING_IDX = 20
THORAX_EXTENSION_IDX = 21
THORAX_TWIST_IDX = 22
HEAD_BENDING_IDX = 23
HEAD_EXTENSION_IDX = 24
HEAD_TWIST_IDX = 25


def fix_head_poses(
    poses: np.ndarray,
    bend_factor: float = 0.2,
    ext_factor: float = 0.3,
    twist_factor: float = 0.3,
    counter_rotation: bool = True,
) -> np.ndarray:
    """
    Fix head poses based on thorax orientation.

    The head naturally follows the thorax but with some damping.
    When counter_rotation=True, the head counter-rotates slightly
    to keep gaze direction more stable (like a chicken's head).

    SKEL pose indices:
    - 20: thorax_bending (side tilt)
    - 21: thorax_extension (forward/backward)
    - 22: thorax_twist (rotation)
    - 23: head_bending
    - 24: head_extension
    - 25: head_twist

    Args:
        poses: [T, 46] SKEL pose parameters
        bend_factor: Coupling factor for bending (default: 0.2)
        ext_factor: Coupling factor for extension (default: 0.3)
        twist_factor: Coupling factor for twist (default: 0.3)
        counter_rotation: If True, head counter-rotates to stay level

    Returns:
        fixed_poses: [T, 46] poses with corrected head parameters
    """
    fixed_poses = poses.copy()

    thorax_bend = poses[:, THORAX_BENDING_IDX]
    thorax_ext = poses[:, THORAX_EXTENSION_IDX]
    thorax_twist = poses[:, THORAX_TWIST_IDX]

    if counter_rotation:
        # Counter-rotation: when body leans forward, head tilts back slightly
        # This keeps the gaze direction more horizontal
        fixed_poses[:, HEAD_BENDING_IDX] = -thorax_bend * bend_factor
        fixed_poses[:, HEAD_EXTENSION_IDX] = -thorax_ext * ext_factor
        fixed_poses[:, HEAD_TWIST_IDX] = thorax_twist * twist_factor  # twist follows, not counters
    else:
        # Simple coupling: head follows thorax with damping
        fixed_poses[:, HEAD_BENDING_IDX] = thorax_bend * bend_factor
        fixed_poses[:, HEAD_EXTENSION_IDX] = thorax_ext * ext_factor
        fixed_poses[:, HEAD_TWIST_IDX] = thorax_twist * twist_factor

    return fixed_poses


# Additional pose indices for spine correction
LUMBAR_BENDING_IDX = 17
LUMBAR_EXTENSION_IDX = 18
LUMBAR_TWIST_IDX = 19


def fix_spine_poses(
    poses: np.ndarray,
    lumbar_ext_offset: float = 0.0,
    thorax_ext_offset: float = 0.0,
    head_ext_offset: float = 0.0,
) -> np.ndarray:
    """
    Fix spine poses by adding offsets to extension angles.

    Positive offset = more upright (less forward lean)
    Negative offset = more forward lean

    Args:
        poses: [T, 46] SKEL pose parameters
        lumbar_ext_offset: Offset for lumbar extension (radians)
        thorax_ext_offset: Offset for thorax extension (radians)
        head_ext_offset: Offset for head extension (radians)

    Returns:
        fixed_poses: [T, 46] poses with corrected spine
    """
    fixed_poses = poses.copy()

    fixed_poses[:, LUMBAR_EXTENSION_IDX] += lumbar_ext_offset
    fixed_poses[:, THORAX_EXTENSION_IDX] += thorax_ext_offset
    fixed_poses[:, HEAD_EXTENSION_IDX] += head_ext_offset

    return fixed_poses


def analyze_poses(poses: np.ndarray, prefix: str = "") -> dict:
    """Analyze thorax and head pose statistics."""
    thorax_bend = np.degrees(poses[:, THORAX_BENDING_IDX])
    thorax_ext = np.degrees(poses[:, THORAX_EXTENSION_IDX])
    thorax_twist = np.degrees(poses[:, THORAX_TWIST_IDX])

    head_bend = np.degrees(poses[:, HEAD_BENDING_IDX])
    head_ext = np.degrees(poses[:, HEAD_EXTENSION_IDX])
    head_twist = np.degrees(poses[:, HEAD_TWIST_IDX])

    stats = {
        'thorax_bending': {'mean': thorax_bend.mean(), 'std': thorax_bend.std(),
                          'min': thorax_bend.min(), 'max': thorax_bend.max()},
        'thorax_extension': {'mean': thorax_ext.mean(), 'std': thorax_ext.std(),
                            'min': thorax_ext.min(), 'max': thorax_ext.max()},
        'thorax_twist': {'mean': thorax_twist.mean(), 'std': thorax_twist.std(),
                        'min': thorax_twist.min(), 'max': thorax_twist.max()},
        'head_bending': {'mean': head_bend.mean(), 'std': head_bend.std(),
                        'min': head_bend.min(), 'max': head_bend.max()},
        'head_extension': {'mean': head_ext.mean(), 'std': head_ext.std(),
                          'min': head_ext.min(), 'max': head_ext.max()},
        'head_twist': {'mean': head_twist.mean(), 'std': head_twist.std(),
                      'min': head_twist.min(), 'max': head_twist.max()},
    }

    if prefix:
        print(f"\n[{prefix}] Pose Statistics (degrees):")
    else:
        print("\nPose Statistics (degrees):")

    for name, s in stats.items():
        print(f"  {name:18s}: mean={s['mean']:6.2f}, std={s['std']:5.2f}, "
              f"range=[{s['min']:6.2f}, {s['max']:6.2f}]")

    return stats


def export_skeleton_objs(
    poses: np.ndarray,
    betas: np.ndarray,
    trans: np.ndarray,
    output_dir: str,
    prefix: str = "fixed",
    gender: str = "male",
    export_skin: bool = True,
):
    """Export skeleton and skin OBJ files using SKEL model."""
    import torch
    from models.skel_model import SKELModelWrapper

    device = torch.device('cpu')
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender=gender,
        device=device
    )

    os.makedirs(output_dir, exist_ok=True)
    T = poses.shape[0]

    print(f"\n[Export] Generating {T} skeleton" + (" + skin" if export_skin else "") + " OBJ files...")

    poses_t = torch.from_numpy(poses).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)

    # Get faces from underlying SKEL model
    skel_faces = skel.model.skel_f.cpu().numpy()
    skin_faces = skel.model.skin_f.cpu().numpy()

    for i in range(T):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Frame {i+1}/{T}")

        with torch.no_grad():
            # Forward pass through SKEL
            output = skel.model(
                betas=betas_t.unsqueeze(0),
                poses=poses_t[i:i+1],
                trans=trans_t[i:i+1]
            )

            # Get skeleton vertices
            skel_verts = output.skel_verts[0].cpu().numpy()
            obj_path = os.path.join(output_dir, f"{prefix}_skeleton_{i:04d}.obj")
            _write_obj(obj_path, skel_verts, skel_faces)

            # Get skin vertices
            if export_skin:
                skin_verts = output.skin_verts[0].cpu().numpy()
                skin_path = os.path.join(output_dir, f"{prefix}_skin_{i:04d}.obj")
                _write_obj(skin_path, skin_verts, skin_faces)

    print(f"[Export] Done! Output: {output_dir}")


def _write_obj(path: str, vertices: np.ndarray, faces: np.ndarray):
    """Write OBJ file."""
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    parser = argparse.ArgumentParser(description="Fix SKEL head poses based on thorax orientation")
    parser.add_argument("--input", "-i", required=True, help="Input skel_params.npz file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--bend-factor", type=float, default=0.2, help="Bending coupling factor")
    parser.add_argument("--ext-factor", type=float, default=0.3, help="Extension coupling factor")
    parser.add_argument("--twist-factor", type=float, default=0.3, help="Twist coupling factor")
    parser.add_argument("--no-counter", action="store_true", help="Disable counter-rotation")
    parser.add_argument("--no-head-fix", action="store_true", help="Skip head pose fix")
    # Spine offset options (in degrees, converted to radians internally)
    # Default: thorax +15°, head +8° (upright torso, natural head position)
    parser.add_argument("--lumbar-offset", type=float, default=0.0, help="Lumbar extension offset (degrees, + = upright)")
    parser.add_argument("--thorax-offset", type=float, default=15.0, help="Thorax extension offset (degrees, + = upright, default: 15)")
    parser.add_argument("--head-offset", type=float, default=8.0, help="Head extension offset (degrees, + = upright, default: 8)")
    parser.add_argument("--export-obj", action="store_true", help="Export skeleton OBJ files")
    parser.add_argument("--gender", default="male", choices=["male", "female"], help="Gender")

    args = parser.parse_args()

    # Load input
    print(f"Loading: {args.input}")
    data = np.load(args.input)
    poses = data['poses']
    betas = data['betas']
    trans = data['trans']

    print(f"  poses: {poses.shape}")
    print(f"  betas: {betas.shape}")
    print(f"  trans: {trans.shape}")

    # Analyze before
    analyze_poses(poses, prefix="Before")

    fixed_poses = poses.copy()

    # Apply spine offset first
    if args.lumbar_offset != 0 or args.thorax_offset != 0 or args.head_offset != 0:
        print(f"\nApplying spine offset...")
        print(f"  lumbar_offset: {args.lumbar_offset}°")
        print(f"  thorax_offset: {args.thorax_offset}°")
        print(f"  head_offset: {args.head_offset}°")

        fixed_poses = fix_spine_poses(
            fixed_poses,
            lumbar_ext_offset=np.radians(args.lumbar_offset),
            thorax_ext_offset=np.radians(args.thorax_offset),
            head_ext_offset=np.radians(args.head_offset),
        )

    # Fix head poses
    if not args.no_head_fix:
        print(f"\nApplying head pose fix...")
        print(f"  bend_factor: {args.bend_factor}")
        print(f"  ext_factor: {args.ext_factor}")
        print(f"  twist_factor: {args.twist_factor}")
        print(f"  counter_rotation: {not args.no_counter}")

        fixed_poses = fix_head_poses(
            fixed_poses,
            bend_factor=args.bend_factor,
            ext_factor=args.ext_factor,
            twist_factor=args.twist_factor,
            counter_rotation=not args.no_counter,
        )

    # Analyze after
    analyze_poses(fixed_poses, prefix="After")

    # Save fixed parameters
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "skel_params_fixed.npz")
    np.savez(output_path, poses=fixed_poses, betas=betas, trans=trans)
    print(f"\nSaved fixed params: {output_path}")

    # Export OBJ files
    if args.export_obj:
        export_skeleton_objs(
            fixed_poses, betas, trans,
            output_dir=args.output,
            prefix="fixed",
            gender=args.gender,
        )


if __name__ == "__main__":
    main()
