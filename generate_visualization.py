#!/usr/bin/env python3
"""
Generate Force Visualization with Pose Fix

End-to-end pipeline that:
1. Loads SKEL parameters from addb2skel output
2. Applies spine/head pose corrections (thorax +15°, head +8° by default)
3. Generates skeleton + body meshes with corrected poses
4. Runs force visualization on the corrected meshes

Usage:
    python -m skel_force_vis.generate_visualization \
        --subject Falisse2017_subject_1 \
        --output /path/to/output

    # Process multiple subjects
    python -m skel_force_vis.generate_visualization \
        --subject Falisse2017_subject_0 Falisse2017_subject_1 Falisse2017_subject_2 \
        --output /path/to/output

    # Custom pose corrections
    python -m skel_force_vis.generate_visualization \
        --subject Falisse2017_subject_1 \
        --thorax-offset 15 \
        --head-offset 10
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from skel_force_vis.fix_head_pose import fix_spine_poses, fix_head_poses, analyze_poses
from skel_force_vis.visualizer import SKELForceVisualizer
from skel_force_vis.colormaps import torque_to_color_plasma, torque_to_color_green

COLORMAPS = {
    'plasma': torque_to_color_plasma,
    'green': torque_to_color_green,
}


# Default paths
ADDB2SKEL_OUTPUT = "/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/full_with_arm"
SKEL_FORCE_VIS_OUTPUT = "/egr/research-zijunlab/kwonjoon/03_Output/skel_force_vis"
SKEL_MODEL_PATH = "/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1"


def generate_corrected_meshes(
    subject_name: str,
    output_dir: str,
    thorax_offset: float = 15.0,
    head_offset: float = 8.0,
    lumbar_offset: float = 0.0,
    gender: str = "male",
    frame_indices: list = None,
    verbose: bool = True,
):
    """
    Generate skeleton and body meshes with pose corrections.

    Args:
        subject_name: Name of subject (e.g., "Falisse2017_subject_1")
        output_dir: Output directory
        thorax_offset: Thorax extension offset in degrees (default: 15)
        head_offset: Head extension offset in degrees (default: 8)
        lumbar_offset: Lumbar extension offset in degrees (default: 0)
        gender: "male" or "female"
        frame_indices: List of frame indices to process, or None for all
        verbose: Print progress

    Returns:
        dict with paths to generated files
    """
    from models.skel_model import SKELModelWrapper

    # Load SKEL parameters
    subject_dir = os.path.join(ADDB2SKEL_OUTPUT, subject_name)
    # Try different locations for skel_params.npz
    params_path = os.path.join(subject_dir, "skel", "skel_params.npz")
    if not os.path.exists(params_path):
        params_path = os.path.join(subject_dir, "skel_params.npz")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"SKEL params not found: {params_path}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Subject: {subject_name}")
        print(f"{'='*60}")
        print(f"Loading: {params_path}")

    data = np.load(params_path)
    poses = data['poses']
    betas = data['betas']
    trans = data['trans']

    if verbose:
        print(f"  poses: {poses.shape}")
        print(f"  betas: {betas.shape}")
        print(f"  trans: {trans.shape}")

    # Select frames
    if frame_indices is not None:
        poses = poses[frame_indices]
        trans = trans[frame_indices]
        if verbose:
            print(f"  Selected {len(frame_indices)} frames")

    T = poses.shape[0]

    # Apply pose corrections
    if verbose:
        print(f"\nApplying pose corrections:")
        print(f"  thorax_offset: +{thorax_offset}°")
        print(f"  head_offset: +{head_offset}°")
        print(f"  lumbar_offset: +{lumbar_offset}°")

    fixed_poses = fix_spine_poses(
        poses,
        lumbar_ext_offset=np.radians(lumbar_offset),
        thorax_ext_offset=np.radians(thorax_offset),
        head_ext_offset=np.radians(head_offset),
    )

    if verbose:
        analyze_poses(fixed_poses, prefix="After fix")

    # Initialize SKEL model
    device = torch.device('cpu')
    skel = SKELModelWrapper(
        model_path=SKEL_MODEL_PATH,
        gender=gender,
        device=device
    )

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    mesh_dir = os.path.join(output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # Get faces
    skel_faces = skel.model.skel_f.cpu().numpy()
    skin_faces = skel.model.skin_f.cpu().numpy()

    # Convert to tensors
    poses_t = torch.from_numpy(fixed_poses).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)

    if verbose:
        print(f"\nGenerating {T} frames...")

    frame_dirs = []

    for i in range(T):
        if verbose and ((i + 1) % 20 == 0 or i == 0):
            print(f"  Frame {i+1}/{T}")

        frame_idx = frame_indices[i] if frame_indices else i
        frame_dir = os.path.join(mesh_dir, f"frame_{frame_idx:04d}")
        os.makedirs(frame_dir, exist_ok=True)
        frame_dirs.append(frame_dir)

        with torch.no_grad():
            output = skel.model(
                betas=betas_t.unsqueeze(0),
                poses=poses_t[i:i+1],
                trans=trans_t[i:i+1]
            )

            # Save skeleton mesh
            skel_verts = output.skel_verts[0].cpu().numpy()
            skel_path = os.path.join(frame_dir, "skeleton.obj")
            _write_obj(skel_path, skel_verts, skel_faces)

            # Save body mesh
            skin_verts = output.skin_verts[0].cpu().numpy()
            skin_path = os.path.join(frame_dir, "body.obj")
            _write_obj(skin_path, skin_verts, skin_faces)

    # Save corrected parameters
    params_out = os.path.join(output_dir, "skel_params_fixed.npz")
    np.savez(params_out, poses=fixed_poses, betas=betas, trans=trans)

    if verbose:
        print(f"\nSaved corrected params: {params_out}")
        print(f"Meshes saved to: {mesh_dir}")

    return {
        'mesh_dir': mesh_dir,
        'frame_dirs': frame_dirs,
        'params_path': params_out,
        'num_frames': T,
    }


def _write_obj(path: str, vertices: np.ndarray, faces: np.ndarray):
    """Write OBJ file."""
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def run_force_visualization(
    subject_name: str,
    mesh_dir: str,
    output_dir: str,
    gender: str = "male",
    max_torque: float = 300.0,
    verbose: bool = True,
    coloring_mode: str = "lbs_blend",
    smooth_sigma: float = 0.02,
    distance_falloff: float = 0.1,
    colormap_name: str = "plasma",
):
    """
    Run force visualization on corrected meshes.

    Args:
        subject_name: Subject name for loading force data
        mesh_dir: Directory containing corrected meshes
        output_dir: Output directory for visualization
        gender: "male" or "female"
        max_torque: Max torque for colormap normalization
        verbose: Print progress
        coloring_mode: 'lbs_blend', 'gaussian', or 'distance'
        smooth_sigma: Sigma for Gaussian smoothing (meters)
        distance_falloff: Falloff for distance-based gradient (meters)
        colormap_name: 'plasma' or 'green'
    """
    # Input for force data (from skel_force_vis output, has frame_XXXX folders with force_data.json)
    input_base = os.path.join(SKEL_FORCE_VIS_OUTPUT, subject_name)

    colormap_func = COLORMAPS.get(colormap_name, torque_to_color_plasma)

    if verbose:
        print(f"\nRunning force visualization...")
        print(f"  Force data: {input_base}")
        print(f"  Mesh override: {mesh_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Colormap: {colormap_name}")

    vis = SKELForceVisualizer(
        input_base=input_base,
        output_dir=output_dir,
        colormap=colormap_func,
        max_torque=max_torque,
        skel_model_path=SKEL_MODEL_PATH,
        gender=gender,
        use_lbs_coloring=True,
        unit_arrow_length=0.05,
        mesh_override_dir=mesh_dir,  # Use corrected meshes
        coloring_mode=coloring_mode,
        smooth_sigma=smooth_sigma,
        distance_falloff=distance_falloff,
    )

    vis.process_all_frames(verbose=verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Generate force visualization with pose corrections"
    )
    parser.add_argument(
        "--subject", "-s",
        nargs="+",
        required=True,
        help="Subject name(s) to process"
    )
    parser.add_argument(
        "--output", "-o",
        default="/egr/research-zijunlab/kwonjoon/skel_force_vis/output",
        help="Base output directory"
    )
    parser.add_argument(
        "--thorax-offset",
        type=float,
        default=15.0,
        help="Thorax extension offset in degrees (default: 15)"
    )
    parser.add_argument(
        "--head-offset",
        type=float,
        default=8.0,
        help="Head extension offset in degrees (default: 8)"
    )
    parser.add_argument(
        "--lumbar-offset",
        type=float,
        default=0.0,
        help="Lumbar extension offset in degrees (default: 0)"
    )
    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default="male",
        help="Gender for SKEL model"
    )
    parser.add_argument(
        "--max-torque",
        type=float,
        default=300.0,
        help="Max torque for colormap (default: 300)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="Specific frame indices to process (default: all)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N frames evenly (e.g., --sample 10)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--coloring-mode",
        choices=["lbs_blend", "gaussian", "distance", "hotspot"],
        default="lbs_blend",
        help="Vertex coloring method: lbs_blend (default), gaussian, distance, hotspot"
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.02,
        help="Sigma for Gaussian smoothing in meters (default: 0.02)"
    )
    parser.add_argument(
        "--distance-falloff",
        type=float,
        default=0.1,
        help="Falloff for distance-based gradient in meters (default: 0.1)"
    )
    parser.add_argument(
        "--colormap",
        choices=["plasma", "green"],
        default="plasma",
        help="Colormap: plasma (purple->yellow), green (dark teal->lime)"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    for subject in args.subject:
        # Determine frame indices
        if args.frames:
            frame_indices = args.frames
        elif args.sample:
            # Sample evenly from available frames
            subject_dir = os.path.join(ADDB2SKEL_OUTPUT, subject)
            params_path = os.path.join(subject_dir, "skel", "skel_params.npz")
            if not os.path.exists(params_path):
                params_path = os.path.join(subject_dir, "skel_params.npz")
            params = np.load(params_path)
            total_frames = params['poses'].shape[0]
            frame_indices = np.linspace(0, total_frames-1, args.sample, dtype=int).tolist()
            if verbose:
                print(f"Sampling {args.sample} frames from {total_frames} total")
        else:
            frame_indices = None

        # Output directory for this subject
        subject_output = os.path.join(args.output, f"{subject}_vis")

        # Step 1: Generate corrected meshes
        result = generate_corrected_meshes(
            subject_name=subject,
            output_dir=subject_output,
            thorax_offset=args.thorax_offset,
            head_offset=args.head_offset,
            lumbar_offset=args.lumbar_offset,
            gender=args.gender,
            frame_indices=frame_indices,
            verbose=verbose,
        )

        # Step 2: Run force visualization
        run_force_visualization(
            subject_name=subject,
            mesh_dir=result['mesh_dir'],
            output_dir=subject_output,
            gender=args.gender,
            max_torque=args.max_torque,
            verbose=verbose,
            coloring_mode=args.coloring_mode,
            smooth_sigma=args.smooth_sigma,
            distance_falloff=args.distance_falloff,
            colormap_name=args.colormap,
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Done: {subject}")
            print(f"Output: {subject_output}")
            print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
