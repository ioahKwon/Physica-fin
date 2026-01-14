#!/usr/bin/env python3
"""
Command-line interface for SKEL Force Visualization.

Usage:
    python -m skel_force_vis.run --input /path/to/frames --output /path/to/output

    # Or directly:
    python run.py --input /path/to/frames --output /path/to/output
"""

import argparse
from .visualizer import SKELForceVisualizer
from .colormaps import torque_to_color_plasma, torque_to_color_jet, torque_to_color_dark_purple


def main():
    parser = argparse.ArgumentParser(
        description="Visualize joint torques on SKEL skeleton mesh"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing frame_XXXX folders"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for generated OBJ files"
    )
    parser.add_argument(
        "--colormap", "-c",
        choices=["plasma", "jet", "dark_purple"],
        default="dark_purple",
        help="Colormap for torque visualization (default: dark_purple)"
    )
    parser.add_argument(
        "--max-torque",
        type=float,
        default=300.0,
        help="Maximum torque for colormap normalization in Nm (default: 300)"
    )
    parser.add_argument(
        "--line-scale",
        type=float,
        default=0.002,
        help="Scale factor for axis line length (default: 0.002, i.e., 1 Nm = 2mm)"
    )
    parser.add_argument(
        "--line-radius",
        type=float,
        default=0.004,
        help="Radius of axis line cylinders in meters (default: 0.004)"
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=0.012,
        help="Radius of endpoint spheres in meters (default: 0.012)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum torque to show axis line in Nm (default: 0.5)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--skel-model-path",
        type=str,
        default="/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1",
        help="Path to SKEL model directory for LBS weights"
    )
    parser.add_argument(
        "--gender",
        type=str,
        choices=["male", "female"],
        default="male",
        help="Gender for SKEL model (default: male)"
    )
    parser.add_argument(
        "--no-lbs",
        action="store_true",
        help="Disable LBS-based coloring (use nearest-joint instead)"
    )
    parser.add_argument(
        "--unit-arrow-length",
        type=float,
        default=0.05,
        help="Fixed length for torque arrows in meters (default: 0.05). Set to 0 for magnitude-scaled."
    )
    parser.add_argument(
        "--grf-scale",
        type=float,
        default=0.0005,
        help="Scale factor for GRF arrow length (default: 0.0005)"
    )
    parser.add_argument(
        "--grf-radius",
        type=float,
        default=0.012,
        help="Radius of GRF arrow cylinder in meters (default: 0.012)"
    )

    args = parser.parse_args()

    # Select colormap
    if args.colormap == "plasma":
        colormap = torque_to_color_plasma
    elif args.colormap == "jet":
        colormap = torque_to_color_jet
    else:  # dark_purple (default)
        colormap = torque_to_color_dark_purple

    # Create visualizer
    vis = SKELForceVisualizer(
        input_base=args.input,
        output_dir=args.output,
        colormap=colormap,
        max_torque=args.max_torque,
        line_scale=args.line_scale,
        line_radius=args.line_radius,
        sphere_radius=args.sphere_radius,
        torque_threshold=args.threshold,
        skel_model_path=args.skel_model_path,
        gender=args.gender,
        use_lbs_coloring=not args.no_lbs,
        unit_arrow_length=args.unit_arrow_length,
        grf_scale=args.grf_scale,
        grf_radius=args.grf_radius,
    )

    # Process all frames
    vis.process_all_frames(verbose=not args.quiet)


if __name__ == "__main__":
    main()
