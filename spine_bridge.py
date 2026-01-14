#!/usr/bin/env python3
"""
Bridge mesh generator to fill the gap between pelvis and lumbar in SKEL skeleton.

The SKEL model has a structural gap between pelvis (bone 0) and lumbar_body (bone 11)
in the spine region. This module creates bridge vertices/faces to fill that gap.
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import Tuple, Optional


def find_spine_gap(
    verts: np.ndarray,
    dominant_joints: np.ndarray,
    pelvis_joint_id: int = 0,
    lumbar_joint_id: int = 11,
    spine_x_radius: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Find the gap region between pelvis and lumbar in the spine.

    Args:
        verts: (N, 3) vertex positions
        dominant_joints: (N,) dominant joint ID for each vertex
        pelvis_joint_id: Joint ID for pelvis (default 0)
        lumbar_joint_id: Joint ID for lumbar (default 11)
        spine_x_radius: X-radius from center to consider as spine

    Returns:
        pelvis_boundary_idx: Indices of pelvis vertices near the gap
        lumbar_boundary_idx: Indices of lumbar vertices near the gap
        gap_bottom: Y coordinate of gap bottom (pelvis top)
        gap_top: Y coordinate of gap top (lumbar bottom)
    """
    # Find center
    x_center = (verts[:, 0].min() + verts[:, 0].max()) / 2

    # Spine region mask
    spine_mask = np.abs(verts[:, 0] - x_center) < spine_x_radius

    pelvis_spine = spine_mask & (dominant_joints == pelvis_joint_id)
    lumbar_spine = spine_mask & (dominant_joints == lumbar_joint_id)

    if pelvis_spine.sum() == 0 or lumbar_spine.sum() == 0:
        raise ValueError("No pelvis or lumbar vertices found in spine region")

    pelvis_y_max = verts[pelvis_spine, 1].max()
    lumbar_y_min = verts[lumbar_spine, 1].min()

    # Gap boundaries (with margin)
    gap_bottom = pelvis_y_max
    gap_top = lumbar_y_min

    # Find boundary vertices (near the gap)
    margin = 0.03  # 3cm margin
    pelvis_boundary = pelvis_spine & (verts[:, 1] > gap_bottom - margin)
    lumbar_boundary = lumbar_spine & (verts[:, 1] < gap_top + margin)

    return (
        np.where(pelvis_boundary)[0],
        np.where(lumbar_boundary)[0],
        gap_bottom,
        gap_top
    )


def create_bridge_mesh(
    verts: np.ndarray,
    pelvis_boundary_idx: np.ndarray,
    lumbar_boundary_idx: np.ndarray,
    gap_bottom: float,
    gap_top: float,
    n_layers: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create bridge vertices and faces to fill the gap.

    Args:
        verts: (N, 3) original vertex positions
        pelvis_boundary_idx: Indices of pelvis boundary vertices
        lumbar_boundary_idx: Indices of lumbar boundary vertices
        gap_bottom: Y coordinate of gap bottom
        gap_top: Y coordinate of gap top
        n_layers: Number of intermediate layers

    Returns:
        bridge_verts: (M, 3) new vertices to add
        bridge_faces: (F, 3) new faces (indices relative to bridge_verts)
    """
    pelvis_boundary = verts[pelvis_boundary_idx]
    lumbar_boundary = verts[lumbar_boundary_idx]

    # Find overlapping XZ region
    x_min = max(pelvis_boundary[:, 0].min(), lumbar_boundary[:, 0].min())
    x_max = min(pelvis_boundary[:, 0].max(), lumbar_boundary[:, 0].max())
    z_min = max(pelvis_boundary[:, 2].min(), lumbar_boundary[:, 2].min())
    z_max = min(pelvis_boundary[:, 2].max(), lumbar_boundary[:, 2].max())

    # Filter to overlapping region
    pelvis_overlap_mask = (
        (pelvis_boundary[:, 0] >= x_min - 0.01) &
        (pelvis_boundary[:, 0] <= x_max + 0.01) &
        (pelvis_boundary[:, 2] >= z_min - 0.01) &
        (pelvis_boundary[:, 2] <= z_max + 0.01)
    )
    lumbar_overlap_mask = (
        (lumbar_boundary[:, 0] >= x_min - 0.01) &
        (lumbar_boundary[:, 0] <= x_max + 0.01) &
        (lumbar_boundary[:, 2] >= z_min - 0.01) &
        (lumbar_boundary[:, 2] <= z_max + 0.01)
    )

    pelvis_overlap = pelvis_boundary[pelvis_overlap_mask]
    lumbar_overlap = lumbar_boundary[lumbar_overlap_mask]

    if len(pelvis_overlap) < 3 or len(lumbar_overlap) < 3:
        print(f"Warning: Not enough overlap vertices (pelvis: {len(pelvis_overlap)}, lumbar: {len(lumbar_overlap)})")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3).astype(int)

    # Create grid of bridge vertices
    # Sample XZ positions from both boundaries
    n_samples = max(10, min(len(pelvis_overlap), len(lumbar_overlap)) // 2)

    # Use mean XZ from both boundaries
    all_xz = np.vstack([pelvis_overlap[:, [0, 2]], lumbar_overlap[:, [0, 2]]])

    # Create a grid covering the XZ range
    x_samples = np.linspace(x_min, x_max, max(5, int((x_max - x_min) / 0.01)))
    z_samples = np.linspace(z_min, z_max, max(5, int((z_max - z_min) / 0.01)))

    # Y layers between gap
    y_samples = np.linspace(gap_bottom, gap_top, n_layers + 2)[1:-1]  # Exclude endpoints

    # Generate bridge vertices
    bridge_verts = []
    for y in y_samples:
        # Interpolate between pelvis and lumbar positions
        t = (y - gap_bottom) / (gap_top - gap_bottom)  # 0 at bottom, 1 at top

        for x in x_samples:
            for z in z_samples:
                # Check if this XZ is within the spine region
                # Simple check: is it close to any boundary vertex?
                dist_to_pelvis = np.min(np.linalg.norm(pelvis_overlap[:, [0, 2]] - np.array([x, z]), axis=1))
                dist_to_lumbar = np.min(np.linalg.norm(lumbar_overlap[:, [0, 2]] - np.array([x, z]), axis=1))

                if dist_to_pelvis < 0.03 or dist_to_lumbar < 0.03:
                    bridge_verts.append([x, y, z])

    if len(bridge_verts) == 0:
        print("Warning: No bridge vertices generated")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3).astype(int)

    bridge_verts = np.array(bridge_verts)

    # Create faces using Delaunay triangulation on XZ plane for each layer
    # This is a simplified approach - just return vertices for now
    # Faces would need more sophisticated handling

    return bridge_verts, np.array([]).reshape(0, 3).astype(int)


def fill_spine_gap_simple(
    verts: np.ndarray,
    dominant_joints: np.ndarray,
    pelvis_joint_id: int = 0,
    lumbar_joint_id: int = 11,
) -> np.ndarray:
    """
    Simple gap filling by moving lumbar vertices down to meet pelvis.

    This modifies vertex positions directly to close the gap.

    Args:
        verts: (N, 3) vertex positions (will be modified in place)
        dominant_joints: (N,) dominant joint ID for each vertex

    Returns:
        Modified vertices
    """
    verts = verts.copy()

    # Find center
    x_center = (verts[:, 0].min() + verts[:, 0].max()) / 2

    # Spine region
    spine_x_radius = 0.06
    spine_mask = np.abs(verts[:, 0] - x_center) < spine_x_radius

    pelvis_spine = spine_mask & (dominant_joints == pelvis_joint_id)
    lumbar_spine = spine_mask & (dominant_joints == lumbar_joint_id)

    if pelvis_spine.sum() == 0 or lumbar_spine.sum() == 0:
        return verts

    pelvis_y_max = verts[pelvis_spine, 1].max()
    lumbar_y_min = verts[lumbar_spine, 1].min()

    gap = lumbar_y_min - pelvis_y_max

    if gap <= 0:
        # No gap
        return verts

    print(f"Filling spine gap: {gap*100:.2f} cm")

    # Move lumbar spine vertices down by gap amount
    # Use smooth falloff based on Y distance from gap
    lumbar_y = verts[lumbar_spine, 1]

    # Smooth falloff: full shift at gap, zero shift at 10cm above gap
    falloff_range = 0.10  # 10cm
    shift_amount = gap + 0.005  # Close gap plus small overlap

    for i in np.where(lumbar_spine)[0]:
        y = verts[i, 1]
        dist_from_gap = y - lumbar_y_min

        if dist_from_gap < falloff_range:
            # Linear falloff
            factor = 1.0 - (dist_from_gap / falloff_range)
            verts[i, 1] -= shift_amount * factor

    return verts


def fill_gap_bidirectional(
    verts: np.ndarray,
    dominant_joints: np.ndarray,
    pelvis_joint_id: int = 0,
    lumbar_joint_id: int = 11,
    verbose: bool = False,
) -> np.ndarray:
    """
    Fill gap by moving both pelvis up and lumbar down to meet in the middle.

    Uses narrow X radius (0.05) to accurately detect spine gap, then applies
    fix to all spine vertices with smooth falloff.

    Args:
        verts: (N, 3) vertex positions
        dominant_joints: (N,) dominant joint ID for each vertex
        verbose: If True, print debug info

    Returns:
        Modified vertices
    """
    verts = verts.copy()

    # Find center
    x_center = (verts[:, 0].min() + verts[:, 0].max()) / 2
    z_center = (verts[:, 2].min() + verts[:, 2].max()) / 2

    # Use NARROW X radius for gap detection (wider radius masks the real gap)
    detect_x_radius = 0.05
    # Use wider radius for applying the fix (to affect more spine vertices)
    fix_x_radius = 0.08

    detect_mask = np.abs(verts[:, 0] - x_center) < detect_x_radius
    fix_mask = np.abs(verts[:, 0] - x_center) < fix_x_radius

    # Detect gap using narrow mask
    pelvis_detect = detect_mask & (dominant_joints == pelvis_joint_id)
    lumbar_detect = detect_mask & (dominant_joints == lumbar_joint_id)

    if pelvis_detect.sum() == 0 or lumbar_detect.sum() == 0:
        if verbose:
            print("No pelvis/lumbar vertices in detection region")
        return verts

    pelvis_y_max = verts[pelvis_detect, 1].max()
    lumbar_y_min = verts[lumbar_detect, 1].min()
    gap = lumbar_y_min - pelvis_y_max

    if gap <= 0.005:  # Less than 5mm gap = no fix needed
        if verbose:
            print(f"Gap is small ({gap*100:.2f}cm), no fix needed")
        return verts

    if verbose:
        print(f"Detected spine gap: {gap*100:.2f}cm")

    # Calculate fix amount
    # Move pelvis up by half_gap, lumbar down by half_gap, plus small overlap
    half_gap = gap / 2 + 0.005  # 5mm overlap

    # Falloff range - scale with gap size, minimum 4cm
    falloff_range = max(0.04, gap * 0.8)

    # Apply fix using wider mask
    pelvis_fix = fix_mask & (dominant_joints == pelvis_joint_id)
    lumbar_fix = fix_mask & (dominant_joints == lumbar_joint_id)

    # Move pelvis up (toward the gap)
    pelvis_y_max_fix = verts[pelvis_fix, 1].max()
    for i in np.where(pelvis_fix)[0]:
        y = verts[i, 1]
        dist_from_gap = pelvis_y_max_fix - y

        if dist_from_gap < falloff_range:
            # Smooth falloff: 1.0 at gap boundary, 0.0 at falloff_range
            factor = 1.0 - (dist_from_gap / falloff_range)
            factor = factor ** 0.5  # Gentler falloff curve
            verts[i, 1] += half_gap * factor

    # Move lumbar down (toward the gap)
    lumbar_y_min_fix = verts[lumbar_fix, 1].min()
    for i in np.where(lumbar_fix)[0]:
        y = verts[i, 1]
        dist_from_gap = y - lumbar_y_min_fix

        if dist_from_gap < falloff_range:
            factor = 1.0 - (dist_from_gap / falloff_range)
            factor = factor ** 0.5  # Gentler falloff curve
            verts[i, 1] -= half_gap * factor

    if verbose:
        # Verify fix
        new_pelvis_y_max = verts[pelvis_detect, 1].max()
        new_lumbar_y_min = verts[lumbar_detect, 1].min()
        new_gap = new_lumbar_y_min - new_pelvis_y_max
        print(f"After fix: gap={new_gap*100:.2f}cm")

    return verts


if __name__ == "__main__":
    import torch
    import sys
    sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
    from models.skel_model import SKELModelWrapper

    # Test
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device='cpu'
    )

    data = np.load("/egr/research-zijunlab/kwonjoon/skel_force_vis/output/head8_thorax0/skel_params_fixed.npz")
    poses = torch.from_numpy(data['poses'][0:1]).float()
    betas = torch.from_numpy(data['betas']).float().unsqueeze(0)
    trans = torch.from_numpy(data['trans'][0:1]).float()

    with torch.no_grad():
        output = skel.model(betas=betas, poses=poses, trans=trans)
        verts = output.skel_verts[0].numpy()
        faces = skel.model.skel_f.cpu().numpy()

    skel_weights = skel.model.skel_weights.to_dense().cpu().numpy()
    dominant = np.argmax(skel_weights, axis=1)

    # Test gap filling
    print("Before:")
    x_center = (verts[:, 0].min() + verts[:, 0].max()) / 2
    spine_mask = np.abs(verts[:, 0] - x_center) < 0.05
    pelvis_spine = spine_mask & (dominant == 0)
    lumbar_spine = spine_mask & (dominant == 11)
    print(f"  Pelvis Y max: {verts[pelvis_spine, 1].max():.4f}")
    print(f"  Lumbar Y min: {verts[lumbar_spine, 1].min():.4f}")
    print(f"  Gap: {(verts[lumbar_spine, 1].min() - verts[pelvis_spine, 1].max())*100:.2f} cm")

    # Fill gap
    verts_fixed = fill_gap_bidirectional(verts, dominant, verbose=True)

    print("\nAfter:")
    print(f"  Pelvis Y max: {verts_fixed[pelvis_spine, 1].max():.4f}")
    print(f"  Lumbar Y min: {verts_fixed[lumbar_spine, 1].min():.4f}")
    print(f"  Gap: {(verts_fixed[lumbar_spine, 1].min() - verts_fixed[pelvis_spine, 1].max())*100:.2f} cm")

    # Save test OBJ
    output_path = "/egr/research-zijunlab/kwonjoon/skel_force_vis/output/test_gap_filled.obj"
    with open(output_path, 'w') as f:
        for v in verts_fixed:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"\nSaved: {output_path}")
