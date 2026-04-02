"""Quick visualization for checking fitting results.

Generates a 2x3 grid (front + side views, 3 key frames) with:
- SKEL mesh (semi-transparent blue)
- SKEL skeleton (blue)
- AddB GT skeleton (orange)
- MPJPE in title

Also saves OBJ files for 3D inspection.

Usage:
    from addb2skel.viz.quick_viz import visualize_result
    visualize_result(result, gt_joints, output_dir, name='Subject1')
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..utils.visualization import save_comparison_objs

# =============================================================================
# Skeleton connectivity
# =============================================================================

SKEL_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]

SKEL_BONES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),       # right leg
    (0,6),(6,7),(7,8),(8,9),(9,10),       # left leg
    (0,11),(11,12),(12,13),               # spine
    (12,14),(14,15),(15,16),(16,17),(17,18),  # right arm
    (12,19),(19,20),(20,21),(21,22),(22,23),  # left arm
]

# AddB OpenSim skeleton (20 joints)
ADDB_BONES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),        # right leg
    (0,6),(6,7),(7,8),(8,9),(9,10),        # left leg
    (0,11),(11,12),(12,13),(13,14),(14,15), # spine + right arm
    (11,16),(16,17),(17,18),(18,19),        # left arm
]


# =============================================================================
# Rendering helpers
# =============================================================================

def _np(x):
    """Convert torch tensor to numpy."""
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)


def _render_mesh(ax, verts, faces, color, view='front', alpha=0.6):
    """Render triangulated mesh with simple shading."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    normals /= norms

    if view == 'front':
        light = np.array([0, 0.3, 1.0])
        proj = [0, 1]
        dax, ds = 2, 1
    else:
        light = np.array([-1, 0.3, 0])
        proj = [2, 1]
        dax, ds = 0, -1

    light /= np.linalg.norm(light)
    intensity = 0.3 + 0.7 * np.abs(normals @ light)
    centers = (v0 + v1 + v2) / 3
    order = np.argsort(ds * centers[:, dax])
    c = np.array(color[:3])

    for idx in order:
        tri = verts[faces[idx]][:, proj].copy()
        if view == 'side':
            tri[:, 0] = -tri[:, 0]
        fc = np.clip(c * intensity[idx], 0, 1)
        ax.add_patch(plt.Polygon(tri, facecolor=fc, edgecolor=fc * 0.8,
                                 linewidth=0.1, alpha=alpha))


def _render_skeleton(ax, joints, bones, color, view='front', ms=4, lw=1.5):
    """Render skeleton as joints + bones."""
    proj = {'front': [0, 1], 'side': [2, 1]}[view]
    j2d = joints[:, proj].copy()
    if view == 'side':
        j2d[:, 0] = -j2d[:, 0]
    for (i, j) in bones:
        if i < len(joints) and j < len(joints):
            ax.plot([j2d[i, 0], j2d[j, 0]], [j2d[i, 1], j2d[j, 1]],
                    '-', color=color, linewidth=lw, alpha=0.85)
    ax.scatter(j2d[:, 0], j2d[:, 1], c=[color], s=ms, zorder=5, alpha=0.9)


def _set_lims(ax, pts, view='front', pad=0.05):
    """Set axis limits based on point cloud."""
    if view == 'front':
        ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    else:
        ax.set_xlim(-pts[:, 2].max() - pad, -pts[:, 2].min() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect('equal')
    ax.axis('off')


# =============================================================================
# Main API
# =============================================================================

def generate_quick_viz(
    name: str,
    verts: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    mpjpe: float,
    output_path: str,
    gt_joints: np.ndarray = None,
    key_frames: list = None,
):
    """Generate 2x3 quick check visualization (front + side, 3 frames).

    Args:
        name: Subject name for title.
        verts: SKEL mesh vertices [T, V, 3].
        joints: SKEL joint positions [T, 24, 3].
        faces: Mesh faces [F, 3].
        mpjpe: MPJPE value in mm.
        output_path: Path to save PNG.
        gt_joints: AddB GT joint positions [T, 20, 3]. Orange skeleton if provided.
        key_frames: List of 3 frame indices. Auto-selected if None.
    """
    T = len(verts)
    if key_frames is None:
        key_frames = sorted([T // 6, T // 2, 5 * T // 6])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{name} — MPJPE={mpjpe:.1f}mm (blue=SKEL, orange=AddB GT)',
                 fontsize=14, fontweight='bold')

    for ci, fi in enumerate(key_frames):
        for ri, view in enumerate(['front', 'side']):
            ax = axes[ri, ci]
            _render_mesh(ax, verts[fi], faces, color=(0.45, 0.7, 0.88),
                        view=view, alpha=0.35)
            _render_skeleton(ax, joints[fi, :24], SKEL_BONES,
                           color=(0.2, 0.5, 0.8), view=view, ms=5, lw=1.8)
            if gt_joints is not None and fi < len(gt_joints):
                _render_skeleton(ax, gt_joints[fi], ADDB_BONES,
                               color=(1.0, 0.5, 0.0), view=view, ms=6, lw=2.0)
            all_pts = verts[fi]
            if gt_joints is not None and fi < len(gt_joints):
                all_pts = np.vstack([all_pts, gt_joints[fi]])
            _set_lims(ax, all_pts, view=view, pad=0.15)
            if ri == 0:
                ax.set_title(f'Frame {fi}', fontsize=11)
            if ci == 0:
                ax.set_ylabel('Front' if ri == 0 else 'Side',
                             fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_objs(
    output_dir: str,
    verts: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    gt_joints: np.ndarray = None,
    skel_model=None,
    skel_betas=None,
    skel_poses=None,
    skel_trans=None,
    key_frames: list = None,
):
    """Save OBJ files for 3 key frames.

    Args:
        output_dir: Base output directory.
        verts: SKEL mesh vertices [T, V, 3].
        joints: SKEL joint positions [T, 24, 3].
        faces: Mesh faces [F, 3].
        gt_joints: AddB GT joint positions [T, 20, 3].
        skel_model: SKEL model instance (for bone mesh OBJ).
        skel_betas: SKEL betas tensor.
        skel_poses: SKEL poses tensor [T, 46].
        skel_trans: SKEL trans tensor [T, 3].
        key_frames: List of 3 frame indices.
    """
    T = len(verts)
    if key_frames is None:
        key_frames = sorted([T // 6, T // 2, 5 * T // 6])

    for fi in key_frames:
        obj_dir = os.path.join(output_dir, f'frame_{fi:04d}')
        gt_j = gt_joints[fi] if gt_joints is not None and fi < len(gt_joints) else None
        save_comparison_objs(
            output_dir=obj_dir,
            body_verts=verts[fi],
            body_faces=faces,
            skel_joints=joints[fi],
            parents=SKEL_PARENTS,
            gt_joints=gt_j,
            skel_model=skel_model,
            skel_betas=skel_betas,
            skel_poses_frame=skel_poses[fi] if skel_poses is not None else None,
            skel_trans_frame=skel_trans[fi] if skel_trans is not None else None,
        )


def visualize_result(
    result,
    gt_joints: np.ndarray,
    output_dir: str,
    name: str = 'Subject',
    skel_interface=None,
    key_frames: list = None,
):
    """All-in-one: generate quick viz PNG + save OBJ files.

    Args:
        result: ConversionResult from convert_addb_to_skel().
        gt_joints: AddB GT joint positions [T, 20, 3] in original AddB coords.
        output_dir: Output directory (will be created).
        name: Subject name for title.
        skel_interface: SKELInterface instance (for OBJ bone mesh). Optional.
        key_frames: List of 3 frame indices. Auto-selected if None.
    """
    os.makedirs(output_dir, exist_ok=True)

    verts = _np(result.skel_vertices)
    joints = _np(result.skel_joints)
    faces = skel_interface.faces if skel_interface is not None else None

    em = result.evaluation_metrics or {}
    mpjpe = em.get('mpjpe_mm', result.mpjpe_mm)

    T = len(verts)
    if key_frames is None:
        key_frames = sorted([T // 6, T // 2, 5 * T // 6])

    # Quick viz PNG
    if faces is not None:
        viz_path = os.path.join(output_dir, f'{name}_viz.png')
        generate_quick_viz(name, verts, joints, faces, mpjpe, viz_path,
                          gt_joints=gt_joints, key_frames=key_frames)

    # OBJ files
    save_objs(
        output_dir=output_dir,
        verts=verts, joints=joints, faces=faces,
        gt_joints=gt_joints,
        skel_model=skel_interface.model if skel_interface else None,
        skel_betas=result.skel_betas,
        skel_poses=result.skel_poses,
        skel_trans=result.skel_trans,
        key_frames=key_frames,
    )

    # Pelvis rotation plot
    poses_np = _np(result.skel_poses)
    pelvis_rot = poses_np[:, 2]
    pelvis_std = float(np.std(np.degrees(pelvis_rot)))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(np.degrees(pelvis_rot), linewidth=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Pelvis Rot (°)')
    ax.set_title(f'{name} — Pelvis Rotation (std={pelvis_std:.1f}°)')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{name}_pelvis_rot.png'), dpi=100)
    plt.close(fig)
