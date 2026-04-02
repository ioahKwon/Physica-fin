"""Publication-quality visualization using pyrender.

Generates high-resolution 3D renderings with:
- Panel 1: Bone skeleton + AddB GT joints (orange spheres + gray sticks)
- Panel 2: Skin mesh (semi-transparent) + Bone skeleton
- Panel 3 (optional): Joint torque coloring (LBS) + GRF arrows

Each panel shows N frames overlaid (walking sequence).

Also saves OBJ files.

Usage:
    from addb2skel.viz.paper_viz import render_paper_figure
    render_paper_figure(result, gt_joints, output_path, name='Subject1')

Requirements:
    pip install pyrender trimesh
    export PYOPENGL_PLATFORM=egl  (for headless rendering)
"""

import os
import numpy as np

# Lazy imports (pyrender/trimesh may not be installed)
_pyrender = None
_trimesh = None


def _ensure_imports():
    global _pyrender, _trimesh
    if _pyrender is None:
        import pyrender
        import trimesh
        _pyrender = pyrender
        _trimesh = trimesh


# =============================================================================
# Constants
# =============================================================================

ADDB_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 11, 16, 17, 18]
SKEL_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]

# Materials
_MATERIALS = None

def _get_materials():
    global _MATERIALS
    if _MATERIALS is None:
        _ensure_imports()
        pyrender = _pyrender
        _MATERIALS = {
            'bone': pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.05, roughnessFactor=0.6, alphaMode='OPAQUE',
                baseColorFactor=(0.945, 0.749, 0.600, 1.0)),  # ivory bone
            'skin': pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, alphaMode='BLEND',
                baseColorFactor=(0.537, 0.765, 0.922, 0.55)),  # semi-transparent blue
            'skin_opaque': pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, alphaMode='OPAQUE',
                baseColorFactor=(0.537, 0.765, 0.922, 1.0)),
            'gt_joint': pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE',
                baseColorFactor=(1.0, 0.5, 0.0, 1.0)),  # orange
            'stick': pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE',
                baseColorFactor=(0.5, 0.5, 0.5, 1.0)),  # gray
            'grf': pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE',
                baseColorFactor=(0.15, 0.85, 0.15, 1.0)),  # green
        }
    return _MATERIALS


# =============================================================================
# Geometry helpers
# =============================================================================

def _make_spheres(positions, radius=0.012):
    _ensure_imports()
    trimesh = _trimesh
    all_v, all_f = [], []
    off = 0
    for p in positions:
        if np.any(np.isnan(p)):
            continue
        s = trimesh.creation.uv_sphere(radius=radius, count=[8, 8])
        s.apply_translation(p)
        all_v.append(s.vertices)
        all_f.append(s.faces + off)
        off += len(s.vertices)
    if not all_v:
        return None
    return trimesh.Trimesh(np.vstack(all_v), np.vstack(all_f), process=False)


def _make_sticks(joints, parents, radius=0.005):
    _ensure_imports()
    trimesh = _trimesh
    all_v, all_f = [], []
    off = 0
    for i, par in enumerate(parents):
        if par < 0 or i >= len(joints) or par >= len(joints):
            continue
        p0, p1 = joints[par], joints[i]
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
            continue
        d = np.linalg.norm(p1 - p0)
        if d < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=d)
        mid = (p0 + p1) / 2
        direction = (p1 - p0) / d
        z = np.array([0, 0, 1.0])
        ax = np.cross(z, direction)
        al = np.linalg.norm(ax)
        if al > 1e-6:
            cyl.apply_transform(trimesh.transformations.rotation_matrix(
                np.arccos(np.clip(np.dot(z, direction), -1, 1)), ax / al))
        elif np.dot(z, direction) < 0:
            cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        cyl.apply_translation(mid)
        all_v.append(cyl.vertices)
        all_f.append(cyl.faces + off)
        off += len(cyl.vertices)
    if not all_v:
        return None
    return trimesh.Trimesh(np.vstack(all_v), np.vstack(all_f), process=False)


def _make_grf_arrow(cop_pos, force_vec, scale=0.0009):
    _ensure_imports()
    trimesh = _trimesh
    mag = np.linalg.norm(force_vec)
    if mag < 10:
        return None
    direction = force_vec / mag
    arrow_len = mag * scale
    end = cop_pos + direction * arrow_len
    z_offset = np.array([0.0, 0.0, -0.08])
    shaft = trimesh.creation.cylinder(
        radius=0.008, segment=np.array([cop_pos + z_offset, end + z_offset]))
    head = trimesh.creation.uv_sphere(radius=0.016, count=[8, 8])
    head.apply_translation(end + z_offset)
    return trimesh.util.concatenate([shaft, head])


# =============================================================================
# Rendering
# =============================================================================

def _create_camera(K4):
    _ensure_imports()
    pyrender = _pyrender
    fx, fy, cx, cy = K4
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
    cam_pose = np.eye(4)
    return camera, cam_pose


def _create_raymond_lights():
    _ensure_imports()
    pyrender = _pyrender
    nodes = []
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix))
    return nodes


def render_scene(meshes_list, cam_t, K4, S, view='front'):
    """Render a list of (trimesh, material) into a single image.

    Args:
        meshes_list: List of (trimesh.Trimesh, pyrender.Material) tuples.
        cam_t: Camera translation [3].
        K4: Camera intrinsics [fx, fy, cx, cy].
        S: Image size (square).
        view: 'front' or 'side'.

    Returns:
        np.ndarray: RGB image [S, S, 3] uint8.
    """
    _ensure_imports()
    pyrender = _pyrender
    trimesh = _trimesh

    white_bg = np.ones((S, S, 3), dtype=np.uint8) * 255
    renderer = pyrender.OffscreenRenderer(viewport_width=S, viewport_height=S, point_size=1.0)
    camera, cam_pose = _create_camera(K4)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    for node in _create_raymond_lights():
        scene.add_node(node)

    for idx, (tm, mat) in enumerate(meshes_list):
        m = tm.copy()
        if view == 'side':
            m.apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(-90), [0, 1, 0]))
        m.apply_transform(trimesh.transformations.translation_matrix(cam_t))
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
        scene.add(pyrender.Mesh.from_trimesh(m, material=mat), 'm%d' % idx)

    scene.add(camera, pose=cam_pose)
    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask = rgba.astype(np.float32)[:, :, [-1]] / 255.0
    final = rgba[:, :, :3] * mask + white_bg * (1 - mask)
    renderer.delete()
    return final.astype(np.uint8)


# =============================================================================
# Main API
# =============================================================================

def render_paper_figure(
    result,
    gt_joints: np.ndarray,
    output_path: str,
    name: str = 'Subject',
    skel_interface=None,
    key_frames: list = None,
    n_frames: int = 4,
    patch_size: int = 768,
    focal: float = 5000,
    panels: list = None,
):
    """Render publication-quality figure with overlaid frames.

    Args:
        result: ConversionResult from convert_addb_to_skel().
        gt_joints: AddB GT joints [T, 20, 3].
        output_path: Path to save PNG.
        name: Subject name for title.
        skel_interface: SKELInterface (for faces).
        key_frames: Frame indices. Auto-selected if None.
        n_frames: Number of frames to overlay (default 4).
        patch_size: Render resolution per panel.
        focal: Camera focal length.
        panels: List of panel types. Default: ['skeleton+gt', 'mesh+skeleton'].
                Options: 'skeleton+gt', 'mesh+skeleton', 'mesh_only'.
    """
    _ensure_imports()
    trimesh = _trimesh
    pyrender = _pyrender
    from PIL import Image, ImageDraw, ImageFont

    if panels is None:
        panels = ['skeleton+gt', 'mesh+skeleton']

    # Extract data
    skin_v = result.skel_vertices.cpu().numpy() if hasattr(result.skel_vertices, 'cpu') else np.array(result.skel_vertices)
    skel_v = None
    if skel_interface is not None and hasattr(skel_interface, 'model'):
        import torch
        with torch.no_grad():
            out = skel_interface.model(
                betas=result.skel_betas.unsqueeze(0).expand(len(skin_v), -1),
                poses=result.skel_poses,
                trans=result.skel_trans)
            skel_v = out.skel_verts.cpu().numpy()

    faces_skin = skel_interface.faces if skel_interface else None
    faces_bone = skel_interface.model.skel_f.cpu().numpy() if skel_interface else None

    T = len(skin_v)
    if key_frames is None:
        key_frames = np.linspace(T // 10, T - T // 10, n_frames, dtype=int).tolist()

    S = patch_size
    mats = _get_materials()

    # Camera setup: center on all selected frames
    all_skin = np.concatenate([skin_v[fi] for fi in key_frames], axis=0)
    gc = all_skin.mean(axis=0)
    ac = all_skin - gc
    ac[:, 1] *= -1
    y_range = ac[:, 1].max() - ac[:, 1].min()
    tz = focal * y_range / (S * 0.80)
    cam_t = np.array([0.0, 0.0, tz])
    K4 = [focal, focal, S // 2, S // 2]

    def prep(v):
        out = v - gc
        out[:, 1] *= -1
        return out

    rendered_panels = []

    for panel_type in panels:
        meshes = []
        for fi in key_frames:
            if panel_type == 'skeleton+gt':
                # Bone skeleton
                if skel_v is not None and faces_bone is not None:
                    meshes.append((trimesh.Trimesh(prep(skel_v[fi]), faces_bone, process=False), mats['bone']))
                # GT joints + sticks
                if gt_joints is not None and fi < len(gt_joints):
                    j = prep(gt_joints[fi].copy())
                    sm = _make_spheres(j[:20], radius=0.012)
                    if sm:
                        meshes.append((sm, mats['gt_joint']))
                    bm = _make_sticks(j[:20], ADDB_PARENTS, radius=0.005)
                    if bm:
                        meshes.append((bm, mats['stick']))

            elif panel_type == 'mesh+skeleton':
                # Bone skeleton
                if skel_v is not None and faces_bone is not None:
                    meshes.append((trimesh.Trimesh(prep(skel_v[fi]), faces_bone, process=False), mats['bone']))
                # Semi-transparent skin
                if faces_skin is not None:
                    meshes.append((trimesh.Trimesh(prep(skin_v[fi]), faces_skin, process=False), mats['skin']))

            elif panel_type == 'mesh_only':
                if faces_skin is not None:
                    meshes.append((trimesh.Trimesh(prep(skin_v[fi]), faces_skin, process=False), mats['skin_opaque']))

        img = render_scene(meshes, cam_t, K4, S, 'front')
        rendered_panels.append(img)

    # Compose panels side by side
    n_panels = len(rendered_panels)
    canvas_w = S * n_panels
    canvas_h = S + 40  # extra space for title
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for i, panel in enumerate(rendered_panels):
        canvas[40:40 + S, i * S:(i + 1) * S] = panel

    # Add title and panel labels
    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    em = result.evaluation_metrics or {}
    mpjpe = em.get('mpjpe_mm', result.mpjpe_mm)
    title = f'{name} | {n_frames} frames | MPJPE={mpjpe:.1f}mm'
    draw.text((10, 8), title, fill=(0, 0, 0), font=font)

    panel_labels = {
        'skeleton+gt': 'Skeleton + AddB',
        'mesh+skeleton': 'Mesh + Skeleton',
        'mesh_only': 'Mesh',
    }
    for i, pt in enumerate(panels):
        label = panel_labels.get(pt, pt)
        draw.text((i * S + 10, S + 12), label, fill=(100, 100, 100), font=font_small)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    img_pil.save(output_path, quality=95)
