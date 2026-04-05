"""
Generalized pyrender-based visualization for addb2skel pipeline.

Renders SKEL body model with optional overlays:
  - Biomechanical skeleton mesh (bone mesh)
  - Skin mesh (translucent)
  - GT joint positions (AddB joints, orange spheres + sticks)
  - SKEL retargeted joints (teal spheres + sticks)
  - Joint torque heatmap (LBS-based plasma coloring)
  - Ground reaction forces (GRF arrows)
  - Colorbar

Usage:
    from addb2skel.utils.render import render_subject, load_skel_data, load_b3d_data

    skel_data = load_skel_data(npz_path, gender)
    b3d_data = load_b3d_data(b3d_path, num_frames=skel_data['skin_v'].shape[0])
    skel_data.update(b3d_data)

    render_subject(skel_data, output_path, panels=['skeleton', 'mesh', 'torque'],
                   num_frames=4, title='Subject 01')
"""

import os
import sys
import numpy as np

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Lazy imports for heavy dependencies
# ---------------------------------------------------------------------------
_nimble = None
_SKELInterface = None
_torch = None


def _ensure_nimble():
    global _nimble
    if _nimble is None:
        import nimblephysics as nimble
        _nimble = nimble
    return _nimble


def _ensure_skel():
    global _SKELInterface
    if _SKELInterface is None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from addb2skel.skel_interface import SKELInterface
        _SKELInterface = SKELInterface
    return _SKELInterface


def _ensure_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ===========================================================================
# Constants
# ===========================================================================

# AddB 20-joint parent hierarchy
ADDB_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 11, 16, 17, 18]

# SKEL 24-joint parent hierarchy
SKEL_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]

# AddB joint name -> DOF start, DOF count (37 DOF total, excluding 6 root DOF)
JOINT_DOF_MAPPING = [
    ("ground_pelvis", 0, 6), ("hip_r", 6, 3), ("walker_knee_r", 9, 1),
    ("ankle_r", 10, 1), ("subtalar_r", 11, 1), ("mtp_r", 12, 1),
    ("hip_l", 13, 3), ("walker_knee_l", 16, 1), ("ankle_l", 17, 1),
    ("subtalar_l", 18, 1), ("mtp_l", 19, 1), ("back", 20, 3),
    ("acromial_r", 23, 3), ("elbow_r", 26, 1), ("radioulnar_r", 27, 1),
    ("radius_hand_r", 28, 2), ("acromial_l", 30, 3), ("elbow_l", 33, 1),
    ("radioulnar_l", 34, 1), ("radius_hand_l", 35, 2),
]

# AddB joint -> SKEL joint index mapping (for torque coloring)
ADDB_TO_SKEL_JOINT_MAP = {
    'ground_pelvis': [0], 'hip_r': [1], 'walker_knee_r': [2],
    'ankle_r': [3], 'subtalar_r': [4], 'mtp_r': [5],
    'hip_l': [6], 'walker_knee_l': [7], 'ankle_l': [8],
    'subtalar_l': [9], 'mtp_l': [10],
    'back': [11, 12, 13],
    'acromial_r': [14, 15], 'elbow_r': [16], 'radioulnar_r': [17], 'radius_hand_r': [18],
    'acromial_l': [19, 20], 'elbow_l': [21], 'radioulnar_l': [22], 'radius_hand_l': [23],
}

# Parent fallback for torque inheritance (when a joint has no torque, use parent's)
ADDB_PARENT_FALLBACK = {
    'mtp_r': 'subtalar_r', 'mtp_l': 'subtalar_l',
    'radioulnar_r': 'elbow_r', 'radioulnar_l': 'elbow_l',
    'radius_hand_r': 'elbow_r', 'radius_hand_l': 'elbow_l',
    'subtalar_r': 'ankle_r', 'subtalar_l': 'ankle_l',
    'ankle_r': 'walker_knee_r', 'ankle_l': 'walker_knee_l',
}

# Ankle joints that get boosted torque coloring
ANKLE_BOOST_JOINTS = {'ankle_r', 'ankle_l', 'subtalar_r', 'subtalar_l', 'mtp_r', 'mtp_l'}
ANKLE_BOOST_FACTOR = 2.0

# Plasma colormap control points (5-point, matches matplotlib plasma)
_PLASMA_COLORS = [
    np.array([0.050383, 0.029803, 0.527975]),
    np.array([0.417642, 0.000564, 0.658390]),
    np.array([0.798216, 0.280197, 0.469538]),
    np.array([0.988362, 0.557937, 0.231441]),
    np.array([0.940015, 0.975158, 0.131326]),
]

_GRAY = np.array([0.85, 0.85, 0.85])

# Default SKEL model path
DEFAULT_SKEL_MODEL = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'

# Default B3D and fitted data paths
DEFAULT_B3D_BASE = '/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm'
DEFAULT_FITTED_BASE = '/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/0. Final_fitted_dataset/full_with_arm'


# ===========================================================================
# Colormap
# ===========================================================================

def plasma_color(t):
    """Map t in [0,1] to plasma colormap RGB (numpy array [3])."""
    t = np.clip(t, 0, 1)
    idx = t * 4
    i = int(idx)
    if i >= 4:
        return _PLASMA_COLORS[4].copy()
    frac = idx - i
    return _PLASMA_COLORS[i] * (1 - frac) + _PLASMA_COLORS[i + 1] * frac


# ===========================================================================
# Data loading
# ===========================================================================

def load_skel_data(npz_path, gender, device='cuda', batch_size=10):
    """Load fitted SKEL parameters and run FK to get vertices/joints.

    Args:
        npz_path: Path to skel_params.npz (contains betas, poses, trans).
        gender: 'male' or 'female'.
        device: Torch device.
        batch_size: FK batch size.

    Returns:
        dict with keys: skin_v [T,6890,3], skel_v [T,V_skel,3], joints [T,24,3],
                         skin_f [F,3], skel_f [F,3]  (all numpy or torch cpu tensors)
    """
    torch = _ensure_torch()
    SKELInterface = _ensure_skel()

    data = np.load(npz_path)
    betas = torch.from_numpy(data['betas']).float().to(device)
    poses = torch.from_numpy(data['poses']).float().to(device)
    trans = torch.from_numpy(data['trans']).float().to(device)

    skel = SKELInterface(sex=gender, device=device)
    T = poses.shape[0]
    skin_v_all, skel_v_all, joints_all = [], [], []

    for s in range(0, T, batch_size):
        e = min(s + batch_size, T)
        b = betas.unsqueeze(0).expand(e - s, -1)
        with torch.no_grad():
            sv, j, skv = skel.forward(b, poses[s:e], trans[s:e], return_skeleton=True)
        skin_v_all.append(sv.cpu())
        joints_all.append(j.cpu())
        if skv is not None:
            skel_v_all.append(skv.cpu())

    skin_f = torch.tensor(skel.faces, dtype=torch.long) if isinstance(skel.faces, np.ndarray) else skel.faces
    skel_f = torch.tensor(skel.skel_faces, dtype=torch.long) if isinstance(skel.skel_faces, np.ndarray) else skel.skel_faces

    return {
        'skin_v': torch.cat(skin_v_all, dim=0),
        'skel_v': torch.cat(skel_v_all, dim=0) if skel_v_all else None,
        'joints': torch.cat(joints_all, dim=0),
        'skin_f': skin_f,
        'skel_f': skel_f,
    }


def load_b3d_data(b3d_path, num_frames=None):
    """Load torques, GT joints, GRF, and markers from .b3d file.

    Args:
        b3d_path: Path to .b3d file.
        num_frames: Max frames to read (None = all).

    Returns:
        dict with keys: tau [T,D], gt_joints [T,J,3], grf_data (list of tuples),
                         markers (list of arrays)
    """
    nimble = _ensure_nimble()
    sub = nimble.biomechanics.SubjectOnDisk(b3d_path)
    trial_length = sub.getTrialLength(0)
    actual = trial_length - 1 if num_frames is None else min(num_frames, trial_length - 1)
    skel = sub.readSkel(processingPass=2)
    n_dof = skel.getNumDofs()

    frames = sub.readFrames(
        trial=0, startFrame=1, numFramesToRead=actual,
        includeProcessingPasses=True, includeSensorData=True,
        stride=1, contactThreshold=1.0,
    )

    T = len(frames)
    tau = np.zeros((T, n_dof), dtype=np.float64)
    gt_joints_all = []
    grf_data = []
    markers_all = []

    for t, frame in enumerate(frames):
        # Torques from pass 2 (dynamics)
        pp = frame.processingPasses[2]
        tau_raw = np.array(pp.tau, dtype=np.float64)
        tau[t] = tau_raw[:n_dof] if len(tau_raw) >= n_dof else np.pad(tau_raw, (0, n_dof - len(tau_raw)))

        # GT joint positions from pass 0 (kinematics).
        # Apply X/Z flip (pipeline.py:121-123 convention): AddB has X/Z flipped relative
        # to SKEL frame. The flip aligns relative bone directions, but shifts the world
        # origin — caller should shift all to match SKEL pelvis position (done post-hoc).
        pp0 = frame.processingPasses[0]
        jc_raw = np.array(pp0.jointCenters, dtype=np.float64)
        n_jc = len(jc_raw) // 3
        jc = jc_raw[:n_jc * 3].reshape(n_jc, 3)
        jc[:, 0] = -jc[:, 0]; jc[:, 2] = -jc[:, 2]
        gt_joints_all.append(jc)

        # GRF: flip X/Z for both CoP (position) and force vector
        contact = np.array(pp.contact)
        grf = np.array(pp.groundContactForce).reshape(-1, 3)
        cop = np.array(pp.groundContactCenterOfPressure).reshape(-1, 3)
        grf[:, 0] = -grf[:, 0]; grf[:, 2] = -grf[:, 2]
        cop[:, 0] = -cop[:, 0]; cop[:, 2] = -cop[:, 2]
        grf_data.append((contact, grf, cop))

        # Markers
        marker_obs = frame.markerObservations
        marker_positions = [np.array(pos) for name, pos in marker_obs]
        markers_all.append(np.array(marker_positions) if marker_positions else np.zeros((0, 3)))

    return {
        'tau': tau,
        'gt_joints': np.array(gt_joints_all),
        'grf_data': grf_data,
        'markers': markers_all,
    }


# ===========================================================================
# pyrender core: camera, lights, scene rendering
# ===========================================================================

def create_camera(K4, Rt=None):
    """Create pyrender camera from intrinsics."""
    if Rt is not None:
        cam_R = np.array(Rt[0], dtype=np.float64).copy()
        cam_t = np.array(Rt[1], dtype=np.float64).copy()
        cam_t[0] *= -1
    else:
        cam_R = np.eye(3)
        cam_t = np.zeros(3)
    fx, fy, cx, cy = K4
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = cam_R
    cam_pose[:3, 3] = cam_t
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e12)
    return camera, cam_pose


def create_raymond_lights():
    """Standard 3-point directional lighting (from HSMR)."""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
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
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=0.75),
            matrix=matrix
        ))
    return nodes


def render_scene(meshes_list, cam_t, K4, size, view='front', bg_color=None):
    """Render a list of (trimesh, material_or_None) in one pyrender scene.

    Args:
        meshes_list: List of (trimesh.Trimesh, pyrender.Material or None).
        cam_t: Camera translation [3].
        K4: [fx, fy, cx, cy].
        size: Image size (square).
        view: 'front' or 'side90d'.
        bg_color: Background RGB [3] uint8. Default white.

    Returns:
        np.ndarray [size, size, 3] uint8 image.
    """
    if bg_color is None:
        bg_color = np.ones((size, size, 3), dtype=np.uint8) * 255

    renderer = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size, point_size=1.0)
    camera, cam_pose = create_camera(K4, None)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    for node in create_raymond_lights():
        scene.add_node(node)

    for idx, (tm, mat) in enumerate(meshes_list):
        m = tm.copy()
        if view == 'side90d':
            m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0]))
        elif view == 'side60d':
            m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-60), [0, 1, 0]))
        m.apply_transform(trimesh.transformations.translation_matrix(cam_t))
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
        if mat is not None:
            scene.add(pyrender.Mesh.from_trimesh(m, material=mat), 'm%d' % idx)
        else:
            scene.add(pyrender.Mesh.from_trimesh(m), 'm%d' % idx)

    scene.add(camera, pose=cam_pose)
    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask = rgba.astype(np.float32)[:, :, [-1]] / 255.0
    final = rgba[:, :, :3] * mask + bg_color * (1 - mask)
    renderer.delete()
    return final.astype(np.uint8)


# ===========================================================================
# Geometry helpers
# ===========================================================================

def make_spheres(positions, radius=0.012):
    """Create merged trimesh of spheres at given positions."""
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


def make_sticks(joints, parents, radius=0.005):
    """Create merged trimesh of cylinder sticks connecting joints by parent hierarchy."""
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


def make_grf_arrow(cop_pos, force_vec, scale=0.0009, z_offset=-0.08):
    """Create GRF arrow mesh (shaft + sphere head).

    Args:
        cop_pos: Center of pressure [3].
        force_vec: Ground reaction force [3] (in display coordinates).
        scale: Force-to-length scale factor.
        z_offset: Z offset to bring arrow in front of body.

    Returns:
        trimesh.Trimesh or None if force too small.
    """
    mag = np.linalg.norm(force_vec)
    if mag < 10:
        return None
    direction = force_vec / mag
    arrow_len = mag * scale
    end = cop_pos + direction * arrow_len
    offset = np.array([0.0, 0.0, z_offset])
    cop_front = cop_pos + offset
    end_front = end + offset
    shaft = trimesh.creation.cylinder(radius=0.008, segment=np.array([cop_front, end_front]))
    head = trimesh.creation.uv_sphere(radius=0.016, count=[8, 8])
    head.apply_translation(end_front)
    return trimesh.util.concatenate([shaft, head])


# ===========================================================================
# Materials (standard set)
# ===========================================================================

def _materials():
    """Return dict of standard pyrender materials."""
    return {
        'bone': pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.05, roughnessFactor=0.6, alphaMode='OPAQUE',
            baseColorFactor=(0.945, 0.749, 0.600, 1.0)),
        'skin_trans': pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode='BLEND',
            baseColorFactor=(0.537, 0.765, 0.922, 0.55)),
        'skin_very_trans': pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode='BLEND',
            baseColorFactor=(0.537, 0.765, 0.922, 0.30)),
        'joint_addb': pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE',
            baseColorFactor=(1.0, 0.5, 0.0, 1.0)),
        'bone_stick': pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE',
            baseColorFactor=(0.5, 0.5, 0.5, 1.0)),
        'joint_skel': pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE',
            baseColorFactor=(0.2, 0.75, 0.5, 1.0)),
        'stick_skel': pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE',
            baseColorFactor=(0.15, 0.6, 0.4, 1.0)),
        'grf': pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE',
            baseColorFactor=(0.15, 0.85, 0.15, 1.0)),
    }


# ===========================================================================
# LBS torque coloring
# ===========================================================================

def load_lbs_weights(skel_model_path=None, gender='male'):
    """Load SKEL LBS skinning weights [V_skel, 24] for skeleton mesh.

    Returns:
        (dominant_joints [V], weights [V, 24])
    """
    import pickle
    if skel_model_path is None:
        skel_model_path = DEFAULT_SKEL_MODEL
    pkl_path = os.path.join(skel_model_path, f'skel_{gender}.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    weights = data['skel_weights'].toarray()
    dominant_joints = np.argmax(weights, axis=1)
    return dominant_joints, weights


def compute_lbs_torque_colors(tau_frame, lbs_weights, torque_percentiles=None):
    """Compute per-vertex RGB colors from torque via LBS blending.

    Args:
        tau_frame: Torque vector [D] for one frame.
        lbs_weights: LBS weight matrix [V, 24].
        torque_percentiles: Sorted log-torque values for percentile mapping.
            If None, uses per-frame max normalization instead.

    Returns:
        vertex_colors: [V, 3] RGB in [0, 1].
    """
    # Compute per-SKEL-joint torque magnitude
    skel_idx_to_torque = np.zeros(24)
    joint_torques = {}
    for jn, ds, nd in JOINT_DOF_MAPPING:
        mag = float(np.linalg.norm(tau_frame[ds:ds + nd]))
        joint_torques[jn] = mag
        if jn not in ADDB_TO_SKEL_JOINT_MAP:
            continue
        for si in ADDB_TO_SKEL_JOINT_MAP[jn]:
            if mag > skel_idx_to_torque[si]:
                skel_idx_to_torque[si] = mag

    # Parent fallback
    for addb_name, parent_name in ADDB_PARENT_FALLBACK.items():
        if addb_name not in ADDB_TO_SKEL_JOINT_MAP:
            continue
        for si in ADDB_TO_SKEL_JOINT_MAP[addb_name]:
            if skel_idx_to_torque[si] == 0:
                pn = parent_name
                pt = joint_torques.get(pn, 0)
                while pt == 0 and pn in ADDB_PARENT_FALLBACK:
                    pn = ADDB_PARENT_FALLBACK[pn]
                    pt = joint_torques.get(pn, 0)
                if pt > 0:
                    skel_idx_to_torque[si] = pt

    # Ankle boost
    for jn in ANKLE_BOOST_JOINTS:
        if jn not in ADDB_TO_SKEL_JOINT_MAP:
            continue
        for si in ADDB_TO_SKEL_JOINT_MAP[jn]:
            skel_idx_to_torque[si] *= ANKLE_BOOST_FACTOR

    # Map each joint to color
    if torque_percentiles is not None and len(torque_percentiles) > 0:
        # Percentile-based mapping
        def _color(mag):
            if mag < 0.05:
                return _GRAY
            log_mag = np.log10(mag)
            idx = np.searchsorted(torque_percentiles, log_mag)
            t = idx / len(torque_percentiles)
            t = t ** 2.0  # power curve: stretch dark region
            return plasma_color(t)
    else:
        # Per-frame max normalization (log scale)
        max_t = max(skel_idx_to_torque.max(), 1.0)
        log_max = np.log10(max_t)

        def _color(mag):
            if mag < 0.05:
                return _GRAY
            t = np.clip(np.log10(mag) / log_max, 0, 1)
            return plasma_color(t)

    joint_colors = np.array([_color(t) for t in skel_idx_to_torque])
    return np.clip(lbs_weights @ joint_colors, 0, 1)


def build_torque_percentiles(tau_all, frame_indices):
    """Build sorted log-torque distribution from selected frames for percentile mapping.

    Args:
        tau_all: [T, D] torque array.
        frame_indices: List of frame indices to include.

    Returns:
        sorted_log_torques: 1D sorted array of log10(torque) values.
    """
    all_torques = []
    for fi in frame_indices:
        tau = tau_all[min(fi, len(tau_all) - 1)]
        for jn, ds, nd in JOINT_DOF_MAPPING:
            if jn == 'ground_pelvis':
                continue
            mag = float(np.linalg.norm(tau[ds:ds + nd]))
            if mag > 0.05:
                all_torques.append(mag)
    if not all_torques:
        return np.array([])
    return np.sort(np.log10(np.array(all_torques)))


# ===========================================================================
# Camera setup
# ===========================================================================

def compute_camera(skel_data, frame_indices, size=768, focal=5000):
    """Compute camera translation and prep function for centering meshes.

    Centers all selected frames together. Flips Y for pyrender's 180°-X convention.

    Args:
        skel_data: dict from load_skel_data().
        frame_indices: List of frame indices.
        size: Image size.
        focal: Focal length.

    Returns:
        cam_t: [3] camera translation.
        K4: [fx, fy, cx, cy].
        global_center: [3] center of all frames' skin vertices.
    """
    torch = _ensure_torch()
    all_skin = []
    for fi in frame_indices:
        v = skel_data['skin_v'][fi]
        if hasattr(v, 'numpy'):
            v = v.numpy()
        all_skin.append(v)
    all_skin = np.concatenate(all_skin, axis=0)
    gc = all_skin.mean(axis=0)
    ac = all_skin - gc
    ac[:, 1] *= -1
    # Use max of X extent (horizontal walking spread) and Y extent (height) for zoom
    # so all 4 frames fit horizontally
    y_extent = ac[:, 1].max() - ac[:, 1].min()
    x_extent = ac[:, 0].max() - ac[:, 0].min()
    max_extent = max(y_extent, x_extent)
    tz = focal * max_extent / (size * 0.80)
    cam_t = np.array([0.0, 0.0, tz])
    K4 = [focal, focal, size // 2, size // 2]
    return cam_t, K4, gc


def prep_vertices(v, global_center):
    """Center vertices and flip Y for pyrender display.

    Args:
        v: [N, 3] vertices (numpy or torch tensor).
        global_center: [3] from compute_camera().

    Returns:
        [N, 3] numpy array ready for rendering.
    """
    if hasattr(v, 'numpy'):
        v = v.numpy()
    out = v - global_center
    out[:, 1] *= -1
    return out


# ===========================================================================
# Panel rendering functions
# ===========================================================================

def _to_numpy(arr):
    """Convert torch tensor or numpy to numpy."""
    if hasattr(arr, 'numpy'):
        return arr.numpy()
    return arr


def render_panel_skeleton(skel_data, frame_indices, cam_t, K4, gc, size,
                          view='front', show_gt_joints=True):
    """Panel: Bone mesh + optional GT joint spheres/sticks.

    Args:
        skel_data: dict from load_skel_data() + optional 'gt_joints' key.
        frame_indices: List of frame indices.
        cam_t, K4, gc: From compute_camera().
        size: Image size.
        view: 'front' or 'side90d'.
        show_gt_joints: Whether to overlay AddB GT joints.

    Returns:
        [size, size, 3] uint8 image.
    """
    mat = _materials()
    sf_bone = _to_numpy(skel_data['skel_f'])
    meshes = []
    for fi in frame_indices:
        sv = prep_vertices(skel_data['skel_v'][fi], gc)
        meshes.append((trimesh.Trimesh(sv, sf_bone, process=False), mat['bone']))
        if show_gt_joints and 'gt_joints' in skel_data and skel_data['gt_joints'] is not None:
            gt = skel_data['gt_joints']
            fi_gt = min(fi, len(gt) - 1)
            j = prep_vertices(gt[fi_gt].copy(), gc)
            sm = make_spheres(j[:20])
            if sm:
                meshes.append((sm, mat['joint_addb']))
            bm = make_sticks(j[:20], ADDB_PARENTS)
            if bm:
                meshes.append((bm, mat['bone_stick']))
    return render_scene(meshes, cam_t, K4, size, view)


def render_panel_mesh(skel_data, frame_indices, cam_t, K4, gc, size,
                      view='front', show_skel_joints=False):
    """Panel: Bone mesh + translucent skin + optional SKEL joints.

    Args:
        show_skel_joints: Whether to overlay SKEL retargeted joints.
    """
    mat = _materials()
    sf_bone = _to_numpy(skel_data['skel_f'])
    sf_skin = _to_numpy(skel_data['skin_f'])
    meshes = []
    for fi in frame_indices:
        sv = prep_vertices(skel_data['skel_v'][fi], gc)
        meshes.append((trimesh.Trimesh(sv, sf_bone, process=False), mat['bone']))
        if show_skel_joints:
            j = prep_vertices(skel_data['joints'][fi], gc)
            sm = make_spheres(j[:24])
            if sm:
                meshes.append((sm, mat['joint_skel']))
            bm = make_sticks(j[:24], SKEL_PARENTS)
            if bm:
                meshes.append((bm, mat['stick_skel']))
        skv = prep_vertices(skel_data['skin_v'][fi], gc)
        meshes.append((trimesh.Trimesh(skv, sf_skin, process=False), mat['skin_trans']))
    return render_scene(meshes, cam_t, K4, size, view)


def render_panel_torque(skel_data, frame_indices, cam_t, K4, gc, size,
                        view='front', lbs_weights=None, torque_percentiles=None,
                        show_grf=True, show_skin=True):
    """Panel: Torque-colored skeleton + optional GRF arrows + optional skin.

    Args:
        lbs_weights: [V, 24] LBS weight matrix. If None, loads from default path.
        torque_percentiles: From build_torque_percentiles(). If None, uses per-frame max.
        show_grf: Whether to render GRF arrows.
        show_skin: Whether to overlay translucent skin.
    """
    mat = _materials()
    sf_bone = _to_numpy(skel_data['skel_f'])
    sf_skin = _to_numpy(skel_data['skin_f'])
    tau_all = skel_data.get('tau')
    grf_data = skel_data.get('grf_data')

    if lbs_weights is None:
        gender = skel_data.get('gender', 'male')
        _, lbs_weights = load_lbs_weights(gender=gender)

    meshes = []

    # GRF arrows first (behind skeleton)
    if show_grf and grf_data is not None:
        for fi in frame_indices:
            fi_grf = min(fi, len(grf_data) - 1)
            contact, grf, cop = grf_data[fi_grf]
            for body_idx in range(len(contact)):
                if contact[body_idx] > 0 and body_idx < len(grf):
                    cop_centered = prep_vertices(cop[body_idx:body_idx + 1].copy(), gc)[0]
                    grf_vec = grf[body_idx].copy()
                    grf_vec[1] = -grf_vec[1]  # flip Y for display
                    arrow = make_grf_arrow(cop_centered, grf_vec)
                    if arrow is not None:
                        meshes.append((arrow, mat['grf']))

    # Torque-colored skeleton + skin
    for fi in frame_indices:
        sv = prep_vertices(skel_data['skel_v'][fi], gc)
        if tau_all is not None:
            tau = tau_all[min(fi, len(tau_all) - 1)]
            vc = compute_lbs_torque_colors(tau, lbs_weights, torque_percentiles)
            colors = (np.column_stack([vc, np.ones(len(vc))]) * 255).astype(np.uint8)
            meshes.append((trimesh.Trimesh(sv, sf_bone, vertex_colors=colors, process=False), None))
        else:
            meshes.append((trimesh.Trimesh(sv, sf_bone, process=False), mat['bone']))

        if show_skin:
            skv = prep_vertices(skel_data['skin_v'][fi], gc)
            meshes.append((trimesh.Trimesh(skv, sf_skin, process=False), mat['skin_trans']))

    return render_scene(meshes, cam_t, K4, size, view)


# ===========================================================================
# Colorbar
# ===========================================================================

def make_colorbar(height, width=100, torque_values=None, torque_percentiles=None):
    """Create vertical plasma colorbar image.

    Args:
        height: Colorbar image height (matches panel height).
        width: Colorbar width in pixels.
        torque_values: List of all torque magnitudes (for percentile tick labels).
        torque_percentiles: Sorted log-torques (for percentile positioning).

    Returns:
        [height, width, 3] uint8 numpy array.
    """
    cbar = np.ones((height, width, 3), dtype=np.uint8) * 255
    bar_top = 60
    bar_bot = height - 60
    bar_h = bar_bot - bar_top
    bar_left = 12
    bar_right = 30

    # Draw gradient
    for y in range(bar_h):
        t_norm = 1.0 - y / bar_h
        color = plasma_color(t_norm)
        rgb = tuple(int(c * 255) for c in color)
        cbar[bar_top + y, bar_left:bar_right] = rgb

    cbar_pil = Image.fromarray(cbar)
    draw = ImageDraw.Draw(cbar_pil)
    draw.rectangle([bar_left, bar_top, bar_right - 1, bar_bot - 1], outline=(80, 80, 80), width=1)

    try:
        font_cb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        font_cb_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    except (OSError, IOError):
        font_cb = ImageFont.load_default()
        font_cb_title = font_cb

    # Tick labels
    if torque_values is not None and torque_percentiles is not None and len(torque_values) > 0:
        tv = np.array(torque_values)
        for p in [0, 25, 50, 75, 100]:
            val = float(np.percentile(tv, p)) if p > 0 and p < 100 else (tv.min() if p == 0 else tv.max())
            log_val = np.log10(max(val, 0.05))
            rank = np.searchsorted(torque_percentiles, log_val) / len(torque_percentiles)
            y_pos = bar_bot - int(rank * bar_h)
            draw.line([(bar_right, y_pos), (bar_right + 5, y_pos)], fill=(80, 80, 80), width=1)
            label = f'{val:.0f}' if val >= 1 else f'{val:.1f}'
            draw.text((bar_right + 7, y_pos - 10), label, fill=(50, 50, 50), font=font_cb)

    # Title
    draw.text((bar_left - 2, bar_top - 45), 'Torque', fill=(50, 50, 50), font=font_cb_title)
    draw.text((bar_left - 2, bar_top - 25), '(Nm)', fill=(80, 80, 80), font=font_cb)

    return np.array(cbar_pil)


# ===========================================================================
# Frame selection helpers
# ===========================================================================

def select_frames(skel_data, num_frames=4, mode='evenly_spaced', grf_only=False):
    """Select frame indices for visualization.

    Args:
        skel_data: dict from load_skel_data() + optional 'grf_data'.
        num_frames: Number of frames to select.
        mode: 'evenly_spaced' (default) — evenly spaced across mid-range.
        grf_only: If True, only select frames with GRF contact.

    Returns:
        List of frame indices.
    """
    T = skel_data['skin_v'].shape[0]

    if grf_only and 'grf_data' in skel_data and skel_data['grf_data'] is not None:
        contact_frames = []
        for fi, (contact, _, _) in enumerate(skel_data['grf_data']):
            if np.any(contact > 0):
                contact_frames.append(fi)
        if contact_frames:
            pick = np.linspace(0, len(contact_frames) - 1, num_frames, dtype=int)
            return [contact_frames[i] for i in pick]

    # Default: evenly spaced across 10%-90% range
    return list(np.linspace(T // 10, T - T // 10, num_frames, dtype=int))


# ===========================================================================
# Grid composition with labels
# ===========================================================================

def compose_grid(panels, panel_labels=None, title=None, colorbar=None):
    """Compose panels into a horizontal grid with optional labels and title.

    Args:
        panels: List of [H, W, 3] uint8 images.
        panel_labels: List of label strings (same length as panels).
        title: Optional title string at top-left.
        colorbar: Optional colorbar image [H, W_cb, 3] to append on right.

    Returns:
        [H, W_total, 3] uint8 composed image.
    """
    if not panels:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    grid = np.concatenate(panels, axis=1)
    if colorbar is not None:
        grid = np.concatenate([grid, colorbar], axis=1)

    pil = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil)

    try:
        font_panel = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 36)
        font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 36)
    except (OSError, IOError):
        font_panel = ImageFont.load_default()
        font_title = font_panel

    S = panels[0].shape[1]  # panel width

    # Panel labels at bottom
    if panel_labels:
        for i, text in enumerate(panel_labels):
            if text is None:
                continue
            x, y = i * S + 10, panels[0].shape[0] - 48
            bb = draw.textbbox((x, y), text, font=font_panel)
            draw.rectangle([bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3], fill=(30, 30, 30, 200))
            draw.text((x, y), text, fill=(255, 255, 255), font=font_panel)

    # Title at top-left
    if title:
        bb = draw.textbbox((10, 8), title, font=font_title)
        draw.rectangle([bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3], fill=(255, 255, 255, 230))
        draw.text((10, 8), title, fill=(50, 50, 50), font=font_title)

    return np.array(pil)


# ===========================================================================
# High-level API
# ===========================================================================

def render_subject(
    skel_data,
    output_path,
    panels=('skeleton', 'mesh', 'torque'),
    num_frames=4,
    size=768,
    focal=5000,
    view='front',
    title=None,
    show_colorbar=True,
    grf_only=False,
    skel_model_path=None,
):
    """Render a multi-panel visualization for one subject and save as PNG.

    This is the main entry point. Pass `skel_data` from load_skel_data() merged
    with load_b3d_data() results.

    Args:
        skel_data: dict with keys from load_skel_data() + optional b3d keys
            (tau, gt_joints, grf_data). Also supports optional 'gender' key.
        output_path: Path to save PNG.
        panels: Tuple of panel names. Supported:
            'skeleton' — bone mesh + GT joints
            'mesh' — bone mesh + skin + SKEL joints
            'torque' — LBS torque heatmap + GRF + skin
        num_frames: Number of frames to overlay per panel.
        size: Panel size in pixels (square).
        focal: Camera focal length.
        view: 'front', 'side90d', or 'side60d'.
        title: Optional title string.
        show_colorbar: Whether to add torque colorbar (only if 'torque' panel present).
        grf_only: If True, select only frames with GRF contact.
        skel_model_path: Path to SKEL model dir (for LBS weights).

    Returns:
        output_path (for chaining).
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    frame_indices = select_frames(skel_data, num_frames, grf_only=grf_only)
    cam_t, K4, gc = compute_camera(skel_data, frame_indices, size, focal)

    # Precompute LBS/torque data if needed
    lbs_w = None
    torque_pct = None
    all_torque_values = None
    tau_all = skel_data.get('tau')
    if 'torque' in panels and tau_all is not None:
        gender = skel_data.get('gender', 'male')
        _, lbs_w = load_lbs_weights(skel_model_path, gender)
        torque_pct = build_torque_percentiles(tau_all, frame_indices)
        # Collect all torque values for colorbar
        all_torque_values = []
        for fi in frame_indices:
            tau = tau_all[min(fi, len(tau_all) - 1)]
            for jn, ds, nd in JOINT_DOF_MAPPING:
                if jn == 'ground_pelvis':
                    continue
                mag = float(np.linalg.norm(tau[ds:ds + nd]))
                if mag > 0.05:
                    all_torque_values.append(mag)

    # Render each panel
    panel_images = []
    panel_labels = []
    label_map = {
        'skeleton': 'Skeleton + AddB',
        'mesh': 'Mesh + Skeleton',
        'torque': 'Joint Torque + GRF',
    }
    for pname in panels:
        if pname == 'skeleton':
            img = render_panel_skeleton(skel_data, frame_indices, cam_t, K4, gc, size, view)
        elif pname == 'mesh':
            img = render_panel_mesh(skel_data, frame_indices, cam_t, K4, gc, size, view,
                                    show_skel_joints=True)
        elif pname == 'torque':
            img = render_panel_torque(skel_data, frame_indices, cam_t, K4, gc, size, view,
                                      lbs_weights=lbs_w, torque_percentiles=torque_pct)
        else:
            raise ValueError(f"Unknown panel: {pname}")
        panel_images.append(img)
        panel_labels.append(label_map.get(pname, pname))

    # Colorbar
    colorbar = None
    if show_colorbar and 'torque' in panels and all_torque_values:
        colorbar = make_colorbar(size, 100, all_torque_values, torque_pct)

    # Compose
    grid = compose_grid(panel_images, panel_labels, title, colorbar)
    Image.fromarray(grid).save(output_path, quality=95)
    print(f"Saved: {output_path}")
    return output_path


# ===========================================================================
# Convenience: render from file paths
# ===========================================================================

def _align_addb_to_skel_pelvis_v2(skel_data):
    """Shift AddB joints and CoP by per-frame shift = SKEL_pelvis - AddB_pelvis."""
    if 'gt_joints' not in skel_data or skel_data['gt_joints'] is None:
        return
    skel_j = skel_data.get('joints')
    if skel_j is None:
        return
    if hasattr(skel_j, 'numpy'):
        skel_j = skel_j.numpy()
    gt = skel_data['gt_joints']
    T = min(len(gt), len(skel_j))
    shifts = skel_j[:T, 0] - gt[:T, 0]  # [T, 3]
    # Apply to gt joints
    skel_data['gt_joints'] = gt[:T] + shifts[:, None, :]  # broadcast over joints
    # Apply to CoP
    grf_data = skel_data.get('grf_data')
    if grf_data is not None:
        new_grf = []
        for t in range(min(len(grf_data), T)):
            contact, grf, cop = grf_data[t]
            cop_shifted = cop + shifts[t][None, :]  # broadcast over contacts
            new_grf.append((contact, grf, cop_shifted))
        skel_data['grf_data'] = new_grf


def render_from_paths(
    npz_path,
    b3d_path,
    gender,
    output_path,
    panels=('skeleton', 'mesh', 'torque'),
    num_frames=4,
    size=768,
    title=None,
    view='front',
    grf_only=False,
):
    """Load data from file paths and render visualization.

    Args:
        npz_path: Path to skel_params.npz.
        b3d_path: Path to .b3d file (or None to skip torque/GRF/GT joints).
        gender: 'male' or 'female'.
        output_path: Where to save PNG.
        panels, num_frames, size, title, view, grf_only: See render_subject().

    Returns:
        output_path.
    """
    print(f"Loading SKEL data from {npz_path}...")
    skel_data = load_skel_data(npz_path, gender)
    skel_data['gender'] = gender
    T = skel_data['skin_v'].shape[0]
    print(f"  Frames: {T}")

    if b3d_path and os.path.exists(b3d_path):
        print(f"Loading b3d data from {b3d_path}...")
        b3d_data = load_b3d_data(b3d_path, num_frames=T)
        skel_data.update(b3d_data)
        print(f"  Torque: {b3d_data['tau'].shape}, GT joints: {b3d_data['gt_joints'].shape}")
        # Align AddB pelvis to SKEL pelvis per-frame (post-flip coord alignment)
        _align_addb_to_skel_pelvis_v2(skel_data)
        print(f"  AddB aligned to SKEL pelvis frame")
    else:
        if 'torque' in panels:
            print("  Warning: no b3d file, torque panel will show plain bone mesh")

    return render_subject(skel_data, output_path, panels=panels, num_frames=num_frames,
                          size=size, title=title, view=view, grf_only=grf_only)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Render addb2skel visualization')
    parser.add_argument('--npz', required=True, help='Path to skel_params.npz')
    parser.add_argument('--b3d', default=None, help='Path to .b3d file')
    parser.add_argument('--gender', default='male', choices=['male', 'female'])
    parser.add_argument('--output', '-o', required=True, help='Output PNG path')
    parser.add_argument('--panels', nargs='+', default=['skeleton', 'mesh', 'torque'],
                        choices=['skeleton', 'mesh', 'torque'])
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--view', default='front', choices=['front', 'side90d', 'side60d'])
    parser.add_argument('--title', default=None)
    parser.add_argument('--grf_only', action='store_true')
    args = parser.parse_args()

    render_from_paths(
        npz_path=args.npz,
        b3d_path=args.b3d,
        gender=args.gender,
        output_path=args.output,
        panels=args.panels,
        num_frames=args.num_frames,
        size=args.size,
        title=args.title,
        view=args.view,
        grf_only=args.grf_only,
    )
