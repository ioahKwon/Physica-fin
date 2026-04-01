"""
I/O utilities for loading and saving data.
"""

import os
import warnings
from fractions import Fraction
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from scipy.signal import resample_poly


def _compute_resample_params(
    original_fps: float,
    target_fps: float,
    tolerance: float = 0.05,
) -> Tuple[int, int, float]:
    """Compute up/down factors for scipy.signal.resample_poly.

    Returns (up, down, actual_fps). Returns (1, 1, original_fps) if
    resampling is unnecessary (ratio within tolerance of 1.0).
    """
    ratio = target_fps / original_fps
    if abs(ratio - 1.0) < tolerance:
        return 1, 1, original_fps

    # Round to integers to avoid large up/down from floating-point fps
    # (e.g. 100.1Hz -> 100, avoids up=401/down=669)
    frac = Fraction(round(target_fps), round(original_fps))

    up = frac.numerator
    down = frac.denominator
    actual_fps = original_fps * up / down
    return up, down, actual_fps


def _resample_array(
    data: np.ndarray, up: int, down: int, axis: int = 0,
) -> np.ndarray:
    """Anti-aliased polyphase resampling (Kaiser window, beta=5.0)."""
    if up == 1 and down == 1:
        return data
    return resample_poly(data, up, down, axis=axis).astype(data.dtype)


def _resample_grf_mask(
    mask: np.ndarray, T_in: int, T_out: int,
) -> np.ndarray:
    """Resample boolean GRF validity mask via nearest-neighbor."""
    if T_in == T_out:
        return mask
    src = np.round(
        np.arange(T_out) * (T_in - 1) / max(T_out - 1, 1)
    ).astype(int)
    src = np.clip(src, 0, T_in - 1)
    return mask[src]


def load_b3d(
    b3d_path: str,
    num_frames: Optional[int] = None,
    start_frame: int = 0,
    processing_pass: int = 0,
    load_markers: bool = False,
    target_fps: Optional[float] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Load AddBiomechanics .b3d file.

    Args:
        b3d_path: Path to .b3d file.
        num_frames: Number of frames to load. None for all.
        start_frame: Starting frame index.
        processing_pass: Processing pass index (0 for kinematics, 1 for dynamics).
        load_markers: If True, also extract marker observations and markersMap.
                      Adds 'markers_map' and 'marker_observations' to metadata.
        target_fps: If set, resample frames to this exact frame rate using
                    scipy.signal.resample_poly with anti-aliasing.
                    E.g. target_fps=60.0 for 60Hz output. None = no resampling.

    Returns:
        joints: Joint positions [T, N_joints, 3] in meters.
        joint_names: List of joint names.
        metadata: Dictionary with subject info (height, mass, sex, etc.).
                  If load_markers=True, also contains:
                  - 'markers_map': osim.markersMap (nimblephysics object)
                  - 'marker_observations': List[Dict[str, np.ndarray]] per frame
                  Always contains:
                  - 'grf_valid_mask': bool array [T], True if GRF is reliable
                  - 'grf_n_valid': number of valid GRF frames
                  - 'grf_n_missing': number of missing GRF frames
    """
    try:
        import nimblephysics as nimble
    except ImportError:
        raise ImportError(
            "nimblephysics is required to read .b3d files. "
            "Install: pip install nimblephysics"
        )

    # Load subject
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # Get skeleton
    skel = subject.readSkel(0, ignoreGeometry=True)

    # Determine frame rate and resampling parameters
    timestep = subject.getTrialTimestep(0)  # seconds per frame
    original_fps = 1.0 / timestep if timestep > 0 else 100.0

    if target_fps is not None:
        up, down, actual_fps = _compute_resample_params(original_fps, target_fps)
        needs_resample = (up != 1 or down != 1)
    else:
        up, down, actual_fps = 1, 1, original_fps
        needs_resample = False

    # Determine frame range
    trial_length = subject.getTrialLength(0)
    end_frame = start_frame + num_frames if num_frames else trial_length
    end_frame = min(end_frame, trial_length)
    frames_to_read = end_frame - start_frame

    # GRF reliability filtering via getMissingGRF
    missing_grf_all = subject.getMissingGRF(0)
    notMissing = nimble.biomechanics.MissingGRFReason.notMissingGRF

    # Load OpenSim file for markers if requested
    markers_map = None
    addb_skel = None  # Skeleton from osim.markersMap body nodes (for Tier 2.5b)
    if load_markers:
        try:
            osim = subject.readOpenSimFile(
                processingPass=processing_pass, ignoreGeometry=True
            )
            markers_map = osim.markersMap
            # Extract the skeleton that markers_map body_nodes belong to
            # (different from readSkel; setPositions on this skel moves body_nodes)
            for _, val in markers_map.items():
                try:
                    body_node, _ = val
                    addb_skel = body_node.getSkeleton()
                    break
                except Exception:
                    pass
        except Exception:
            markers_map = None

    # Read all frames at full resolution (stride=1) for proper resampling
    trial = subject.readFrames(
        trial=0,
        startFrame=start_frame,
        numFramesToRead=frames_to_read,
        stride=1,
    )
    T_raw = len(trial)

    # Get joint names from skeleton using correct nimblephysics API
    # Build mapping from joint centers to joint names
    pp_idx = min(processing_pass, len(trial[0].processingPasses) - 1)
    pp = trial[0].processingPasses[pp_idx]

    # Set skeleton positions
    pos = np.asarray(pp.pos, dtype=np.float32)
    if pos.size == skel.getNumDofs():
        skel.setPositions(pos)

    # Get world joint positions and names
    world_joints = []
    for i in range(skel.getNumJoints()):
        joint = skel.getJoint(i)
        world = joint.getChildBodyNode().getWorldTransform().translation()
        joint_name = joint.getName()
        world_joints.append((joint_name, world))

    # Use jointCenters from processing pass and match to joint names
    joint_centers_first = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
    num_joints = joint_centers_first.shape[0]

    joint_names = []
    for center in joint_centers_first:
        dists = [np.linalg.norm(center - w) for _, w in world_joints]
        best = int(np.argmin(dists))
        joint_names.append(world_joints[best][0])

    # Extract joint positions (and optionally markers) for all frames
    joints_raw = np.zeros((T_raw, num_joints, 3))
    marker_obs_raw = [] if load_markers else None
    addb_poses_raw = [] if load_markers else None

    for t, frame in enumerate(trial):
        pp = frame.processingPasses[pp_idx]
        joint_centers = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
        joints_raw[t] = joint_centers

        if load_markers:
            marker_dict = {}
            try:
                for mname, mpos in frame.markerObservations:
                    mname = mname.strip()  # strip \r\n from Windows-style CSV marker names
                    marker_dict[mname] = np.array(
                        [float(mpos[0]), float(mpos[1]), float(mpos[2])],
                        dtype=np.float32
                    )
            except Exception:
                pass
            marker_obs_raw.append(marker_dict)
            addb_poses_raw.append(np.asarray(pp.pos, dtype=np.float32))

    # Build GRF validity mask at full resolution
    grf_mask_raw = np.zeros(T_raw, dtype=bool)
    for t in range(T_raw):
        frame_idx = start_frame + t
        if frame_idx < len(missing_grf_all):
            grf_mask_raw[t] = (missing_grf_all[frame_idx] == notMissing)

    # Apply anti-aliased resampling if needed
    if needs_resample:
        joints = _resample_array(joints_raw, up, down, axis=0)
        T = joints.shape[0]
        grf_valid_mask = _resample_grf_mask(grf_mask_raw, T_raw, T)

        if load_markers and addb_poses_raw:
            poses_arr = np.stack(addb_poses_raw, axis=0)  # [T_raw, N_dof]
            poses_resampled = _resample_array(poses_arr, up, down, axis=0)
            addb_poses = [poses_resampled[t] for t in range(T)]
        else:
            addb_poses = addb_poses_raw

        if load_markers and marker_obs_raw:
            # Nearest-neighbor mapping for marker observation dicts
            src_idx = np.round(
                np.arange(T) * (T_raw - 1) / max(T - 1, 1)
            ).astype(int)
            src_idx = np.clip(src_idx, 0, T_raw - 1)
            marker_observations = [marker_obs_raw[i] for i in src_idx]
        else:
            marker_observations = marker_obs_raw
    else:
        joints = joints_raw
        T = T_raw
        grf_valid_mask = grf_mask_raw
        marker_observations = marker_obs_raw
        addb_poses = addb_poses_raw

    n_valid = int(grf_valid_mask.sum())
    n_missing = T - n_valid

    # Get subject info
    metadata = {
        'height_m': subject.getHeightM(),
        'mass_kg': subject.getMassKg(),
        'sex': subject.getBiologicalSex(),
        'age': subject.getAgeYears() if hasattr(subject, 'getAgeYears') else None,
        'num_frames': T,
        'num_joints': num_joints,
        'grf_valid_mask': grf_valid_mask,
        'grf_n_valid': n_valid,
        'grf_n_missing': n_missing,
        'original_fps': original_fps,
        'fps': actual_fps,
        'resample_up': up,
        'resample_down': down,
    }

    if load_markers:
        metadata['markers_map'] = markers_map
        metadata['marker_observations'] = marker_observations
        metadata['addb_poses'] = addb_poses   # List[np.ndarray[n_dofs]] per frame
        metadata['addb_skel'] = addb_skel    # nimble Skeleton (markers_map body_node's skel)

    return joints, joint_names, metadata


def save_npy(data: np.ndarray, path: str):
    """Save numpy array to .npy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)


def load_npy(path: str) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(path)


def save_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str,
    vertex_colors: Optional[np.ndarray] = None,
):
    """
    Save mesh to OBJ file.

    Args:
        vertices: Vertex positions [V, 3].
        faces: Face indices [F, 3] (0-indexed).
        path: Output path.
        vertex_colors: Optional vertex colors [V, 3] in [0, 1].
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        for i, v in enumerate(vertices):
            if vertex_colors is not None:
                c = vertex_colors[i]
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n')
            else:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

        for face in faces:
            # OBJ uses 1-indexed faces
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def save_npz(
    path: str,
    **arrays: np.ndarray,
):
    """Save multiple arrays to .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load arrays from .npz file."""
    return dict(np.load(path))
