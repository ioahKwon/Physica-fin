"""
Configuration and constants for the addb2skel pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
import torch


# =============================================================================
# Path Configuration
# =============================================================================

SKEL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'
ADDB_DATA_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/AddB'
OUTPUT_PATH = '/egr/research-zijunlab/kwonjoon/03_Output/addb2skel'


# =============================================================================
# SKEL Model Constants
# =============================================================================

SKEL_NUM_JOINTS = 24
SKEL_NUM_BETAS = 10
SKEL_NUM_POSE_DOF = 46  # Euler angles for all DOFs
SKEL_NUM_VERTICES = 6890  # Same as SMPL

# Scapula DOF indices in SKEL pose vector
SCAPULA_DOF_INDICES = {
    'right': {
        'abduction': 26,
        'elevation': 27,
        'upward_rot': 28,
    },
    'left': {
        'abduction': 36,
        'elevation': 37,
        'upward_rot': 38,
    },
}

# Humerus (shoulder) DOF indices
HUMERUS_DOF_INDICES = {
    'right': [29, 30, 31],  # shoulder_x, y, z
    'left': [39, 40, 41],
}

# Spine DOF indices (from working compare_smpl_skel.py)
# DOF 17-19: lumbar, DOF 20-22: thorax
# Head DOF (23-25) excluded — driven by marker loss instead of regularized to zero
SPINE_DOF_INDICES = list(range(17, 23))  # lumbar + thorax only

# Arm DOF indices (elbow + wrist — tend to over-flex without strong regularization)
# DOF 32: elbow_r, 33: pro_sup_r, 34-35: wrist_r
# DOF 42: elbow_l, 43: pro_sup_l, 44-45: wrist_l
ARM_DOF_INDICES = [32, 33, 34, 35, 42, 43, 44, 45]

# Scapula DOF bounds (radians)
SCAPULA_DOF_BOUNDS = (-0.5, 0.5)

# Per-joint weights for optimization (critical for good results!)
# Based on compare_smpl_skel.py: shoulders 10x, spine 5x, pelvis/femurs 2x
# Tuned to reduce high-error joints (acromial, walker_knee)
#
# NOTE: acromial (AddB surface landmark) → scapula (SKEL lateral shoulder)
# Scapula is more lateral than humerus, providing better shoulder width match.
# Humerus (glenohumeral) is not directly used in joint loss.
SKEL_JOINT_WEIGHTS = {
    'pelvis': 20.0,      # ROOT - highest weight! Must anchor the skeleton
    'femur_r': 5.0,      # hip
    'femur_l': 5.0,      # hip
    'tibia_r': 3.0,      # knee
    'tibia_l': 3.0,      # knee
    'talus_r': 10.0,     # ankle — GRF critical (aggressive foot fitting)
    'talus_l': 10.0,     # ankle — GRF critical
    'calcn_r': 10.0,     # heel — heel strike position (aggressive)
    'calcn_l': 10.0,     # heel — heel strike position
    'toes_r': 10.0,      # toes — toe push-off position (aggressive)
    'toes_l': 10.0,      # toes — toe push-off position
    'lumbar': 15.0,      # spine - very high weight
    'thorax': 5.0,       # spine
    'head': 1.0,
    'scapula_r': 10.0,   # acromial→scapula: HIGH weight for shoulder width
    'scapula_l': 10.0,   # acromial→scapula: HIGH weight for shoulder width
    'humerus_r': 1.0,    # humerus: not directly in loss, low weight
    'humerus_l': 1.0,    # humerus: not directly in loss, low weight
    'ulna_r': 10.0,      # elbow - HIGH weight
    'ulna_l': 10.0,      # elbow - HIGH weight
    'radius_r': 8.0,     # wrist - HIGH weight
    'radius_l': 8.0,     # wrist - HIGH weight
    'hand_r': 15.0,      # hand - HIGHEST weight for arm chain
    'hand_l': 15.0,      # hand - HIGHEST weight for arm chain
}

# SKEL joint name to index mapping
SKEL_JOINT_TO_IDX = {
    'pelvis': 0,
    'femur_r': 1, 'tibia_r': 2, 'talus_r': 3, 'calcn_r': 4, 'toes_r': 5,
    'femur_l': 6, 'tibia_l': 7, 'talus_l': 8, 'calcn_l': 9, 'toes_l': 10,
    'lumbar': 11, 'thorax': 12, 'head': 13,
    'scapula_r': 14, 'humerus_r': 15, 'ulna_r': 16, 'radius_r': 17, 'hand_r': 18,
    'scapula_l': 19, 'humerus_l': 20, 'ulna_l': 21, 'radius_l': 22, 'hand_l': 23,
}


# =============================================================================
# AddB Constants
# =============================================================================

ADDB_NUM_JOINTS = 20


# =============================================================================
# Optimization Configuration
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for the optimization pipeline."""

    # Device
    device: str = 'cuda'

    # Scale estimation (Stage 1)
    scale_lr: float = 0.01
    scale_iters: int = 200

    # Stage 2 beta learning rate (controls how much beta can drift from Stage 1)
    # Lower = preserve Stage 1 bone lengths, Higher = allow more adaptation
    # 0.02 (original) → beta drifts heavily, shin -20mm
    # 0.002 → shin preserved but MPJPE +2mm
    # 0.005 → balanced compromise
    stage2_beta_lr: float = 0.002

    # Per-bone weights for scale estimation (Stage 1 beta optimization)
    # Shin (knee→ankle) gets higher weight because:
    # - SKEL shape space couples thigh and shin (beta increases both)
    # - GT has shin > thigh but SKEL default has shin < thigh
    # - Short shin → ankle too high → tip-toe appearance (visual artifact)
    # - Slightly long thigh is visually less noticeable than short shin
    scale_bone_weights: Optional[Dict[str, float]] = None  # Set in __post_init__

    # Pose optimization (Stage 2)
    pose_lr: float = 0.01
    pose_iters: int = 500

    # Loss weights (tuned based on compare_smpl_skel.py)
    weight_joint: float = 1.0
    weight_bone_dir: float = 0.0    # DISABLED — PROXY joints (acromial) in bone pairs cause issues
    weight_bone_len: float = 1.0    # 0.3 → 1.0 (critical!)
    weight_shoulder: float = 1.0
    weight_width: float = 2.0       # L/R acromial distance — low weight to prevent narrow shoulders without over-abducting scapula
    weight_pose_reg: float = 0.01   # Keep original — increasing degrades MPJPE significantly
    weight_spine_reg: float = 0.3   # 0.1 → 0.3 (prevents thorax over-extension when spine markers absent)
    weight_arm_reg: float = 0.0     # Arm DOFs (elbow/wrist) regularization — disabled by default
    weight_scapula_reg: float = 0.05  # Scapula DOF regularization (anatomical prior, keeps shoulder neutral)
    weight_temporal: float = 0.01   # Keep original — increasing degrades MPJPE significantly
    weight_acceleration: float = 0.1  # 2nd-order (acceleration) smoothness regularization
    adaptive_temporal: bool = False   # If True, normalize temporal loss by dt (dt-adaptive smoothing)
    dt: float = 0.01                  # Timestep in seconds (used when adaptive_temporal=True)
    weight_cartesian_temporal: float = 0.0    # Cartesian joint velocity smoothness (0=disabled)
    weight_cartesian_acceleration: float = 0.0  # Cartesian joint acceleration smoothness (0=disabled)
    weight_q_match: float = 0.1            # DOF-space q matching vs AddB GT (prevents Euler wrapping)
    q_match_warmup_fraction: float = 0.3   # Warmup: Cartesian 수렴 후 q-match 시작 (0.3=30%)
    q_match_confidence_weighted: bool = True  # |scale|로 DOF별 가중치 (낮은 scale = 낮은 신뢰)

    # --- Direction correction methods (mutually exclusive) ---
    # Method A: Chirality loss — penalizes L/R swap via cross product constraint
    weight_chirality: float = 0.0      # 0=disabled. Recommended: 5.0-10.0
    # Method B: GT-based R_align — compute R_align from GT joint positions instead of pelvis FK
    use_gt_align: bool = False
    # Method C: Two-pass — try 0° and 180° pelvis rotation, pick lower MPJPE
    use_two_pass: bool = True             # Two-pass direction correction (0°/180° pelvis probe)
    two_pass_iters: int = 20           # iterations per pass (short probe)
    # R_align for rotation matrix mapping: R_90_Y (90° Y) vs R_FLIP_XZ (180° Y = X,Z flip)
    use_r_flip_xz: bool = False        # True: use 180° Y (matches joint X,Z flip), False: use 90° Y (verified best)

    # --- Sequential / Sliding Window optimization ---
    use_sequential: bool = False       # Frame-by-frame with warm start from previous frame
    sequential_iters: int = 100        # Per-frame iterations (warm start → fewer needed)
    use_sliding_window: bool = False   # Sliding window batch optimization
    window_size: int = 50              # Frames per window
    window_stride: int = 25            # Stride between windows (overlap = window_size - stride)
    window_blend_frames: int = 10      # Linear blending zone at window boundaries
    freeze_pelvis_in_phaseA: bool = False  # Fix B: freeze pelvis DOFs (0-2) during Phase A direction-first
    use_tight_dof_limits: bool = False     # Fix D: tighten DOF limits around q_reference ± margin
    tight_dof_margin_rad: float = 0.5      # Margin in radians (~29°) around q_ref for tight limits
    pelvis_rotation_unlimited: bool = False # Fix F: remove pelvis_rotation limit (SKEL original has no limit)
    use_adaptive_pelvis_limit: bool = False # Fix H: set pelvis_rot limit to q_ref_median ± margin
    adaptive_pelvis_margin_rad: float = 0.785  # ±45° default
    auto_adaptive_pelvis: bool = False  # Auto-detect: apply adaptive only when q_ref pelvis_rot is outside ±45°
    weight_foot_reg: float = 0.0    # Subtalar/mtp DOF regularization (0=disabled; >0 degrades foot direction)
    weight_foot_height: float = 15.0  # Foot height (Y-axis) loss — GRF critical: aggressive foot fitting

    # Foot-only fine-tune pass after Stage 2
    # Only optimizes ankle/subtalar/mtp DOFs to match foot joints precisely
    foot_finetune_iters: int = 400
    foot_finetune_lr: float = 0.02
    foot_xyz_weights: List[float] = field(default_factory=lambda: [5.0, 20.0, 5.0])  # [X, Y, Z] weights for foot fine-tune
    foot_finetune_use_joint_weights: bool = False  # Apply SKEL_JOINT_WEIGHTS in Stage 2e

    # Runtime override for SKEL_JOINT_WEIGHTS (None = use default)
    joint_weights_override: Optional[Dict[str, float]] = None

    # Factor for APPROXIMATE-mapped foot joints (subtalar→calcn) in foot-specific losses.
    # Affects: foot_height_loss (Y-axis), foot fine-tune lower body joint loss, foot XYZ loss.
    # 1.0 = no change (default), 0.0 = exclude calcn from all foot losses.
    approximate_foot_weight_factor: float = 1.0

    # Humerus alignment/regularization weights (extracted from hardcoded values in scapula_handler)
    weight_humerus_align: float = 0.0    # DISABLED — humerus has no DIRECT mapping
    weight_humerus_reg: float = 0.05     # Humerus DOF regularization (anatomical prior, keeps arm neutral)

    # Head marker activation (remove LFHD/RFHD/LBHD/RBHD from blacklist at runtime)
    use_head_markers: bool = False

    # Virtual acromial vertex indices (computed at runtime)
    acromial_vertex_indices: Optional[Dict[str, List[int]]] = None

    # ==========================================================================
    # Phase 1 Improvements: Quick Wins
    # ==========================================================================

    # Cosine annealing learning rate schedule
    use_cosine_lr: bool = True

    # Huber loss (robust to outliers) instead of MSE
    use_huber_loss: bool = True
    huber_delta: float = 0.05  # 50mm in meters

    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 80  # Stop if no improvement for N iterations

    # ==========================================================================
    # Phase 2 Improvements: Dynamic Weighting
    # ==========================================================================

    # Dynamic joint weighting based on per-joint errors
    use_dynamic_weights: bool = False  # Disabled by default, enable for experiments
    dynamic_weight_scale: float = 0.5  # How much to scale weights by error

    # Soft pose constraints (penalty instead of hard clamp)
    use_soft_constraints: bool = False  # Disabled by default
    soft_constraint_weight: float = 0.1  # Weight for constraint penalty

    # ==========================================================================
    # Phase 3 Improvements: Advanced Techniques
    # ==========================================================================

    # Multi-Resolution Optimization: optimize key joints first, then all joints
    use_multi_resolution: bool = False  # Disabled by default
    # Key joints: pelvis, hips, shoulders (scapula), hands
    # SKEL indices: pelvis=0, femur_r=1, femur_l=6, scapula_r=14, scapula_l=19, hand_r=18, hand_l=23
    multi_res_key_joints: List[int] = None  # Will be set to default in __post_init__
    multi_res_stage1_iters: int = 100  # Iterations for key joints only

    # Scapula-Humerus Coupling Constraint (scapulohumeral rhythm)
    # Anatomically, scapula rotates ~1/3 of humerus elevation (2:1 ratio)
    use_scapulohumeral_coupling: bool = False  # Disabled by default
    scapulohumeral_weight: float = 0.1  # Weight for coupling constraint
    scapulohumeral_ratio: float = 3.0  # humerus_elevation / scapula_upward_rot ratio

    # ==========================================================================
    # Marker-Based Loss (Phase 3)
    # ==========================================================================

    # Enable marker position loss (requires markers_map + marker_observations)
    use_marker_loss: bool = True   # Marker position loss (BSM vertex matching)
    # Upper body marker boost: multiplier for shoulder/elbow/wrist markers in marker loss.
    # Compensates for lack of DIRECT joint matching in upper body (only elbow~hand have direct match).
    upper_body_marker_boost: float = 5.0  # Boost shoulder/elbow/wrist BSM markers (verified optimal)
    head_marker_boost: float = 1.0        # No head boost (head boost harms spine chain, verified)

    # Weight for marker loss term
    weight_marker: float = 10.0  # Marker loss weight (10.0 verified optimal for L/R + fitting)

    # Enable marker loss per optimization stage
    marker_in_stage1: bool = False  # BSMVertexMarkerHandler needs mesh (Stage 1 has no mesh yet)
    marker_in_stage2: bool = True

    # Weight for marker loss in scale estimation (Stage 1)
    weight_marker_scale: float = 0.1
    # Weight for shoulder width matching in scale estimation (Stage 1)
    # Matches SKEL scapula width to AddB acromial width via betas
    weight_shoulder_width_in_scale: float = 0.0  # DISABLED — beta distorts body shape to match width

    # Use VertexMarkerHandler (SMPL2AddBiomechanics approach) instead of MarkerHandler
    use_vertex_marker_handler: bool = True
    bsm_markers_path: str = '/egr/research-zijunlab/kwonjoon/01_Code/SMPL2AddBiomechanics/smpl2ab/data/bsm_markers.yaml'

    # Marker fine-tuning (2-pass approach):
    # Pass 1: normal optimization without marker loss
    # Pass 2: short fine-tuning with marker loss added
    marker_finetune_iters: int = 200   # Grid search v3 optimal (200 > 100)
    marker_finetune_lr: float = 0.01   # Grid search v3 optimal (0.01 > 0.003)

    # Per-tier marker weight multipliers: Tier 1 (exact) > Tier 2 (alias) > Tier 3 (position)
    # Tier 3 cluster markers can distort joint optimization if weighted too heavily
    tier_weight_factors: Dict[int, float] = field(default_factory=lambda: {1: 2.0, 2: 1.5, 3: 0.2})

    # Stage 2b: Tier 1/2 pre-fine-tune before Tier 3 matching
    # Only effective when use_marker_in_pose=False (2-pass approach).
    # When use_marker_in_pose=True, Stage 2 already includes full marker loss
    # optimization, so Stage 2b would re-disturb an already-converged solution.
    # 0 = disabled (default when use_marker_in_pose=True)
    tier12_finetune_iters: int = 0

    # Adaptive marker weight: auto-downweight markers with large residuals
    # Markers like T10 (119mm) or RBAK (111mm) may have wrong vertex mapping
    # even in Tier 1; adaptive weight reduces their gradient impact.
    # NOTE: Empirically this does NOT improve results — wrong vertex mappings
    # cause persistent errors regardless of weighting. Disabled by default.
    use_adaptive_marker_weight: bool = False
    adaptive_weight_threshold_mm: float = 50.0  # marker err >= 50mm → weight < 1.0

    # Marker blacklist: marker names to exclude from matching entirely (case-insensitive)
    # These markers were excluded based on empirical error analysis, but some may be
    # recoverable with better vertex mapping or fitting. Needs re-evaluation.
    marker_blacklist: List[str] = field(default_factory=lambda: [
        'LFHD', 'RFHD', 'LBHD', 'RBHD',              # head markers — limited by neck joint 3-DOF control
        'LLATFOOT', 'RLATFOOT', 'LANKMED', 'RANKMED',  # badly aliased foot markers (wrong vertex)
        'T10', 'RBAK', 'STRN',                         # thorax/back — needs re-evaluation (T10 added to yaml)
        'LPSIS', 'RPSIS', 'LASIS', 'RASIS',            # pelvis surface markers — skin vs bone prominence offset
        'LPSI', 'RPSI',                                # posterior iliac spine — needs re-evaluation (added to yaml)
        'LTOE', 'RTOE',                                # toe tip vertex — mismatch when GT LTOE is actually MTP not tip
        # C7/CV7 removed from blacklist (2026-02-27): ablation shows ~42mm error, MPJPE unaffected.
        # See 03_Output/addb2skel/20260226/20260226_blacklist_ablation/
        'SACR',                                        # sacrum — needs re-evaluation (added to yaml)
        # Force plate / calibration markers (not body markers)
        'FL', 'FR', 'BL', 'BR',                       # force plate corners (Carter2023 etc.)
        'LPT', 'RPT',                                 # force plate markers (Y~0.07m, near ground)
        'FP1', 'FP2', 'FP3', 'FP4',                   # force plate numbered markers
    ])

    # ==========================================================================
    # Virtual Marker IK (New Pipeline)
    # ==========================================================================

    # Use BSMVertexMarkerHandler (direct BSM name matching) instead of VertexMarkerHandler
    use_bsm_marker_handler: bool = True

    # Use marker pairwise distance for beta estimation
    use_marker_beta: bool = False  # Disabled — Phase 2 test showed MPJPE degradation (+2.4mm)

    # Marker warm-up schedule: 0→warmup_fraction = joints only,
    # warmup_fraction→1.0 = progressive marker weight increase
    marker_warmup_fraction: float = 0.0  # Use markers from the start

    # Enable marker loss during pose optimization (Stage 2)
    use_marker_in_pose: bool = True

    # Save virtual markers after fitting
    save_virtual_markers: bool = True
    save_trc: bool = True

    # Weight for marker pairwise distance in beta estimation
    weight_marker_pdist: float = 1.0

    # Position-based marker matching fallback (Tier 3) — DISABLED permanently
    # T3 matches by 3D position proximity, prone to wrong matches → removed
    use_position_based_fallback: bool = False
    position_fallback_max_dist_mm: float = 50.0  # unused (T3 disabled)

    # AddB body-local offset based vertex matching (Tier 2.5)
    # Uses markers_map (body, osim_offset) → SKEL T-pose nearest vertex
    # Runs after Stage 1 (betas needed) and Tier 1/2 matching
    # NOTE: Only adds 1 marker per body (skips cluster markers)
    use_offset_matching: bool = False  # Disabled: cluster markers cause MPJPE regression
    offset_matching_max_dist_mm: float = 60.0

    # Coordinate flip for marker observations: AddB obs → SKEL coords
    # True: flip X and Z (AddB→SKEL: negate X and Z axes)
    # Required for correct marker loss computation
    marker_flip_xz: bool = True

    # Force verification (Stage 4, standalone script)
    run_force_verification: bool = False

    # ==========================================================================
    # Direction-First Staged Optimization (P1)
    # ==========================================================================
    # Phase A: 방향 수렴 (DOF 제한 없음) → Phase B: DOF 제한 적용
    # 기존 Stage 1 (250 iters) = Phase A (100) + Phase B (150)

    use_direction_first: bool = True
    phaseA_iters: int = 100           # Phase A: 방향 수렴 (DOF 제한 없음)
    phaseA_lr: float = 0.025
    phaseA_weight_bone_dir: float = 1.5   # ↑ (기본 0.3)
    phaseA_weight_joint: float = 0.5      # ↓ (기본 1.0)
    phaseA_weight_bone_len: float = 0.5
    phaseA_weight_pose_reg: float = 0.005
    phaseB_iters: int = 150           # Phase B: DOF 제한 적용
    phaseB_lr: float = 0.02
    phaseB_soft_constraint_weight: float = 0.5  # soft penalty 초기 weight
    phaseB_hard_clamp_start: float = 0.5  # Phase B의 50% 이후에 hard clamp

    # ==========================================================================
    # Marker-pair Direction Loss (P2a)
    # ==========================================================================
    # bsm_markers.yaml 마커 쌍으로 관절별 방향 벡터 감독

    weight_marker_direction: float = 0.0   # Marker-pair direction loss (0=disabled)
    phaseA_weight_marker_direction: float = 1.0  # Phase A에서 방향 감독 강화

    # ==========================================================================
    # Bony/Soft Marker Weighting (P2b)
    # ==========================================================================
    # SKEL 논문: bony markers (뼈 돌출부) 높은 가중치, soft markers (연부조직) 낮은 가중치
    # Tier weights와 곱셈으로 결합

    use_bony_soft_weighting: bool = True
    bony_marker_weight: float = 1.5     # SKEL 논문: 높은 λk
    soft_marker_weight: float = 0.5     # SKEL 논문: 낮은 λk
    head_marker_weight_boost: float = 2.0  # head markers 추가 boost (bony weight에 곱해짐)
    phaseA_bony_only: bool = True       # Phase A에서 bony markers만 사용 (방향 정확도↑)

    # ==========================================================================
    # dJ Joint Offset Correction (SKEL forward dJ parameter)
    # ==========================================================================
    # SKEL J_regressor_osim regresses approximate joint positions.
    # dJ [24, 3] provides per-joint 3D offsets for fine correction.
    use_dj_optimization: bool = True      # Enable dJ optimization in Stage 1
    weight_dj_reg: float = 1.0            # L2 regularization on dJ (strong: force betas to do main work)
    dj_lr_factor: float = 0.1             # dJ learning rate = scale_lr * this factor

    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.multi_res_key_joints is None:
            # Key joints: pelvis, both hips, both scapulas, both hands
            self.multi_res_key_joints = [0, 1, 6, 14, 19, 18, 23]
        if self.scale_bone_weights is None:
            # Bone weights for Stage 1 beta estimation
            # Key: (addb_parent, addb_child) as string "parent→child"
            # Shin gets 3x weight to prioritize ankle height accuracy
            self.scale_bone_weights = {
                'ground_pelvis→hip_r': 1.0,
                'ground_pelvis→hip_l': 1.0,
                'hip_r→walker_knee_r': 1.0,        # thigh
                'hip_l→walker_knee_l': 1.0,         # thigh
                'walker_knee_r→ankle_r': 10.0,      # shin — 10x weight (SKEL thigh-shin coupling)
                'walker_knee_l→ankle_l': 10.0,      # shin — 10x weight
                'ankle_r→mtp_r': 3.0,                # foot bone — GRF critical
                'ankle_l→mtp_l': 3.0,                # foot bone — GRF critical
                'elbow_r→radioulnar_r': 1.0,
                'elbow_l→radioulnar_l': 1.0,
            }

    def get_device(self) -> torch.device:
        """Get torch device."""
        if self.device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')


@dataclass
class ConversionResult:
    """Result of AddB to SKEL conversion."""

    # Core outputs
    skel_joints: 'torch.Tensor'  # [T, 24, 3]
    skel_poses: 'torch.Tensor'   # [T, 46]
    skel_betas: 'torch.Tensor'   # [10] or [T, 10]
    skel_trans: 'torch.Tensor'   # [T, 3]

    # Optionally mesh
    skel_vertices: Optional['torch.Tensor'] = None  # [T, V, 3]

    # Virtual markers
    virtual_markers: Optional['np.ndarray'] = None  # [T, K, 3] marker trajectories
    virtual_marker_names: Optional[List[str]] = None  # marker names
    marker_quality: Optional[Dict[str, float]] = None  # per-marker error in mm
    marker_tier_counts: Optional[Dict[int, int]] = None  # {1: N, 2: N, 3: N} tier matching stats

    # Foot marker offsets (per-subject correction for structural mismatch)
    foot_marker_offsets: Optional[Dict[str, 'np.ndarray']] = None  # {name: [3] in SKEL coords}

    # Diagnostics
    mpjpe_mm: float = 0.0
    per_joint_error: Optional[Dict[str, float]] = None
    scapula_dofs: Optional[Dict[str, float]] = None

    # Metadata
    num_frames: int = 0
    sex: str = 'male'

    # Evaluation metrics (all 7: MPJPE, W-MPJPE, PA-MPJPE, ACCL, VEL, FS, GP)
    evaluation_metrics: Optional[Dict[str, float]] = None

    # Extended stats (scale estimation, finetune)
    scale_stats: Optional[Dict] = None
    finetune_stats: Optional[Dict] = None


# =============================================================================
# Default configuration instance
# =============================================================================

DEFAULT_CONFIG = OptimizationConfig()
