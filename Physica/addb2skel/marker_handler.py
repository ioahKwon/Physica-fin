"""
Marker-based loss module for SKEL pose optimization.

Two approaches:
1. MarkerHandler: Uses AddB markersMap body-local offsets + nearest vertex per segment.
   Has ~160mm error due to missing body rotation in T-pose matching.
2. VertexMarkerHandler (SMPL2AddBiomechanics approach): Uses bsm_markers.yaml vertex indices.
   For each subject marker, finds nearest SKEL vertex in first-frame posed mesh.
   Simpler and more accurate (~20-30mm error).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import yaml


# AddB OpenSim body name → SKEL joint index
# Only DIRECT correspondences (same anatomical body, same local frame orientation)
# torso excluded: AddB has single 'torso', SKEL splits into lumbar/thorax/head
ADDB_BODY_TO_SKEL_JOINT = {
    'pelvis': 0,
    'femur_r': 1,
    'tibia_r': 2,
    'talus_r': 3,
    'calcn_r': 4,
    'toes_r': 5,
    'femur_l': 6,
    'tibia_l': 7,
    'talus_l': 8,
    'calcn_l': 9,
    'toes_l': 10,
    'torso': 12,      # thorax joint (AddB single 'torso' → SKEL thorax, best approximation)
    'head': 13,       # head joint
    'humerus_r': 15,
    'ulna_r': 16,
    'radius_r': 17,
    'hand_r': 18,
    'humerus_l': 20,
    'ulna_l': 21,
    'radius_l': 22,
    'hand_l': 23,
}


class MarkerHandler:
    """
    Bone-transform based marker loss for SKEL pose optimization.

    Uses AddB markersMap (body_node, local_offset) to:
    1. Identify which SKEL body segment each marker belongs to
    2. Find nearest SKEL mesh vertex within that segment
    3. Compute differentiable marker position loss during optimization
    """

    def __init__(self, device: torch.device):
        self.device = device

        # Populated by prepare_from_markersmap()
        self.skel_joint_indices: Optional[torch.Tensor] = None   # [K]
        self.local_offsets: Optional[torch.Tensor] = None         # [K, 3]
        self.marker_targets: Optional[torch.Tensor] = None        # [T, K, 3]
        self.marker_mask: Optional[torch.Tensor] = None           # [T, K]
        self.marker_vertex_indices: Optional[torch.Tensor] = None # [K]
        self.marker_names: List[str] = []
        self.num_markers: int = 0

    def prepare_from_markersmap(
        self,
        markers_map,
        marker_observations: List[Dict[str, np.ndarray]],
        flip_xz: bool = True,
    ) -> int:
        """
        Extract marker info from AddB markersMap and build optimization targets.

        Args:
            markers_map: osim.markersMap from nimblephysics (dict-like,
                         each entry has .first=BodyNode, .second=offset)
            marker_observations: Per-frame observed marker positions.
                                 List of dicts: {marker_name: np.ndarray[3]}
            flip_xz: Apply AddB→SKEL coordinate conversion (negate X and Z)

        Returns:
            Number of matched markers (those with a DIRECT body correspondence)
        """
        matched = []

        for name in markers_map:
            entry = markers_map[name]
            # nimblephysics markersMap entries are (BodyNode, np.ndarray) tuples
            body_node, offset = entry[0], entry[1]
            body_name = body_node.getName()

            if body_name not in ADDB_BODY_TO_SKEL_JOINT:
                continue

            skel_idx = ADDB_BODY_TO_SKEL_JOINT[body_name]
            offset_np = np.array([float(offset[0]), float(offset[1]), float(offset[2])],
                                 dtype=np.float32)
            matched.append((name, skel_idx, offset_np))

        if not matched:
            self.num_markers = 0
            return 0

        K = len(matched)
        self.marker_names = [m[0] for m in matched]

        skel_indices = np.array([m[1] for m in matched], dtype=np.int64)
        offsets = np.stack([m[2] for m in matched])  # [K, 3]

        # Build per-frame target positions [T, K, 3]
        T = len(marker_observations)
        targets = np.full((T, K, 3), np.nan, dtype=np.float32)
        for t, obs in enumerate(marker_observations):
            for k, name in enumerate(self.marker_names):
                if name in obs:
                    pos = obs[name]
                    targets[t, k] = [float(pos[0]), float(pos[1]), float(pos[2])]

        # Apply coordinate conversion: AddB → SKEL (flip X and Z)
        if flip_xz:
            offsets[:, 0] = -offsets[:, 0]
            offsets[:, 2] = -offsets[:, 2]
            targets[:, :, 0] = -targets[:, :, 0]
            targets[:, :, 2] = -targets[:, :, 2]

        # Build mask for valid (non-NaN) observations
        valid_mask = ~np.isnan(targets[:, :, 0])  # [T, K]
        targets = np.nan_to_num(targets, nan=0.0)

        # Convert to tensors
        self.skel_joint_indices = torch.tensor(skel_indices, device=self.device)
        self.local_offsets = torch.tensor(offsets, dtype=torch.float32, device=self.device)
        self.marker_targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        self.marker_mask = torch.tensor(valid_mask, device=self.device)
        self.num_markers = K

        n_valid = int(valid_mask.sum())
        n_total = T * K
        return K

    def find_nearest_vertices(self, skel_interface, betas: torch.Tensor):
        """
        Find nearest SKEL mesh vertex for each marker within its body segment.

        Uses T-pose + body-local offset to identify the closest vertex.
        This is called once after prepare_from_markersmap() and before optimization.

        Args:
            skel_interface: SKELInterface instance
            betas: Shape parameters [10]
        """
        if self.num_markers == 0:
            return

        with torch.no_grad():
            poses_zero = torch.zeros(1, 46, device=self.device)
            trans_zero = torch.zeros(1, 3, device=self.device)
            verts, joints, _ = skel_interface.forward(
                betas.unsqueeze(0), poses_zero, trans_zero
            )
            verts = verts[0]    # [6890, 3]
            joints = joints[0]  # [24, 3]

            # Get body segment assignment for each vertex
            skin_weights = skel_interface.model.skin_weights  # [6890, 24]
            if skin_weights.is_sparse:
                skin_weights = skin_weights.to_dense()
            vertex_segment = skin_weights.argmax(dim=1)  # [6890]

            matched_vidx = []
            for k in range(self.num_markers):
                seg = self.skel_joint_indices[k].item()

                # Target position: joint center + local offset (approximate, rotation ignored)
                target = joints[seg] + self.local_offsets[k]

                # Get vertices belonging to this body segment
                seg_mask = (vertex_segment == seg)
                if seg_mask.sum() == 0:
                    # Fallback: use all vertices if segment is empty
                    seg_mask = torch.ones(6890, dtype=torch.bool, device=self.device)

                seg_verts = verts[seg_mask]
                seg_global_idx = seg_mask.nonzero(as_tuple=True)[0]

                # Find nearest vertex
                dists = torch.norm(seg_verts - target, dim=1)
                best_local = dists.argmin()
                matched_vidx.append(seg_global_idx[best_local].item())

            self.marker_vertex_indices = torch.tensor(
                matched_vidx, dtype=torch.long, device=self.device
            )

    def compute_marker_loss(
        self,
        skel_joints: torch.Tensor,
        skel_verts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute marker position loss.

        If vertex indices are available (from find_nearest_vertices), uses vertex positions.
        Otherwise falls back to joint + offset (rotation not accounted for).

        Args:
            skel_joints: SKEL joint positions [T, 24, 3]
            skel_verts: SKEL mesh vertices [T, 6890, 3] (optional)

        Returns:
            Scalar loss value
        """
        if self.num_markers == 0:
            return torch.tensor(0.0, device=self.device)

        # Use vertex positions if available (more accurate)
        if skel_verts is not None and self.marker_vertex_indices is not None:
            pred = skel_verts[:, self.marker_vertex_indices, :]  # [T, K, 3]
        else:
            # Fallback: joint position + offset (no rotation)
            pred = skel_joints[:, self.skel_joint_indices, :] + self.local_offsets  # [T, K, 3]

        diff = pred - self.marker_targets  # [T, K, 3]

        # Huber loss: L2 for small errors (< delta), L1 for large errors (>= delta)
        delta = 0.02  # 20mm threshold in meters
        abs_diff = diff.norm(dim=-1)  # [T, K]
        sq_error = torch.where(
            abs_diff < delta,
            0.5 * (diff ** 2).sum(dim=-1),          # L2 region
            delta * abs_diff - 0.5 * delta ** 2,    # L1 region
        )  # [T, K]

        # Apply mask for missing markers — per-frame normalization
        # (prevents bias when some frames have many missing markers)
        mask_f = self.marker_mask.float()  # [T, K]
        n_valid_per_frame = mask_f.sum(dim=1).clamp(min=1.0)  # [T]
        per_frame_loss = (sq_error * mask_f).sum(dim=1) / n_valid_per_frame  # [T]
        return per_frame_loss.mean()

    def get_marker_statistics(
        self,
        skel_joints: torch.Tensor,
        skel_verts: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute per-marker error statistics (in mm) for diagnostics.

        Returns:
            Dict of marker_name → mean error in mm
        """
        if self.num_markers == 0:
            return {}

        with torch.no_grad():
            if skel_verts is not None and self.marker_vertex_indices is not None:
                pred = skel_verts[:, self.marker_vertex_indices, :]
            else:
                pred = skel_joints[:, self.skel_joint_indices, :] + self.local_offsets

            diff = pred - self.marker_targets  # [T, K, 3]
            error = torch.norm(diff, dim=-1)   # [T, K] in meters

            stats = {}
            for k, name in enumerate(self.marker_names):
                valid = self.marker_mask[:, k]
                if valid.sum() > 0:
                    mean_err = error[:, k][valid].mean().item() * 1000  # → mm
                    stats[name] = mean_err

            return stats

    def get_summary(self) -> str:
        """Return a summary string for logging."""
        if self.num_markers == 0:
            return "MarkerHandler: 0 markers matched"

        # Count per body segment
        seg_counts = {}
        for k in range(self.num_markers):
            seg = self.skel_joint_indices[k].item()
            seg_counts[seg] = seg_counts.get(seg, 0) + 1

        n_valid_frames = self.marker_mask.all(dim=1).sum().item() if self.marker_mask is not None else 0
        T = self.marker_mask.shape[0] if self.marker_mask is not None else 0

        return (f"MarkerHandler: {self.num_markers} markers matched, "
                f"{len(seg_counts)} body segments, "
                f"{n_valid_frames}/{T} frames with all markers visible")


# =============================================================================
# T1 Redirect table: BSM name → corrected BSM name
#
# When a dataset marker has the SAME NAME as a BSM marker (T1 exact match),
# but the BSM marker is at a DIFFERENT anatomical location than the standard
# biomechanics convention, redirect to the correct BSM marker.
#
# Example: PiG "LTOE" = 2nd metatarsal head, but BSM "LTOE" = toe tip (67mm apart).
# =============================================================================
T1_REDIRECT_TABLE = {
    'LTOE': 'LTOS',   # PiG LTOE = 2nd MT head; BSM LTOE = toe tip (67mm error). Redirect to BSM LTOS.
    'RTOE': 'RTOS',   # Same for right side.
}

# =============================================================================
# Marker name alias table: alternative name (UPPER) → BSM name (UPPER)
#
# Maps dataset-specific anatomical marker names to standard BSM names.
# Each entry is annotated with:
#   - Source dataset(s)
#   - Anatomical basis for the mapping
#   - Confidence: HIGH = same anatomical landmark
#                 MEDIUM = nearby landmark (~15-50mm offset)
#                 LOW = distant/approximate mapping (>50mm, use with caution)
#
# Vertex indices are determined by BSM marker definitions (bsm_markers.yaml).
# =============================================================================
MARKER_ALIAS_TABLE = {
    # ---- Knee (무릎) ---- [HIGH confidence: epicondyle = knee marker]
    'LFLE': 'LKNE',     # [Carter] Lateral Femoral Epicondyle → Knee lateral (same landmark) HIGH
    'RFLE': 'RKNE',     # [Carter] HIGH
    'LFME': 'LKNI',     # [Carter] Medial Femoral Epicondyle → Knee medial (same landmark) HIGH
    'RFME': 'RKNI',     # [Carter] HIGH
    'LKNEMED': 'LKNI',  # [Falisse] Knee medial HIGH
    'RKNEMED': 'RKNI',  # [Falisse] HIGH

    # ---- Pelvis (골반) ---- [HIGH: ASIS/PSIS are standard landmarks]
    'LASIS': 'LFWT',    # [Carter] Anterior Superior Iliac Spine → Front Waist HIGH
    'RASIS': 'RFWT',    # [Carter] HIGH
    'LASI': 'LFWT',     # [Falisse] ASIS → Front Waist HIGH
    'RASI': 'RFWT',     # [Falisse] HIGH
    'LPSIS': 'LPSI',    # [Carter] Posterior SIS → LPSI (exact match) HIGH
    'RPSIS': 'RPSI',    # [Carter] HIGH

    # ---- Ankle (발목) ---- [HIGH: malleolus = ankle marker]
    'LFAL': 'LANK',     # [Carter] Lateral Fibular Ankle (lat. malleolus) HIGH
    'RFAL': 'RANK',     # [Carter] HIGH
    'LTAM': 'LAKI',     # [Carter] Medial Tibial Ankle (medial malleolus) → BSM LAKI (NOT LTIC — LTIC is 80mm above ankle) HIGH
    'RTAM': 'RAKI',     # [Carter] HIGH
    'LANKMED': 'LAKI',  # [Falisse] Medial ankle → BSM LAKI (medial malleolus) HIGH
    'RANKMED': 'RAKI',  # [Falisse] HIGH

    # ---- Heel / Toe (발) ---- [HIGH for heel, MEDIUM for MTP]
    'LCAL': 'LHEE',     # [Carter] Calcaneus → Heel (same landmark) HIGH
    'RCAL': 'RHEE',     # [Carter] HIGH
    'L1MTP': 'LTOS',    # [Carter] 1st MTP head → LTOS (BSM v3350, ~42mm offset) MEDIUM
    'R1MTP': 'RTOS',    # [Carter] 1st MTP head → RTOS (BSM v6750, ~45mm offset) MEDIUM
    'L5MTP': 'LFPI',    # [Carter] 5th MTP head → Foot Pinky (BSM v3223, ~15mm) HIGH
    'R5MTP': 'RFPI',    # [Carter] 5th MTP head → Foot Pinky (BSM v6623, ~24mm) HIGH

    # ---- Elbow (팔꿈치) ---- [HIGH: humeral epicondyle = elbow marker]
    'LHLE': 'LELB',     # [Carter] Lateral Humeral Epicondyle HIGH
    'RHLE': 'RELB',     # [Carter] HIGH

    # ---- Wrist (손목) ---- [HIGH: styloid process = wrist marker]
    'LRSP': 'LWRA',     # [Carter] Radial Styloid → Wrist A HIGH
    'RRSP': 'RWRA',     # [Carter] HIGH
    'LUSP': 'LWRB',     # [Carter] Ulnar Styloid → Wrist B HIGH
    'RUSP': 'RWRB',     # [Carter] HIGH

    # ---- Spine / Thorax (척추) ----
    'CV7': 'C7',        # [Carter] C7 vertebra HIGH
    'SJN': 'CLAV',      # [Carter] Sternal Jugular Notch → Clavicle MEDIUM (~30mm, both anterior upper thorax)
    'STRN': 'CLAV',     # [Falisse] Sternum → Clavicle MEDIUM
    'RBAK': 'RBAC',     # [Carter] Right back → BSM Right Back HIGH

    # ---- Foot lateral ----
    'LLATFOOT': 'LMT5', # Left lateral foot → 5th metatarsal MEDIUM
    'RLATFOOT': 'RMT5', # Right lateral foot MEDIUM

    # ---- Shoulder (어깨) ---- [HIGH: acromial = shoulder marker]
    'LSAJ': 'LSHO',     # [Carter] Shoulder Acromial Joint HIGH
    'RSAJ': 'RSHO',     # [Carter] HIGH

    # ---- Han2023 convention (prefix_anatomicalname) ----
    'TRUNK_C7':      'C7',      # [Han] C7 vertebra HIGH
    'TRUNK_STRN':    'CLAV',    # [Han] sternum → clavicle MEDIUM
    'TRUNK_MANU':    'CLAV',    # [Han] manubrium → clavicle MEDIUM (~30mm)
    # REMOVED: TRUNK_XYPH→LUMB (226mm apart, xyphoid=anterior chest, LUMB=posterior lumbar)
    # REMOVED: TRUNK_T8→LUMB (97mm apart, T8=mid-thoracic, LUMB=L3/L4)
    # REMOVED: TRUNK_T12→LUMB (different vertebral level, multiple markers→same BSM causes conflict)
    # REMOVED: TRUNK_ABDL→LUMB (281mm apart, abdomen=anterior, LUMB=posterior)
    'PELVIS_SAC':    'SACR',    # [Han] sacrum → SACR HIGH
    'PELVIS_L_PSIS': 'LPSI',   # [Han] L PSIS → LPSI HIGH
    'PELVIS_R_PSIS': 'RPSI',   # [Han] R PSIS HIGH
    'PELVIS_L_ASIS': 'LFWT',   # [Han] L ASIS → Front Waist HIGH
    'PELVIS_R_ASIS': 'RFWT',   # [Han] HIGH
    'PELVIS_L_ILCR': 'LBWT',   # [Han] iliac crest → back waist MEDIUM (~30mm)
    'PELVIS_R_ILCR': 'RBWT',   # [Han] MEDIUM
    'L_ACRO':        'LSHO',   # [Han] acromion → shoulder HIGH
    'R_ACRO':        'RSHO',   # [Han] HIGH
    'L_ANK_LAT':     'LANK',   # [Han] ankle lateral HIGH
    'R_ANK_LAT':     'RANK',   # [Han] HIGH
    'L_ANK_MED':     'LAKI',   # [Han] ankle medial (medial malleolus) → BSM LAKI HIGH
    'R_ANK_MED':     'RAKI',   # [Han] HIGH
    'L_KNEE_LAT':    'LKNE',   # [Han] knee lateral HIGH
    'R_KNEE_LAT':    'RKNE',   # [Han] HIGH
    'L_KNEE_MED':    'LKNI',   # [Han] knee medial HIGH
    'R_KNEE_MED':    'RKNI',   # [Han] HIGH
    'L_ELB_LAT':     'LELB',   # [Han] elbow lateral HIGH
    'R_ELB_LAT':     'RELB',   # [Han] HIGH
    'L_WRI_LAT':     'LWRA',   # [Han] wrist lateral HIGH
    'R_WRI_LAT':     'RWRA',   # [Han] HIGH
    'L_WRI_MED':     'LWRB',   # [Han] wrist medial HIGH
    'R_WRI_MED':     'RWRB',   # [Han] HIGH
    # REMOVED: L_HIP_GRTR→LFWT (128mm apart, trochanter=lateral femur, LFWT=ASIS anterior pelvis)
    # REMOVED: R_HIP_GRTR→RFWT (same issue)

    # ---- Head (머리) ----
    # NOTE: FL/FR/BL/BR are force plate corners in Carter2023, NOT head markers.
    'FRHD': 'RFHD',    # alternate naming for front/back head markers HIGH
    'FLHD': 'LFHD',    # HIGH
    'BRHD': 'RBHD',    # HIGH
    'BLHD': 'LBHD',    # HIGH

    # ---- Foot — Carter2023 (metatarsal base / hallux) ----
    # REMOVED: LB5MTP→LMT5 (5th MT BASE/styloid ≠ HEAD, ~50mm apart)
    # REMOVED: RB5MTP→RMT5 (same)
    # REMOVED: LB1MTP→LTOS (1st MT BASE ≠ HEAD, ~60mm apart)
    # REMOVED: RB1MTP→RTOS (same)
    'LHAL': 'LTOE',    # [Carter] Hallux (big toe tip) → LTOE HIGH
    'RHAL': 'RTOE',    # [Carter] HIGH
    # NOTE: LPT/RPT are force plate markers in Carter2023 (Y~0.07m), NOT thigh markers.

    # ---- Fregly2012 (dot-separated names, Cleveland Clinic marker set) ----
    # Ref: Fregly et al. 2012 J Orthop Res; PiG standard conventions
    'L.ASIS':      'LFWT',   # [Fregly] ASIS HIGH
    'R.ASIS':      'RFWT',   # [Fregly] HIGH
    'L.PSIS':      'LPSI',   # [Fregly] PSIS HIGH
    'R.PSIS':      'RPSI',   # [Fregly] HIGH
    'L.HEEL':      'LHEE',   # [Fregly] Calcaneus HIGH
    'R.HEEL':      'RHEE',   # [Fregly] HIGH
    'L.TOE':       'LTOS',   # [Fregly] 2nd metatarsal head (PiG standard) → BSM LTOS (NOT LTOE, which is toe tip) HIGH
    'R.TOE':       'RTOS',   # [Fregly] HIGH
    'L.SHOULDER':  'LSHO',   # [Fregly] Acromion HIGH
    'R.SHOULDER':  'RSHO',   # [Fregly] HIGH
    'L.ELBOW':     'LELB',   # [Fregly] Lateral humeral epicondyle HIGH
    'R.ELBOW':     'RELB',   # [Fregly] HIGH
    'L.RADIUS':    'LWRA',   # [Fregly] Radial styloid HIGH
    'R.RADIUS':    'RWRA',   # [Fregly] HIGH
    'L.ULNA':      'LWRB',   # [Fregly] Ulnar styloid HIGH
    'R.ULNA':      'RWRB',   # [Fregly] HIGH
    'SACRAL':      'SACR',   # [Fregly] Sacrum midpoint HIGH
    'NECK':        'C7',     # [Fregly] C7 spinous process (ISB standard) HIGH
    'LUMBAR':      'LUMB',   # [Fregly] Lumbar spine (L3-L5 spinous process) HIGH

    # ---- Tiziana2019 (Lx/Rx prefix, LAMB protocol) ----
    # Ref: Lencioni et al. 2019 Sci Data; Rabuffetti et al. 2019 (LAMB protocol)
    'LXASIS':      'LFWT',   # [Tiziana] ASIS (paper confirmed) HIGH
    'RXASIS':      'RFWT',   # [Tiziana] HIGH
    'LXLATMAL':    'LANK',   # [Tiziana] Lateral malleolus (paper confirmed) HIGH
    'RXLATMAL':    'RANK',   # [Tiziana] HIGH
    'LXLATCON':    'LKNE',   # [Tiziana] Lateral femoral condyle (paper confirmed) HIGH
    'RXLATCON':    'RKNE',   # [Tiziana] HIGH
    'LXMETA5':     'LMT5',   # [Tiziana] 5th MT head (paper confirmed: "META5") HIGH
    'RXMETA5':     'RMT5',   # [Tiziana] HIGH
    'LXHEEL':      'LHEE',   # [Tiziana] Calcaneus (calcn body attachment) HIGH
    'RXHEEL':      'RHEE',   # [Tiziana] HIGH
    'LXSHOULD':    'LSHO',   # [Tiziana] Acromion HIGH
    'RXSHOULD':    'RSHO',   # [Tiziana] HIGH
    'LXSHOULDER':  'LSHO',   # [Tiziana] alternate spelling
    'RXSHOULDER':  'RSHO',   # [Tiziana] alternate spelling
    'LXELBOW':     'LELB',   # [Tiziana] Lateral humeral epicondyle HIGH
    'RXELBOW':     'RELB',   # [Tiziana] HIGH
    # NOTE: LxToe1 = 1st MT head (META1), NOT same as BSM LTOE (2nd MT head). Do NOT alias.
    # NOTE: LxFH = Fibular Head (NOT femoral head). No BSM equivalent. Do NOT alias.

    # ---- vanderZee2022 ----
    # Ref: van der Zee et al. 2022 Sci Data, Table 3
    'LAC':         'LSHO',   # [vanderZee] Acromion (paper confirmed) HIGH
    'RAC':         'RSHO',   # [vanderZee] HIGH
    'LLML':        'LANK',   # [vanderZee] Lateral malleolus (paper confirmed) HIGH
    'RLML':        'RANK',   # [vanderZee] HIGH
    'LMML':        'LAKI',   # [vanderZee] Medial malleolus → BSM LAKI (NOT LTIC!) HIGH
    'RMML':        'RAKI',   # [vanderZee] HIGH
    'LLEP':        'LKNE',   # [vanderZee] Lateral KNEE epicondyle (NOT elbow!) HIGH
    'RLEP':        'RKNE',   # [vanderZee] HIGH
    'LMEP':        'LKNI',   # [vanderZee] Medial KNEE epicondyle HIGH
    'RMEP':        'RKNI',   # [vanderZee] HIGH
    'L5TH':        'LMT5',   # [vanderZee] 5th metatarsal HIGH
    'R5TH':        'RMT5',   # [vanderZee] HIGH
    'LEP':         'LELB',   # [vanderZee] Elbow epicondyle (single marker, lateral) HIGH
    'REP':         'RELB',   # [vanderZee] HIGH
    # NOTE: LWR/RWR = single wrist marker, side unknown. Do NOT alias.
    # NOTE: LGTR/RGTR = greater trochanter, no BSM match. Do NOT alias.

    # ---- Hammer2013 ----
    # Ref: Hamner et al. 2010/2013 J Biomech; OpenSim model Setup_Scale.xml
    'LACR':        'LSHO',   # [Hammer] Acromion (OpenSim "L.Acromium") HIGH
    'RACR':        'RSHO',   # [Hammer] HIGH
    'LLEL':        'LELB',   # [Hammer] Lateral elbow epicondyle HIGH
    'RLEL':        'RELB',   # [Hammer] HIGH
    'LFARADIUS':   'LWRA',   # [Hammer] Radial styloid (OpenSim "Wrist.Lat") HIGH
    'RFARADIUS':   'RWRA',   # [Hammer] HIGH
    'LFAULNA':     'LWRB',   # [Hammer] Ulnar styloid (OpenSim "Wrist.Med") HIGH
    'RFAULNA':     'RWRB',   # [Hammer] HIGH

    # ---- Moore2015 ----
    # Ref: Moore et al. 2015 PeerJ Table 2; HBM protocol (van den Bogert 2013)
    'LLEE':        'LELB',   # [Moore] Lateral elbow epicondyle (Table 2) HIGH
    'RLEE':        'RELB',   # [Moore] HIGH
    'LLM':         'LANK',   # [Moore] Lateral malleolus (Table 2) HIGH
    'RLM':         'RANK',   # [Moore] HIGH
    'LLW':         'LWRB',   # [Moore] "Lateral Wrist" = ulnar styloid (Table 2: "styloid process ulna, pinky side") HIGH
    'RLW':         'RWRB',   # [Moore] HIGH
    'LMW':         'LWRA',   # [Moore] "Medial Wrist" = radial styloid (Table 2: "styloid process radius, thumb side") HIGH
    'RMW':         'RWRA',   # [Moore] HIGH
    # NOTE: LGTRO/RGTRO = greater trochanter, no BSM match. Do NOT alias.
}


# =============================================================================
# Region-aware Tier 3 matching constants
# Maps marker name keywords → body region, and BSM marker names → region.
# Used by match_remaining_by_position() to constrain search space.
# =============================================================================

# Ordered list of (region, keywords): first match wins, so put specific before general
_REGION_KEYWORDS = [
    ('pelvis',   ['sacr', 'psis', 'asis', 'pelv', 'iliac', 'ilcr', 'bwt', 'fwt', 'psi']),
    ('spine',    ['t10', 't8', 't12', 't7', 'c7', 'cv7', 'strn', 'manu', 'xyph',
                  'bak', 'bac', 'abdl', 'thorax', 'trunk', 'lumb', 'spine']),
    ('shoulder', ['acro', 'acr', 'sho', 'scap', 'sca', 'clav', 'sjn']),
    ('elbow',    ['elb', 'hle', 'hme', 'bicep', 'tricep']),
    ('wrist',    ['wri', 'wra', 'wrb', 'rsp', 'usp']),
    ('thigh',    ['thigh', 'thi', 'fem']),
    ('knee',     ['knee', 'kne', 'kni', 'fle', 'fme']),
    ('shin',     ['shank', 'shin', 'shn', 'farm']),   # farm=forearm (arm chain)
    ('ankle',    ['ankle', 'ank', 'tic', 'aki', 'fal', 'tam', 'mal']),
    ('heel',     ['heel', 'hee', 'cal']),
    ('toe',      ['latfoot', 'toe', 'toes', 'mt5', 'mtp', 'foot']),
]

# BSM marker name → body region (covers all 112 BSM markers in yaml)
_BSM_REGION: Dict[str, str] = {
    # Spine / Thorax
    'C7': 'spine', 'CLAV': 'shoulder', 'LUMB': 'spine', 'T10': 'spine',
    'RBAK': 'spine', 'LBAC': 'spine', 'RBAC': 'spine',
    'NOSE': 'spine', 'THD': 'spine',  # head → spine-like region
    # Pelvis
    'LFWT': 'pelvis', 'RFWT': 'pelvis', 'LBWT': 'pelvis', 'RBWT': 'pelvis',
    'SACR': 'pelvis', 'LPSI': 'pelvis', 'RPSI': 'pelvis',
    # NOTE: 'LUMB' already mapped to 'spine' above (line 541). Duplicate key removed.
    # Shoulder / Upper arm
    'LSHO': 'shoulder', 'RSHO': 'shoulder',
    'LSCA': 'shoulder', 'RSCA': 'shoulder',
    'LELS': 'shoulder', 'RELS': 'shoulder',
    'LELSO': 'shoulder', 'RELSO': 'shoulder',
    # Elbow / Forearm
    'LELB': 'elbow', 'RELB': 'elbow',
    'LFRM': 'elbow', 'RFRM': 'elbow',
    # Wrist / Hand
    'LWRA': 'wrist', 'LWRB': 'wrist', 'RWRA': 'wrist', 'RWRB': 'wrist',
    'LFIN': 'wrist', 'RFIN': 'wrist',
    'LHAP': 'wrist', 'RHAP': 'wrist',
    'LHBA': 'wrist', 'RHBA': 'wrist',
    'LHME': 'wrist', 'RHME': 'wrist',
    'LHPI': 'wrist', 'RHPI': 'wrist',
    'LHTH': 'wrist', 'RHTH': 'wrist',
    'LHTO': 'wrist', 'RHTO': 'wrist',
    'LHEB': 'wrist', 'RHEB': 'wrist',
    'LHFR': 'wrist', 'RHFR': 'wrist',
    # Thigh
    'LTIA': 'thigh', 'RTIA': 'thigh',
    'LTIB': 'shin',  'RTIB': 'shin',
    # Knee
    'LKNE': 'knee', 'RKNE': 'knee', 'LKNI': 'knee', 'RKNI': 'knee',
    # Shin
    'LSHN': 'shin', 'RSHN': 'shin',
    'LTIC': 'shin', 'RTIC': 'shin',
    # Ankle
    'LANK': 'ankle', 'RANK': 'ankle',
    'LAKI': 'ankle', 'RAKI': 'ankle',
    # Heel
    'LHEE': 'heel', 'RHEE': 'heel',
    # Toe / Foot
    'LTOE': 'toe', 'RTOE': 'toe',
    'LMT5': 'toe', 'RMT5': 'toe',
    'LLATFOOT': 'toe', 'RLATFOOT': 'toe',
    'LTOP': 'toe', 'RTOP': 'toe',
    'LTOS': 'toe', 'RTOS': 'toe',
    # Other foot markers (BSM-specific)
    'LFBT': 'toe', 'RFBT': 'toe',
    'LFFB': 'toe', 'RFFB': 'toe',
    'LFFT': 'toe', 'RFFT': 'toe',
    'LFLB': 'toe', 'RFLB': 'toe',
    'LFLT': 'toe', 'RFLT': 'toe',
    'LFMB': 'toe', 'RFMB': 'toe',
    'LFMT': 'toe', 'RFMT': 'toe',
    'LFPI': 'toe', 'RFPI': 'toe',
    'LFBB': 'toe', 'RFBB': 'toe',
    'LFHD': 'spine', 'RFHD': 'spine',  # front head markers (SMPL+H v136, v3648)
    'LUMC': 'ankle',
    # Head (treat as spine region for Tier 3)
    'LBHD': 'spine', 'RBHD': 'spine',
    'CHIN': 'spine',
    # Back / Trunk cluster markers (BSM-specific)
    'RIBL': 'spine', 'RIBR': 'spine',
    'RITL': 'pelvis', 'RITR': 'pelvis',
    'BLTI': 'pelvis', 'BRTI': 'pelvis',
    'FBBT': 'pelvis', 'FRBT': 'pelvis',
    'FLTI': 'pelvis', 'FRTI': 'pelvis',
}

# Region-specific max matching distance (mm) for Tier 3 position-based matching
_REGION_MAX_DIST_MM: Dict[str, float] = {
    'pelvis':   80.0,
    'spine':   120.0,   # spine spans large area, need generous threshold
    'shoulder': 70.0,
    'elbow':    50.0,
    'wrist':    40.0,
    'thigh':    90.0,
    'knee':     55.0,
    'shin':     80.0,
    'ankle':    50.0,
    'heel':     45.0,
    'toe':      60.0,
}


def _classify_region(marker_name: str) -> Optional[str]:
    """Infer body region from marker name using keyword matching.

    Returns region string (e.g. 'pelvis', 'spine') or None if unknown.
    None → Tier 3 falls back to full BSM search with default threshold.
    """
    lower = marker_name.lower().replace('-', '_')
    for region, keywords in _REGION_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return region
    return None


# =============================================================================
# Bony/Soft Marker Classification (SKEL paper: Keller et al., SIGGRAPH Asia 2023)
# =============================================================================
# Bony markers: on bone prominences (epicondyles, malleoli, ASIS, styloid processes)
#   → Low fitting error expected → high weight λk
# Soft markers: on soft tissue (muscle bellies, fat pads, trunk)
#   → Higher fitting error expected → low weight λk
# Classification based on bsm.osim parent body frame + anatomical knowledge.

BONY_MARKERS = {
    # Head/Spine (bony landmarks)
    'C7', 'THD', 'CLAV', 'LUMB', 'T10',
    # Pelvis (ASIS, PSIS, sacrum)
    'LFWT', 'RFWT', 'LPSI', 'RPSI', 'SACR',
    # Knee (epicondyles)
    'LKNE', 'RKNE', 'LKNI', 'RKNI',
    # Ankle (malleoli)
    'LANK', 'RANK', 'LAKI', 'RAKI',
    # Foot (calcaneus, metatarsal heads, toe tips)
    'LHEE', 'RHEE', 'LMT5', 'RMT5', 'LTOE', 'RTOE', 'LTOS', 'RTOS',
    # Shoulder (acromion)
    'LSHO', 'RSHO',
    # Elbow (epicondyle)
    'LELB', 'RELB',
    # Wrist (styloid processes)
    'LWRA', 'RWRA', 'LWRB', 'RWRB',
    # Tibia (tibial tuberosity, direct bone surface)
    'LTIB', 'RTIB', 'LTIC', 'RTIC', 'LTIA', 'RTIA',
    # Scapula (acromion/spine)
    'LSCA', 'RSCA',
    # Head (bony prominences)
    'LFHD', 'RFHD', 'LBHD', 'RBHD', 'CHIN', 'NOSE',
}  # ~52 markers; remaining ~60 are soft

# Head markers — subset of BONY_MARKERS for additional weight boost
_HEAD_MARKERS = {'LFHD', 'RFHD', 'LBHD', 'RBHD', 'CHIN', 'NOSE', 'THD'}


# =============================================================================
# Marker-pair Direction Pairs (P2a: direction vector supervision)
# =============================================================================
# Each pair defines a direction vector from marker1 → marker2.
# pred direction (from SKEL mesh) is compared to obs direction (from mocap markers).
# Uses cosine similarity loss to supervise limb/joint directions directly.

MARKER_DIRECTION_PAIRS = {
    # Joint axis pairs (lateral → medial)
    'knee_axis_r': ('RKNE', 'RKNI'),    # knee flexion axis
    'knee_axis_l': ('LKNE', 'LKNI'),
    'ankle_axis_r': ('RANK', 'RAKI'),   # ankle axis
    'ankle_axis_l': ('LANK', 'LAKI'),
    # Limb direction pairs (proximal → distal)
    'thigh_dir_r': ('RFWT', 'RKNE'),    # hip → knee = thigh direction
    'thigh_dir_l': ('LFWT', 'LKNE'),
    'shin_dir_r': ('RKNE', 'RANK'),     # knee → ankle = shin direction
    'shin_dir_l': ('LKNE', 'LANK'),
    'foot_dir_r': ('RHEE', 'RTOE'),     # heel → toe = foot direction
    'foot_dir_l': ('LHEE', 'LTOE'),
    # Upper body direction pairs
    'upper_arm_r': ('RSHO', 'RELB'),    # shoulder → elbow
    'upper_arm_l': ('LSHO', 'LELB'),
    'forearm_r': ('RELB', 'RWRA'),      # elbow → wrist
    'forearm_l': ('LELB', 'LWRA'),
}


class BSMVertexMarkerHandler:
    """
    BSM name-based marker handler with 3-tier hybrid matching.

    Matching tiers:
    1. Exact case-insensitive BSM name match
    2. Alias table lookup (MARKER_ALIAS_TABLE)
    3. Position-based nearest-BSM-marker fallback (after mesh fitting)

    Advantages over VertexMarkerHandler:
    - Uses exact BSM vertex indices (no search error for Tier 1+2)
    - Handles naming variations across datasets
    - Position fallback for completely unknown naming conventions
    """

    def __init__(self, device: torch.device, bsm_markers_path: str,
                 alias_table: Optional[Dict[str, str]] = None):
        self.device = device

        # Load BSM marker definitions
        with open(bsm_markers_path) as f:
            self.bsm_markers: Dict[str, int] = yaml.safe_load(f)

        # Build case-insensitive lookup: UPPER_NAME -> (original_name, vertex_idx)
        self._bsm_lookup: Dict[str, Tuple[str, int]] = {}
        for name, vidx in self.bsm_markers.items():
            self._bsm_lookup[name.upper()] = (name, vidx)

        # Build alias lookup: ALT_UPPER -> (bsm_name, vertex_idx)
        alias = alias_table if alias_table is not None else MARKER_ALIAS_TABLE
        self._alias_lookup: Dict[str, Tuple[str, int]] = {}
        for alt_name, bsm_name in alias.items():
            bsm_upper = bsm_name.upper()
            if bsm_upper in self._bsm_lookup:
                original_bsm, vidx = self._bsm_lookup[bsm_upper]
                self._alias_lookup[alt_name.upper()] = (original_bsm, vidx)

        # Populated by prepare_from_markersmap()
        self.marker_names: List[str] = []
        self.vertex_indices: Optional[torch.Tensor] = None  # [K]
        self.marker_targets: Optional[torch.Tensor] = None   # [T, K, 3]
        self.marker_mask: Optional[torch.Tensor] = None       # [T, K]
        self.num_markers: int = 0
        self._match_info: List[Dict] = []
        self._tier_counts = {1: 0, 2: 0, 3: 0}
        self._unmatched_names: List[str] = []
        self._marker_observations: Optional[List[Dict[str, np.ndarray]]] = None
        self._flip_xz: bool = True


    def prepare_from_markersmap(
        self,
        markers_map,
        marker_observations: List[Dict[str, np.ndarray]],
        flip_xz: bool = True,
        blacklist: Optional[List[str]] = None,
    ) -> int:
        """
        Build marker targets by matching AddB marker names to BSM markers.

        Uses 3-tier matching:
        1. Exact case-insensitive BSM name match
        2. Alias table lookup (MARKER_ALIAS_TABLE)
        3. Position-based fallback (deferred to match_remaining_by_position())

        Args:
            markers_map: osim.markersMap from nimblephysics.
            marker_observations: Per-frame observed marker positions.
            flip_xz: Apply AddB→SKEL coordinate conversion.
            blacklist: List of marker names to exclude (case-insensitive).

        Returns:
            Number of matched markers (Tier 1+2 only).
        """
        blacklist_upper = {n.upper() for n in (blacklist or [])}

        matched_names = []
        matched_vidx = []
        match_info = []
        unmatched_names = []
        used_bsm = set()  # Track used BSM markers to prevent duplicates

        for addb_name in markers_map:
            addb_name = addb_name.strip()  # strip \r\n from Windows-style CSV marker names
            upper_name = addb_name.upper()

            # Skip blacklisted markers
            if upper_name in blacklist_upper:
                continue

            # T1 Redirect: fix BSM naming conflicts (e.g. PiG LTOE ≠ BSM LTOE)
            redirected_name = T1_REDIRECT_TABLE.get(upper_name, upper_name)

            # Tier 1: Exact case-insensitive BSM name match (after redirect)
            if redirected_name in self._bsm_lookup:
                bsm_name, vidx = self._bsm_lookup[redirected_name]
                if vidx not in used_bsm:
                    matched_names.append(addb_name)
                    matched_vidx.append(vidx)
                    used_bsm.add(vidx)
                    match_info.append({
                        'addb_name': addb_name, 'bsm_name': bsm_name,
                        'vertex_index': vidx, 'match_tier': 1,
                    })
                    continue

            # Tier 2: Alias table lookup
            if upper_name in self._alias_lookup:
                bsm_name, vidx = self._alias_lookup[upper_name]
                if vidx not in used_bsm:
                    matched_names.append(addb_name)
                    matched_vidx.append(vidx)
                    used_bsm.add(vidx)
                    match_info.append({
                        'addb_name': addb_name, 'bsm_name': bsm_name,
                        'vertex_index': vidx, 'match_tier': 2,
                        'alias_from': upper_name,
                    })
                    continue

            # Unmatched — collect for Tier 3
            unmatched_names.append(addb_name)

        # Also try matching markers that exist in observations but NOT in markers_map
        # (e.g., LFHD, RFHD, LBHD, RBHD — head markers observed but not in OpenSim model)
        obs_only_names = set()
        markers_map_stripped = {k.strip() for k in markers_map} if markers_map else set()
        for obs in (marker_observations or [])[:min(5, len(marker_observations or []))]:
            for name in obs.keys():
                name = name.strip()
                if name not in markers_map_stripped and name not in obs_only_names:
                    obs_only_names.add(name)

        matched_from_obs = 0
        for addb_name in sorted(obs_only_names):
            upper_name = addb_name.upper()

            # Skip blacklisted markers
            if upper_name in blacklist_upper:
                continue

            # T1 Redirect
            redirected_name = T1_REDIRECT_TABLE.get(upper_name, upper_name)

            # Tier 1: BSM name match (after redirect)
            if redirected_name in self._bsm_lookup:
                bsm_name, vidx = self._bsm_lookup[redirected_name]
                if vidx not in used_bsm:
                    matched_names.append(addb_name)
                    matched_vidx.append(vidx)
                    used_bsm.add(vidx)
                    match_info.append({
                        'addb_name': addb_name, 'bsm_name': bsm_name,
                        'vertex_index': vidx, 'match_tier': 1,
                        'source': 'observation_only',
                    })
                    matched_from_obs += 1
                    continue

            # Tier 2: Alias match
            if upper_name in self._alias_lookup:
                bsm_name, vidx = self._alias_lookup[upper_name]
                if vidx not in used_bsm:
                    matched_names.append(addb_name)
                    matched_vidx.append(vidx)
                    used_bsm.add(vidx)
                    match_info.append({
                        'addb_name': addb_name, 'bsm_name': bsm_name,
                        'vertex_index': vidx, 'match_tier': 2,
                        'alias_from': upper_name, 'source': 'observation_only',
                    })
                    matched_from_obs += 1
                    continue

            unmatched_names.append(addb_name)

        if matched_from_obs > 0:
            print(f"    Matched {matched_from_obs} observation-only markers "
                  f"(not in markersMap)")

        self._unmatched_names = unmatched_names
        self._used_bsm = used_bsm
        self._marker_observations = marker_observations
        self._flip_xz = flip_xz

        K = len(matched_names)
        if K == 0:
            self.num_markers = 0
            self._match_info = match_info
            self._tier_counts = {1: 0, 2: 0, 3: 0}
            return 0

        self.marker_names = matched_names
        self._match_info = match_info
        self._tier_counts = {
            1: sum(1 for m in match_info if m.get('match_tier') == 1),
            2: sum(1 for m in match_info if m.get('match_tier') == 2),
            3: 0,
        }
        self.vertex_indices = torch.tensor(
            matched_vidx, dtype=torch.long, device=self.device
        )

        # Build per-frame target positions [T, K, 3]
        self._build_targets(marker_observations, flip_xz)

        return K

    def match_remaining_by_position(
        self,
        fitted_verts: torch.Tensor,
        marker_observations: Optional[List[Dict[str, np.ndarray]]] = None,
        flip_xz: Optional[bool] = None,
        max_distance_mm: float = 50.0,
        use_region_aware: bool = True,
        markers_map=None,
        skel_interface=None,
        addb_skel=None,
        addb_poses: Optional[List[np.ndarray]] = None,
    ) -> int:
        """
        Tier 3: Match remaining unmatched AddB markers to nearest BSM marker
        by 3D position on the fitted mesh.

        Args:
            fitted_verts: Fitted SKEL mesh [T, 6890, 3] in SKEL coords.
            marker_observations: Per-frame observed marker positions (AddB coords).
            flip_xz: Convert observed markers from AddB to SKEL coords.
            max_distance_mm: Maximum distance for a valid match (default fallback).
            use_region_aware: If True, constrain candidates to same body region
                and use region-specific distance thresholds (more robust across datasets).
            markers_map: Optional osim.markersMap for body-aware vertex matching.
            skel_interface: Optional SKELInterface for skinning weight access.
            addb_skel: Optional nimblephysics Skeleton from markers_map body_node.
                If provided with addb_poses, uses AddB model predictions instead of
                noisy observations for vertex matching (Tier 2.5b quality).
            addb_poses: Optional list of pp.pos arrays per frame (AddB DOF positions).
                Used with addb_skel to compute clean marker predictions.

        Returns:
            Number of newly matched markers.
        """
        if not self._unmatched_names:
            return 0

        if marker_observations is None:
            marker_observations = self._marker_observations
        if flip_xz is None:
            flip_xz = self._flip_xz
        if marker_observations is None:
            return 0

        # Get all BSM marker positions on fitted mesh (frame 0)
        all_bsm_vidx = list(self.bsm_markers.values())
        all_bsm_names = list(self.bsm_markers.keys())
        used_bsm = set(self._used_bsm)

        with torch.no_grad():
            bsm_positions = fitted_verts[0, all_bsm_vidx, :].cpu().numpy()  # [N_bsm, 3]
            all_verts_np = fitted_verts[0].cpu().numpy()  # [6890, 3] for body-aware matching

        # Build body → vertex mapping via skinning weights (for body-aware matching)
        body_vertex_candidates: Optional[Dict[str, np.ndarray]] = None
        if markers_map is not None and skel_interface is not None:
            try:
                w_dense = skel_interface.model.skin_weights.to_dense().cpu().numpy()  # [6890, 24]
                body_vertex_candidates = {}
                for body_name, joint_idx in ADDB_BODY_TO_SKEL_JOINT.items():
                    cand_mask = w_dense[:, joint_idx] > 0.3
                    body_vertex_candidates[body_name] = np.where(cand_mask)[0]
            except Exception:
                body_vertex_candidates = None

        # Build marker → body_name lookup from markers_map
        marker_body_map: Dict[str, str] = {}
        if markers_map is not None:
            for name, val in markers_map.items():
                try:
                    bn, _ = val
                    marker_body_map[name] = bn.getName()
                except Exception:
                    pass

        # Tier 2.5b: Compute AddB model predictions using osim_skel + pp.pos
        # This gives cleaner positions than noisy observations (3-45mm vs obs noise)
        addb_model_preds: Dict[str, np.ndarray] = {}
        if (addb_skel is not None and addb_poses is not None
                and markers_map is not None and len(addb_poses) > 0):
            try:
                # Use first available frame's pp.pos
                frame_pos = addb_poses[0]
                addb_skel.setPositions(frame_pos)
                raw_preds = addb_skel.getMarkerMapWorldPositions(markers_map)
                for mname, mpos in raw_preds.items():
                    p = np.array(mpos, dtype=np.float64)
                    # Convert AddB model coords → target coords
                    # (apply same flip_xz as observations)
                    if flip_xz:
                        p_target = np.array([-p[0], p[1], -p[2]])
                    else:
                        p_target = p
                    addb_model_preds[mname] = p_target
            except Exception:
                addb_model_preds = {}

        new_matched_names = []
        new_matched_vidx = []
        new_match_info = []

        for addb_name in self._unmatched_names:
            # Prefer AddB model prediction over noisy observation (Tier 2.5b)
            if addb_name in addb_model_preds:
                obs_pos = addb_model_preds[addb_name].copy()
            else:
                # Fallback: use first valid observed position
                obs_pos = None
                for obs in marker_observations:
                    if addb_name in obs:
                        obs_pos = np.array(obs[addb_name], dtype=np.float64)
                        if flip_xz:
                            obs_pos[0] = -obs_pos[0]
                            obs_pos[2] = -obs_pos[2]
                        break
            if obs_pos is None:
                continue

            matched = False

            # Body-aware matching (Tier 3a): use skinning weights to restrict candidates
            # This is more accurate than region-aware BSM lookup for cluster/variant markers
            body_name = marker_body_map.get(addb_name)
            if (body_vertex_candidates is not None
                    and body_name is not None
                    and body_name in body_vertex_candidates):
                cand_vidx = body_vertex_candidates[body_name]
                if len(cand_vidx) > 0:
                    cand_positions = all_verts_np[cand_vidx]  # [M, 3]
                    cand_dists = np.linalg.norm(cand_positions - obs_pos, axis=1)
                    sorted_j = np.argsort(cand_dists)
                    for j in sorted_j:
                        vidx = int(cand_vidx[j])
                        dist_mm = cand_dists[j] * 1000
                        if vidx not in used_bsm and dist_mm < max_distance_mm * 3.0:
                            new_matched_names.append(addb_name)
                            new_matched_vidx.append(vidx)
                            used_bsm.add(vidx)
                            new_match_info.append({
                                'addb_name': addb_name,
                                'bsm_name': f'BODY_{body_name}',
                                'vertex_index': vidx,
                                'match_tier': 3,
                                'distance_mm': dist_mm,
                                'region': body_name,
                                'method': 'body_aware',
                            })
                            matched = True
                            break

            if matched:
                continue

            # Compute distances to all BSM markers (fallback)
            dists = np.linalg.norm(bsm_positions - obs_pos, axis=1)

            if use_region_aware:
                # Classify marker name into body region
                region = _classify_region(addb_name)
                region_threshold = _REGION_MAX_DIST_MM.get(region, max_distance_mm)
                # Restrict candidates to same body region
                if region is not None:
                    cand_indices = [i for i, n in enumerate(all_bsm_names)
                                    if _BSM_REGION.get(n) == region]
                    if not cand_indices:
                        # No BSM marker in this region → fallback to full search
                        cand_indices = list(range(len(all_bsm_names)))
                        region_threshold = max_distance_mm
                else:
                    # Unknown region → skip (avoid wrong body mapping)
                    continue
            else:
                cand_indices = list(range(len(all_bsm_names)))
                region_threshold = max_distance_mm

            # Find nearest unused BSM marker within region + threshold
            cand_dists = dists[cand_indices]
            for j in np.argsort(cand_dists):
                i = cand_indices[j]
                vidx = all_bsm_vidx[i]
                dist_mm = cand_dists[j] * 1000
                if vidx not in used_bsm and dist_mm < region_threshold:
                    new_matched_names.append(addb_name)
                    new_matched_vidx.append(vidx)
                    used_bsm.add(vidx)
                    new_match_info.append({
                        'addb_name': addb_name,
                        'bsm_name': all_bsm_names[i],
                        'vertex_index': vidx,
                        'match_tier': 3,
                        'distance_mm': dist_mm,
                        'region': region if use_region_aware else None,
                    })
                    matched = True
                    break

        if not new_matched_names:
            return 0

        # Extend existing matched data
        self.marker_names.extend(new_matched_names)
        self._match_info.extend(new_match_info)
        self._tier_counts[3] = len(new_match_info)
        self._used_bsm = used_bsm
        self._unmatched_names = [n for n in self._unmatched_names
                                  if n not in new_matched_names]

        # Rebuild vertex indices and targets
        all_vidx = (list(self.vertex_indices.cpu().numpy()) if self.vertex_indices is not None else [])
        all_vidx.extend(new_matched_vidx)
        self.vertex_indices = torch.tensor(all_vidx, dtype=torch.long, device=self.device)

        # Rebuild targets for the full marker set
        obs = marker_observations if marker_observations is not None else self._marker_observations
        if obs is not None:
            self._build_targets(obs, self._flip_xz)

        return len(new_matched_names)

    def _build_targets(
        self,
        marker_observations: List[Dict[str, np.ndarray]],
        flip_xz: bool,
    ):
        """Build per-frame target positions tensor from marker observations."""
        K = len(self.marker_names)
        T = len(marker_observations)
        targets = np.full((T, K, 3), np.nan, dtype=np.float32)
        for t, obs in enumerate(marker_observations):
            for k, name in enumerate(self.marker_names):
                if name in obs:
                    pos = obs[name]
                    targets[t, k] = [float(pos[0]), float(pos[1]), float(pos[2])]

        if flip_xz:
            targets[:, :, 0] = -targets[:, :, 0]
            targets[:, :, 2] = -targets[:, :, 2]

        valid_mask = ~np.isnan(targets[:, :, 0])
        targets = np.nan_to_num(targets, nan=0.0)

        self.marker_targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        self.marker_mask = torch.tensor(valid_mask, device=self.device)
        self.num_markers = K

    def compute_marker_loss(
        self,
        skel_joints: torch.Tensor,
        skel_verts: Optional[torch.Tensor] = None,
        tier_weight_factors: Optional[Dict[int, float]] = None,
        use_adaptive_weight: bool = False,
        adaptive_threshold_mm: float = 50.0,
        bony_soft_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute marker position loss using BSM vertex indices.

        Args:
            skel_joints: [T, 24, 3] (unused, for API compat).
            skel_verts: [T, 6890, 3] SKEL skin vertices.
            tier_weight_factors: Per-tier weight multipliers, e.g. {1: 2.0, 2: 1.5, 3: 0.2}.
            use_adaptive_weight: If True, auto-downweight markers with large residuals.
            adaptive_threshold_mm: Marker residual threshold for adaptive weighting.
            bony_soft_weights: {'bony': 1.5, 'soft': 0.5} — differential weighting.
            **kwargs: Additional options:
                upper_body_marker_boost (float): Multiplier for upper body markers (default 1.0).

        Returns:
            Scalar loss.
        """
        if self.num_markers == 0 or self.vertex_indices is None or skel_verts is None:
            return torch.tensor(0.0, device=self.device)

        pred = skel_verts[:, self.vertex_indices, :]  # [T, K, 3]

        diff = pred - self.marker_targets

        # Huber loss: L2 for small errors (< delta), L1 for large errors (>= delta)
        # Prevents large-error markers (e.g. mismatched SACR) from dominating the loss
        delta = 0.02  # 20mm threshold in meters
        abs_diff = diff.norm(dim=-1)  # [T, K]
        sq_error = torch.where(
            abs_diff < delta,
            0.5 * (diff ** 2).sum(dim=-1),          # L2 region
            delta * abs_diff - 0.5 * delta ** 2,    # L1 region
        )  # [T, K]

        # Build combined marker weights: tier weight × bony/soft weight × adaptive weight
        marker_weights = torch.ones(len(self.marker_names), device=self.device)

        # Per-tier weight multipliers: Tier 1 (exact) > Tier 2 (alias) > Tier 3 (position)
        if tier_weight_factors is not None and self._match_info:
            for k, info in enumerate(self._match_info):
                tier = info.get('match_tier', 1)
                marker_weights[k] = tier_weight_factors.get(tier, 1.0)

        # Bony/soft marker weighting (SKEL paper: bony=high λk, soft=low λk)
        if bony_soft_weights is not None:
            bony_w = bony_soft_weights.get('bony', 1.0)
            soft_w = bony_soft_weights.get('soft', 1.0)
            head_boost = bony_soft_weights.get('head_boost', 1.0)
            for k in range(len(self.marker_names)):
                name_upper = self.marker_names[k].upper()
                if name_upper in BONY_MARKERS:
                    marker_weights[k] *= bony_w
                    # Head markers get additional boost
                    if head_boost > 1.0 and name_upper in _HEAD_MARKERS:
                        marker_weights[k] *= head_boost
                else:
                    marker_weights[k] *= soft_w

        # Upper body marker boost: increase weight for shoulder/arm/wrist markers
        # to compensate for lack of DIRECT joint matching in upper body
        upper_body_boost = kwargs.get('upper_body_marker_boost', 1.0)
        head_boost = kwargs.get('head_marker_boost', 1.0)
        if upper_body_boost > 1.0 or head_boost > 1.0:
            _ARM_MARKERS = {
                'LSHO', 'RSHO',            # shoulder
                'LELB', 'RELB',            # elbow
                'LELS', 'RELS',            # elbow surface
                'LELSO', 'RELSO',          # elbow outer
                'LFRM', 'RFRM',           # forearm
                'LWRA', 'LWRB',            # wrist L
                'RWRA', 'RWRB',            # wrist R
                'LFIN', 'RFIN',            # finger
                'C7', 'CLAV',             # spine/clavicle
            }
            _HEAD_MARKERS_BOOST = {'LFHD', 'RFHD', 'LBHD', 'RBHD'}
            for k in range(len(self.marker_names)):
                name = self.marker_names[k].upper()
                if name in _HEAD_MARKERS_BOOST and head_boost > 1.0:
                    marker_weights[k] *= head_boost
                elif name in _ARM_MARKERS and upper_body_boost > 1.0:
                    marker_weights[k] *= upper_body_boost

        # Adaptive weight: auto-downweight markers with large mean residuals
        if use_adaptive_weight:
            mask_f_temp = self.marker_mask.float()  # [T, K]
            with torch.no_grad():
                n_valid = mask_f_temp.sum(0).clamp(min=1.0)            # [K]
                per_marker_err = (abs_diff * mask_f_temp).sum(0) / n_valid  # [K], meters
            threshold_m = adaptive_threshold_mm / 1000.0
            adaptive_w = 1.0 / (1.0 + per_marker_err / threshold_m)   # [K]
            marker_weights = marker_weights * adaptive_w

        # Apply combined weights: [K] → [1, K] for broadcasting over [T, K]
        sq_error = sq_error * marker_weights.unsqueeze(0)

        # Per-frame normalization: prevents bias from frames with many missing markers
        mask_f = self.marker_mask.float()  # [T, K]
        n_valid_per_frame = mask_f.sum(dim=1).clamp(min=1.0)  # [T]
        per_frame_loss = (sq_error * mask_f).sum(dim=1) / n_valid_per_frame  # [T]
        return per_frame_loss.mean()

    def compute_marker_direction_loss(
        self,
        skel_verts: torch.Tensor,
        bony_only: bool = False,
    ) -> torch.Tensor:
        """
        Compute marker-pair direction loss (cosine similarity).

        For each defined marker pair (m1→m2), compare:
        - pred direction: normalized(skel_verts[vidx_m2] - skel_verts[vidx_m1])
        - obs direction:  normalized(obs_m2 - obs_m1)

        Loss = 1 - cosine_similarity(pred_dir, obs_dir), averaged over valid pairs.

        Args:
            skel_verts: [T, 6890, 3] SKEL skin vertices.
            bony_only: If True, only use pairs where BOTH markers are bony.

        Returns:
            Scalar loss.
        """
        if skel_verts is None or self.num_markers == 0:
            return torch.tensor(0.0, device=skel_verts.device if skel_verts is not None else 'cpu')

        # Build name→index lookup for matched markers (BSM name → index in self.marker_names)
        name_to_idx = {}
        for k, name in enumerate(self.marker_names):
            name_to_idx[name.upper()] = k

        losses = []
        for pair_name, (m1_name, m2_name) in MARKER_DIRECTION_PAIRS.items():
            m1_upper = m1_name.upper()
            m2_upper = m2_name.upper()

            # Both markers must be matched
            if m1_upper not in name_to_idx or m2_upper not in name_to_idx:
                continue

            # If bony_only, skip pairs with any soft marker
            if bony_only:
                if m1_upper not in BONY_MARKERS or m2_upper not in BONY_MARKERS:
                    continue

            k1 = name_to_idx[m1_upper]
            k2 = name_to_idx[m2_upper]

            # Predicted direction from SKEL mesh vertices
            vidx1 = self.vertex_indices[k1]
            vidx2 = self.vertex_indices[k2]
            pred_vec = skel_verts[:, vidx2, :] - skel_verts[:, vidx1, :]  # [T, 3]
            pred_dir = pred_vec / (pred_vec.norm(dim=-1, keepdim=True) + 1e-8)

            # Observed direction from marker observations
            obs_m1 = self.marker_targets[:, k1, :]  # [T, 3]
            obs_m2 = self.marker_targets[:, k2, :]  # [T, 3]
            obs_vec = obs_m2 - obs_m1  # [T, 3]
            obs_dir = obs_vec / (obs_vec.norm(dim=-1, keepdim=True) + 1e-8)

            # Mask: both markers must be valid in each frame
            mask = self.marker_mask[:, k1] & self.marker_mask[:, k2]  # [T]

            # Cosine similarity loss: 1 - cos(pred, obs)
            cos_sim = (pred_dir * obs_dir).sum(dim=-1)  # [T]
            pair_loss = (1.0 - cos_sim) * mask.float()  # [T]

            n_valid = mask.float().sum().clamp(min=1.0)
            losses.append(pair_loss.sum() / n_valid)

        if not losses:
            return torch.tensor(0.0, device=skel_verts.device)

        return torch.stack(losses).mean()

    def get_marker_statistics(
        self,
        skel_joints: torch.Tensor,
        skel_verts: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Per-marker error in mm."""
        if self.num_markers == 0 or self.vertex_indices is None or skel_verts is None:
            return {}

        with torch.no_grad():
            pred = skel_verts[:, self.vertex_indices, :]
            diff = pred - self.marker_targets
            error = torch.norm(diff, dim=-1)

            stats = {}
            for k, name in enumerate(self.marker_names):
                valid = self.marker_mask[:, k]
                if valid.sum() > 0:
                    stats[name] = error[:, k][valid].mean().item() * 1000
            return stats

    def get_summary(self) -> str:
        if self.num_markers == 0:
            return "BSMVertexMarkerHandler: 0 markers matched"
        n_valid_frames = self.marker_mask.all(dim=1).sum().item() if self.marker_mask is not None else 0
        T = self.marker_mask.shape[0] if self.marker_mask is not None else 0
        t1 = self._tier_counts.get(1, 0)
        t2 = self._tier_counts.get(2, 0)
        t3 = self._tier_counts.get(3, 0)
        unmatched = len(self._unmatched_names)
        return (f"BSMVertexMarkerHandler: {self.num_markers} matched "
                f"(T1={t1} name, T2={t2} alias, T3={t3} position), "
                f"{unmatched} unmatched, "
                f"{n_valid_frames}/{T} frames all visible")

    def prepare_from_markersmap_with_offsets(
        self,
        markers_map,
        marker_observations: List[Dict[str, np.ndarray]],
        skel_interface,
        betas: 'torch.Tensor',
        flip_xz: bool = True,
        max_dist_mm: float = 60.0,
    ) -> int:
        """
        Tier 2.5: AddB body-local offset 기반 자동 vertex 매핑.

        markers_map[name] = (BodyNode, osim_offset) 정보를 이용해
        SKEL T-pose mesh에서 nearest vertex를 직접 찾아 매핑.
        Tier 1/2에서 이미 매칭된 마커와 already used vertex는 건너뜀.

        좌표 변환 (OpenSim → SKEL):
            X_skel = -Z_osim, Y_skel = Y_osim, Z_skel = X_osim

        Args:
            markers_map: osim.markersMap (Dict[name, (BodyNode, ndarray[3])])
            marker_observations: per-frame observation dicts
            skel_interface: SKELInterface instance (T-pose forward 용)
            betas: [10] subject shape parameters
            flip_xz: whether to flip X/Z for AddB→SKEL coord conversion
            max_dist_mm: maximum allowed distance to nearest vertex (mm)

        Returns:
            Number of newly matched markers.
        """
        # 이미 매칭된 마커 이름 집합
        already_matched = set(self.marker_names)
        used_vidx = set(self.vertex_indices.tolist()) if self.vertex_indices is not None else set()

        # SKEL T-pose forward pass (pose=0, betas=subject shape)
        with torch.no_grad():
            b = betas.to(skel_interface.device)
            if b.dim() == 1:
                b = b.unsqueeze(0)  # [1, 10]
            poses_zero = torch.zeros(1, 46, device=skel_interface.device)
            trans_zero = torch.zeros(1, 3, device=skel_interface.device)

            # raw SKEL model forward (joints_ori 필요)
            output = skel_interface.model(
                poses=poses_zero,
                betas=b,
                trans=trans_zero,
                poses_type='skel',
                skelmesh=False,
                pose_dep_bs=True,
            )
            verts_np = output.skin_verts[0].cpu().numpy()   # [6890, 3]
            joints_np = output.joints[0].cpu().numpy()       # [24, 3]
            # joints_ori: [1, 24, 3, 3] world rotation matrix per joint
            if hasattr(output, 'joints_ori') and output.joints_ori is not None:
                joints_ori_np = output.joints_ori[0].cpu().numpy()  # [24, 3, 3]
            else:
                # joints_ori 없으면 identity rotation 사용 (T-pose이므로 거의 identity)
                joints_ori_np = np.tile(np.eye(3), (24, 1, 1))

        new_names: List[str] = []
        new_vidx: List[int] = []
        new_match_info: List[dict] = []

        # 이미 사용된 vertex 위치들 (deduplication용)
        used_vidx_positions = verts_np[list(used_vidx)] if used_vidx else np.zeros((0, 3))
        dedup_dist_mm = 30.0  # 이미 사용된 vertex의 30mm 이내는 skip (cluster dedup)

        # nimblephysics body_node.getWorldTransform()을 직접 사용하려면
        # skeleton.setPositions(zeros)가 필요함. markers_map에서 skeleton 접근.
        skeleton = None
        try:
            for _, val in markers_map.items():
                body_node, _ = val
                skeleton = body_node.getSkeleton()
                break
        except Exception:
            skeleton = None

        if skeleton is not None:
            # AddB T-pose (q=0)에서 body world transforms 계산
            orig_pos = skeleton.getPositions().copy()
            skeleton.setPositions(np.zeros(skeleton.getNumDofs()))
        else:
            skeleton = None

        for addb_name, marker_val in markers_map.items():
            if addb_name in already_matched:
                continue

            # markers_map value: (BodyNode, osim_offset_ndarray)
            try:
                body_node, osim_offset = marker_val
                body_name = body_node.getName()
                osim_off = np.array(osim_offset, dtype=np.float64)
            except Exception:
                continue

            if ADDB_BODY_TO_SKEL_JOINT.get(body_name) is None:
                # ADDB_BODY_TO_SKEL_JOINT에 없는 body → skip
                continue

            # AddB T-pose에서 마커 world position 계산
            if skeleton is not None:
                try:
                    T_world = body_node.getWorldTransform()
                    R_addb = np.array(T_world.rotation())   # [3, 3]
                    t_addb = np.array(T_world.translation()) # [3]
                    world_pos_addb = R_addb @ osim_off + t_addb
                except Exception:
                    continue
            else:
                # Fallback: joints_ori 기반 (T-pose이므로 거의 동일)
                joint_idx = ADDB_BODY_TO_SKEL_JOINT[body_name]
                skel_off = np.array([-osim_off[2], osim_off[1], osim_off[0]], dtype=np.float64)
                R = joints_ori_np[joint_idx]
                t = joints_np[joint_idx]
                # SKEL T-pose coords에서 계산 → 직접 vertex 탐색
                world_pos_skel = R @ skel_off + t
                dists_v = np.linalg.norm(verts_np - world_pos_skel, axis=1)
                nearest_idx = int(np.argmin(dists_v))
                dist_mm = dists_v[nearest_idx] * 1000.0
                if dist_mm > max_dist_mm or nearest_idx in used_vidx:
                    continue
                # Dedup check
                if len(used_vidx_positions) > 0:
                    dedup_dists = np.linalg.norm(used_vidx_positions - verts_np[nearest_idx], axis=1)
                    if dedup_dists.min() * 1000 < dedup_dist_mm:
                        continue
                new_names.append(addb_name)
                new_vidx.append(nearest_idx)
                used_vidx.add(nearest_idx)
                used_vidx_positions = np.vstack([used_vidx_positions, verts_np[nearest_idx]])
                new_match_info.append({
                    'addb_name': addb_name,
                    'bsm_name': f'OFFSET_{body_name}',
                    'vertex_index': nearest_idx,
                    'match_tier': 2,
                    'dist_mm': dist_mm,
                    'body_name': body_name,
                })
                continue

            # AddB T-pose world → SKEL T-pose world 좌표 변환
            # AddB: X=forward, Y=up, Z=right (OpenSim convention)
            # SKEL: X=-Z_addb, Y=Y_addb, Z=X_addb (검증된 변환)
            world_pos_skel = np.array([
                -world_pos_addb[2],
                world_pos_addb[1],
                world_pos_addb[0],
            ], dtype=np.float64)

            # SKEL T-pose mesh에서 nearest vertex
            dists_v = np.linalg.norm(verts_np - world_pos_skel, axis=1)  # [6890]
            nearest_idx = int(np.argmin(dists_v))
            dist_mm = dists_v[nearest_idx] * 1000.0

            if dist_mm > max_dist_mm:
                continue
            if nearest_idx in used_vidx:
                continue
            # Dedup: 이미 사용된 vertex와 너무 가까우면 skip (cluster markers 방지)
            if len(used_vidx_positions) > 0:
                dedup_dists = np.linalg.norm(used_vidx_positions - verts_np[nearest_idx], axis=1)
                if dedup_dists.min() * 1000 < dedup_dist_mm:
                    continue

            new_names.append(addb_name)
            new_vidx.append(nearest_idx)
            used_vidx.add(nearest_idx)
            used_vidx_positions = np.vstack([used_vidx_positions, verts_np[nearest_idx]])
            new_match_info.append({
                'addb_name': addb_name,
                'bsm_name': f'OFFSET_{body_name}',
                'vertex_index': nearest_idx,
                'match_tier': 2,  # Tier 2.5 → 2로 기록
                'dist_mm': dist_mm,
                'body_name': body_name,
            })

        # AddB skeleton 포즈 복원
        if skeleton is not None:
            try:
                skeleton.setPositions(orig_pos)
            except Exception:
                pass

        if not new_names:
            return 0

        # 기존 매칭에 추가
        self.marker_names = self.marker_names + new_names
        if self.vertex_indices is not None:
            new_vidx_tensor = torch.tensor(new_vidx, dtype=torch.long, device=self.device)
            self.vertex_indices = torch.cat([self.vertex_indices, new_vidx_tensor])
        else:
            self.vertex_indices = torch.tensor(new_vidx, dtype=torch.long, device=self.device)

        self._match_info = self._match_info + new_match_info
        self._tier_counts[2] = self._tier_counts.get(2, 0) + len(new_names)

        # marker_observations에 없는 마커를 _unmatched_names에서 제거
        self._unmatched_names = [n for n in self._unmatched_names if n not in set(new_names)]

        # target tensor 재구성
        self._build_targets(marker_observations, flip_xz)

        return len(new_names)


class VertexMarkerHandler:
    """
    Vertex-based marker loss following the SMPL2AddBiomechanics approach.

    For each marker in the subject's markersMap, finds the nearest SKEL vertex
    in the first-frame posed mesh. Uses that vertex index for all subsequent
    frames: marker_pred = skel_verts[:, vertex_idx, :].

    This avoids the body-rotation problem in MarkerHandler and gives ~20-30mm error
    (limited by mesh resolution vs. marker placement on skin surface).
    """

    def __init__(self, device: torch.device, bsm_markers_path: Optional[str] = None):
        self.device = device
        self.bsm_markers: Optional[Dict[str, int]] = None
        if bsm_markers_path is not None:
            with open(bsm_markers_path) as f:
                self.bsm_markers = yaml.safe_load(f)

        # Populated by prepare_from_markersmap()
        self.marker_names: List[str] = []
        self.vertex_indices: Optional[torch.Tensor] = None  # [K]
        self.marker_targets: Optional[torch.Tensor] = None   # [T, K, 3]
        self.marker_mask: Optional[torch.Tensor] = None       # [T, K]
        self.num_markers: int = 0

    def prepare_from_markersmap(
        self,
        markers_map,
        marker_observations: List[Dict[str, np.ndarray]],
        flip_xz: bool = True,
    ) -> int:
        """
        Build marker targets from AddB markersMap observations.

        Vertex indices are NOT set here — they are set later by
        find_nearest_vertices() using the first-frame posed SKEL mesh.

        Args:
            markers_map: osim.markersMap from nimblephysics.
            marker_observations: Per-frame observed marker positions.
            flip_xz: Apply AddB→SKEL coordinate conversion (negate X and Z).

        Returns:
            Number of markers.
        """
        marker_names = list(markers_map.keys())
        K = len(marker_names)
        if K == 0:
            self.num_markers = 0
            return 0

        self.marker_names = marker_names

        # Build per-frame target positions [T, K, 3]
        T = len(marker_observations)
        targets = np.full((T, K, 3), np.nan, dtype=np.float32)
        for t, obs in enumerate(marker_observations):
            for k, name in enumerate(self.marker_names):
                if name in obs:
                    pos = obs[name]
                    targets[t, k] = [float(pos[0]), float(pos[1]), float(pos[2])]

        # Apply coordinate conversion: AddB → SKEL (flip X and Z)
        if flip_xz:
            targets[:, :, 0] = -targets[:, :, 0]
            targets[:, :, 2] = -targets[:, :, 2]

        # Build mask for valid (non-NaN) observations
        valid_mask = ~np.isnan(targets[:, :, 0])  # [T, K]
        targets = np.nan_to_num(targets, nan=0.0)

        self.marker_targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        self.marker_mask = torch.tensor(valid_mask, device=self.device)
        self.num_markers = K

        return K

    def find_nearest_vertices(self, skel_interface, betas: torch.Tensor):
        """
        Find nearest SKEL vertex for each marker using the first-frame posed mesh.

        Uses the first valid observation of each marker as the target position,
        then finds the globally nearest vertex (not segment-constrained).

        Args:
            skel_interface: SKELInterface instance.
            betas: Shape parameters [10].
        """
        if self.num_markers == 0:
            return

        with torch.no_grad():
            # Use T-pose to find nearest vertices
            poses_zero = torch.zeros(1, 46, device=self.device)
            trans_zero = torch.zeros(1, 3, device=self.device)
            verts, _, _ = skel_interface.forward(
                betas.unsqueeze(0), poses_zero, trans_zero
            )
            verts = verts[0]  # [6890, 3]

            matched_vidx = []
            for k in range(self.num_markers):
                # Find first valid frame for this marker
                valid_frames = self.marker_mask[:, k].nonzero(as_tuple=True)[0]
                if len(valid_frames) == 0:
                    # No valid observations — use vertex 0 as fallback
                    matched_vidx.append(0)
                    continue

                t0 = valid_frames[0].item()
                target = self.marker_targets[t0, k]  # [3]

                # Find nearest vertex globally
                dists = torch.norm(verts - target, dim=1)
                matched_vidx.append(dists.argmin().item())

            self.vertex_indices = torch.tensor(
                matched_vidx, dtype=torch.long, device=self.device
            )

    def find_nearest_vertices_from_mesh(self, fitted_verts: torch.Tensor):
        """
        Find nearest SKEL vertex for each marker using the ALREADY-FITTED mesh.

        Unlike find_nearest_vertices() which uses T-pose, this uses the actual
        optimized mesh vertices. This gives much better matching because the
        mesh is in the same pose/position as the marker observations.

        Args:
            fitted_verts: SKEL mesh vertices from optimized pose [T, 6890, 3].
        """
        if self.num_markers == 0:
            return

        with torch.no_grad():
            matched_vidx = []
            match_dists = []
            for k in range(self.num_markers):
                # Find first valid frame for this marker
                valid_frames = self.marker_mask[:, k].nonzero(as_tuple=True)[0]
                if len(valid_frames) == 0:
                    matched_vidx.append(0)
                    match_dists.append(float('inf'))
                    continue

                t0 = valid_frames[0].item()
                target = self.marker_targets[t0, k]  # [3]

                # Use the fitted mesh at the SAME frame
                verts_t0 = fitted_verts[t0]  # [6890, 3]
                dists = torch.norm(verts_t0 - target, dim=1)
                min_dist = dists.min().item()
                matched_vidx.append(dists.argmin().item())
                match_dists.append(min_dist * 1000)  # mm

            self.vertex_indices = torch.tensor(
                matched_vidx, dtype=torch.long, device=self.device
            )
            self._match_dists_mm = match_dists

    def compute_marker_loss(
        self,
        skel_joints: torch.Tensor,
        skel_verts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute marker position loss using vertex positions.

        marker_pred = skel_verts[:, vertex_indices, :]

        Args:
            skel_joints: SKEL joint positions [T, 24, 3] (unused, for API compat).
            skel_verts: SKEL mesh vertices [T, 6890, 3].

        Returns:
            Scalar loss value.
        """
        if self.num_markers == 0 or self.vertex_indices is None:
            return torch.tensor(0.0, device=self.device)

        if skel_verts is None:
            return torch.tensor(0.0, device=self.device)

        pred = skel_verts[:, self.vertex_indices, :]  # [T, K, 3]
        diff = pred - self.marker_targets  # [T, K, 3]
        sq_error = (diff ** 2).sum(dim=-1)  # [T, K]

        masked_error = sq_error * self.marker_mask.float()
        n_valid = self.marker_mask.float().sum().clamp(min=1.0)

        return masked_error.sum() / n_valid

    def get_marker_statistics(
        self,
        skel_joints: torch.Tensor,
        skel_verts: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute per-marker error statistics (in mm)."""
        if self.num_markers == 0 or self.vertex_indices is None or skel_verts is None:
            return {}

        with torch.no_grad():
            pred = skel_verts[:, self.vertex_indices, :]
            diff = pred - self.marker_targets
            error = torch.norm(diff, dim=-1)  # [T, K] in meters

            stats = {}
            for k, name in enumerate(self.marker_names):
                valid = self.marker_mask[:, k]
                if valid.sum() > 0:
                    stats[name] = error[:, k][valid].mean().item() * 1000
            return stats

    def get_summary(self) -> str:
        """Return a summary string for logging."""
        if self.num_markers == 0:
            return "VertexMarkerHandler: 0 markers"
        has_vidx = self.vertex_indices is not None
        n_valid_frames = self.marker_mask.all(dim=1).sum().item() if self.marker_mask is not None else 0
        T = self.marker_mask.shape[0] if self.marker_mask is not None else 0
        return (f"VertexMarkerHandler: {self.num_markers} markers, "
                f"vertex_indices={'set' if has_vidx else 'pending'}, "
                f"{n_valid_frames}/{T} frames all visible")
