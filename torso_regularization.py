"""
Torso Regularization Losses for Skeleton Fitting

Prevents forward collapse when fitting skeleton models to AddBiomechanics data
that lacks head/neck observations.

Two soft hinge losses:
1. Torso Upright Loss - prevents excessive torso pitch
2. Head Forward Offset Loss - prevents head from drifting too far forward

Author: SKEL Force Vis Team
"""

import numpy as np

# Optional: PyTorch for differentiable optimization
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# NumPy Implementation (for analysis / non-differentiable use)
# ============================================================================

def torso_upright_loss_numpy(
    R_torso: np.ndarray,
    thresh_deg: float = 25.0,
    weight: float = 0.05
) -> float:
    """
    Compute torso upright loss (soft hinge).

    Penalizes when torso deviates from world up by more than threshold.

    Args:
        R_torso: [3, 3] rotation matrix of torso body in world frame
        thresh_deg: Threshold angle in degrees (default: 25)
        weight: Loss weight (default: 0.05)

    Returns:
        Loss value (scalar)

    Math:
        u_torso = R_torso @ [0, 1, 0]  # Torso up vector
        cos_theta = u_torso . [0, 1, 0]  # Alignment with world up
        loss = weight * max(0, cos_thresh - cos_theta)^2
    """
    y_local = np.array([0.0, 1.0, 0.0])
    y_world = np.array([0.0, 1.0, 0.0])

    # Torso up vector in world space
    u_torso = R_torso @ y_local

    # Cosine of angle with world up
    cos_theta = np.dot(u_torso, y_world)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Threshold in cosine space
    cos_thresh = np.cos(np.radians(thresh_deg))

    # Hinge loss: penalize when cos_theta < cos_thresh (angle > threshold)
    violation = max(0.0, cos_thresh - cos_theta)

    return weight * violation ** 2


def head_forward_loss_numpy(
    p_head: np.ndarray,
    p_pelvis: np.ndarray,
    R_pelvis: np.ndarray,
    thresh_m: float = 0.20,
    weight: float = 0.05,
    forward_axis: str = 'z'
) -> float:
    """
    Compute head forward offset loss (soft hinge).

    Penalizes when head is too far forward relative to pelvis.

    Args:
        p_head: [3] head position in world frame
        p_pelvis: [3] pelvis position in world frame
        R_pelvis: [3, 3] pelvis rotation matrix in world frame
        thresh_m: Threshold in meters (default: 0.20)
        weight: Loss weight (default: 0.05)
        forward_axis: Local axis for forward direction ('x' or 'z')

    Returns:
        Loss value (scalar)

    Math:
        d_fwd = R_pelvis @ [0, 0, 1]  # Pelvis forward direction
        delta_fwd = (p_head - p_pelvis) . d_fwd  # Forward offset
        loss = weight * max(0, delta_fwd - thresh)^2
    """
    # Local forward axis
    if forward_axis == 'z':
        fwd_local = np.array([0.0, 0.0, 1.0])
    elif forward_axis == 'x':
        fwd_local = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError(f"forward_axis must be 'x' or 'z', got {forward_axis}")

    # Pelvis forward direction in world
    d_fwd = R_pelvis @ fwd_local
    d_fwd = d_fwd / (np.linalg.norm(d_fwd) + 1e-8)

    # Forward offset of head relative to pelvis
    head_offset = p_head - p_pelvis
    delta_fwd = np.dot(head_offset, d_fwd)

    # Hinge loss: penalize when delta_fwd > threshold
    violation = max(0.0, delta_fwd - thresh_m)

    return weight * violation ** 2


def compute_torso_angle_deg(R_torso: np.ndarray) -> float:
    """
    Compute torso pitch angle in degrees.

    Args:
        R_torso: [3, 3] rotation matrix of torso

    Returns:
        Angle in degrees (0 = upright, positive = forward lean)
    """
    y_local = np.array([0.0, 1.0, 0.0])
    y_world = np.array([0.0, 1.0, 0.0])

    u_torso = R_torso @ y_local
    cos_theta = np.clip(np.dot(u_torso, y_world), -1.0, 1.0)

    return np.degrees(np.arccos(cos_theta))


def compute_head_forward_offset(
    p_head: np.ndarray,
    p_pelvis: np.ndarray,
    R_pelvis: np.ndarray,
    forward_axis: str = 'z'
) -> float:
    """
    Compute head forward offset in meters.

    Args:
        p_head: [3] head position
        p_pelvis: [3] pelvis position
        R_pelvis: [3, 3] pelvis rotation
        forward_axis: 'x' or 'z'

    Returns:
        Forward offset in meters (positive = head in front)
    """
    if forward_axis == 'z':
        fwd_local = np.array([0.0, 0.0, 1.0])
    else:
        fwd_local = np.array([1.0, 0.0, 0.0])

    d_fwd = R_pelvis @ fwd_local
    d_fwd = d_fwd / (np.linalg.norm(d_fwd) + 1e-8)

    return np.dot(p_head - p_pelvis, d_fwd)


# ============================================================================
# PyTorch Implementation (for differentiable optimization)
# ============================================================================

if TORCH_AVAILABLE:

    class TorsoRegularizer:
        """
        Soft regularization losses to prevent forward collapse
        during skeleton fitting to AddBiomechanics data.

        Example:
            regularizer = TorsoRegularizer(
                upright_thresh_deg=25.0,
                upright_weight=0.05,
                head_fwd_thresh_m=0.20,
                head_fwd_weight=0.05,
            )

            losses = regularizer.compute_losses(body_rotations, body_positions)
            total_loss = data_loss + losses['total_reg_loss']
        """

        def __init__(
            self,
            # Torso upright loss params
            upright_thresh_deg: float = 25.0,
            upright_weight: float = 0.05,
            # Head forward offset loss params
            head_fwd_thresh_m: float = 0.20,
            head_fwd_weight: float = 0.05,
            # Body names (adjust for your skeleton)
            torso_body_name: str = "torso",
            head_body_name: str = "head",
            pelvis_body_name: str = "pelvis",
            # Forward axis
            forward_axis: str = 'z',
            # Smoothing
            use_smooth_relu: bool = False,
            smooth_beta: float = 10.0,
        ):
            """
            Initialize regularizer.

            Args:
                upright_thresh_deg: Max allowed torso pitch (degrees)
                upright_weight: Weight for upright loss
                head_fwd_thresh_m: Max allowed head forward offset (meters)
                head_fwd_weight: Weight for head forward loss
                torso_body_name: Name of torso body in skeleton
                head_body_name: Name of head body in skeleton
                pelvis_body_name: Name of pelvis body in skeleton
                forward_axis: Local forward axis ('x' or 'z')
                use_smooth_relu: Use softplus instead of ReLU for smoother gradients
                smooth_beta: Beta parameter for softplus
            """
            self.upright_thresh_cos = np.cos(np.radians(upright_thresh_deg))
            self.upright_weight = upright_weight
            self.head_fwd_thresh = head_fwd_thresh_m
            self.head_fwd_weight = head_fwd_weight

            self.torso_body = torso_body_name
            self.head_body = head_body_name
            self.pelvis_body = pelvis_body_name

            self.forward_axis = forward_axis
            self.use_smooth_relu = use_smooth_relu
            self.smooth_beta = smooth_beta

        def _hinge(self, x: torch.Tensor) -> torch.Tensor:
            """Apply hinge (ReLU or smooth version)."""
            if self.use_smooth_relu:
                return F.softplus(x * self.smooth_beta) / self.smooth_beta
            else:
                return F.relu(x)

        def compute_losses(
            self,
            body_rotations: dict,
            body_positions: dict,
        ) -> dict:
            """
            Compute regularization losses.

            Args:
                body_rotations: Dict mapping body name to [3,3] rotation matrix (torch.Tensor)
                body_positions: Dict mapping body name to [3] position vector (torch.Tensor)

            Returns:
                Dict with 'upright_loss', 'head_fwd_loss', 'total_reg_loss'
            """
            losses = {}
            device = body_rotations[self.torso_body].device

            # ============================================
            # Loss 1: Torso Upright (Soft Hinge)
            # ============================================
            R_torso = body_rotations[self.torso_body]  # [3, 3]
            y_local = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=R_torso.dtype)
            y_world = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=R_torso.dtype)

            # Torso up vector in world space
            u_torso = R_torso @ y_local  # [3]

            # Cosine of angle with world up
            cos_theta = torch.dot(u_torso, y_world)
            cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

            # Hinge loss: penalize when cos_theta < cos_thresh
            upright_violation = self._hinge(self.upright_thresh_cos - cos_theta)
            losses['upright_loss'] = self.upright_weight * upright_violation ** 2

            # ============================================
            # Loss 2: Head Forward Offset (Soft Hinge)
            # ============================================
            p_head = body_positions[self.head_body]      # [3]
            p_pelvis = body_positions[self.pelvis_body]  # [3]
            R_pelvis = body_rotations[self.pelvis_body]  # [3, 3]

            # Pelvis forward direction
            if self.forward_axis == 'z':
                fwd_local = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=R_pelvis.dtype)
            else:
                fwd_local = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=R_pelvis.dtype)

            d_fwd = R_pelvis @ fwd_local  # [3]
            d_fwd = d_fwd / (torch.norm(d_fwd) + 1e-8)

            # Forward offset of head relative to pelvis
            head_offset = p_head - p_pelvis  # [3]
            delta_fwd = torch.dot(head_offset, d_fwd)  # scalar

            # Hinge loss: penalize when delta_fwd > threshold
            head_violation = self._hinge(delta_fwd - self.head_fwd_thresh)
            losses['head_fwd_loss'] = self.head_fwd_weight * head_violation ** 2

            # ============================================
            # Total Regularization Loss
            # ============================================
            losses['total_reg_loss'] = losses['upright_loss'] + losses['head_fwd_loss']

            # Debug info
            losses['torso_angle_deg'] = torch.acos(cos_theta) * 180.0 / np.pi
            losses['head_fwd_offset_m'] = delta_fwd

            return losses

        def compute_losses_batch(
            self,
            body_rotations: dict,
            body_positions: dict,
        ) -> dict:
            """
            Compute regularization losses for a batch of frames.

            Args:
                body_rotations: Dict mapping body name to [B, 3, 3] rotation matrices
                body_positions: Dict mapping body name to [B, 3] position vectors

            Returns:
                Dict with losses (each is [B] tensor)
            """
            losses = {}
            device = body_rotations[self.torso_body].device
            B = body_rotations[self.torso_body].shape[0]

            # Torso upright
            R_torso = body_rotations[self.torso_body]  # [B, 3, 3]
            y_local = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0)  # [1, 3]
            y_world = torch.tensor([0.0, 1.0, 0.0], device=device)

            u_torso = torch.bmm(R_torso, y_local.expand(B, -1).unsqueeze(-1)).squeeze(-1)  # [B, 3]
            cos_theta = (u_torso * y_world).sum(dim=-1)  # [B]
            cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

            upright_violation = self._hinge(self.upright_thresh_cos - cos_theta)
            losses['upright_loss'] = self.upright_weight * upright_violation ** 2

            # Head forward
            p_head = body_positions[self.head_body]      # [B, 3]
            p_pelvis = body_positions[self.pelvis_body]  # [B, 3]
            R_pelvis = body_rotations[self.pelvis_body]  # [B, 3, 3]

            if self.forward_axis == 'z':
                fwd_local = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0)
            else:
                fwd_local = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0)

            d_fwd = torch.bmm(R_pelvis, fwd_local.expand(B, -1).unsqueeze(-1)).squeeze(-1)  # [B, 3]
            d_fwd = d_fwd / (torch.norm(d_fwd, dim=-1, keepdim=True) + 1e-8)

            head_offset = p_head - p_pelvis  # [B, 3]
            delta_fwd = (head_offset * d_fwd).sum(dim=-1)  # [B]

            head_violation = self._hinge(delta_fwd - self.head_fwd_thresh)
            losses['head_fwd_loss'] = self.head_fwd_weight * head_violation ** 2

            losses['total_reg_loss'] = losses['upright_loss'] + losses['head_fwd_loss']

            return losses


# ============================================================================
# NimblePhysics Integration Helper
# ============================================================================

def get_body_transforms_nimble(skel, pos: np.ndarray) -> tuple:
    """
    Get body rotations and positions from NimblePhysics skeleton.

    Args:
        skel: nimblephysics.dynamics.Skeleton
        pos: [N_DOF] joint positions

    Returns:
        body_rotations: Dict[str, np.ndarray] - {body_name: [3,3] rotation}
        body_positions: Dict[str, np.ndarray] - {body_name: [3] position}
    """
    skel.setPositions(pos)

    body_rotations = {}
    body_positions = {}

    for i in range(skel.getNumBodyNodes()):
        body = skel.getBodyNode(i)
        name = body.getName()
        transform = body.getWorldTransform()

        body_rotations[name] = np.array(transform.rotation())
        body_positions[name] = np.array(transform.translation())

    return body_rotations, body_positions


def compute_regularization_nimble(
    skel,
    pos: np.ndarray,
    upright_thresh_deg: float = 25.0,
    upright_weight: float = 0.05,
    head_fwd_thresh_m: float = 0.20,
    head_fwd_weight: float = 0.05,
    torso_name: str = "torso",
    head_name: str = "head",
    pelvis_name: str = "pelvis",
) -> dict:
    """
    Compute regularization losses for NimblePhysics skeleton.

    Args:
        skel: nimblephysics.dynamics.Skeleton
        pos: [N_DOF] joint positions
        ... (threshold and weight parameters)

    Returns:
        Dict with 'upright_loss', 'head_fwd_loss', 'total_reg_loss',
        'torso_angle_deg', 'head_fwd_offset_m'
    """
    body_rotations, body_positions = get_body_transforms_nimble(skel, pos)

    # Compute losses
    upright_loss = torso_upright_loss_numpy(
        body_rotations[torso_name],
        thresh_deg=upright_thresh_deg,
        weight=upright_weight
    )

    head_fwd_loss = head_forward_loss_numpy(
        body_positions[head_name],
        body_positions[pelvis_name],
        body_rotations[pelvis_name],
        thresh_m=head_fwd_thresh_m,
        weight=head_fwd_weight
    )

    # Debug info
    torso_angle = compute_torso_angle_deg(body_rotations[torso_name])
    head_offset = compute_head_forward_offset(
        body_positions[head_name],
        body_positions[pelvis_name],
        body_rotations[pelvis_name]
    )

    return {
        'upright_loss': upright_loss,
        'head_fwd_loss': head_fwd_loss,
        'total_reg_loss': upright_loss + head_fwd_loss,
        'torso_angle_deg': torso_angle,
        'head_fwd_offset_m': head_offset,
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== Torso Regularization Demo ===\n")

    # Example: Create sample rotation matrices

    # Upright torso (0 degrees pitch)
    R_upright = np.eye(3)
    print(f"Upright torso angle: {compute_torso_angle_deg(R_upright):.1f} deg")
    print(f"Upright loss: {torso_upright_loss_numpy(R_upright):.6f}")

    # Forward leaning torso (30 degrees pitch)
    pitch_rad = np.radians(30)
    R_forward = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    print(f"\n30 deg forward lean angle: {compute_torso_angle_deg(R_forward):.1f} deg")
    print(f"30 deg forward loss (thresh=25): {torso_upright_loss_numpy(R_forward, thresh_deg=25.0):.6f}")

    # Excessive forward lean (50 degrees)
    pitch_rad = np.radians(50)
    R_collapsed = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    print(f"\n50 deg collapsed angle: {compute_torso_angle_deg(R_collapsed):.1f} deg")
    print(f"50 deg collapsed loss (thresh=25): {torso_upright_loss_numpy(R_collapsed, thresh_deg=25.0):.6f}")

    # Head forward offset example
    print("\n--- Head Forward Offset ---")
    p_pelvis = np.array([0, 1.0, 0])
    R_pelvis = np.eye(3)

    p_head_normal = np.array([0, 1.7, 0.1])  # Slightly forward
    p_head_collapsed = np.array([0, 1.5, 0.4])  # Too far forward

    print(f"Normal head offset: {compute_head_forward_offset(p_head_normal, p_pelvis, R_pelvis):.3f} m")
    print(f"Normal head loss: {head_forward_loss_numpy(p_head_normal, p_pelvis, R_pelvis):.6f}")

    print(f"\nCollapsed head offset: {compute_head_forward_offset(p_head_collapsed, p_pelvis, R_pelvis):.3f} m")
    print(f"Collapsed head loss (thresh=0.2): {head_forward_loss_numpy(p_head_collapsed, p_pelvis, R_pelvis, thresh_m=0.2):.6f}")

    # PyTorch example
    if TORCH_AVAILABLE:
        print("\n--- PyTorch Example ---")
        regularizer = TorsoRegularizer(
            upright_thresh_deg=25.0,
            upright_weight=0.05,
            head_fwd_thresh_m=0.20,
            head_fwd_weight=0.05,
        )

        body_rotations = {
            'torso': torch.from_numpy(R_collapsed).float(),
            'pelvis': torch.eye(3),
        }
        body_positions = {
            'head': torch.tensor([0.0, 1.5, 0.4]),
            'pelvis': torch.tensor([0.0, 1.0, 0.0]),
        }

        losses = regularizer.compute_losses(body_rotations, body_positions)
        print(f"Upright loss: {losses['upright_loss'].item():.6f}")
        print(f"Head fwd loss: {losses['head_fwd_loss'].item():.6f}")
        print(f"Total reg loss: {losses['total_reg_loss'].item():.6f}")
        print(f"Torso angle: {losses['torso_angle_deg'].item():.1f} deg")
        print(f"Head fwd offset: {losses['head_fwd_offset_m'].item():.3f} m")
