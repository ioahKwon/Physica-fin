#!/usr/bin/env python3
"""
Torso Regularization Finetuning for SKEL Model

Applies soft torso regularization losses to existing SKEL pose parameters
to prevent forward collapse issues common in AddBiomechanics data.

Usage:
    python -m skel_force_vis.finetune_torso \
        --input /path/to/skel_params.npz \
        --output /path/to/output_dir \
        --epochs 100
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, Tuple

import torch
import torch.optim as optim

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from skel_force_vis.torso_regularization import (
    compute_torso_angle_deg,
    compute_head_forward_offset,
)


# SKEL model path
SKEL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'


def euler_to_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (XYZ order) to rotation matrix.

    Args:
        angles: [3] tensor of Euler angles in radians (rx, ry, rz)

    Returns:
        R: [3, 3] rotation matrix
    """
    rx, ry, rz = angles[0], angles[1], angles[2]

    cos_x, sin_x = torch.cos(rx), torch.sin(rx)
    cos_y, sin_y = torch.cos(ry), torch.sin(ry)
    cos_z, sin_z = torch.cos(rz), torch.sin(rz)

    # Rotation matrices
    Rx = torch.stack([
        torch.stack([torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx)]),
        torch.stack([torch.zeros_like(rx), cos_x, -sin_x]),
        torch.stack([torch.zeros_like(rx), sin_x, cos_x])
    ])

    Ry = torch.stack([
        torch.stack([cos_y, torch.zeros_like(ry), sin_y]),
        torch.stack([torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry)]),
        torch.stack([-sin_y, torch.zeros_like(ry), cos_y])
    ])

    Rz = torch.stack([
        torch.stack([cos_z, -sin_z, torch.zeros_like(rz)]),
        torch.stack([sin_z, cos_z, torch.zeros_like(rz)]),
        torch.stack([torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)])
    ])

    # R = Rz @ Ry @ Rx
    R = torch.mm(torch.mm(Rz, Ry), Rx)
    return R


def euler_to_rotation_matrix_batch(angles: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of Euler angles to rotation matrices.

    Args:
        angles: [B, 3] tensor of Euler angles

    Returns:
        R: [B, 3, 3] rotation matrices
    """
    B = angles.shape[0]
    Rs = []
    for i in range(B):
        R = euler_to_rotation_matrix(angles[i])
        Rs.append(R)
    return torch.stack(Rs)


class SKELTorsoFinetuner:
    """
    Finetuner for SKEL poses with torso regularization.

    Only optimizes torso-related pose parameters:
    - Lumbar (17-19): lumbar_bending, lumbar_extension, lumbar_twist
    - Thorax (20-22): thorax_bending, thorax_extension, thorax_twist
    - Head (23-25): head_bending, head_extension, head_twist
    """

    # SKEL pose parameter indices for spine/head
    LUMBAR_IDX = [17, 18, 19]  # lumbar_bending, lumbar_extension, lumbar_twist
    THORAX_IDX = [20, 21, 22]  # thorax_bending, thorax_extension, thorax_twist
    HEAD_IDX = [23, 24, 25]    # head_bending, head_extension, head_twist

    # Joint indices in SKEL
    PELVIS_JOINT = 0
    THORAX_JOINT = 12
    HEAD_JOINT = 13

    def __init__(
        self,
        model_path: str = SKEL_MODEL_PATH,
        gender: str = 'male',
        device: str = 'cuda',
        # Regularization params
        upright_thresh_deg: float = 25.0,
        upright_weight: float = 1.0,
        head_fwd_thresh_m: float = 0.15,
        head_fwd_weight: float = 1.0,
        # Pose prior weight (keep close to original)
        pose_prior_weight: float = 0.1,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.gender = gender

        # Regularization thresholds
        self.upright_thresh_cos = np.cos(np.radians(upright_thresh_deg))
        self.upright_weight = upright_weight
        self.head_fwd_thresh = head_fwd_thresh_m
        self.head_fwd_weight = head_fwd_weight
        self.pose_prior_weight = pose_prior_weight

        # Load SKEL model
        self._load_model()

    def _load_model(self):
        """Load SKEL model."""
        try:
            from models.skel_model import SKELModelWrapper
            self.skel = SKELModelWrapper(
                model_path=self.model_path,
                gender=self.gender,
                device=self.device
            )
            print(f"[Finetuner] SKEL model loaded (device={self.device})")
        except Exception as e:
            print(f"[Finetuner] Failed to load SKEL model: {e}")
            print("[Finetuner] Running without SKEL model (will use simplified kinematics)")
            self.skel = None

    def compute_body_transforms_simplified(
        self,
        poses: torch.Tensor,
        trans: torch.Tensor
    ) -> Tuple[dict, dict]:
        """
        Simplified kinematic computation for torso/head without full SKEL model.

        Uses simplified forward kinematics chain:
        pelvis → lumbar → thorax → head

        Args:
            poses: [46] or [B, 46] pose parameters
            trans: [3] or [B, 3] translation

        Returns:
            body_rotations: Dict[str, Tensor] - {body_name: [3,3] or [B,3,3]}
            body_positions: Dict[str, Tensor] - {body_name: [3] or [B,3]}
        """
        single = poses.ndim == 1
        if single:
            poses = poses.unsqueeze(0)
            trans = trans.unsqueeze(0)

        B = poses.shape[0]

        # Extract Euler angles
        pelvis_euler = poses[:, 0:3]   # pelvis_tilt, list, rotation
        lumbar_euler = poses[:, 17:20]  # lumbar_bending, extension, twist
        thorax_euler = poses[:, 20:23]  # thorax_bending, extension, twist
        head_euler = poses[:, 23:26]    # head_bending, extension, twist

        # Convert to rotation matrices
        R_pelvis = euler_to_rotation_matrix_batch(pelvis_euler)   # [B, 3, 3]
        R_lumbar = euler_to_rotation_matrix_batch(lumbar_euler)   # [B, 3, 3]
        R_thorax = euler_to_rotation_matrix_batch(thorax_euler)   # [B, 3, 3]
        R_head = euler_to_rotation_matrix_batch(head_euler)       # [B, 3, 3]

        # Chain: R_world = R_pelvis @ R_lumbar @ R_thorax (for thorax)
        #        R_world = R_pelvis @ R_lumbar @ R_thorax @ R_head (for head)
        R_thorax_world = torch.bmm(torch.bmm(R_pelvis, R_lumbar), R_thorax)
        R_head_world = torch.bmm(R_thorax_world, R_head)

        # Approximate positions (simplified - not accurate but useful for regularization)
        # Pelvis at translation
        p_pelvis = trans  # [B, 3]

        # Thorax roughly 0.3m above pelvis
        thorax_offset = torch.tensor([0.0, 0.3, 0.0], device=self.device).expand(B, -1)
        p_thorax = p_pelvis + torch.bmm(R_pelvis, thorax_offset.unsqueeze(-1)).squeeze(-1)

        # Head roughly 0.3m above thorax
        head_offset = torch.tensor([0.0, 0.3, 0.0], device=self.device).expand(B, -1)
        p_head = p_thorax + torch.bmm(R_thorax_world, head_offset.unsqueeze(-1)).squeeze(-1)

        body_rotations = {
            'pelvis': R_pelvis[0] if single else R_pelvis,
            'torso': R_thorax_world[0] if single else R_thorax_world,
            'head': R_head_world[0] if single else R_head_world,
        }

        body_positions = {
            'pelvis': p_pelvis[0] if single else p_pelvis,
            'torso': p_thorax[0] if single else p_thorax,
            'head': p_head[0] if single else p_head,
        }

        return body_rotations, body_positions

    def compute_body_transforms_skel(
        self,
        poses: torch.Tensor,
        betas: torch.Tensor,
        trans: torch.Tensor
    ) -> Tuple[dict, dict]:
        """
        Compute body transforms using full SKEL model.
        """
        if self.skel is None:
            return self.compute_body_transforms_simplified(poses, trans)

        single = poses.ndim == 1
        if single:
            poses = poses.unsqueeze(0)
            trans = trans.unsqueeze(0)
            betas = betas.unsqueeze(0)

        # Forward pass through SKEL
        vertices, joints = self.skel.forward(betas, poses, trans)

        # joints: [B, 24, 3]
        p_pelvis = joints[:, self.PELVIS_JOINT]  # [B, 3]
        p_thorax = joints[:, self.THORAX_JOINT]  # [B, 3]
        p_head = joints[:, self.HEAD_JOINT]      # [B, 3]

        # For rotations, use simplified computation
        # (SKEL doesn't directly expose body rotation matrices)
        B = poses.shape[0]
        pelvis_euler = poses[:, 0:3]
        lumbar_euler = poses[:, 17:20]
        thorax_euler = poses[:, 20:23]
        head_euler = poses[:, 23:26]

        R_pelvis = euler_to_rotation_matrix_batch(pelvis_euler)
        R_lumbar = euler_to_rotation_matrix_batch(lumbar_euler)
        R_thorax = euler_to_rotation_matrix_batch(thorax_euler)
        R_head = euler_to_rotation_matrix_batch(head_euler)

        R_thorax_world = torch.bmm(torch.bmm(R_pelvis, R_lumbar), R_thorax)
        R_head_world = torch.bmm(R_thorax_world, R_head)

        body_rotations = {
            'pelvis': R_pelvis[0] if single else R_pelvis,
            'torso': R_thorax_world[0] if single else R_thorax_world,
            'head': R_head_world[0] if single else R_head_world,
        }

        body_positions = {
            'pelvis': p_pelvis[0] if single else p_pelvis,
            'torso': p_thorax[0] if single else p_thorax,
            'head': p_head[0] if single else p_head,
        }

        return body_rotations, body_positions

    def compute_regularization_loss(
        self,
        body_rotations: dict,
        body_positions: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute torso regularization losses.

        Returns:
            total_loss: Scalar tensor
            loss_dict: Dict with individual losses
        """
        device = body_rotations['torso'].device

        # Handle batch vs single frame
        if body_rotations['torso'].ndim == 3:
            # Batch mode [B, 3, 3]
            return self._compute_reg_loss_batch(body_rotations, body_positions)

        # Single frame mode
        R_torso = body_rotations['torso']  # [3, 3]
        p_head = body_positions['head']    # [3]
        p_pelvis = body_positions['pelvis']  # [3]
        R_pelvis = body_rotations['pelvis']  # [3, 3]

        # === Loss 1: Torso Upright ===
        y_local = torch.tensor([0.0, 1.0, 0.0], device=device)
        y_world = torch.tensor([0.0, 1.0, 0.0], device=device)

        u_torso = R_torso @ y_local
        cos_theta = torch.dot(u_torso, y_world)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        upright_violation = torch.relu(self.upright_thresh_cos - cos_theta)
        upright_loss = self.upright_weight * upright_violation ** 2

        # === Loss 2: Head Forward Offset ===
        fwd_local = torch.tensor([0.0, 0.0, 1.0], device=device)
        d_fwd = R_pelvis @ fwd_local
        d_fwd = d_fwd / (torch.norm(d_fwd) + 1e-8)

        head_offset = p_head - p_pelvis
        delta_fwd = torch.dot(head_offset, d_fwd)

        head_fwd_violation = torch.relu(delta_fwd - self.head_fwd_thresh)
        head_fwd_loss = self.head_fwd_weight * head_fwd_violation ** 2

        total_loss = upright_loss + head_fwd_loss

        loss_dict = {
            'upright_loss': upright_loss.item(),
            'head_fwd_loss': head_fwd_loss.item(),
            'torso_angle_deg': (torch.acos(cos_theta) * 180.0 / np.pi).item(),
            'head_fwd_offset_m': delta_fwd.item(),
        }

        return total_loss, loss_dict

    def _compute_reg_loss_batch(
        self,
        body_rotations: dict,
        body_positions: dict
    ) -> Tuple[torch.Tensor, dict]:
        """Batch version of regularization loss."""
        device = body_rotations['torso'].device
        B = body_rotations['torso'].shape[0]

        R_torso = body_rotations['torso']     # [B, 3, 3]
        p_head = body_positions['head']       # [B, 3]
        p_pelvis = body_positions['pelvis']   # [B, 3]
        R_pelvis = body_rotations['pelvis']   # [B, 3, 3]

        # Torso upright
        y_local = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 3, 1)
        y_world = torch.tensor([0.0, 1.0, 0.0], device=device)

        u_torso = torch.bmm(R_torso, y_local.expand(B, -1, -1)).squeeze(-1)  # [B, 3]
        cos_theta = (u_torso * y_world).sum(dim=-1)  # [B]
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        upright_violation = torch.relu(self.upright_thresh_cos - cos_theta)
        upright_loss = self.upright_weight * (upright_violation ** 2).mean()

        # Head forward
        fwd_local = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1)
        d_fwd = torch.bmm(R_pelvis, fwd_local.expand(B, -1, -1)).squeeze(-1)  # [B, 3]
        d_fwd = d_fwd / (torch.norm(d_fwd, dim=-1, keepdim=True) + 1e-8)

        head_offset = p_head - p_pelvis  # [B, 3]
        delta_fwd = (head_offset * d_fwd).sum(dim=-1)  # [B]

        head_fwd_violation = torch.relu(delta_fwd - self.head_fwd_thresh)
        head_fwd_loss = self.head_fwd_weight * (head_fwd_violation ** 2).mean()

        total_loss = upright_loss + head_fwd_loss

        loss_dict = {
            'upright_loss': upright_loss.item(),
            'head_fwd_loss': head_fwd_loss.item(),
            'mean_torso_angle_deg': (torch.acos(cos_theta) * 180.0 / np.pi).mean().item(),
            'mean_head_fwd_offset_m': delta_fwd.mean().item(),
        }

        return total_loss, loss_dict

    def finetune_poses(
        self,
        poses: np.ndarray,
        betas: np.ndarray,
        trans: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Finetune pose parameters with torso regularization.

        Only optimizes spine/head parameters (indices 17-25).

        Args:
            poses: [T, 46] original pose parameters
            betas: [10] shape parameters
            trans: [T, 3] translations
            epochs: Number of optimization epochs
            lr: Learning rate
            verbose: Print progress

        Returns:
            refined_poses: [T, 46] finetuned pose parameters
        """
        T = poses.shape[0]

        # Convert to tensors
        poses_t = torch.from_numpy(poses).float().to(self.device)
        betas_t = torch.from_numpy(betas).float().to(self.device)
        trans_t = torch.from_numpy(trans).float().to(self.device)

        # Store original poses for prior
        original_poses = poses_t.clone()

        # Only optimize spine/head parameters
        spine_head_idx = self.LUMBAR_IDX + self.THORAX_IDX + self.HEAD_IDX  # [17-25]

        # Create optimizable parameters for spine/head only
        spine_head_params = poses_t[:, spine_head_idx].clone().requires_grad_(True)

        optimizer = optim.Adam([spine_head_params], lr=lr)

        if verbose:
            print(f"\n[Finetuning] {T} frames, {epochs} epochs, lr={lr}")
            print(f"[Finetuning] Optimizing indices: {spine_head_idx}")

        # Analyze before optimization
        if verbose:
            self._analyze_poses(poses_t, betas_t, trans_t, prefix="Before")

        best_loss = float('inf')
        best_params = spine_head_params.clone()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Build full pose tensor
            current_poses = poses_t.clone()
            current_poses[:, spine_head_idx] = spine_head_params

            # Compute body transforms
            body_rotations, body_positions = self.compute_body_transforms_simplified(
                current_poses, trans_t
            )

            # Regularization loss
            reg_loss, loss_dict = self.compute_regularization_loss(
                body_rotations, body_positions
            )

            # Pose prior loss (stay close to original)
            prior_loss = self.pose_prior_weight * (
                (spine_head_params - original_poses[:, spine_head_idx]) ** 2
            ).mean()

            total_loss = reg_loss + prior_loss

            # Backprop
            total_loss.backward()
            optimizer.step()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_params = spine_head_params.clone()

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}: total={total_loss.item():.6f}, "
                      f"reg={reg_loss.item():.6f}, prior={prior_loss.item():.6f}, "
                      f"torso_angle={loss_dict.get('mean_torso_angle_deg', loss_dict.get('torso_angle_deg', 0)):.1f}°")

        # Use best parameters
        refined_poses = poses_t.clone()
        refined_poses[:, spine_head_idx] = best_params.detach()

        # Analyze after optimization
        if verbose:
            self._analyze_poses(refined_poses, betas_t, trans_t, prefix="After")

        return refined_poses.cpu().numpy()

    def _analyze_poses(
        self,
        poses: torch.Tensor,
        betas: torch.Tensor,
        trans: torch.Tensor,
        prefix: str = ""
    ):
        """Analyze pose statistics."""
        with torch.no_grad():
            body_rotations, body_positions = self.compute_body_transforms_simplified(
                poses, trans
            )

            # Handle batch
            R_torso = body_rotations['torso']  # [B, 3, 3]
            if R_torso.ndim == 2:
                R_torso = R_torso.unsqueeze(0)

            B = R_torso.shape[0]
            y_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).view(1, 3, 1)
            y_world = torch.tensor([0.0, 1.0, 0.0], device=self.device)

            u_torso = torch.bmm(R_torso, y_local.expand(B, -1, -1)).squeeze(-1)
            cos_theta = (u_torso * y_world).sum(dim=-1)
            angles_deg = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)) * 180.0 / np.pi

            print(f"[{prefix}] Torso angle: mean={angles_deg.mean():.1f}°, "
                  f"max={angles_deg.max():.1f}°, min={angles_deg.min():.1f}°")

    def export_obj_files(
        self,
        poses: np.ndarray,
        betas: np.ndarray,
        trans: np.ndarray,
        output_dir: str,
        prefix: str = "refined",
        export_skin: bool = True,
        export_skeleton: bool = True,
    ):
        """
        Export OBJ mesh files for all frames.

        Args:
            poses: [T, 46] pose parameters
            betas: [10] shape parameters
            trans: [T, 3] translations
            output_dir: Output directory
            prefix: Filename prefix
            export_skin: Export skin mesh
            export_skeleton: Export skeleton mesh
        """
        if self.skel is None:
            print("[Export] SKEL model not loaded, cannot export OBJ files")
            return

        os.makedirs(output_dir, exist_ok=True)
        T = poses.shape[0]

        print(f"[Export] Exporting {T} frames to {output_dir}")

        poses_t = torch.from_numpy(poses).float().to(self.device)
        betas_t = torch.from_numpy(betas).float().to(self.device)
        trans_t = torch.from_numpy(trans).float().to(self.device)

        for i in range(T):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Frame {i+1}/{T}")

            # Get meshes
            if export_skeleton:
                vertices, joints, skel_verts = self.skel.forward(
                    betas_t, poses_t[i], trans_t[i], return_skeleton=True
                )
            else:
                vertices, joints = self.skel.forward(
                    betas_t, poses_t[i], trans_t[i]
                )

            # Export skin mesh
            if export_skin:
                skin_path = os.path.join(output_dir, f"{prefix}_skin_{i:04d}.obj")
                self._write_obj(skin_path, vertices.cpu().numpy(), self.skel.faces)

            # Export skeleton mesh
            if export_skeleton and self.skel.skel_faces is not None:
                skel_path = os.path.join(output_dir, f"{prefix}_skeleton_{i:04d}.obj")
                self._write_obj(skel_path, skel_verts.cpu().numpy(), self.skel.skel_faces)

        print(f"[Export] Done! Saved to {output_dir}")

    def _write_obj(self, path: str, vertices: np.ndarray, faces: np.ndarray):
        """Write OBJ file."""
        with open(path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    parser = argparse.ArgumentParser(description="Finetune SKEL poses with torso regularization")
    parser.add_argument("--input", "-i", required=True, help="Input skel_params.npz file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Optimization epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--upright-thresh", type=float, default=25.0, help="Upright threshold (degrees)")
    parser.add_argument("--upright-weight", type=float, default=1.0, help="Upright loss weight")
    parser.add_argument("--head-fwd-thresh", type=float, default=0.15, help="Head forward threshold (meters)")
    parser.add_argument("--head-fwd-weight", type=float, default=1.0, help="Head forward loss weight")
    parser.add_argument("--pose-prior-weight", type=float, default=0.1, help="Pose prior weight")
    parser.add_argument("--export-obj", action="store_true", help="Export OBJ files")
    parser.add_argument("--gender", default="male", choices=["male", "female"], help="Gender")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Load input
    print(f"Loading: {args.input}")
    data = np.load(args.input)
    poses = data['poses']
    betas = data['betas']
    trans = data['trans']

    print(f"  poses: {poses.shape}")
    print(f"  betas: {betas.shape}")
    print(f"  trans: {trans.shape}")

    # Create finetuner
    finetuner = SKELTorsoFinetuner(
        gender=args.gender,
        device=args.device,
        upright_thresh_deg=args.upright_thresh,
        upright_weight=args.upright_weight,
        head_fwd_thresh_m=args.head_fwd_thresh,
        head_fwd_weight=args.head_fwd_weight,
        pose_prior_weight=args.pose_prior_weight,
    )

    # Finetune
    refined_poses = finetuner.finetune_poses(
        poses, betas, trans,
        epochs=args.epochs,
        lr=args.lr,
        verbose=True
    )

    # Save refined parameters
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "skel_params_refined.npz")
    np.savez(output_path, poses=refined_poses, betas=betas, trans=trans)
    print(f"\nSaved refined params: {output_path}")

    # Export OBJ files
    if args.export_obj:
        finetuner.export_obj_files(
            refined_poses, betas, trans,
            output_dir=args.output,
            prefix="refined",
            export_skin=True,
            export_skeleton=True,
        )


if __name__ == "__main__":
    main()
