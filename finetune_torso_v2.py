#!/usr/bin/env python3
"""
Torso Regularization Finetuning for SKEL Model - Version 2

Uses actual SKEL model forward pass to compute torso orientation,
ensuring accurate kinematics.

Usage:
    python -m skel_force_vis.finetune_torso_v2 \
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

from models.skel_model import SKELModelWrapper


# SKEL model path
SKEL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'

# SKEL joint indices
PELVIS_JOINT = 0
THORAX_JOINT = 12
HEAD_JOINT = 13

# SKEL pose indices for spine/head
LUMBAR_IDX = [17, 18, 19]   # lumbar_bending, lumbar_extension, lumbar_twist
THORAX_IDX = [20, 21, 22]   # thorax_bending, thorax_extension, thorax_twist
HEAD_IDX = [23, 24, 25]     # head_bending, head_extension, head_twist


class SKELTorsoFinetunerV2:
    """
    Finetuner for SKEL poses with torso regularization.

    Uses actual SKEL model forward pass to get accurate joint positions,
    then computes torso angle from thorax-pelvis vector vs world up.
    """

    def __init__(
        self,
        model_path: str = SKEL_MODEL_PATH,
        gender: str = 'male',
        device: str = 'cuda',
        # Regularization params - using joint positions directly
        min_torso_height_ratio: float = 0.85,  # thorax.y should be >= 0.85 * pelvis.y + offset
        head_behind_pelvis_weight: float = 1.0,  # penalize head being in front of pelvis
        pose_prior_weight: float = 0.5,  # keep poses close to original
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.gender = gender

        self.min_torso_height_ratio = min_torso_height_ratio
        self.head_behind_pelvis_weight = head_behind_pelvis_weight
        self.pose_prior_weight = pose_prior_weight

        # Load SKEL model
        self._load_model()

    def _load_model(self):
        """Load SKEL model."""
        try:
            self.skel = SKELModelWrapper(
                model_path=self.model_path,
                gender=self.gender,
                device=self.device
            )
            print(f"[Finetuner] SKEL model loaded (device={self.device})")
        except Exception as e:
            print(f"[Finetuner] Failed to load SKEL model: {e}")
            raise

    def compute_posture_metrics(
        self,
        joints: torch.Tensor  # [B, 24, 3] or [24, 3]
    ) -> dict:
        """
        Compute posture metrics from SKEL joint positions.

        Args:
            joints: Joint positions from SKEL forward pass

        Returns:
            Dict with torso_angle_deg, head_forward_offset, etc.
        """
        single = joints.ndim == 2
        if single:
            joints = joints.unsqueeze(0)

        B = joints.shape[0]

        pelvis = joints[:, PELVIS_JOINT]   # [B, 3]
        thorax = joints[:, THORAX_JOINT]   # [B, 3]
        head = joints[:, HEAD_JOINT]       # [B, 3]

        # Torso vector (pelvis -> thorax)
        torso_vec = thorax - pelvis  # [B, 3]

        # World up vector
        y_world = torch.tensor([0.0, 1.0, 0.0], device=joints.device)

        # Compute angle between torso and world up
        torso_len = torch.norm(torso_vec, dim=-1, keepdim=True) + 1e-8
        torso_unit = torso_vec / torso_len

        cos_theta = (torso_unit * y_world).sum(dim=-1)  # [B]
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        torso_angle_rad = torch.acos(cos_theta)
        torso_angle_deg = torso_angle_rad * 180.0 / np.pi

        # Head forward offset (in pelvis local frame, approximated)
        # Using Z axis as forward (positive Z = forward)
        head_offset = head - pelvis  # [B, 3]
        head_forward = head_offset[:, 2]  # Z component

        if single:
            return {
                'torso_angle_deg': torso_angle_deg[0],
                'head_forward_m': head_forward[0],
                'torso_vec': torso_vec[0],
            }
        else:
            return {
                'torso_angle_deg': torso_angle_deg,
                'head_forward_m': head_forward,
                'torso_vec': torso_vec,
            }

    def compute_regularization_loss(
        self,
        joints: torch.Tensor,  # [B, 24, 3]
        target_torso_angle_deg: float = 15.0,  # Target: ~15 degrees forward lean is natural
        max_head_forward: float = 0.15,  # Max head forward offset in meters
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute posture regularization loss.

        Instead of using rotation matrices, we directly use joint positions
        to compute posture errors.
        """
        single = joints.ndim == 2
        if single:
            joints = joints.unsqueeze(0)

        B = joints.shape[0]

        pelvis = joints[:, PELVIS_JOINT]   # [B, 3]
        thorax = joints[:, THORAX_JOINT]   # [B, 3]
        head = joints[:, HEAD_JOINT]       # [B, 3]

        # === Loss 1: Torso should be mostly upright ===
        # Torso vector should align with world Y (up)
        torso_vec = thorax - pelvis  # [B, 3]
        torso_len = torch.norm(torso_vec, dim=-1, keepdim=True) + 1e-8
        torso_unit = torso_vec / torso_len

        y_world = torch.tensor([0.0, 1.0, 0.0], device=joints.device)
        cos_theta = (torso_unit * y_world).sum(dim=-1)  # [B]

        # Target: cos(15 deg) ≈ 0.966
        target_cos = np.cos(np.radians(target_torso_angle_deg))

        # Penalize when cos_theta < target_cos (too much forward lean)
        upright_violation = torch.relu(target_cos - cos_theta)
        upright_loss = (upright_violation ** 2).mean()

        # === Loss 2: Head should not be too far forward ===
        head_offset = head - pelvis  # [B, 3]
        head_forward = head_offset[:, 2]  # Z = forward

        head_violation = torch.relu(head_forward - max_head_forward)
        head_loss = (head_violation ** 2).mean()

        # === Loss 3: Thorax should be above pelvis (not collapsed down) ===
        thorax_height = thorax[:, 1] - pelvis[:, 1]  # Y difference
        # Expect thorax to be ~0.3-0.4m above pelvis
        min_height = 0.25
        height_violation = torch.relu(min_height - thorax_height)
        height_loss = (height_violation ** 2).mean()

        total_loss = upright_loss + self.head_behind_pelvis_weight * head_loss + height_loss

        # Debug metrics
        cos_theta_clamped = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        angles_deg = torch.acos(cos_theta_clamped) * 180.0 / np.pi

        loss_dict = {
            'upright_loss': upright_loss.item(),
            'head_loss': head_loss.item(),
            'height_loss': height_loss.item(),
            'mean_torso_angle_deg': angles_deg.mean().item(),
            'mean_head_forward_m': head_forward.mean().item(),
            'mean_thorax_height_m': thorax_height.mean().item(),
        }

        return total_loss, loss_dict

    def finetune_poses(
        self,
        poses: np.ndarray,
        betas: np.ndarray,
        trans: np.ndarray,
        epochs: int = 100,
        lr: float = 0.005,
        target_torso_angle_deg: float = 15.0,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Finetune pose parameters with torso regularization.

        Only optimizes spine/head parameters (indices 17-25).
        """
        T = poses.shape[0]

        # Convert to tensors
        poses_t = torch.from_numpy(poses).float().to(self.device)
        betas_t = torch.from_numpy(betas).float().to(self.device)
        trans_t = torch.from_numpy(trans).float().to(self.device)

        # Store original poses for prior
        original_poses = poses_t.clone()

        # Only optimize spine/head parameters
        spine_head_idx = LUMBAR_IDX + THORAX_IDX + HEAD_IDX  # [17, 18, 19, 20, 21, 22, 23, 24, 25]

        # Create optimizable parameters
        spine_head_params = poses_t[:, spine_head_idx].clone().requires_grad_(True)

        optimizer = optim.Adam([spine_head_params], lr=lr)

        if verbose:
            print(f"\n[Finetuning V2] {T} frames, {epochs} epochs, lr={lr}")
            print(f"[Finetuning V2] Target torso angle: {target_torso_angle_deg}°")
            print(f"[Finetuning V2] Optimizing indices: {spine_head_idx}")

        # Analyze before optimization
        if verbose:
            with torch.no_grad():
                _, joints = self.skel.forward(betas_t, poses_t, trans_t)
                metrics = self.compute_posture_metrics(joints)
                print(f"[Before] Torso angle: mean={metrics['torso_angle_deg'].mean():.1f}°, "
                      f"max={metrics['torso_angle_deg'].max():.1f}°")

        best_loss = float('inf')
        best_params = spine_head_params.clone()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Build full pose tensor
            current_poses = poses_t.clone()
            current_poses[:, spine_head_idx] = spine_head_params

            # Forward through SKEL model
            _, joints = self.skel.forward(betas_t, current_poses, trans_t)

            # Regularization loss
            reg_loss, loss_dict = self.compute_regularization_loss(
                joints, target_torso_angle_deg=target_torso_angle_deg
            )

            # Pose prior loss
            prior_loss = self.pose_prior_weight * (
                (spine_head_params - original_poses[:, spine_head_idx]) ** 2
            ).mean()

            total_loss = reg_loss + prior_loss

            # Backprop
            total_loss.backward()
            optimizer.step()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_params = spine_head_params.clone().detach()

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}: total={total_loss.item():.6f}, "
                      f"reg={reg_loss.item():.6f}, prior={prior_loss.item():.6f}, "
                      f"torso={loss_dict['mean_torso_angle_deg']:.1f}°")

        # Use best parameters
        refined_poses = poses_t.clone()
        refined_poses[:, spine_head_idx] = best_params

        # Analyze after optimization
        if verbose:
            with torch.no_grad():
                _, joints = self.skel.forward(betas_t, refined_poses, trans_t)
                metrics = self.compute_posture_metrics(joints)
                print(f"[After] Torso angle: mean={metrics['torso_angle_deg'].mean():.1f}°, "
                      f"max={metrics['torso_angle_deg'].max():.1f}°")

        return refined_poses.cpu().detach().numpy()

    def export_skeleton_obj(
        self,
        poses: np.ndarray,
        betas: np.ndarray,
        trans: np.ndarray,
        output_dir: str,
        prefix: str = "refined",
    ):
        """Export skeleton OBJ files using SKEL model."""
        os.makedirs(output_dir, exist_ok=True)
        T = poses.shape[0]

        print(f"[Export] Exporting {T} frames to {output_dir}")

        poses_t = torch.from_numpy(poses).float().to(self.device)
        betas_t = torch.from_numpy(betas).float().to(self.device)
        trans_t = torch.from_numpy(trans).float().to(self.device)

        for i in range(T):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Frame {i+1}/{T}")

            with torch.no_grad():
                # Get skeleton mesh
                output = self.skel.forward_with_skeleton(
                    betas_t.unsqueeze(0) if betas_t.ndim == 1 else betas_t,
                    poses_t[i:i+1],
                    trans_t[i:i+1]
                )

                if 'skeleton_vertices' in output:
                    skel_verts = output['skeleton_vertices'][0].cpu().numpy()
                    skel_faces = self.skel.skel_faces

                    obj_path = os.path.join(output_dir, f"{prefix}_skeleton_{i:04d}.obj")
                    self._write_obj(obj_path, skel_verts, skel_faces)

        print(f"[Export] Done!")

    def _write_obj(self, path: str, vertices: np.ndarray, faces: np.ndarray):
        """Write OBJ file."""
        with open(path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    parser = argparse.ArgumentParser(description="Finetune SKEL poses with torso regularization V2")
    parser.add_argument("--input", "-i", required=True, help="Input skel_params.npz file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Optimization epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--target-angle", type=float, default=15.0, help="Target torso angle (degrees)")
    parser.add_argument("--pose-prior-weight", type=float, default=0.5, help="Pose prior weight")
    parser.add_argument("--export-obj", action="store_true", help="Export OBJ files")
    parser.add_argument("--gender", default="male", choices=["male", "female"], help="Gender")
    parser.add_argument("--device", default="cuda", help="Device")

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
    finetuner = SKELTorsoFinetunerV2(
        gender=args.gender,
        device=args.device,
        pose_prior_weight=args.pose_prior_weight,
    )

    # Finetune
    refined_poses = finetuner.finetune_poses(
        poses, betas, trans,
        epochs=args.epochs,
        lr=args.lr,
        target_torso_angle_deg=args.target_angle,
        verbose=True
    )

    # Save refined parameters
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "skel_params_refined.npz")
    np.savez(output_path, poses=refined_poses, betas=betas, trans=trans)
    print(f"\nSaved refined params: {output_path}")

    # Export OBJ files
    if args.export_obj:
        finetuner.export_skeleton_obj(
            refined_poses, betas, trans,
            output_dir=args.output,
            prefix="refined",
        )


if __name__ == "__main__":
    main()
