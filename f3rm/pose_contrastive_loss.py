"""
pose_contrastive_loss.py

Pose-Level Contrastive Loss for F3RM.

Rays in a training batch are drawn from several camera poses at once. This
loss pools the rendered ("predicted") feature-field output and the target
(ground-truth CLIP/DINO) feature per camera, then runs an InfoNCE-style
objective that:

  - pulls a pose's *predicted* pooled embedding toward its *own* target
    pooled embedding (the positive pair), and
  - pushes it away from the target pooled embeddings of every *other* pose
    present in the batch (in-batch negatives).

This is complementary to the per-ray MSE reconstruction loss: MSE only cares
that each ray's rendered feature is close to its own target in absolute
terms, whereas this loss explicitly encourages different poses to be
*discriminable* from one another, which helps sharpen view-dependent detail
and counteracts collapse toward a scene-average feature.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PoseContrastiveLossConfig:
    temperature: float = 0.1
    """InfoNCE temperature. Lower = sharper/more confident distribution over poses."""
    symmetric: bool = True
    """If True, average the pred->target and target->pred directions (as in CLIP's loss)."""
    min_poses: int = 2
    """Need at least this many distinct poses in the batch to have any negatives."""


class PoseContrastiveLoss(torch.nn.Module):
    """Contrasts pooled per-pose predicted features against pooled per-pose target features."""

    def __init__(self, config: PoseContrastiveLossConfig = None):
        super().__init__()
        self.config = config or PoseContrastiveLossConfig()

    @staticmethod
    def _pool_by_camera(
        features: torch.Tensor, camera_idx: torch.Tensor, unique_cams: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool `features` per camera id using efficient tensor operations."""
        # Map camera IDs to sequential indices [0, num_poses-1]
        # unique_cams is already sorted by unique()
        # inverse_indices gives us an [N] tensor where each entry is the position in unique_cams
        _, inverse_indices = torch.unique(camera_idx, return_inverse=True)
        num_poses = unique_cams.shape[0]
        
        # 1. Sum up features per camera group
        summed = torch.zeros((num_poses, features.shape[-1]), device=features.device, dtype=features.dtype)
        summed.index_add_(0, inverse_indices, features)
        
        # 2. Count rays per camera group
        counts = torch.zeros(num_poses, device=features.device, dtype=features.dtype)
        counts.index_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=features.dtype))
        
        # 3. Safe division for mean pooling
        return summed / counts.unsqueeze(1).clamp(min=1e-6)

    def forward(
        self,
        pred_features: torch.Tensor,  # [N, D] rendered features for the batch's rays
        target_features: torch.Tensor,  # [N, D] ground-truth features for the same rays
        camera_idx: torch.Tensor,  # [N] camera/pose id for each ray
    ) -> torch.Tensor:
        if pred_features.shape[0] == 0:
            return pred_features.sum() * 0.0

        camera_idx = camera_idx.to(pred_features.device)
        unique_cams = camera_idx.unique()
        num_poses = unique_cams.shape[0]

        # Need at least a couple of distinct poses in the batch to form negatives
        if num_poses < self.config.min_poses:
            return pred_features.sum() * 0.0

        pred_pooled = self._pool_by_camera(pred_features.float(), camera_idx, unique_cams)
        target_pooled = self._pool_by_camera(target_features.float(), camera_idx, unique_cams)

        pred_pooled = F.normalize(pred_pooled, dim=-1)
        target_pooled = F.normalize(target_pooled, dim=-1)

        # [P, P]: row i = similarity of pose i's *prediction* against every pose's *target*
        logits = (pred_pooled @ target_pooled.T) / self.config.temperature
        labels = torch.arange(num_poses, device=pred_features.device)

        loss_pred_to_target = F.cross_entropy(logits, labels)

        if not self.config.symmetric:
            return loss_pred_to_target

        loss_target_to_pred = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_pred_to_target + loss_target_to_pred)