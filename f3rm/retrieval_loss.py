"""
retrieval_loss.py

Global-Context Multi-Text Contrastive Retrieval Loss for F3RM.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
import sys


@dataclass
class RetrievalLossConfig:
    temperature: float = 0.05   # InfoNCE language temperature
    n_pos_samples: int = 32     # GT bbox rays to use as positives per annotation
    min_pos: int = 2            # skip annotation if fewer positives than this


class RetrievalLoss(torch.nn.Module):

    def __init__(self, config: RetrievalLossConfig = None):
        super().__init__()
        self.config = config or RetrievalLossConfig()

    def forward(
        self,
        features: torch.Tensor,      
        ray_cam_idx: torch.Tensor,   
        ray_y: torch.Tensor,         
        ray_x: torch.Tensor,         
        annotations: list,
        global_annotations: list = None,
    ) -> torch.Tensor:

        # Force unbuffered output text to confirm active optimization execution
        print(f"!!! RETRIEVAL LOSS EXECUTING - Current Batch Local Anns: {len(annotations)}", file=sys.stderr, flush=True)

        if features.shape[0] == 0:
            return features.sum() * 0.0

        # Use global annotations to assemble our distractor pool if local is empty
        source_for_text = global_annotations if (global_annotations and len(global_annotations) > 0) else annotations
        if not source_for_text:
            return features.sum() * 0.0

        feats_norm = F.normalize(features.float(), dim=-1)  
        device = features.device

        # Deduplicate text descriptors to build a clean competitive text matrix
        unique_texts = {}
        for ann in source_for_text:
            norm_emb = F.normalize(ann["text_embed"].float().to(device), dim=-1)
            emb_key = tuple(norm_emb[0, :5].tolist())  # check a small fingerprint prefix for speed
            if emb_key not in unique_texts:
                unique_texts[emb_key] = norm_emb

        # If we don't have enough classes to compete against, we can't build a denominator
        if len(unique_texts) < 2:
            return features.sum() * 0.0

        # Stack unique embeddings into a matrix: [N_unique_labels, D]
        stacked_text_embs = torch.cat(list(unique_texts.values()), dim=0)
        losses = []

        # If no local labels are active in this specific camera bundle, return 0.0 safely
        if not annotations:
            return features.sum() * 0.0

        for ann in annotations:
            x1, y1, x2, y2 = ann["bbox_feat"]   
            cam = ann["cam_idx"]
            
            target_emb = F.normalize(ann["text_embed"].float().to(device), dim=-1)
            target_idx = torch.argmax(stacked_text_embs @ target_emb.T).item()

            # Isolate matching rays inside this specific object bounding box
            same_cam = (ray_cam_idx == cam)
            inside = (
                same_cam
                & (ray_x >= x1) & (ray_x < x2)
                & (ray_y >= y1) & (ray_y < y2)
            )

            pos_idx = inside.nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() < self.config.min_pos:
                continue  

            if pos_idx.numel() > self.config.n_pos_samples:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:self.config.n_pos_samples]]

            pos_feats = feats_norm[pos_idx]

            # Matrix multiplication sets up cross-entropy matching evaluation structure
            # [N_pos, D] @ [D, N_unique_labels] -> [N_pos, N_unique_labels]
            logits = torch.matmul(pos_feats, stacked_text_embs.T)
            logits = logits / self.config.temperature

            targets = torch.full((pos_feats.shape[0],), target_idx, dtype=torch.long, device=device)

            loss_ann = F.cross_entropy(logits, targets)
            losses.append(loss_ann)

        if not losses:
            return features.sum() * 0.0

        return torch.stack(losses).mean()