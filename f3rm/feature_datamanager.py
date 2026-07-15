import gc
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Type

import numpy as np
import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from f3rm.features.clip_extract import CLIPArgs, extract_clip_features
from f3rm.features.dino_extract import DINOArgs, extract_dino_features

import json
from pathlib import Path
import torch.nn.functional as F


@dataclass
class FeatureDataManagerConfig(VanillaDataManagerConfig):
    retrieval_dataset_path: Optional[str] = None
    _target: Type = field(default_factory=lambda: FeatureDataManager)
    feature_type: Literal["CLIP", "DINO"] = "CLIP"
    """Feature type to extract."""
    enable_cache: bool = True
    """Whether to cache extracted features."""


feat_type_to_extract_fn = {
    "CLIP": extract_clip_features,
    "DINO": extract_dino_features,
}

feat_type_to_args = {
    "CLIP": CLIPArgs,
    "DINO": DINOArgs,
}


class FeatureDataManager(VanillaDataManager):
    config: FeatureDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract features
        features = self.extract_features()

        # Split into train and eval features
        self.train_features = features[: len(self.train_dataset)]
        self.eval_features = features[len(self.train_dataset) :]
        assert len(self.eval_features) == len(self.eval_dataset)

        # Set metadata, so we can initialize model with feature dimensionality
        self.train_dataset.metadata["feature_type"] = self.config.feature_type
        self.train_dataset.metadata["feature_dim"] = self.train_features.shape[-1]

        # Determine scaling factors for nearest neighbor interpolation
        feat_h, feat_w = features.shape[1:3]
        im_h = set(self.train_dataset.cameras.image_height.squeeze().tolist())
        im_w = set(self.train_dataset.cameras.image_width.squeeze().tolist())
        assert len(im_h) == 1, "All images must have the same height"
        assert len(im_w) == 1, "All images must have the same width"
        im_h, im_w = im_h.pop(), im_w.pop()
        self.scale_h = feat_h / im_h
        self.scale_w = feat_w / im_w
        assert np.isclose(
            self.scale_h, self.scale_w, atol=1.5e-3
        ), f"Scales must be similar, got h={self.scale_h} and w={self.scale_w}"

        # Garbage collect
        torch.cuda.empty_cache()
        gc.collect()
        self._retrieval_index = {}   # file_path → list of annotation dicts
        self._retrieval_ready = False
 
        if self.config.retrieval_dataset_path is not None:
            self._load_retrieval_annotations(
                Path(self.config.retrieval_dataset_path)
            )


    def extract_features(self) -> Float[torch.Tensor, "n h w c"]:
        """Extract features with support for caching."""
        if self.config.feature_type not in feat_type_to_extract_fn:
            raise ValueError(f"Unknown feature type {self.config.feature_type}")
        extract_fn = feat_type_to_extract_fn[self.config.feature_type]
        extract_args = feat_type_to_args[self.config.feature_type]
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames

        # If cache exists, load it and validate it. We save it to the dataset directory.
        cache_dir = self.config.dataparser.data
        cache_path = cache_dir / f"f3rm_{self.config.feature_type.lower()}_features.pt"
        if self.config.enable_cache and cache_path.exists():
            cache_dict = torch.load(cache_path)
            if cache_dict.get("image_fnames") != image_fnames:
                CONSOLE.print("Image filenames have changed, cache invalidated...")
            elif cache_dict.get("args") != extract_args.id_dict():
                CONSOLE.print("Feature extraction args have changed, cache invalidated...")
            else:
                return cache_dict["features"]

        # Cache is invalid or doesn't exist, so extract features
        CONSOLE.print(f"Extracting {self.config.feature_type} features for {len(image_fnames)} images...")
        features = extract_fn(image_fnames, self.device)
        if self.config.enable_cache:
            cache_dict = {"args": extract_args.id_dict(), "image_fnames": image_fnames, "features": features}
            cache_dir.mkdir(exist_ok=True)
            torch.save(cache_dict, cache_path)
            CONSOLE.print(f"Saved {self.config.feature_type} features to cache at {cache_path}")

        return features

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Nearest-neighbour feature interpolation + retrieval annotation injection."""
        ray_bundle, batch = super().next_train(step)

        # ── Pixel area (unchanged from original) ─────────────────────────────
        if "pixel_area" in batch:
            pa = batch["pixel_area"].to(ray_bundle.origins.device).float().reshape(-1, 1)
            ray_bundle.pixel_area = pa
        else:
            patch_pixel_count = 1.0 / (self.scale_h * self.scale_w)
            ray_bundle.pixel_area = ray_bundle.pixel_area * float(patch_pixel_count)

        # ── Feature lookup (unchanged from original) ─────────────────────────
        ray_indices = batch["indices"]
        camera_idx  = ray_indices[:, 0]
        y_idx = (ray_indices[:, 1] * self.scale_h).long()
        x_idx = (ray_indices[:, 2] * self.scale_w).long()
        batch["feature"] = self.train_features[camera_idx, y_idx, x_idx]

        # ── Retrieval annotations ─────────────────────────────────────────────
        if self._retrieval_ready:
            # Expose feat-space coords so the loss can do bbox membership tests
            batch["feat_y"]    = y_idx        # [N_rays]
            batch["feat_x"]    = x_idx        # [N_rays]
            batch["ray_cam"]   = camera_idx   # [N_rays]

            # Collect annotations for every unique camera in this batch
            unique_cams = camera_idx.unique().tolist()
            anns = []
            for cam in unique_cams:
                anns.extend(self._retrieval_index.get(int(cam), []))
            batch["retrieval_annotations"] = anns

        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Nearest neighbor interpolation of features"""
        ray_bundle, batch = super().next_eval(step)
        ray_indices = batch["indices"]
        camera_idx = ray_indices[:, 0]
        y_idx = (ray_indices[:, 1] * self.scale_h).long()
        x_idx = (ray_indices[:, 2] * self.scale_w).long()
        batch["feature"] = self.eval_features[camera_idx, y_idx, x_idx]
        return ray_bundle, batch
    
    def _load_retrieval_annotations(self, dataset_path: Path):
        """
        Load dataset_normalized.json and pre-encode all text embeddings.
        Builds self._retrieval_index: {file_path -> [annotation_dicts]}
        where each annotation_dict has:
            text_embed : Tensor [1, 768]   (CPU, normalized)
            bbox_feat  : [x1, y1, x2, y2]  in feat-pixel coords
            cam_idx    : int               (index into train_dataset)
            label      : str
        """
        from f3rm.features.clip import load as clip_load, tokenize
        from f3rm.features.clip_extract import CLIPArgs
 
        if not dataset_path.exists():
            CONSOLE.print(f"[yellow]Retrieval dataset not found: {dataset_path}")
            return
 
        CONSOLE.print(f"Loading retrieval annotations from {dataset_path}")
        with open(dataset_path) as f:
            data = json.load(f)
 
        # Load CLIP model on CPU to avoid occupying GPU during data prep
        clip_model, _ = clip_load(CLIPArgs.model_name, device="cpu")
        clip_model.eval()
 
        # Build a lookup: filename → camera index in train_dataset
        train_fnames = {
            str(p): i
            for i, p in enumerate(self.train_dataset.image_filenames)
        }
 
        # Template queries (same as eval script — keep in sync)
        query_templates = {
            "pink cup":     "a pink cup on a wooden block",
            "teddy bear":   "a plush teddy bear lying on the table",
            "foam block":   "a white rectangular foam block",
            "wooden block": "a small wooden block on the table",
            "screwdriver":  "a screwdriver with a blue handle",
            "duct tape":    "a roll of black duct tape",
            "scissors":     "scissors with orange handles",
            "measuring cup":"a white plastic measuring cup",
            "measuring jug":"a clear plastic measuring jug",
        }
 
        # Cache text embeddings so we don't re-encode the same label 200 times
        embed_cache = {}
 
        def get_text_embed(label):
            if label not in embed_cache:
                query = query_templates.get(label, f"a {label} on the table")
                with torch.no_grad():
                    tokens = tokenize([query])
                    emb = clip_model.encode_text(tokens).float()
                    emb = F.normalize(emb, dim=-1)   # [1, D]
                embed_cache[label] = emb.cpu()
            return embed_cache[label]
 
        feat_h = self.train_features.shape[1]
        feat_w = self.train_features.shape[2]
        im_h   = self.train_dataset.cameras.image_height[0].item()
        im_w   = self.train_dataset.cameras.image_width[0].item()
 
        skipped = 0
        loaded  = 0
 
        for frame in data.get("frames", []):
            file_path = frame["file_path"]
 
            # Match against train image filenames (try both relative and stem)
            cam_idx = None
            for fname, idx in train_fnames.items():
                if fname.endswith(file_path) or file_path in fname:
                    cam_idx = idx
                    break
 
            if cam_idx is None:
                skipped += 1
                continue
 
            anns = []
            for obj in frame.get("objects", []):
                label = obj.get("label", "")
                bbox  = obj.get("bbox_norm", [])
                if len(bbox) != 4 or not label:
                    continue
 
                # Convert normalised bbox → feat-pixel coords
                x1 = int(bbox[0] * feat_w)
                y1 = int(bbox[1] * feat_h)
                x2 = int(bbox[2] * feat_w)
                y2 = int(bbox[3] * feat_h)
                x1, x2 = max(0, x1), min(feat_w, x2)
                y1, y2 = max(0, y1), min(feat_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
 
                anns.append({
                    "text_embed": get_text_embed(label),  # [1, D] CPU tensor
                    "bbox_feat":  [x1, y1, x2, y2],
                    "cam_idx":    cam_idx,
                    "label":      label,
                })
 
            if anns:
                # Key by cam_idx for fast lookup in next_train
                self._retrieval_index[cam_idx] = (
                    self._retrieval_index.get(cam_idx, []) + anns
                )
                loaded += len(anns)
 
        del clip_model
        torch.cuda.empty_cache()
        self._retrieval_ready = len(self._retrieval_index) > 0
        CONSOLE.print(
            f"Retrieval annotations: {loaded} loaded, {skipped} frames skipped "
            f"(not in train set). Ready={self._retrieval_ready}"
        )
 
# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — replace next_train() entirely with this version
# ══════════════════════════════════════════════════════════════════════════════
 
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Nearest-neighbour feature interpolation + retrieval annotation injection."""
        ray_bundle, batch = super().next_train(step)
 
        # ── Pixel area (unchanged from original) ─────────────────────────────
        if "pixel_area" in batch:
            pa = batch["pixel_area"].to(ray_bundle.origins.device).float().reshape(-1, 1)
            ray_bundle.pixel_area = pa
        else:
            patch_pixel_count = 1.0 / (self.scale_h * self.scale_w)
            ray_bundle.pixel_area = ray_bundle.pixel_area * float(patch_pixel_count)
 
        # ── Feature lookup (unchanged from original) ─────────────────────────
        ray_indices = batch["indices"]
        camera_idx  = ray_indices[:, 0]
        y_idx = (ray_indices[:, 1] * self.scale_h).long()
        x_idx = (ray_indices[:, 2] * self.scale_w).long()
        batch["feature"] = self.train_features[camera_idx, y_idx, x_idx]
 
        # ── Retrieval annotations ─────────────────────────────────────────────
        if self._retrieval_ready:
            # Expose feat-space coords so the loss can do bbox membership tests
            batch["feat_y"]    = y_idx        # [N_rays]
            batch["feat_x"]    = x_idx        # [N_rays]
            batch["ray_cam"]   = camera_idx   # [N_rays]
 
            # Collect annotations for every unique camera in this batch
            unique_cams = camera_idx.unique().tolist()
            anns = []
            for cam in unique_cams:
                anns.extend(self._retrieval_index.get(int(cam), []))
            batch["retrieval_annotations"] = anns
 
        return ray_bundle, batch
 
