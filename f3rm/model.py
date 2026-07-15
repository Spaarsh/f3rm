from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Type

import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerNumber,
    ViewerText,
)
from torch.nn import Parameter

from f3rm.feature_field import FeatureField, FeatureFieldHeadNames
from f3rm.pca_colormap import apply_pca_colormap_return_proj
from f3rm.renderer import FeatureRenderer

import time
from contextlib import contextmanager

"""
PATCH for model.py  — 4 targeted changes, nothing else.
"""

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 1: New import — add after existing imports
# ══════════════════════════════════════════════════════════════════════════════

from f3rm.retrieval_loss import RetrievalLoss, RetrievalLossConfig
from f3rm.pose_contrastive_loss import PoseContrastiveLoss, PoseContrastiveLossConfig


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2: Two new fields in FeatureFieldModelConfig
# Add these inside the dataclass, after feat_num_layers:
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 3: Instantiate loss in populate_modules()
# Add at the END of populate_modules(), after the self.field.forward = ... line
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 4: Replace get_loss_dict() entirely
# ══════════════════════════════════════════════════════════════════════════════

from collections import defaultdict
# add a module-level timing store: {name -> [elapsed_ms, ...]}
PROFILE_TIMINGS = defaultdict(list)
@contextmanager
def profile_cuda(name: str):
    """Context manager to measure exact CUDA execution time."""
    # Ensure any previous CUDA operations are done before starting timer
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    yield
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    PROFILE_TIMINGS[name].append(elapsed_time_ms)
    #print(f"[Profile] {name:<30} : {elapsed_time_ms:>8.2f} ms")

@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    """Note: make sure to use naming that doesn't conflict with NerfactoModelConfig"""

    _target: Type = field(default_factory=lambda: FeatureFieldModel)
    # Weighing for the feature loss
    feat_loss_weight: float = 1e-3
    # Feature Field Positional Encoding
    feat_use_pe: bool = True
    feat_pe_n_freq: int = 6
    # Feature Field Hash Grid
    feat_num_levels: int = 12
    feat_log2_hashmap_size: int = 19
    feat_start_res: int = 16
    feat_max_res: int = 128
    feat_features_per_level: int = 8
    # Feature Field MLP Head
    feat_hidden_dim: int = 64
    feat_num_layers: int = 2

    # Retrieval loss params (added)
    retrieval_loss_weight: float = 1e-2
    retrieval_loss_config: RetrievalLossConfig = field(default_factory=RetrievalLossConfig)

    # Pose contrastive loss params (added)
    pose_contrastive_loss_weight: float = 1e-2
    """Weight for the cross-pose contrastive loss. Set to 0.0 to disable."""
    pose_contrastive_loss_config: PoseContrastiveLossConfig = field(
        default_factory=PoseContrastiveLossConfig
    )


@dataclass
class ViewerUtils:
    pca_proj: Optional[torch.Tensor] = None
    positives: List[str] = field(default_factory=list)
    pos_embed: Optional[torch.Tensor] = None
    negatives: List[str] = field(default_factory=list)
    neg_embed: Optional[torch.Tensor] = None
    softmax_temp: float = 0.1
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @cached_property
    def clip(self):
        from f3rm.features.clip import load
        from f3rm.features.clip_extract import CLIPArgs

        CONSOLE.print(f"Loading CLIP {CLIPArgs.model_name} for viewer")
        model, _ = load(CLIPArgs.model_name, device=self.device)
        model.eval()
        return model

    @torch.no_grad()
    def handle_language_queries(self, raw_text: str, is_positive: bool):
        """Compute CLIP embeddings based on queries and update state"""
        from f3rm.features.clip import tokenize

        texts = [x.strip() for x in raw_text.split(",") if x.strip()]
        # Clear the GUI state if there are no texts
        if not texts:
            self.clear_positives() if is_positive else self.clear_negatives()
            return
        # Embed text queries
        tokens = tokenize(texts).to(self.device)
        embed = self.clip.encode_text(tokens).float()
        if is_positive:
            self.positives = texts
            # Average embedding if we have multiple positives
            embed = embed.mean(dim=0, keepdim=True)
            embed /= embed.norm(dim=-1, keepdim=True)
            self.pos_embed = embed
        else:
            self.negatives = texts
            # We don't average the negatives as we compute pair-wise softmax
            embed /= embed.norm(dim=-1, keepdim=True)
            self.neg_embed = embed

    @property
    def has_positives(self) -> bool:
        return self.positives and self.pos_embed is not None

    def clear_positives(self):
        self.positives.clear()
        self.pos_embed = None

    @property
    def has_negatives(self) -> bool:
        return self.negatives and self.neg_embed is not None

    def clear_negatives(self):
        self.negatives.clear()
        self.neg_embed = None

    def update_softmax_temp(self, temp: float):
        self.softmax_temp = temp

    def reset_pca_proj(self):
        self.pca_proj = None
        CONSOLE.print("Reset PCA projection")


viewer_utils = ViewerUtils()


class FeatureFieldModel(NerfactoModel):
    config: FeatureFieldModelConfig

    feature_field: FeatureField
    renderer_feature: FeatureRenderer

    def populate_modules(self):
        super().populate_modules()

        # Create feature field
        feature_dim = 1024 #self.kwargs["metadata"]["feature_dim"]
        if feature_dim <= 0:
            raise ValueError(f"Feature dimensionality must be positive, not {feature_dim}")

        self.feature_field = FeatureField(
            feature_dim=feature_dim,
            spatial_distortion=self.field.spatial_distortion,
            use_pe=self.config.feat_use_pe,
            pe_n_freq=self.config.feat_pe_n_freq,
            num_levels=self.config.feat_num_levels,
            log2_hashmap_size=self.config.feat_log2_hashmap_size,
            start_res=self.config.feat_start_res,
            max_res=self.config.feat_max_res,
            features_per_level=self.config.feat_features_per_level,
            hidden_dim=self.config.feat_hidden_dim,
            num_layers=self.config.feat_num_layers,
        )
        self.renderer_feature = FeatureRenderer()
        self.setup_gui()
        
        # Store the original forward method of the MLP head
        original_mlp_forward = self.field.mlp_head.forward

        # 1. Create a profiled forward method for training
        def profiled_mlp_forward(*args, **kwargs):
            with profile_cuda("   [Layer] Base Color MLP (3 Layers)"):
                return original_mlp_forward(*args, **kwargs)

        # 2. Create a dummy/cheap forward method for skipping
        def dummy_mlp_forward(in_tensor):
            # tinycudann expects a specific output shape matching its configuration.
            # We fetch n_output_dims dynamically so tinycudann doesn't throw a shape error.
            out_dim = self.field.mlp_head.tcnn_encoding.n_output_dims
            return torch.zeros((*in_tensor.shape[:-1], out_dim), device=in_tensor.device)

        # Wrap the main field forward pass to split the profiling
        original_field_forward = self.field.forward

        def profiled_field_forward(ray_samples, compute_normals=False):
            # --- STEP A: Profile the HashGrid + Density MLP ---
            with profile_cuda("   [Layer] Base HashGrid + Density MLP"):
                # Force the MLP head to return zeros instantly so its execution time is practically 0
                self.field.mlp_head.forward = dummy_mlp_forward
                outputs = original_field_forward(ray_samples, compute_normals=compute_normals)
            
            # --- STEP B: Run/Profile or Skip the Color MLP Head ---
            if self.training:
                # In training mode, run and measure the real layers
                self.field.mlp_head.forward = profiled_mlp_forward
                outputs = original_field_forward(ray_samples, compute_normals=compute_normals)
            else:
                self.field.mlp_head.forward = profiled_mlp_forward
                outputs = original_field_forward(ray_samples, compute_normals=compute_normals)
                # In eval/viewer mode, we do nothing else! 
                # The outputs dictionary already contains the dummy zeros from Step A.
                pass
                
            # Restore the original forward method for the next loop execution
            self.field.mlp_head.forward = original_mlp_forward
            return outputs

        """# Inject our wrapper into the model's field
        self.field.forward = profiled_field_forward
        # Backwards-compatible: some saved configs may not contain retrieval_loss_config.
        retrieval_cfg = getattr(self.config, "retrieval_loss_config", None)
        if retrieval_cfg is None:
            # lazy-import the default config if missing
            from f3rm.retrieval_loss import RetrievalLossConfig as _RetrievalLossConfig
            retrieval_cfg = _RetrievalLossConfig()
        self.retrieval_loss_fn = RetrievalLoss(retrieval_cfg)"""

        # Backwards-compatible: some saved configs may not contain_contrastive_loss_config.
        pose_cfg = getattr(self.config, "pose_contrastive_loss_config", None)
        if pose_cfg is None:
            from f3rm.pose_contrastive_loss import PoseContrastiveLossConfig as _PoseContrastiveLossConfig
            pose_cfg = _PoseContrastiveLossConfig()
        self.pose_contrastive_loss_fn = PoseContrastiveLoss(pose_cfg)

    def setup_gui(self):
        viewer_utils.device = self.kwargs["device"]
        # Note: the GUI elements are shown based on alphabetical variable names
        self.btn_refresh_pca = ViewerButton("Refresh PCA Projection", cb_hook=lambda _: viewer_utils.reset_pca_proj())

        # Only setup GUI for language features if we're using CLIP
        if self.kwargs["metadata"]["feature_type"] != "CLIP":
            return
        self.hint_text = ViewerText(name="Note:", disabled=True, default_value="Use , to separate labels")
        self.lang_1_pos_text = ViewerText(
            name="Language (Positives)",
            default_value="",
            cb_hook=lambda elem: viewer_utils.handle_language_queries(elem.value, is_positive=True),
        )
        self.lang_2_neg_text = ViewerText(
            name="Language (Negatives)",
            default_value="",
            cb_hook=lambda elem: viewer_utils.handle_language_queries(elem.value, is_positive=False),
        )
        self.softmax_temp = ViewerNumber(
            name="Softmax temperature",
            default_value=viewer_utils.softmax_temp,
            cb_hook=lambda elem: viewer_utils.update_softmax_temp(elem.value),
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["feature_field"] = list(self.feature_field.parameters())
        return param_groups

    """def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        
        # 1. Proposal Sampler
        with profile_cuda("Proposal Sampler"):
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
                ray_bundle, density_fns=self.density_fns
            )
            
        # 2. Main NeRF Field Forward Pass
        with profile_cuda("Base Field Forward"):
            if self.training:
                field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            else:
                # CRITICAL: We only fetch density and skip the color calculations entirely
                density_embedding = self.field.get_density(ray_samples)
                field_outputs = {
                    FieldHeadNames.DENSITY: density_embedding[0]
                }

            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        # 3. Weights calculation
        with profile_cuda("Weights Generation"):
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            weights_list.append(weights)
            ray_samples_list.append(ray_samples)

        # 4. Standard Renderers (Bypassing RGB entirely when evaluating)
        with profile_cuda("Standard Rendering (Depth Only)"):
            if self.training:
                # We still need standard color pipelines during training
                rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            
            with torch.no_grad():
                depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
            accumulation = self.renderer_accumulation(weights=weights)

        # 5. Feature Field Forward Pass
        with profile_cuda("Feature Field Forward"):
            ff_outputs = self.feature_field(ray_samples)
            
        # 6. Feature Rendering Pass
        with profile_cuda("Feature Volume Rendering"):
            features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)

        # 7. Build Output Dictionary
        outputs = {
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "feature": features,
        }
        
        # Only inject the RGB canvas if we are training
        if self.training:
            outputs["rgb"] = rgb

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs"""

    def get_outputs(self, ray_bundle: RayBundle):
        """Modified from nerfacto.get_outputs with granular CUDA profiling."""
        ray_samples: RaySamples
        
        # 1. Proposal Sampler
        with profile_cuda("Proposal Sampler"):
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
                ray_bundle, density_fns=self.density_fns
            )
            
        # 2. Main NeRF Field Forward Pass
        with profile_cuda("Base Field Forward"):
            if self.training:
                field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            else:
                density_embedding = self.field.get_density(ray_samples)
                field_outputs = {
                    FieldHeadNames.DENSITY: density_embedding[0]
                }
            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        # 3. Weights calculation
        with profile_cuda("Weights Generation"):
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            if self.training:
                weights_list.append(weights)
                ray_samples_list.append(ray_samples)

        # 4. Standard Renderers (RGB, Depth, Accumulation) - Fully bypassed in Evaluation
        if self.training:
            with profile_cuda("Standard Rendering (RGB/Depth)"):
                rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
                with torch.no_grad():
                    depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
                expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
                accumulation = self.renderer_accumulation(weights=weights)

        # 5. Feature Field Forward Pass
        with profile_cuda("Feature Field Forward"):
            ff_outputs = self.feature_field(ray_samples)
            
        # 6. Feature Rendering Pass
        with profile_cuda("Feature Volume Rendering"):
            features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)

        # Return Early during inference/evaluation to avoid processing residual data
        if not self.training:
            return {
                "feature": features
            }

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "feature": features,
        }

        # 7. Normals and Losses (if applicable)
        with profile_cuda("Normals & Extra Renderers"):
            if self.config.predict_normals:
                normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
                pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
                outputs["normals"] = self.normals_shader(normals)
                outputs["pred_normals"] = self.normals_shader(pred_normals)
                
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

            if self.config.predict_normals:
                outputs["rendered_orientation_loss"] = orientation_loss(
                    weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
                )
                outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                    weights.detach(),
                    field_outputs[FieldHeadNames.NORMALS].detach(),
                    field_outputs[FieldHeadNames.PRED_NORMALS],
                )

            for i in range(self.config.num_proposal_iterations):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    """def get_outputs(self, ray_bundle: RayBundle):
        ""Modified from nerfacto.get_outputs to include feature field outputs.""
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # Feature outputs
        ff_outputs = self.feature_field(ray_samples)
        features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "feature": features,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs"""

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Compute feature error
        target_feats = batch["feature"].to(self.device)
        metrics_dict["feature_error"] = F.mse_loss(outputs["feature"], target_feats)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
 
        # Existing feature reconstruction loss (MSE)
        target_feats = batch["feature"].to(self.device)
        loss_dict["feature_loss"] = self.config.feat_loss_weight * F.mse_loss(
            outputs["feature"], target_feats
        )
 
        # Contrastive retrieval loss
        if (
            self.config.retrieval_loss_weight > 0.0
            and "retrieval_annotations" in batch
        ):
            # Combine local step annotations and global batch tokens safely
            ret_loss = self.retrieval_loss_fn(
                features     = outputs["feature"],          
                ray_cam_idx  = batch["ray_cam"],            
                ray_y        = batch["feat_y"],             
                ray_x        = batch["feat_x"],             
                annotations  = batch["retrieval_annotations"],
                global_annotations = batch.get("global_retrieval_annotations", [])
            )
            loss_dict["retrieval_loss"] = (
                self.config.retrieval_loss_weight * ret_loss
            )

        # Pose contrastive loss: pulls a pose's rendered feature toward its own
        # target feature and pushes it away from other poses' target features,
        # using every other camera present in the current batch as negatives.
        if self.config.pose_contrastive_loss_weight > 0.0 and "indices" in batch:
            camera_idx = batch["indices"][:, 0].to(self.device)
            pose_loss = self.pose_contrastive_loss_fn(
                pred_features=outputs["feature"],
                target_features=target_feats,
                camera_idx=camera_idx,
            )
            loss_dict["pose_contrastive_loss"] = (
                self.config.pose_contrastive_loss_weight * pose_loss
            )
        
        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        # Compute PCA of features separately, so we can reuse the same projection matrix
        outputs["feature_pca"], viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
            outputs["feature"], viewer_utils.pca_proj
        )

        # Nothing else to do if not CLIP features or no positives
        if self.kwargs["metadata"]["feature_type"] != "CLIP" or not viewer_utils.has_positives:
            return outputs

        # Normalize CLIP features rendered by feature field
        clip_features = outputs["feature"]
        clip_features /= clip_features.norm(dim=-1, keepdim=True)

        # If there are no negatives, just show the cosine similarity with the positives
        if not viewer_utils.has_negatives:
            sims = clip_features @ viewer_utils.pos_embed.T
            # Show the mean similarity if there are multiple positives
            if sims.shape[-1] > 1:
                sims = sims.mean(dim=-1, keepdim=True)
            outputs["similarity"] = sims
            return outputs

        # Use paired softmax method as described in the paper with positive and negative texts
        text_embs = torch.cat([viewer_utils.pos_embed, viewer_utils.neg_embed], dim=0)
        raw_sims = clip_features @ text_embs.T

        # Broadcast positive label similarities to all negative labels
        pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
        pos_sims = pos_sims.broadcast_to(neg_sims.shape)
        paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)

        # Compute paired softmax
        probs = (paired_sims / viewer_utils.softmax_temp).softmax(dim=-1)[..., :1]
        torch.nan_to_num_(probs, nan=0.0)
        sims, _ = probs.min(dim=-1, keepdim=True)
        outputs["similarity"] = sims
        return outputs