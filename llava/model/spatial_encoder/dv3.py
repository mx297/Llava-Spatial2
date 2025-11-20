import os
import sys
from typing import Optional, Tuple
#import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from llava.utils import rank0_print
from depth_anything_3.api import DepthAnything3
import torchvision.transforms as T
#from depth_anything_3.utils.io.output_processor import OutputProcessor
#from depth_anything_3.utils.visualize import visualize_depth

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def _prep_images(img: torch.Tensor,
                 clip_mean: Optional[torch.Tensor] = None,
                 clip_std: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Accept ONE image (C,H,W) or a batch (B,C,H,W).
    Input is expected to be EITHER:
      - CLIP-normalized: x = (img_[0..1] - mean)/std  -> we invert using (mean,std)
      - Or in [-1, 1] space (legacy)                  -> we map to [0,1] via x*0.5+0.5
    Output: (B, 1, C, 378, 378) in [0,1] for VGGT aggregator.
    """
    if img.ndim == 3:        # (C,H,W) -> (1,C,H,W)
        img = img.unsqueeze(0)
    elif img.ndim != 4:
        raise ValueError(f"Expected (C,H,W) or (B,C,H,W), got {img.shape}")

    B, C, H, W = img.shape

    # If CLIP stats were provided, invert CLIP normalization:
    if clip_mean is not None and clip_std is not None:
        # ensure proper shape/device/dtype
        mean = torch.as_tensor(clip_mean, dtype=img.dtype, device=img.device).view(1, C, 1, 1)
        std  = torch.as_tensor(clip_std,  dtype=img.dtype, device=img.device).view(1, C, 1, 1)
        img = img * std + mean  # back to [0,1] (assuming CLIP preproc used 0..1 then norm)
    else:
        # Fallback: assume legacy [-1,1] inputs
        img = img * 0.5 + 0.5

    #print(img.min(),img.max())
    # (Optional) resize here if needed; kept disabled because 336 divisible by 14 as noted
    nh,nw = int((H//14)*14) , int((W//14)*14)
    img = nn.functional.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=False)
    #print(img.shape)
    img = NORMALIZE(img)
    return img.unsqueeze(1)  # (B,1,C,H,W) expected by VGGT aggregator


class DV3SpatialTower(nn.Module):
    """
    Minimal tower:
      - load_model() loads VGGT from <repo_root>/vggt/VGGT-1B
      - forward(image) accepts (C,H,W) or (B,C,H,W)
      - returns (camera_tokens, patch_tokens) with shapes:
          camera_tokens: (B, 1, D)
          patch_tokens : (B, Np, D)
    """
    def __init__(self, spatial_tower: str, spatial_tower_cfg, delay_load: bool = True):
        super().__init__()
        self.is_loaded = False
        self.spatial_tower_name = spatial_tower

        #ADDED BLOCK
        self.clip_image_mean = getattr(spatial_tower_cfg, "clip_image_mean", None)
        self.clip_image_std  = getattr(spatial_tower_cfg, "clip_image_std",  None)
        ###
        
        self.register_buffer("_device_marker", torch.empty(0), persistent=False)


        self.dv3 = None
        self.backbone = None
        self.backbone_metric = None
        #self.output_processor = OutputProcessor()
        if not delay_load:
            self.load_model()

    def load_model(self, device_map: Optional[dict] = None):
        if self.is_loaded:
            rank0_print(f"{self.spatial_tower_name} already loaded; skipping.")
            return
        rank0_print(f"Loading dv3 weights from: depth-anything/DA3NESTED-GIANT-LARGE")
        print("--------------------------------------------------------------")
        self.dv3 = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
        self.backbone = self.dv3.model.da3.backbone
        self.backbone_metric = self.dv3.model.da3_metric.backbone
        self.backbone.eval()
        self.backbone_metric.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone_metric.parameters():
            p.requires_grad = False
        
        dv3_device = self._device_marker.device
        self.backbone.to(dv3_device)
        self.backbone_metric.to(dv3_device)
        #self.dv3.to(dv3_device)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image: (C,H,W) or (B,C,H,W), assumed normalized to [-1,1]
        Returns:
          camera_tokens: (B, 1, D)
          patch_tokens : (B, Np, D)
        """
        if not self.is_loaded:
            self.load_model()

        # Move to the model's device/dtype if available
        tgt_device = self.device if self.dv3 is not None else image.device
        tgt_dtype = self.dtype if self.dv3 is not None else image.dtype
        image = image.to(device=tgt_device, dtype=tgt_dtype)
        
        tgt_device = self._device_marker.device
        tgt_dtype  = self.dtype if self.dv3 is not None else image.dtype
        image = image.to(device=tgt_device, dtype=tgt_dtype, non_blocking=True)

        views = _prep_images(
                    image,
                    clip_mean=self.clip_image_mean,
                    clip_std=self.clip_image_std
                )  # (B,1,C,378,378)
        #print("views_shape",views.shape)
        with torch.cuda.amp.autocast(enabled=views.is_cuda):
            feats_metric,_ = self.backbone_metric(views)
            feats,_ = self.backbone(views)
            #o = self.dv3(views,export_feat_layers=[])

        #output = self.output_processor(o)
        #print(output.depth.shape)
        #depth_vis = visualize_depth(output.depth[0][0], cmap="Spectral")
        #print(depth_vis.shape)
        #plt.imsave("depth1_viridis.png", depth_vis,cmap="Spectral",vmin=0,vmax=255)
        #depth_map, _ = self.vggt.depth_head(aggregated_tokens_list, views, ps_idx)
        #depth_map = depth_map.cpu().detach().numpy()
        #print(depth_map.shape)
        #depth_map = np.squeeze(depth_map[0][0],axis=-1)
        #print(depth_map.shape)
        #depth_map = (depth_map*255).astype('uint8')
        #plt.imsave("depth1_viridis.png", depth_map, cmap="viridis", vmin=0.0, vmax=depth_map.max())
        # VGGT returns list over layers; take last. Common shape: (F, B, N_all, D) with F=1 here.
        
        
        patch_tokens = feats[-1][0].to(image.dtype)
        patch_tokens_metric = feats_metric[-1][0].to(image.dtype)
        camera_tokens = feats[-1][1].to(image.dtype)
        camera_tokens_metric = feats_metric[-1][1].to(image.dtype)
        #print("token_shape",tokens.shape)
        if patch_tokens.dim() == 4:
            # (F, B, N, D) -> (B, N, D); assume F=1
            patch_tokens = patch_tokens.squeeze(dim=1)
        elif patch_tokens.dim() != 3:
            raise RuntimeError(f"Unexpected token shape from dv3: {patch_tokens.shape}")
        
        if patch_tokens_metric.dim() == 4:
            # (F, B, N, D) -> (B, N, D); assume F=1
            patch_tokens_metric = patch_tokens_metric.squeeze(dim=1)
        elif patch_tokens_metric.dim() != 3:
            raise RuntimeError(f"Unexpected token shape from dv3: {patch_tokens_metric.shape}")

        ps_idx = 0
        patch_tokens  = patch_tokens[:, ps_idx:, :] # (B, Np, D)  where to get ps_idx?? 0 for now
        patch_tokens_metric  = patch_tokens_metric[:, ps_idx:, :] # (B, Np, D)  where to get ps_idx?? 0 for now
        
        all_camera_tokens = torch.cat([camera_tokens,camera_tokens_metric],dim=-1)
        all_patch_tokens = torch.cat([patch_tokens,patch_tokens_metric],dim=-1)
        #print("camera_shape",all_camera_tokens.shape)
        #print("patch_shape",all_patch_tokens.shape)
        return all_camera_tokens, all_patch_tokens

    @property
    def dtype(self):
        if self.dv3 is None:
            return torch.float16 if torch.cuda.is_available() else torch.float32
        for p in self.dv3.parameters():
            return p.dtype

    @property
    def device(self):
        #if self.vggt is None:
        #    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #for p in self.vggt.parameters():
        #       return p.device
        
        return self._device_marker.device