import os
import sys
from typing import Optional, Tuple
#import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from llava.utils import rank0_print
from ..vggt.vggt.models.vggt import VGGT
#from ..vggt.vggt.models import VGGT

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

    # (Optional) resize here if needed; kept disabled because 336 divisible by 14 as noted
    nh,nw = int((H//14)*14) , int((W//14)*14)
    img = nn.functional.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=False)

    return img.unsqueeze(1)  # (B,1,C,H,W) expected by VGGT aggregator


class VGGTSpatialTower(nn.Module):
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

        # Default weights at: <repo_root>/vggt/VGGT-1B
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        #repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        #self.weights_path = os.path.join(repo_root, 'vggt', 'VGGT-1B')

        self.vggt = None
        if not delay_load:
            self.load_model()

    def load_model(self, device_map: Optional[dict] = None):
        if self.is_loaded:
            rank0_print(f"{self.spatial_tower_name} already loaded; skipping.")
            return
        rank0_print(f"Loading VGGT weights from: facebook/VGGT-1B")
        print("--------------------------------------------------------------")
        self.vggt = VGGT.from_pretrained("facebook/VGGT-1B")
        self.vggt.eval()
        for p in self.vggt.parameters():
            p.requires_grad = False
        
        vggt_device = self._device_marker.device
        self.vggt.to(vggt_device)
        
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
        tgt_device = self.device if self.vggt is not None else image.device
        tgt_dtype = self.dtype if self.vggt is not None else image.dtype
        image = image.to(device=tgt_device, dtype=tgt_dtype)
        
        #print(tgt_device)
        #print(self.device)

        tgt_device = self._device_marker.device
        tgt_dtype  = self.dtype if self.vggt is not None else image.dtype
        image = image.to(device=tgt_device, dtype=tgt_dtype, non_blocking=True)

        # Debug (optional)
        #rank0_print(f"VGGTSpatialTower running on: {tgt_device}")

        views = _prep_images(
                    image,
                    clip_mean=self.clip_image_mean,
                    clip_std=self.clip_image_std
                )  # (B,1,C,378,378)
        #print("views_shape",views.shape)
        with torch.cuda.amp.autocast(enabled=views.is_cuda):
            aggregated_tokens_list, ps_idx = self.vggt.aggregator(views)
        
        #depth_map, _ = self.vggt.depth_head(aggregated_tokens_list, views, ps_idx)
        #depth_map = depth_map.cpu().detach().numpy()
        #print(depth_map.shape)
        #depth_map = np.squeeze(depth_map[0][0],axis=-1)
        #print(depth_map.shape)
        #depth_map = (depth_map*255).astype('uint8')
        #plt.imsave("depth1_viridis.png", depth_map, cmap="viridis", vmin=0.0, vmax=depth_map.max())
        # VGGT returns list over layers; take last. Common shape: (F, B, N_all, D) with F=1 here.
        tokens = aggregated_tokens_list[-1].to(image.dtype)
        #print("token_shape",tokens.shape)
        if tokens.dim() == 4:
            # (F, B, N, D) -> (B, N, D); assume F=1
            tokens = tokens.squeeze(dim=1)
        elif tokens.dim() != 3:
            raise RuntimeError(f"Unexpected token shape from VGGT: {tokens.shape}")

        camera_tokens = tokens[:, 0:1, :]     # (B, 1, D)
        patch_tokens  = tokens[:, ps_idx:, :] # (B, Np, D)
        #print("camera_shape",camera_tokens.shape)
        #print("patch_shape",patch_tokens.shape)
        return camera_tokens, patch_tokens

    @property
    def dtype(self):
        if self.vggt is None:
            return torch.float16 if torch.cuda.is_available() else torch.float32
        for p in self.vggt.parameters():
            return p.dtype

    @property
    def device(self):
        #if self.vggt is None:
        #    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #for p in self.vggt.parameters():
        #       return p.device
        
        return self._device_marker.device