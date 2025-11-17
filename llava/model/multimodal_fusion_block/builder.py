import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads):
        super(CrossAttentionFusion, self).__init__()
        
        # pre-norm
        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)
        
        # projection
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)
        
        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)
        
        # post-norm
        self.out_norm = nn.LayerNorm(d_attn)
        
        # projection
        self.out_proj = nn.Linear(d_attn, d_clip)
        
        # dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, clip_features, spatial_encoder_features):
        """
        Args:
            clip_features: [B, N, D_clip]
            spatial_encoder_features: [B, N, D_spatial_encoder]
        Returns:
            fused_features: [B, N, D_clip]
        """
        # pre-norm

        clip_features_norm = self.clip_norm(clip_features)  # [B, N, D_clip]
        #print(clip_features_norm.shape)

        device = next(self.parameters()).device
        # Move inputs to that device
        clip_features = clip_features.to(device)
        spatial_encoder_features = spatial_encoder_features.to(device)
        
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)  # [B, N, D_spatial_encoder]
        
        # projection to D_attn dimension
        clip_query_proj = self.clip_query_proj(clip_features_norm)  # [B, N, D_attn]
        spatial_encoder_key_proj = self.spatial_encoder_key_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        spatial_encoder_value_proj = self.spatial_encoder_value_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        
        #print(clip_query_proj.shape)
        #print(spatial_encoder_key_proj.shape)
        #   print(spatial_encoder_value_proj.shape)

        # cross attention
        fused_features, attn_weights = self.cross_attention(
            query=clip_query_proj,
            key=spatial_encoder_key_proj,
            value=spatial_encoder_value_proj
        )
        
        # projection to D_clip dimension
        fused_features = self.out_proj(fused_features)   # [B, N_clip, D_clip]
        
        # residual connection and dropout
        fused_features = self.out_norm(fused_features)
        fused_features = fused_features + clip_features  # [B, N_clip, D_clip]
        # print(f'status_of_fused_features: max:{fused_features.max():.2f}, min:{fused_features.min():.2f}, mean:{fused_features.mean():.2f}, std:{fused_features.std():.2f}')
        # print(f'status_of_clip_features: max:{clip_features.max():.2f}, min:{clip_features.min():.2f}, mean:{clip_features.mean():.2f}, std:{clip_features.std():.2f}')
        fused_features = self.dropout(fused_features)
        
        return fused_features, attn_weights

def build_multimodal_fusion_block(config, delay_load=False, **kwargs):
    fusion_block_type = getattr(config, "fusion_block", "cross_attention")
    d_clip = config.mm_hidden_size
    d_attn = d_clip
    d_spatial_encoder = getattr(config, "spatial_feature_dim", 768)
  
    if fusion_block_type == "cross_attention":
        return CrossAttentionFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            #num_heads=18
            num_heads=16
        )

    raise ValueError(f"Unknown fusion block type: {fusion_block_type}")