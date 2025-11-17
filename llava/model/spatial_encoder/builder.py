def build_spatial_tower(spatial_tower_cfg,**kwargs):
    spatial_tower = getattr(spatial_tower_cfg, "spatial_tower", None)
    if spatial_tower == "vggt":
        # Use relative import for the encoder wrapper/adapter file
        from ..spatial_encoder.vggt import VGGTSpatialTower
        return VGGTSpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)
    elif spatial_tower == "dv3":
        from ..spatial_encoder.dv3 import DV3SpatialTower
        return DV3SpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)
    raise ValueError(f"Unknown vision tower: {spatial_tower}")