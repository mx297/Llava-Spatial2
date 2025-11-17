#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .spatial_encoder.builder import build_spatial_tower
from .multimodal_fusion_block.builder import build_multimodal_fusion_block
from .multimodal_projector.builder import build_vision_projector
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print


class LlavaSpatialMetaModel:

    def __init__(self, config):
        super(LlavaSpatialMetaModel, self).__init__(config)
        
        if hasattr(config, "vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            
            # create spatial tower and fusion block
            if hasattr(config, "spatial_tower"):
                self.spatial_tower = build_spatial_tower(config, delay_load=True)
            if hasattr(config, "fusion_block"):
                self.fusion_block = build_multimodal_fusion_block(config, delay_load=delay_load)

            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_spatial_tower(self):
        spatial_tower = getattr(self, "spatial_tower", None)
        if type(spatial_tower) is list:
            spatial_tower = spatial_tower[0]
        return spatial_tower

    def get_fusion_block(self):
        fusion_block = getattr(self, "fusion_block", None)
        if type(fusion_block) is list:
            fusion_block = fusion_block[0]
        return fusion_block

    def initialize_spatial_tower(self, model_args, fsdp=None):
        spatial_tower = model_args.spatial_tower
        self.config.mm_spatial_tower = spatial_tower

        # === ADD THIS BLOCK HERE ===
        # Propagate CLIP stats from the vision side into model_args so the spatial
        # tower can invert CLIP normalization back to [0,1].
        if getattr(self.config, "clip_image_mean", None) is not None:
            setattr(model_args, "clip_image_mean", self.config.clip_image_mean)
        if getattr(self.config, "clip_image_std", None) is not None:
            setattr(model_args, "clip_image_std", self.config.clip_image_std)
        # === END OF ADDED BLOCK ===
        
        if self.get_spatial_tower() is None:

            spatial_tower = build_spatial_tower(model_args)
            #for k, v in spatial_tower.config.items():
            #    setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.spatial_tower = [spatial_tower]
            else:
                self.spatial_tower = spatial_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                spatial_tower = self.spatial_tower[0]
            else:
                spatial_tower = self.spatial_tower
            
            spatial_tower.load_model()

    def initialize_fusion_block(self, model_args, fsdp=None):
        # initialize the fusion block
        #print(self.config)
        #print(getattr(self, "fusion_block", None))
        if getattr(self, "fusion_block", None) is None:
            self.fusion_block = build_multimodal_fusion_block(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.fusion_block.parameters():
                p.requires_grad = True

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.fusion_block.load_state_dict(get_w(mm_projector_weights, "fusion_block"), strict=False)
            rank0_print(f"Loaded fusion block weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

          
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type


        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        #NEW ADDED
        ip = getattr(vision_tower, "image_processor", None)
        if ip is not None:
            self.config.clip_image_mean = getattr(ip, "image_mean", None)
            self.config.clip_image_std  = getattr(ip, "image_std",  None)
        else:
            self.config.clip_image_mean = None
            self.config.clip_image_std  = None
        ###

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaSpatialMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # Accessors consistent with your first version
    def get_spatial_tower(self):
        m = self.get_model()
        return getattr(m, "get_spatial_tower", lambda: None)()

    def get_fusion_block(self):
        m = self.get_model()
        return getattr(m, "get_fusion_block", lambda: None)()

    def _select_spatial_features(self, camera_tokens, patch_tokens):
        """
        Build the spatial feature matrix according to config.spatial_tower_select_feature.
        Allowed: "patch_tokens" (default), "camera_tokens", "all", or comma list e.g. "camera_tokens,patch_tokens".
        """
        cfg = self.get_model().config
        select = getattr(cfg, "spatial_tower_select_feature", "patch_tokens")
        final_feats = []
        for part in select.split(","):
            p = part.strip()
            if p == "camera_tokens":
                final_feats.append(camera_tokens)
            elif p == "patch_tokens":
                final_feats.append(patch_tokens)
            elif p == "all":
                final_feats = [camera_tokens, patch_tokens]
                break
            else:
                raise ValueError(f"Unexpected spatial_tower_select_feature: {p}")
        return torch.cat(final_feats, dim=1).to(self.get_model().dtype)

    def encode_images(self, images,spatial_features=None,point_maps=None):
        """
        Single-image path. `images` can be (B, C, H, W), or as your data loader
        sometimes passes (list / 5D); we DO NOT normalize here—leave upstream as-is.
        This function expects a batched (B, C, H, W) tensor by the time it is called
        in the simple branch of prepare_inputs_labels_for_multimodal, and the
        concatenated 4D tensor when coming from the list/5D branch.
        """
        #print(images.shape)
        image_features = self.get_model().get_vision_tower()(images)  # (B, Nv, Dv)
        #print(image_features.shape)
        spatial_tower = self.get_spatial_tower()
        fusion_block = self.get_fusion_block()

        if spatial_tower is not None and fusion_block is not None:
            fusion_type = getattr(self.get_model().config, "fusion_block", "cross_attention")
            # Spatial tower returns (camera_tokens, patch_tokens)

            camera_tokens, patch_tokens = spatial_tower(images)
            if fusion_type == "cross_attention":
                spatial_feats = self._select_spatial_features(camera_tokens, patch_tokens)  # (B, Ns, Ds)
                
                device = image_features.device
                if spatial_feats.device != device:
                    spatial_feats = spatial_feats.to(device)
            
                image_features, attn_weights = fusion_block(image_features, spatial_feats)
        # Project to language dimension
        image_features = self.get_model().mm_projector(image_features)  # (B, Nv, D)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, spatial_features=None, point_maps=None, modalities=["image"],image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # --- Leave original behavior if `images` is a list or 5D ---
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)  # (sumB, C, H, W)
            image_features = self.encode_images(concat_images,spatial_features,point_maps)             # (sumB, Nv, D)
            #print("if image_features",image_features.shape)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)

            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx],
                                self.config.image_grid_pinpoints,
                                self.get_vision_tower().config.image_size
                            )
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError

                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(
                                    *image_feature.shape[:-1], 1
                                ).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)

                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

        else:
            # Simple single-image batched tensor (B, C, H, W)
            image_features = self.encode_images(images,spatial_features,point_maps)  # (B, Nv, D)
            #print("else image_features",image_features.shape)
            #image_features = [feat.flatten(0, 1) for feat in torch.unbind(image_features, dim=0)]
            image_features = [feat.contiguous() for feat in torch.unbind(image_features, dim=0)]
            #print("else image_features",image_features.shape)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # ---- Standard bookkeeping (unchanged) ----
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # Do not consume an image when there is no IMAGE_TOKEN
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels_full = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels_full[image_token_indices[i]+1:image_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    assert cur_image_idx < len(image_features), "Mismatch between IMAGE_TOKENs and provided images."
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full(
                        (cur_image_features.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels_full.device,
                        dtype=cur_labels_full.dtype
                    ))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            #new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            #new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
            new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
