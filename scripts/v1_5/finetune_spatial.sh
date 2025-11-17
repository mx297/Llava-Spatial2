#!/bin/bash

SPATIAL_TOWER="vggt"
FUSION_BLOCK="cross_attention"
SPATIAL_TOWER_SELECT_FEATURE="all"
SPATIAL_FEATURE_DIM="2048"
TUNE_MM_MLP_ADAPTER=True
#VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
#MODEL="lmms-lab/llava-onevision-qwen2-7b-si"
#MODEL="lmms-lab/llava-next-interleave-qwen-7b"
MODEL="liuhaotian/llava-v1.5-7b"

#--pretrain_mm_mlp_adapter ./checkpoints/mm_projector.bin \
deepspeed llava/train/train_mem_spatial.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${MODEL} \
    --version v1 \
    --data_path ./formatted_questions_image.json \
    --image_folder /l/users/$USER \
    --eval_data_path ./sampled_val.json \
    --val_image_folder /l/users/$USER \
    --vision_tower ${VISION_MODEL_VERSION} \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --spatial_tower $SPATIAL_TOWER \
    --spatial_tower_select_feature $SPATIAL_TOWER_SELECT_FEATURE \
    --spatial_feature_dim $SPATIAL_FEATURE_DIM \
    --fusion_block $FUSION_BLOCK \
    --tune_spatial_tower False \
    --tune_fusion_block True \
    --tune_mm_mlp_adapter $TUNE_MM_MLP_ADAPTER \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir /l/users/$USER/checkpoints/llava-spatial-wu0.1-lr1e-5-bs32-2epochs-max4096-lora64-128-vicuna7b \
    --num_train_epochs 2    \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --load_best_model_at_end True \
    --greater_is_better False \
    --metric_for_best_model eval_loss \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1000 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name llava-spatial-wu0.1-lr1e-5-bs32-2epochs-max4096-lora64-128-vicuna7b
