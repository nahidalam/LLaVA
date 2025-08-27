#!/bin/bash

export PROMPT_VERSION=plain

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version $PROMPT_VERSION \
    --data_path ~/data/LLaVA_Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder  ~/data/images \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_backbone True \
    --mm_vision_select_layer -2 \
    --vision_tower akshataa/gemma3-4b_it_siglip_encoder \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-gemma3_4b_it_siglip-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --mm_projector_lr 2e-6 \
    --max_grad_norm 1.0 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --tf32 True \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
