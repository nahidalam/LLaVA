import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)



def get_finetune_args(model_name,
    pretrain_mm_mlp_adapter, 
data_path, 
image_folder, 
output_dir, 
logging_dir
):


    model_args = ModelArguments(
                                model_name_or_path=model_name,
                                version='v1',
                                freeze_backbone=False,
                                tune_mm_mlp_adapter=False,
                                vision_tower='openai/clip-vit-large-patch14-336',
                                mm_vision_select_layer=-2,
                                pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
                                mm_projector_type='mlp2x_gelu',
                                mm_use_im_start_end=False,
                                mm_use_im_patch_token=False,
                                mm_patch_merge_type='flat',
                                mm_vision_select_feature='patch')
    
    
    data_args = DataArguments(
        data_path=data_path,
        lazy_preprocess=True,
        is_multimodal=False,
        image_folder=image_folder,
        image_aspect_ratio='pad'
    )
    
    
    training_args = TrainingArguments(
    # _n_gpu=1,  removing as it throws an error
    accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
    adafactor=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    auto_find_batch_size=False,
    batch_eval_metrics=False,
    bf16=True,
    bf16_full_eval=False,
    bits=16,
    cache_dir=None,
    data_seed=None,
    dataloader_drop_last=False,
    dataloader_num_workers=4,
    dataloader_persistent_workers=False,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=None,
    ddp_backend=None,
    ddp_broadcast_buffers=None,
    ddp_bucket_cap_mb=None,
    ddp_find_unused_parameters=None,
    ddp_timeout=1800,
    debug=[],
    # deepspeed='/home/roshan/Documents/GitHub/LLaVA/scripts/zero3.json',
    deepspeed='/content/LLaVA/scripts/zero2.json',   # For Colab
    disable_tqdm=False,
    dispatch_batches=None,
    do_eval=False,
    do_predict=False,
    do_train=False,
    double_quant=True,
    eval_accumulation_steps=None,
    eval_delay=0,
    eval_do_concat_batches=True,
    eval_on_start=False,
    eval_steps=None,
    eval_strategy='no',
    evaluation_strategy='no',
    fp16=False,
    fp16_backend='auto',
    fp16_full_eval=False,
    fp16_opt_level='O1',
    freeze_mm_mlp_adapter=False,
    fsdp=[],
    fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
    fsdp_min_num_params=0,
    fsdp_transformer_layer_cls_to_wrap=None,
    full_determinism=False,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs=None,
    greater_is_better=None,
    group_by_length=False,
    group_by_modality_length=True,
    half_precision_backend='auto',
    hub_always_push=False,
    hub_model_id=None,
    hub_private_repo=False,
    hub_strategy='every_save',
    hub_token=None,
    ignore_data_skip=False,
    include_inputs_for_metrics=False,
    include_num_input_tokens_seen=False,
    include_tokens_per_second=False,
    jit_mode_eval=False,
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=2e-05,
    length_column_name='length',
    load_best_model_at_end=False,
    local_rank=0,
    log_level='passive',
    log_level_replica='warning',
    log_on_each_node=True,
    logging_dir=logging_dir,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=1.0,
    logging_strategy='steps',
    lora_alpha=16,
    lora_bias='none',
    lora_dropout=0.05,
    lora_enable=False,
    lora_r=64,
    lora_weight_path='',
    lr_scheduler_kwargs={},
    lr_scheduler_type='cosine',
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=None,
    mm_projector_lr=None,
    model_max_length=2048,
    mp_parameters=None,
    mpt_attn_impl='triton',
    neftune_noise_alpha=None,
    no_cuda=False,
    num_train_epochs=1.0,
    optim='adamw_torch',
    optim_args=None,
    optim_target_modules=None,
    output_dir=output_dir,
    overwrite_output_dir=False,
    past_index=-1,
    per_device_eval_batch_size=4,
    per_device_train_batch_size=16,
    prediction_loss_only=False,
    push_to_hub=False,
    push_to_hub_model_id=None,
    push_to_hub_organization=None,
    push_to_hub_token=None,
    quant_type='nf4',
    ray_scope='last',
    remove_unused_columns=False,
    report_to=['wandb'],
    restore_callback_states_from_checkpoint=False,
    resume_from_checkpoint=None,
    run_name=output_dir,
    save_on_each_node=False,
    save_only_model=False,
    save_safetensors=True,
    save_steps=500,
    save_strategy='steps',
    save_total_limit=1,
    seed=42,
    skip_memory_metrics=True,
    split_batches=None,
    tf32=True,
    torch_compile=False,
    torch_compile_backend=None,
    torch_compile_mode=None,
    torchdynamo=None,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_cpu=False,
    use_ipex=False,
    use_legacy_prediction_loop=False,
    use_mps_device=False,
    warmup_ratio=0.03,
    warmup_steps=0,
    weight_decay=0.0,
    )

    return model_args, data_args, training_args
    
    
    
    
    
def get_finetune_lora_args(model_name,
    pretrain_mm_mlp_adapter, 
data_path, 
image_folder, 
output_dir, 
logging_dir):

    model_args = ModelArguments(
            model_name_or_path=model_name, 
            version='v1', 
            freeze_backbone=False, 
            tune_mm_mlp_adapter=False, 
            vision_tower='openai/clip-vit-large-patch14-336', 
            mm_vision_select_layer=-2, 
            pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
            mm_projector_type='mlp2x_gelu',
            mm_use_im_start_end=False,
            mm_use_im_patch_token=False,
            mm_patch_merge_type='flat',
            mm_vision_select_feature='patch')

    data_args = DataArguments(
        data_path=data_path,
        lazy_preprocess=True,
        is_multimodal=False,
        image_folder=image_folder,
        image_aspect_ratio='pad')

    training_args = TrainingArguments(
    # _n_gpu=1,
    accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
    adafactor=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    auto_find_batch_size=False,
    batch_eval_metrics=False,
    bf16=True,
    bf16_full_eval=False,
    bits=16,
    cache_dir=None,
    data_seed=None,
    dataloader_drop_last=False,
    dataloader_num_workers=4,
    dataloader_persistent_workers=False,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=None,
    ddp_backend=None,
    ddp_broadcast_buffers=None,
    ddp_bucket_cap_mb=None,
    ddp_find_unused_parameters=None,
    ddp_timeout=1800,
    debug=[],
    deepspeed='/content/LLaVA/scripts/zero2.json',   # For Colab
    disable_tqdm=False,
    dispatch_batches=None,
    do_eval=False,
    do_predict=False,
    do_train=False,
    double_quant=True,
    eval_accumulation_steps=None,
    eval_delay=0,
    eval_do_concat_batches=True,
    eval_on_start=False,
    eval_steps=None,
    eval_strategy='no',
    evaluation_strategy='no',
    fp16=False,
    fp16_backend='auto',
    fp16_full_eval=False,
    fp16_opt_level='O1',
    freeze_mm_mlp_adapter=False,
    fsdp=[],
    fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
    fsdp_min_num_params=0,
    fsdp_transformer_layer_cls_to_wrap=None,
    full_determinism=False,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs=None,
    greater_is_better=None,
    group_by_length=False,
    group_by_modality_length=True,
    half_precision_backend='auto',
    hub_always_push=False,
    hub_model_id=None,
    hub_private_repo=False,
    hub_strategy='every_save',
    hub_token=None,
    ignore_data_skip=False,
    include_inputs_for_metrics=False,
    include_num_input_tokens_seen=False,
    include_tokens_per_second=False,
    jit_mode_eval=False,
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=0.0002,
    length_column_name='length',
    load_best_model_at_end=False,
    local_rank=0,
    log_level='passive',
    log_level_replica='warning',
    log_on_each_node=True,
    logging_dir=logging_dir,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=1.0,
    logging_strategy='steps',
    lora_alpha=256,
    lora_bias='none',
    lora_dropout=0.05,
    lora_enable=True,
    lora_r=128,
    lora_weight_path='',
    lr_scheduler_kwargs={},
    lr_scheduler_type='cosine',
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=None,
    mm_projector_lr=2e-05,
    model_max_length=2048,
    mp_parameters=None,
    mpt_attn_impl='triton',
    neftune_noise_alpha=None,
    no_cuda=False,
    num_train_epochs=1.0,
    optim='adamw_torch',
    optim_args=None,
    optim_target_modules=None,
    output_dir=output_dir,
    overwrite_output_dir=False,
    past_index=-1,
    per_device_eval_batch_size=4,
    per_device_train_batch_size=16,
    prediction_loss_only=False,
    push_to_hub=False,
    push_to_hub_model_id=None,
    push_to_hub_organization=None,
    push_to_hub_token=None,
    quant_type='nf4',
    ray_scope='last',
    remove_unused_columns=False,
    report_to=['wandb'],
    restore_callback_states_from_checkpoint=False,
    resume_from_checkpoint=None,
    run_name=output_dir,
    save_on_each_node=False,
    save_only_model=False,
    save_safetensors=True,
    save_steps=50000,
    save_strategy='steps',
    save_total_limit=1,
    seed=42,
    skip_memory_metrics=True,
    split_batches=None,
    tf32=True,
    torch_compile=False,
    torch_compile_backend=None,
    torch_compile_mode=None,
    torchdynamo=None,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_cpu=False,
    use_ipex=False,
    use_legacy_prediction_loop=False,
    use_mps_device=False,
    warmup_ratio=0.03,
    warmup_steps=0,
    weight_decay=0.0,
    )

    return model_args, data_args, training_args
   
