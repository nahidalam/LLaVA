import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .siglip_encoder import SiglipVisionTower
from .aimv2_encoder import Aimv2VisionTower
from transformers import SiglipVisionModel, SiglipVisionConfig, CLIPVisionModel, Siglip2VisionModel, Aimv2VisionModel
from transformers.models.clip.modeling_clip import CLIPAttention
from transformers.models.siglip.modeling_siglip import SiglipAttention
from transformers.models.siglip2.modeling_siglip2 import Siglip2Attention
from transformers.models.aimv2.modeling_aimv2 import Aimv2Attention
from .custom_rope_vit import convert_vision_encoder_to_rope

from huggingface_hub import hf_hub_download
from safetensors import safe_open

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    use_s2 = getattr(vision_tower_cfg, "s2", False)

    if getattr(vision_tower_cfg, 'use_rope_vision', False):
        ENCODER_SPECS = {
            'clip': {
                'model_class': CLIPVisionModel,
                'attention_class': CLIPAttention,
                'pos_embed_path': 'vision_model.embeddings.position_embedding',
                'tower_wrapper_class': CLIPVisionTower,
                'grid_dim_divisor': 14
            },
            'siglip': {
                'model_class': SiglipVisionModel,
                'attention_class': SiglipAttention,
                'pos_embed_path': 'vision_model.embeddings.position_embedding',
                'tower_wrapper_class': SiglipVisionTower,
                'grid_dim_divisor': 16 # e.g., 384 / 16 = 24
            },
            'aimv2': {
                'model_class': Aimv2VisionModel,
                'attention_class': Aimv2Attention,
                'pos_embed_path': 'embeddings.position_embedding',
                'tower_wrapper_class': Aimv2VisionTower, 
                'grid_dim_divisor': 14
            }
            # TO DO: add specs for other encoders
        }

        spec = None
        if 'clip' in vision_tower.lower():
            spec = ENCODER_SPECS['clip']
        elif 'siglip' in vision_tower.lower():
            spec = ENCODER_SPECS['siglip']
        elif 'aimv2' in vision_tower.lower():
            spec = ENCODER_SPECS['aimv2']
        
        if spec is None:
            raise ValueError(f"RoPE conversion is not yet configured for vision tower: {vision_tower}")

        original_encoder = spec['model_class'].from_pretrained(vision_tower, **kwargs)
        
        image_size = original_encoder.config.image_size
        grid_dim = image_size // spec['grid_dim_divisor']
        grid_dims = (grid_dim, grid_dim)

        rope_encoder = convert_vision_encoder_to_rope(
            encoder=original_encoder,
            attention_module_class=spec['attention_class'],
            positional_embedding_path=spec['pos_embed_path'],
            grid_dims=grid_dims
        )

        tower = spec['tower_wrapper_class'](vision_tower, args=vision_tower_cfg, **kwargs)
        tower.vision_tower = rope_encoder
        tower.is_loaded = True # Mark as loaded since we did it manually
        
        return tower

    if vision_tower and ("apple/aimv2" in vision_tower or "aim-v2" in vision_tower.lower() or "aimv2" in vision_tower.lower()):
        return Aimv2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    if "siglip" in vision_tower.lower():
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if os.path.exists(vision_tower) or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower identifier: {vision_tower}")