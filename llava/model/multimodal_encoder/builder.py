import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .siglip_encoder import SiglipVisionTower
from .aimv2_encoder import Aimv2VisionTower
from .custom_siglip import SiglipVisionTransformerWithRope
from transformers import SiglipVisionModel, SiglipVisionConfig

from huggingface_hub import hf_hub_download
from safetensors import safe_open

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    use_s2 = getattr(vision_tower_cfg, "s2", False)

    if vision_tower and ("apple/aimv2" in vision_tower or "aim-v2" in vision_tower.lower() or "aimv2" in vision_tower.lower()):
        return Aimv2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if 'siglip' in vision_tower and getattr(vision_tower_cfg, 'use_rope_vision', False):
        weights_path = hf_hub_download(repo_id=vision_tower, filename="model.safetensors")
        
        config = SiglipVisionConfig.from_pretrained(vision_tower)
        
        custom_model = SiglipVisionTransformerWithRope(config)
        
        state_dict = {}
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        missing_keys, unexpected_keys = custom_model.load_state_dict(state_dict, strict=False)
        
        tower = SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        tower.vision_tower = custom_model 
        return tower
    
    if "siglip" in vision_tower.lower():
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        
    if os.path.exists(vision_tower) or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower identifier: {vision_tower}")