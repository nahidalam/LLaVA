import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .siglip_encoder import SiglipVisionTower
from .aimv2_encoder import Aimv2VisionTower
from .custom_siglip import SiglipVisionTransformerWithRope
from transformers import SiglipVisionModel, SiglipVisionConfig, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPAttention
from transformers.models.siglip.modeling_siglip import SiglipAttention
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
                'grid_dim_divisor': 14 # e.g., 336 / 14 = 24
            },
            'siglip': {
                'model_class': SiglipVisionModel,
                'attention_class': SiglipAttention,
                'pos_embed_path': 'embeddings.position_embedding',
                'tower_wrapper_class': SiglipVisionTower,
                'grid_dim_divisor': 16 # e.g., 384 / 16 = 24
            }
            # TO DO: add specs for other encoders
        }

        spec = None
        if 'clip' in vision_tower_name.lower():
            spec = ENCODER_SPECS['clip']
        elif 'siglip' in vision_tower_name.lower():
            spec = ENCODER_SPECS['siglip']
        
        if spec is None:
            raise ValueError(f"RoPE conversion is not yet configured for vision tower: {vision_tower_name}")

        # 2. Load the original pre-trained encoder from Hugging Face
        original_encoder = spec['model_class'].from_pretrained(vision_tower_name, **kwargs)
        
        # 3. Determine the grid dimensions
        # We get the image size from the model's config.
        image_size = original_encoder.config.image_size
        grid_dim = image_size // spec['grid_dim_divisor']
        grid_dims = (grid_dim, grid_dim)

        # 4. Call your generalized conversion function to perform the model surgery
        rope_encoder = convert_vision_encoder_to_rope(
            encoder=original_encoder,
            attention_module_class=spec['attention_class'],
            positional_embedding_path=spec['pos_embed_path'],
            grid_dims=grid_dims
        )

        # 5. Create the LLaVA VisionTower wrapper and inject the modified encoder
        # The wrapper handles things like the image processor and device placement.
        tower = spec['tower_wrapper_class'](vision_tower_name, args=vision_tower_cfg, **kwargs)
        tower.vision_tower = rope_encoder
        tower.is_loaded = True # Mark as loaded since we did it manually
        
        print(f"Successfully built vision tower '{vision_tower_name}' with 2D RoPE.")
        return tower

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