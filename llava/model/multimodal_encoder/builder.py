import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .siglip_encoder import SiglipVisionTower
from .aimv2_encoder import Aimv2VisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    use_s2 = getattr(vision_tower_cfg, "s2", False)

    if vision_tower and ("apple/aimv2" in vision_tower or "aim-v2" in vision_tower.lower() or "aimv2" in vision_tower.lower()):
        return Aimv2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if "siglip" in vision_tower.lower() or "gemma3-vision-encoder" in vision_tower:
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if os.path.exists(vision_tower) or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower identifier: {vision_tower}")