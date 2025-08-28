from llava.model.multimodal_encoder.siglip_encoder import SiglipVisionTower
from PIL import Image
import torch

class Args: 
    def __init__(self):
        self.mm_vision_select_layer = -2      # Common choice: second-to-last layer
        self.mm_vision_select_feature = 'patch'  # or 'cls_patch'

args = Args()

local_ckpt_path = "gemma3_siglip_encoder"  # Your extracted encoder path
tower = SiglipVisionTower(
    vision_tower=local_ckpt_path,
    args=args,
    delay_load=False
)

img = Image.new('RGB', (224, 224), color='red')  # simple red square
pix = tower.image_processor(images=img, return_tensors='pt')['pixel_values']

with torch.no_grad():  
    feats = tower(pix)

print("Output shape:", feats.shape)           # expect (1, num_patches, hidden_size)
print("Dummy feature shape:", tower.dummy_feature.shape)

assert feats.ndim == 3, f"Expected 3D tensor, got {feats.ndim}D"
assert tower.dummy_feature.ndim == 2, f"Expected 2D dummy feature, got {tower.dummy_feature.ndim}D"

# Additional checks
print(f"Hidden size: {feats.shape[-1]}")
print(f"Number of patches: {feats.shape[1]}")
print(f"Vision tower config: {tower.config}")
print("SigLIP encoder is working ✓")

try:
    real_img = Image.open('sample.jpg')
    real_pix = tower.image_processor(images=real_img, return_tensors='pt')['pixel_values']
    real_feats = tower(real_pix)
    print("Real image shape:", real_feats.shape)
    pass
except Exception as e:
    print(f"Real image test failed: {e}")