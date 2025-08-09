from llava.model.multimodal_encoder.siglip_encoder import SiglipVisionTower
from PIL import Image

# 1. Configure the minimal args
class Args: pass
args = Args()
args.mm_vision_select_layer   = 5         # pick a layer (0–n)
args.mm_vision_select_feature = 'patch'   # or 'cls_patch'

# 2. Instantiate & load weights
tower = SiglipVisionTower(
    vision_tower='google/siglip-base-patch16-224',  # ← new model id
    args=args,
    delay_load=False
)

# 3. Load & preprocess a sample image
img = Image.new('RGB', (224, 224), color='red')  # simple red square
pix = tower.image_processor(images=img, return_tensors='pt').pixel_values

# 4. Forward pass & inspect
feats = tower(pix)
print("Output shape:", feats.shape)           # expect (1, num_patches, hidden_size)
print("Dummy feature shape:", tower.dummy_feature.shape)

# 5. Quick sanity checks
assert feats.ndim == 3
assert tower.dummy_feature.ndim == 2
print("SigLIP encoder is working")
