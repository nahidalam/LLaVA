from llava.model.multimodal_encoder.aimv2_encoder import Aimv2VisionTower
from PIL import Image
import torch

# ---- fake args object (LLaVA normally passes this) ----
class Args: pass
args = Args()
args.mm_vision_select_layer   = 5        # pick any layer you like
args.mm_vision_select_feature = "patch"  # or "cls_patch"

# ---- build the tower ----
tower = Aimv2VisionTower(
    "apple/aimv2-large-patch14-224",      # public checkpoint name
    args=args,
    delay_load=False                      # download weights immediately
)

# ---- create a dummy red image ----
img   = Image.new("RGB", (224, 224), "red")
pixel = tower.image_processor(images=img, return_tensors="pt").pixel_values

# ---- forward pass ----
features = tower(pixel)
print("Output shape :", features.shape)      # expect (1, 196, 768)
print("Dummy shape  :", tower.dummy_feature.shape)
print("AIM v2 encoder is working")
