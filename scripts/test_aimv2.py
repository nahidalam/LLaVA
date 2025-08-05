from PIL import Image
from transformers import AutoImageProcessor, AutoModel

image = Image.open("/home/shapla/LLaVA/images/llava_logo.png")

processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-native")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-native")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
print(outputs)
