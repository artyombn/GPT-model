"""
Model from Hugging Face:
skyau/dog-breed-classifier-vit from
"""
import torch

from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
from pathlib import Path


model_name = "skyau/dog-breed-classifier-vit"

# Load image processor
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Load model
model = ViTForImageClassification.from_pretrained(model_name)

print("Model is loaded:", model.config.model_type)

DOG_DIR = Path(__file__).resolve().parent / "dog_images"

image_path1 = DOG_DIR / "douglas.jpeg"
image_path2 = DOG_DIR / "grusha.jpeg"
image = Image.open(image_path1).convert("RGB")
# image.show()

# Image preparation
inputs = image_processor(images=image, return_tensors="pt")
print(inputs["pixel_values"].shape)

# Eval mode
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted dog breed:", model.config.id2label[predicted_class_idx])