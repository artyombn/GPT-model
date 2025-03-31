import tensorflow as tf

from transformers import AutoImageProcessor, TFAutoModelForImageClassification
from PIL import Image
from pathlib import Path

model_name = "skyau/dog-breed-classifier-vit"

image_processor = AutoImageProcessor.from_pretrained(model_name)

model = TFAutoModelForImageClassification.from_pretrained(model_name)
print("Model is loaded:", model.config.model_type)

DOG_DIR = Path(__file__).resolve().parent / "dog_images"

image_path1 = DOG_DIR / "douglas.jpeg"
image_path2 = DOG_DIR / "grusha.jpeg"
image = Image.open(image_path1).convert("RGB")

inputs = image_processor(images=image, return_tensors="tf")
print(inputs["pixel_values"].shape)

outputs = model(inputs)
logits = outputs.logits

predicted_class_idx = tf.argmax(logits, axis=-1).numpy()[0]
print("Predicted breed:", model.config.id2label[predicted_class_idx])