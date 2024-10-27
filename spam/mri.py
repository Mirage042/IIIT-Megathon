import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# Load the model and image processor
model_name = "DHEIVER/Alzheimer-MRI"
model = ViTForImageClassification.from_pretrained(model_name)
image_processor = ViTImageProcessor.from_pretrained(model_name)

# Load and preprocess an MRI image
image_path = "image2.jpg"  # Replace with the actual path to your image
image = Image.open(image_path)

# Convert image to RGB if it is in grayscale
if image.mode != "RGB":
    image = image.convert("RGB")

# Preprocess the image
inputs = image_processor(images=image, return_tensors="pt")

# Run the model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

# Output the prediction
if predicted_class == 1:
    print("The MRI scan indicates Alzheimer's presence.")
else:
    print("The MRI scan does not indicate Alzheimer's presence.")
