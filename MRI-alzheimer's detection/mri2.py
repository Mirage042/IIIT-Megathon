import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the saved model
model = load_model("model.h5")

# Set your model's expected input dimensions
image_height, image_width = 224, 224  # Update according to your model's training input size

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(image_height, image_width))  # Resize image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to get the latest image from a directory
def get_latest_image(folder_path):
    # Get list of all files in the directory
    files = os.listdir(folder_path)
    # Filter out non-image files if needed, or add more robust checks here
    images = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
    # Get the full paths and sort by modification time
    full_paths = [os.path.join(folder_path, f) for f in images]
    latest_image = max(full_paths, key=os.path.getmtime)  # Get the latest image based on modification time
    return latest_image

# Path to the folder containing images
folder_path = 'path/to/your/image/folder'  # Replace with your folder path
img_path = get_latest_image(folder_path)  # Get the latest image

# Load and preprocess the latest image
test_image = load_and_preprocess_image(img_path)

# Make predictions
pred = model.predict(test_image)
pred_class = np.argmax(pred, axis=1)  # Get the index of the class with the highest probability

# Mapping class indices to labels
labels = {0: 'Mild Demented', 1: 'Moderate Demented', 2: 'Non Demented', 3: 'Very Mild Demented'}  # Replace with your actual labels
predicted_label = labels[pred_class[0]]

# Formatting the confidence scores
confidence_scores = pred[0]
formatted_scores = [f"{score:.8f}" for score in confidence_scores]

print(f"Predicted Class: {predicted_label}")
print("Confidence Scores:")
for class_label, score in zip(labels.values(), formatted_scores):
    print(f"  {class_label}: {score}")
