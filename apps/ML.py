import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color

# Define your color palette
colors = {
    "primary": "#750f90",
    "secondary": "#931883",
    "background": "#27293d",
    "text": "white",
}


# Function to get the latest image file from the 'uploads' folder
def get_latest_image(upload_folder):
    files = [
        f for f in os.listdir(upload_folder) if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    if not files:
        raise FileNotFoundError("No image files found in the uploads folder.")
    full_paths = [os.path.join(upload_folder, f) for f in files]
    latest_file = max(full_paths, key=os.path.getmtime)
    return latest_file


# Function to handle images with an Alpha channel (convert RGBA to RGB)
def handle_rgba(image):
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


# Function to calculate brightness
def calculate_brightness(image):
    image = handle_rgba(image)
    grayscale_image = color.rgb2gray(image)
    return np.mean(grayscale_image)


# Function to calculate contrast
def calculate_contrast(image):
    image = handle_rgba(image)
    grayscale_image = color.rgb2gray(image)
    p2, p98 = np.percentile(grayscale_image, (2, 98))
    return p98 - p2


# Function to calculate sharpness using Sobel filter
def calculate_sharpness(image):
    image = handle_rgba(image)
    grayscale_image = color.rgb2gray(image)
    edges = filters.sobel(grayscale_image)
    return np.mean(edges)


# Function to calculate edge intensity
def calculate_edge_intensity(image):
    image = handle_rgba(image)
    grayscale_image = color.rgb2gray(image)
    edges = filters.sobel(grayscale_image)
    return np.sum(edges)


# Function to plot metrics
def plot_metrics(metrics, output_plot_path):
    metric_names = ["Brightness", "Contrast", "Sharpness", "Edge Intensity"]

    plt.figure(figsize=(8, 5))  # Slightly larger size for better clarity
    bars = plt.bar(
        metric_names,
        metrics,
        color=[
            colors["primary"],
            colors["secondary"],
            colors["background"],
            colors["primary"],
        ],
    )

    # Add data labels above the bars for clarity
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.02,
            round(yval, 2),
            ha="center",
            va="bottom",
            color="white",
            fontsize=10,
        )

    plt.title("Image Metrics", color=colors["text"], fontsize=14)
    plt.ylabel("Metric Value", color=colors["text"], fontsize=12)
    plt.xticks(color=colors["text"], fontsize=10)
    plt.yticks(color=colors["text"], fontsize=10)
    plt.gca().set_facecolor(colors["background"])  # Set background color

    # Save the plot
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(
        output_plot_path, facecolor=colors["background"], dpi=300
    )  # Increased DPI for better resolution
    plt.close()  # Close the figure to free up memory


# Main processing logic
if __name__ == "__main__":
    # Folder where images are uploaded
    upload_folder = "uploads"

    # Get the latest image
    try:
        latest_image_path = get_latest_image(upload_folder)
        print(f"Processing latest image: {latest_image_path}")

        # Load the latest image
        image = io.imread(latest_image_path)

        # Calculate metrics
        brightness = calculate_brightness(image)
        contrast = calculate_contrast(image)
        sharpness = calculate_sharpness(image)
        edge_intensity = calculate_edge_intensity(image)

        # Print metrics
        print(f"Brightness: {brightness:.2f}")
        print(f"Contrast: {contrast:.2f}")
        print(f"Sharpness: {sharpness:.2f}")
        print(f"Edge Intensity: {edge_intensity:.2f}")

        # Plot metrics
        metrics = [brightness, contrast, sharpness, edge_intensity]

        # Ensure the directory exists (Updated path)
        plot_dir = r"C:\Users\adika\Documents\Codes\IIIT-Megathon\apps\static\images"
        os.makedirs(plot_dir, exist_ok=True)

        # Save the plot as an image file
        plot_path = os.path.join(plot_dir, "metrics_plot.png")
        plot_metrics(metrics, plot_path)

    except FileNotFoundError as e:
        print(e)
