import os
import sys
import cv2
import matplotlib.pyplot as plt

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.image_classification import classify_contrast, classify_weather

# Mapping of numerical labels to descriptive labels
image_type_labels = {0: "normal", 1: "bright", 2: "dimmed"}
weather_labels = {0: "cloudy", 1: "sunny", 2: "rainy", 3: "snowy", 4: "foggy"}

def display_image_with_classification(image_path, image_type, weather, ax):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot open or read the image file at {image_path}")
    
    # Convert the image to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Update the plot
    ax.clear()
    ax.imshow(image_rgb)
    ax.set_title(f"Type: {image_type_labels[image_type]}, Weather: {weather_labels[weather]}")
    ax.axis('off')
    plt.draw()
    plt.pause(10) 

def main():
    input_dir = "/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/input_images/test"
    images = sorted(os.listdir(input_dir))

    # Set up the plot
    fig, ax = plt.subplots()

    for image_name in images:
        img_path = os.path.join(input_dir, image_name)
        
        # Perform image classification
        image_type = classify_contrast(img_path)
        weather = 0

        # Display the image with its classification
        display_image_with_classification(img_path, image_type, weather, ax)

    plt.close(fig)

if __name__ == "__main__":
    main()