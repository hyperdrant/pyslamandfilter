import os
import sys
import cv2
import matplotlib.pyplot as plt

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.image_classification import classify_contrast
from preprocessing.image_preprocessing import preprocess_image

# Mapping of numerical labels to descriptive labels
image_type_labels = {0: "normal", 1: "bright", 2: "dimmed"}

def display_image_with_annotation(image, title, ax):
    # Convert the image to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Update the plot
    ax.clear()
    ax.imshow(image_rgb)
    ax.set_title(title)
    ax.axis('off')
    plt.draw()
    plt.pause(5)  

def main():
    input_dir = "/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/tets"
    images = sorted(os.listdir(input_dir))

    # Set up the plot
    fig, ax = plt.subplots()

    for image_name in images:
        img_path = os.path.join(input_dir, image_name)
        
        # Perform image classification to get the contrast type
        image_contrast = classify_contrast(img_path)
        
        # Preprocess the image based on its contrast type
        preprocessed_image = preprocess_image(img_path, image_contrast)
        
        # Determine the title text
        if image_contrast == 0:
            title = "No contrast enhancement, Image sharpened"
        elif image_contrast == 1:
            title = "Enhanced contrast of bright image, Image sharpened"
        elif image_contrast == 2:
            title = "Enhanced contrast of dimmed image, Image sharpened"

        # Display the image with the title
        display_image_with_annotation(preprocessed_image, title, ax)

    plt.close(fig)

if __name__ == "__main__":
    main()