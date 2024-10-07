import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.semantic_segmentation import segment_image

def display_image_with_segmentation(image, seg_mask, class_labels):
    # Convert the image to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a color palette for visualization
    cmap = plt.get_cmap('tab20', len(class_labels))
    colors = cmap(np.arange(len(class_labels)))
    segmentation_color = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)

    for label in range(len(class_labels)):
        segmentation_color[seg_mask == label] = (colors[label][:3] * 255).astype(np.uint8)

    # Create a plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Overlay the segmentation mask on the original image
    ax.imshow(image_rgb)
    ax.imshow(segmentation_color, alpha=0.5)  # Set alpha for transparency
    ax.set_title("Segmentation Overlay")
    ax.axis('off')

    # Create the legend
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i][:3], markersize=10, label=class_labels[i]) for i in range(len(class_labels))]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
    plt.pause(0.1)  # Pause for 0.1 seconds

    plt.close(fig)  # Close the figure to free up memory

def main():
    input_dir = "/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/tets"
    images = sorted(os.listdir(input_dir))

    for image_name in images:
        img_path = os.path.join(input_dir, image_name)

        # Load the image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot open or read the image file at {img_path}")

        # Perform semantic segmentation
        seg_mask, class_labels = segment_image(image)

        # Display the image with the segmentation overlay
        display_image_with_segmentation(image, seg_mask, class_labels)

if __name__ == "__main__":
    main()