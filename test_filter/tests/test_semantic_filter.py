## Imports ##
import os
import cv2
import matplotlib.pyplot as plt
import sys

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.image_classification import classify_contrast, classify_weather
from preprocessing.image_preprocessing import preprocess_image
from preprocessing.semantic_segmentation import segment_image
from preprocessing.feature_extraction import extract_orb_features
from attributes.semantic_filter import semantic_filter

def display_features(image, keypoint_class_pairs, class_labels):
    """
    Display features on the image. Green for general features, red for features classified as "car",
    blue for features classified as "vegetation".
    
    Parameters:
    - image: The input image
    - keypoint_class_pairs: List of tuples, each containing a keypoint and its corresponding class index
    - class_labels: List of class labels corresponding to the indices in the segmentation mask
    """
    # Create a copy of the image to draw keypoints on
    image_with_keypoints = image.copy()
    
    # Draw keypoints as points
    for kp, class_index in keypoint_class_pairs:
        if class_labels[class_index] == "car":
            color = (0, 0, 255)  # Red for features classified as "car"
        elif class_labels[class_index] == "vegetation":
            color = (255, 0, 0)  # Blue for features classified as "vegetation"
        else:
            color = (0, 255, 0)  # Green for general features
        cv2.circle(image_with_keypoints, (int(kp.pt[0]), int(kp.pt[1])), 2, color, -1)
    
    # Convert the image to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Feature Classification")
    plt.axis('off')
    plt.show()

def main(img_path):
    ## Preprocessing Pipeline ##
    # Image Classification 
    image_contrast = classify_contrast(img_path) # 0: Normal, 1: Bright, 2: Dimmed
    image_weather = 1 # 0: Cloudy, 1: Sunny, 2: Rainy, 3: Snowy, 4: Foggy

    # Image Preprocessing
    preprocessed_image = preprocess_image(img_path, image_contrast)

    # Perform semantic segmentation
    seg_mask, class_labels = segment_image(preprocessed_image)

    # Extract ORB features
    keypoints, descriptors = extract_orb_features(preprocessed_image)

    # Semantic Filter
    keypoint_class_pairs = semantic_filter(keypoints, seg_mask, class_labels)
    
    # Display features
    display_features(preprocessed_image, keypoint_class_pairs, class_labels)

if __name__ == "__main__":
    img_path = "/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/input_images/erding_3_cropped/image_1200.jpg"  # Change this to the path of your test image
    main(img_path)