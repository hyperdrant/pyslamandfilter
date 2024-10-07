## Imports ##
import os
import cv2
import matplotlib.pyplot as plt
import sys

# Add the src and models directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from preprocessing.image_classification import classify_contrast, classify_weather
from preprocessing.image_preprocessing import preprocess_image
from preprocessing.semantic_segmentation import segment_image
from preprocessing.feature_extraction import extract_features
from attributes.depth_attribute import depth_attribute 

def display_keypoints_with_depth(image, keypoint_depth_pairs):
    """
    Display keypoints on the image with their depth values.
    
    Parameters:
    - image: The input image
    - keypoint_depth_pairs: List of tuples, each containing a keypoint and its corresponding depth value
    """
    # Create a copy of the image to draw keypoints on
    image_with_keypoints = image.copy()
    
    for kp, depth in keypoint_depth_pairs:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(image_with_keypoints, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image_with_keypoints, f'{depth:.1f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
    
    # Convert the image to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image_rgb)
    plt.title("Keypoints with Depth")
    plt.axis('off')
    plt.show()

def main(input_dir):
    images = sorted(os.listdir(input_dir))

    for image_name in images:
        img_path = os.path.join(input_dir, image_name)

        ## Preprocessing Pipeline ##
        # Image Classification 
        image_contrast = classify_contrast(img_path) # 0: Normal, 1: Bright, 2: Dimmed

        # Image Preprocessing
        preprocessed_image = preprocess_image(img_path, image_contrast)

        # Extract ORB features
        keypoints, descriptors = extract_features(preprocessed_image)

        # Depth Filter
        keypoint_depth_pairs = depth_attribute(keypoints, preprocessed_image)

        # Display keypoints with depth
        display_keypoints_with_depth(preprocessed_image, keypoint_depth_pairs)



if __name__ == "__main__":
    input_dir = "/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/tets"
    main(input_dir)