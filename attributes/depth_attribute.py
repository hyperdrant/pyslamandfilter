### Imports ###
import sys
import os

### Paths ###
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))

### Imports ###
from models.Depth.depth_estimation import estimate_depth # import function to estimate depth of a monocular image

### Depth Attribute Extraction ###
def depth_attribute(keypoints, image):
    """
    Source: [14] https://depth-anything.github.io/

    Computes the depth values for each keypoint in the image.

    Parameters:
    - keypoints: List of cv2.KeyPoint objects
    - image: Input image as a numpy array in BGR format

    Returns:
    - keypoint_depth_pairs: List of tuples, each containing a keypoint and its corresponding depth value
    - depth_map: Estimated depth map of the image
    """

    depth_map = estimate_depth(image) # Determine depth map of the image
    feature_depth = []
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1]) # Extract keypoints x / y coordinates (convert to int for depth_map)
        depth_value = depth_map[y, x] # Determine keypoint depth based on position in depth_map
        feature_depth.append((kp, depth_value)) # Append keypoint + the corresponding depth value to the list (required for feature storage)
    
    return feature_depth, depth_map
    




