### Imports ###
import cv2
from python_orb_slam3 import ORBExtractor
import numpy as np

### Feature Extraction ###
def extract_features(image):
    """
    Source: [7] https://doi.org/10.1109/TRO.2021.3075644, [8] https://github.com/mnixry/python-orb-slam3

    Extracts ORB features from an image as in ORB SLAM 3.

    Arguments:
    image (ndarray): Input image from which to extract features.

    Returns:
    keypoints (list): Detected FAST keypoints in the image.
    descriptors (ndarray): BRIEF descriptors corresponding to the keypoints.
    """

    # Convert image to grayscale 
    if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = ORBExtractor(n_features=2000) # [8]

    # Detect FAST keypoints and BRIEF descriptors
    keypoints, descriptors = orb.detectAndCompute(image)

    return keypoints, descriptors


