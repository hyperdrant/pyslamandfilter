### Imports ###
import sys
import os
import cv2
import numpy as np

### Parameters ###
alpha_bright = 1.4  # Contrast factor for bright image
beta_bright = 45    # Brightness factor for bright image
alpha_dark = 0.6    # Contrast factor for dark image
beta_dark = -25     # Brightness factor for dark image
alpha_high = 2.0    # Contrast factor for high contrast image
beta_high = 0       # Brightness factor for high contrast image
alpha_low = 0.4    # Contrast factor for low contrast image
beta_low = 0        # Brightness factor for low contrast image

ransac_thresh = 25
max_iters = 2000
confidence = 0.99

### Paths ###
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

### Imports ###
from preprocessing.feature_extraction import extract_features
from preprocessing.feature_matching import match_featuresBF

### Adjust Brightness / Contrast ###
def adjust_brightness_contrast(image, alpha, beta):
    """
    Adjusts the brightness and contrast of an input image.

    Parameters:
    - image (ndarray): Input image in BGR format.
    - alpha (float): Contrast control. Values >1.0 will increase contrast, values between 0 and 1.0 will decrease contrast.
    - beta (int): Brightness control. Positive values will increase brightness, negative values will decrease brightness.

    Returns:
    - new_image (ndarray): The adjusted image with modified brightness and contrast.
    """

    modified_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta) # modified_image = alpha * image + beta
    return modified_image

### Robustness Attribute Extraction ###
def robustness_attribute(preprocessed_image, keypoints, descriptors):
    """
    Source: /

    Evaluates the robustness of keypoints by adjusting brightness and contrast of the image.

    Parameters:
    - preprocessed_image (ndarray): Input preprocessed image.
    - keypoints (list): List of cv2.KeyPoint objects.
    - descriptors (ndarray): Descriptors corresponding to the keypoints.

    Returns:
    - feature_robustness (list): List of tuples, each containing a keypoint and its robustness score.
    """

    # Adjust the brightness and contrast of the preprocessed image
    bright_img = adjust_brightness_contrast(preprocessed_image, alpha=alpha_bright, beta=beta_bright)
    dark_img = adjust_brightness_contrast(preprocessed_image, alpha=alpha_dark, beta=beta_dark)
    high_img = adjust_brightness_contrast(preprocessed_image, alpha=alpha_high, beta=beta_high)
    low_img = adjust_brightness_contrast(preprocessed_image, alpha=alpha_low, beta=beta_low)

    modified_images = [bright_img, dark_img, high_img, low_img] # Store modified images in a list

    robustness = np.zeros(len(keypoints)) # Initialize array
    inlier_matches = [] # Inlier matches initialization 

    for mod_img in modified_images:
        mod_kp, mod_desc = extract_features(mod_img) # Extract features with implemented feature extractor
        inlier_matches, a = match_featuresBF(keypoints, descriptors, mod_kp, mod_desc, ransac_thresh, max_iters, confidence) # Match features with implemented feature matcher (MAGSAC++)
        
        # Update robustness based on matching results
        for match in inlier_matches:
            robustness[match.queryIdx] += 1

    feature_robustness = [(keypoints[i], robustness[i]) for i in range(len(keypoints))] # Append keypoint + the corresponding robustness value to the list (required for feature storage)

    return feature_robustness