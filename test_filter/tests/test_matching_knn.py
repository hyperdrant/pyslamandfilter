### Imports ###
import cv2 
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from python_orb_slam3 import ORBExtractor

### Parameters ###
threshold_contrast = 0.25 # Threshold to classify whether and image has a high or low contrast ([1] AFE-ORB-SLAM Eq. 1)
threshold_illumination = 0 # Threshold to classify whether an image is bright or dimmed ([1] Eq. 8)
exp_in = 128 # Expected global average intensity (experimental verification) ([1] Eq. 7)

### Global ###
image_contrast_distribution = {"Normal": 0, "Bright": 0, "Dimmed": 0} # Global dictionary to store the distribution of image types

### Contrast Classification ### 
def classify_contrast(image_path): 
    """
    Source: [1] AFE-ORB-SLAM (https://doi.org/10.1007/s10846-022-01645-w) 

    The image is classified (normal, bright, dimmed) based on its contrast and illumination

    Arguments:
    image_path (str): The path to the image file.

    Return values::
    image_type (int):
         0 - Normal image
         1 - Bright image
         2 - Dimmed image
    """

    image = cv2.imread(image_path) # Load image from given path
    
    if image is None:
        raise FileNotFoundError(f"Cannot open or read the image file at {image_path}")
    
    channels = image.shape[2] if len(image.shape) == 3 else 1 # Determine number of channels of the image (RGB / grayscale)
    
    if channels == 1:
        L = image.copy()  # Grayscale
    else:
        HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL) # RGB --> HSV
        HSV_channels = cv2.split(HSV) # Split in H, S, V
        L = HSV_channels[2]  # Take V channel (brigthness)
    
    L_norm = L.astype(np.float64) / 255.0 # Normalization (0 - 1)
    mean, stddev = cv2.meanStdDev(L_norm) # Mean and std
    
    lambda_value = stddev[0, 0] # lambda_value = std


    if lambda_value > threshold_contrast: # [1] Eq. 1
        image_type = 0 # Normal image (lambda > threshold)
        image_contrast_distribution["Normal"] += 1
    else:
        YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # Intensity component of the image
        Y = YCrCb[:,:,0]

        # Calculate wheter image is bright / dimmed
        M,N = image.shape[:2]
        mean_in = np.sum(Y/(M*N)) 
        t = (mean_in - exp_in)/ exp_in

        if t >= threshold_illumination: # [1] Eq. 8
            image_type = 1 # Bright image
            image_contrast_distribution["Bright"] += 1
        else: 
            image_type = 2 # Dimmed image
            image_contrast_distribution["Dimmed"] += 1
    return image_type, image_contrast_distribution


### Parameters ### 
alpha = 0.6 # Parameter to control the level of image contrast enhancement ([1] Eq. 17 & p. 5f)
beta = 1.0 # Parameter to control the level of image sharpening enhancement ([1] Eq. 17 & p. 5f)
theta = 0.3 # Threshold used for CDF truncations ([1] Eq. 9 and p. 5)
a_bright = 0.25 # The parameter to control the degree of enhancement for bright images (not specified in [1], specified in [2] p. 10)
a_dimmed = 0.75 # The parameter to control the degree of enhancement for dimmed images (not specified in [1], specified in [2] p. 10)

### Image Preprocessing ### 
def preprocess_image(image_path, image_contrast):
    """
    Source: [1] AFE-ORB-SLAM (https://doi.org/10.1007/s10846-022-01645-w), [2] Contrast Enhancement (https://doi.org/10.48550/arXiv.1709.04427), [3] https://github.com/leowang7/iagcwd/blob/master/IAGCWD.py

    The image is preprocessed based on its classified contrast (contrast enhancement + image sharpening)

    Arguments:
    image_path (str): The path to the image file.
    image_contrast (int): The type of image contrast:
        0 - Normal image
        1 - Bright image
        2 - Dimmed image

    Return values:
    enhanced_image (numpy.ndarray): The processed image with enhanced contrast and sharpness.
    """

    image = cv2.imread(image_path) # Load image from given path
    
    if image is None:
        raise FileNotFoundError(f"Cannot open or read the image file at {image_path}")
    
    # Preprocessing based on classified image contrast based on [1] Eq. 17
    if image_contrast == 1:  # Bright image (1)
        T_mask = np.uint8(alpha*process_bright(image))
        g_mask = np.uint8(beta*sharpen_image(image))
        enhanced_image = image - T_mask
        enhanced_image = cv2.add(enhanced_image, g_mask)
    elif image_contrast == 2:  # Dimmed image (2)
        T_mask = np.uint8(alpha*process_dimmed(image))
        g_mask = np.uint8(beta*sharpen_image(image))
        enhanced_image = image + T_mask
        enhanced_image = cv2.add(enhanced_image, g_mask)
    else: # Normal image (3)
        g_mask = np.uint8(beta*sharpen_image(image))
        enhanced_image = cv2.add(image, g_mask)

    return enhanced_image

def image_agcwd(img, a=0.25, truncated_cdf=False):
    """
    Source: [1] AFE-ORB-SLAM (https://doi.org/10.1007/s10846-022-01645-w), [2] Contrast Enhancement (https://doi.org/10.48550/arXiv.1709.04427), [3] https://github.com/leowang7/iagcwd/blob/master/IAGCWD.py 

    Function to apply Automatic Gamma Correction with Weighting Distribution (AGCWD) to an image (contrast enhancement)

    Arguments:
    img (numpy.ndarray): The input grayscale image.
    a (float): The parameter to control the degree of enhancement (default 0.25).
    truncated_cdf (bool): Flag to indicate if the CDF should be truncated (default false).

    Return values:
    img_new (numpy.ndarray): The contrast enhanced image.
    """

    # Adapted from [3]
    hist,bins = np.histogram(img.flatten(),256,[0,256]) # Histogram and bins of flattened image
    prob_normalized = hist / hist.sum() # Normlize histogram -> Probability distribution ([1] Eq. 2)

    unique_intensity = np.unique(img) # Unique intensitie values in the image
    prob_min = prob_normalized.min() # Minimum probability
    prob_max = prob_normalized.max() # Maximum probability 
    
    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min) # Apply AGCWD ([1] Eq. 3) 
    pn_temp[pn_temp>0] = prob_max * (pn_temp[pn_temp>0]**a)
    pn_temp[pn_temp<0] = prob_max * (-((-pn_temp[pn_temp<0])**a))
    prob_normalized_wd = pn_temp / pn_temp.sum() # Normalize weighted distribution to [0,1]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum() # Compute CDF ([1] Eq. 5)
    
    if truncated_cdf: 
        inverse_cdf = np.maximum(theta,1 - cdf_prob_normalized_wd) # [1] Eq. 9
    else:
        inverse_cdf = 1 - cdf_prob_normalized_wd # [1] Eq. 6
    
    img_new = img.copy() # Copy of original image to apply AGCWD
    for i in unique_intensity:
        img_new[img==i] = np.round(255 * (i / 255)**inverse_cdf[i]) # [1] Eq. 10
   
    return img_new

def process_bright(img):
    """
    Source: [1] AFE-ORB-SLAM (https://doi.org/10.1007/s10846-022-01645-w), [2] Contrast Enhancement (https://doi.org/10.48550/arXiv.1709.04427), [3] https://github.com/leowang7/iagcwd/blob/master/IAGCWD.py
    
    Processes a bright image to generate a mask (contrast enhancement).
    
    Arguments:
    img (ndarray): Input bright image in grayscale.
    
    Returns:
    T_mask (ndarray): Transformation mask for the input bright image.
    """

    img_negative = 255 - img # Negative image of the input ([1] Eq. 11)
    agcwd = image_agcwd(img_negative, a=a_bright, truncated_cdf=False) # Apply AGCWD
    I_ce = 255 - agcwd # [1] Eq. 11
    T_mask = img - I_ce # [1] Eq. 12
    return T_mask

def process_dimmed(img):
    """
    Source: [1] AFE-ORB-SLAM (https://doi.org/10.1007/s10846-022-01645-w), [2] Contrast Enhancement (https://doi.org/10.48550/arXiv.1709.04427), [3] https://github.com/leowang7/iagcwd/blob/master/IAGCWD.py
    
    Processes a dimmed image to generate a mask (contrast enhancement).
    
    Arguments:
    img (ndarray): Input dimmed image in grayscale.
    
    Returns:
    T_mask (ndarray): Transformation mask for the input dimmed image.
    """

    agcwd = image_agcwd(img, a=a_bright, truncated_cdf=True) # Apply AGCWD
    I_ce = agcwd
    T_mask = I_ce - img # [1] Eq. 12
    return T_mask

def sharpen_image(img):
    """
    Source: [1] AFE-ORB-SLAM (https://doi.org/10.1007/s10846-022-01645-w), [2] Contrast Enhancement (https://doi.org/10.48550/arXiv.1709.04427), [3] https://github.com/leowang7/iagcwd/blob/master/IAGCWD.py
    
    Processes the image to generate a mask (sharpening enhancement).
    
    Arguments:
    img (ndarray): Input normal image in grayscale.
    
    Returns:
    g_mask (ndarray): Transformation mask for the input image (sharpening adjustment).
    """

    f = cv2.GaussianBlur(img, (3,3), 0) # [1] Eq. 13
    g_mask = cv2.subtract(img,f) # [1] Eq. 14
    return g_mask



### Paramaters ### 
ransacReprojThreshold = 25  # Reprojection threshold
maxIters = 8000 # maxIters
confidence = 0.99 # Confidence level

### Feature Matching ### 
def match_features(prev_keypoints, prev_descriptors, keypoints, descriptors):
    """
    Source: [9] https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html, [10] https://openaccess.thecvf.com/content_CVPR_2020/papers/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.pdf, [11] https://docs.opencv.org/4.x/de/d3e/tutorial_usac.html, [12] #https://opencv.org/blog/evaluating-opencvs-new-ransacs/ 

    Matches features between two sets of keypoints and descriptors.

    Arguments:
    prev_keypoints (list): List of keypoints from the previous image.
    prev_descriptors (ndarray): Descriptors corresponding to the previous keypoints.
    keypoints (list): List of keypoints from the current image.
    descriptors (ndarray): Descriptors corresponding to the current keypoints.

    Returns:
    inlier_matches (list): List of inlier matches after applying homography and MAGSAC++.
    """

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # BFMatcher with Hamming distance and cross-check

    matches = bf.match(prev_descriptors, descriptors) # Match descriptors
    
    matches = sorted(matches, key=lambda x: x.distance) # Sort matches by distance

    # Initialization of arrays to store keypoints coordinates
    points_prev = np.zeros((len(matches), 2), dtype=np.float32)
    points_current = np.zeros((len(matches), 2), dtype=np.float32)

    # Extract coordinates
    for i, match in enumerate(matches):
        points_prev[i, :] = prev_keypoints[match.queryIdx].pt
        points_current[i, :] = keypoints[match.trainIdx].pt

    # Find homography using MAGSAC++
    homography, mask = cv2.findHomography(points_prev, points_current, cv2.USAC_MAGSAC, ransacReprojThreshold, maxIters=maxIters, confidence=confidence) # [9, 10, 11, 12]

    # Select inlier matches
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    
    return inlier_matches


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
    orb = ORBExtractor() # [8]

    # Detect FAST keypoints and BRIEF descriptors
    keypoints, descriptors = orb.detectAndCompute(image)

    return keypoints, descriptors







###############################################





import cv2
import numpy as np
import matplotlib.pyplot as plt

# Bildpfade
img1_path = '/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/input_images/test/office_spring_winter/1_1.png'
img2_path = '/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/input_images/test/office_spring_winter/1_2.png'

# Kontrastklassifizierung
img1_contrast, dist1 = classify_contrast(img1_path)
img2_contrast, dist2 = classify_contrast(img2_path)
print(img1_contrast)
print(img2_contrast)

# Bildvorverarbeitung
preprocessed_img1 = preprocess_image(img1_path, img1_contrast)
preprocessed_img2 = preprocess_image(img2_path, img2_contrast)

# Zuschneiden des unteren Teils der Bilder
crop_fraction = 0.23  # Anteil des Bildes, der unten abgeschnitten wird
height1 = preprocessed_img1.shape[0]
crop_height1 = int(height1 * crop_fraction)
preprocessed_img1 = preprocessed_img1[:-crop_height1, :]

height2 = preprocessed_img2.shape[0]
crop_height2 = int(height2 * crop_fraction)
preprocessed_img2 = preprocessed_img2[:-crop_height2, :]

# ORB Features extrahieren
keypoints1, descriptors1 = extract_features(preprocessed_img1)
keypoints2, descriptors2 = extract_features(preprocessed_img2)

# Feature Matching mit k-NN Matcher und Distance Ratio Test
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Distance Ratio Test und Euklidische Distanz Filterung
ratio_thresh = 0.75
valid_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        pt1 = np.array(keypoints1[m.queryIdx].pt)
        pt2 = np.array(keypoints2[m.trainIdx].pt)
        if np.linalg.norm(pt1 - pt2) < 40:
            valid_matches.append(m)

print(f'Number of valid matches: {len(valid_matches)}')

# Matches anzeigen
img_matches = cv2.drawMatches(preprocessed_img1, keypoints1, preprocessed_img2, keypoints2, valid_matches, None)

plt.figure(figsize=(20, 10))
plt.imshow(img_matches)
plt.title('Valid Matches')
plt.axis('off')
plt.show()






