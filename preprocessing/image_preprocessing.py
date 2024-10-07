### Imports ### 
import cv2
import numpy as np
import matplotlib.pyplot as plt

### Parameters ### 
alpha = 0.3 # Parameter to control the level of image contrast enhancement ([1] Eq. 17 & p. 5f)
beta = 1.0 # Parameter to control the level of image sharpening enhancement ([1] Eq. 17 & p. 5f)
theta = 0.3 # Threshold used for CDF truncations ([1] Eq. 9 and p. 5)
a_bright = 0.25 # The parameter to control the degree of enhancement for bright images (not specified in [1], specified in [2] p. 10)
a_dimmed = 0.75 # The parameter to control the degree of enhancement for dimmed images (not specified in [1], specified in [2] p. 10)

### Image Preprocessing ### 
def preprocess_image(image, image_contrast):
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

    # image = cv2.imread(image_path) # Load image from given path
    
    # if image is None:
    #     raise FileNotFoundError(f"Cannot open or read the image file at {image_path}")
    
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


