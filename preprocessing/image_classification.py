### Imports ###
import cv2 
import numpy as np
from PIL import Image
import tensorflow as tf

### Parameters ###
threshold_contrast = 0.25 # Threshold to classify whether and image has a high or low contrast ([1] AFE-ORB-SLAM Eq. 1)
threshold_illumination = 0 # Threshold to classify whether an image is bright or dimmed ([1] Eq. 8)
exp_in = 128 # Expected global average intensity (experimental verification) ([1] Eq. 7)

### Global ###
image_contrast_distribution = {"Normal": 0, "Bright": 0, "Dimmed": 0} # Global dictionary to store the distribution of image types

### Contrast Classification ### 
def classify_contrast(image): 
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

    # image = cv2.imread(image_path) # Load image from given path
    
    # if image is None:
    #     raise FileNotFoundError(f"Cannot open or read the image file at {image_path}")
    
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

### Weather Classification ### 
def classify_weather(image_path):
    """
    Source: https://github.com/Qwerty735805/Weather_Classification

    Info: Model shows bad performance --> Replace with more accurate model or use another approach (Onboard vehicle sensors)

    The weather depicted in the image is classified using a pretrained Deep Learning Model

    Args:
    image_path (str): The path to the image file.

    Returns:
    int: The type of weather depicted in the image.
         0 - Cloudy
         1 - Sunny
         2 - Rainy
         3 - Snowy
         4 - Foggy

    Raises:
    FileNotFoundError: If the image file cannot be opened or read.
    """

    model = tf.keras.models.load_model('/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/models/weather_classifier.h5') # Load pretrained model

    image = Image.open(image_path)

    if image is None:
        raise FileNotFoundError(f"Cannot open or read the image file at {image_path}")
    
    if image.mode != "RGB":
        image = image.convert("RGB") 
    image = image.resize((100, 100)) # Resize image to match model input size
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image).tolist()[0]

    image_weather = np.argmax(prediction) # 0 (Cloudy), 1 (Sunny), 2 (Rainy), 3 (Snowy), 4 (Foggy)
    
    return image_weather


