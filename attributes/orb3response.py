import cv2
from matplotlib import pyplot as plt
import numpy as np
from python_orb_slam3 import ORBExtractor


def select_features_by_response(keypoints, threshold):
    selected_keypoints = []
    selected_descriptors = []
    for kp in keypoints:
        if kp[0].response >= threshold:
            selected_keypoints.append(kp[0])
            selected_descriptors.append(kp[1])
    return selected_keypoints, selected_descriptors

source = cv2.imread("munich_000327_000019_leftImg8bit.png")
target = cv2.imread("munich_000327_000019_leftImg8bit.png")

orb_extractor = ORBExtractor()

# Extract features from source image
source_keypoints, source_descriptors = orb_extractor.detectAndCompute(source)
source_info = list(zip(source_keypoints, source_descriptors))
target_keypoints, target_descriptors = orb_extractor.detectAndCompute(target)
target_info = list(zip(target_keypoints, target_descriptors))

# Calculate median response
source_responses = [kp[0].response for kp in source_info]
median_souce_response = np.median(source_responses)
target_responses = [kp[0].response for kp in target_info]
median_target_response = np.median(target_responses)

# Select areas with high response
selected_source_keypoints, selected_source_descriptors = select_features_by_response(source_info, median_souce_response)
selected_target_keypoints, selected_target_descriptors = select_features_by_response(target_info, median_target_response)

# Match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(np.vstack(selected_source_descriptors), np.vstack(selected_target_descriptors))

# Draw matches
source_image = cv2.drawKeypoints(source, selected_source_keypoints, None)
target_image = cv2.drawKeypoints(target, selected_target_keypoints, None)
matches_image = cv2.drawMatches(source_image, source_keypoints, target_image, target_keypoints, matches, None)

# Show matches
plt.imshow(matches_image)
plt.show()
print(len(selected_target_keypoints))