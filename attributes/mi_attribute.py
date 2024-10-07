import numpy as np
from python_orb_slam3 import ORBExtractor
import cv2
from matplotlib import pyplot as plt

def compute_mutual_information(features, target):
    num_features = features.shape[1]
    
    # covariance calculation
    state_covariance = np.cov(target, rowvar=False)
    
    mutual_info_scores = []
    
    # mutual info calculation
    for i in range(num_features):
        # remove i-th feature
        remaining_features = np.delete(features, i, axis=1)
        
        # calculate covairance without i-th feature
        feature_covariance = np.cov(remaining_features, rowvar=False)
        
        # calculate joint covaricance
        joint_data = np.hstack((target, remaining_features))
        joint_covariance = np.cov(joint_data, rowvar=False)
        
        # mutual information
        try:
            det_state_covariance = np.linalg.det(state_covariance)
            det_feature_covariance = np.linalg.det(feature_covariance)
            det_joint_covariance = np.linalg.det(joint_covariance)
            
            if det_state_covariance > 0 and det_feature_covariance > 0 and det_joint_covariance > 0:
                mutual_information = 0.5 * np.log(det_state_covariance * det_feature_covariance / det_joint_covariance)
                mutual_info_scores.append((i, mutual_information))
            else:
                mutual_info_scores.append((i, 0))
        except np.linalg.LinAlgError:
            mutual_info_scores.append((i, 0))

    return mutual_info_scores


def remove_least_important_features(features, keypoints, mutual_info_scores, ratio=0.5):
    mutual_info_scores.sort(key=lambda x: x[1])
    num_to_remove = int(len(mutual_info_scores) * ratio)
    indices_to_remove = [x[0] for x in mutual_info_scores[:num_to_remove]]
    
    reduced_features = np.delete(features, indices_to_remove, axis=1)
    reduced_keypoints = [kp for i, kp in enumerate(keypoints) if i not in indices_to_remove]
    
    return reduced_features, reduced_keypoints, indices_to_remove


target = np.array([38.78752953,	21.42514155,151.4852811])

# import image
source = cv2.imread("ADCAM_FRONT_MAIN_DATA/image_5231215582883_185460.jpg")

orb_extractor = ORBExtractor(n_features=200)
# orb feature detection
source_keypoints, source_descriptors = orb_extractor.detectAndCompute(source)
features = source_descriptors
target = np.tile(target, (len(features), 1))

# mutual info
mutual_info_scores = compute_mutual_information(features, target)

# remove half less informative features
reduced_features, reduced_keypoints, removed_indices = remove_least_important_features(features, source_keypoints, mutual_info_scores)


image_with_keypoints = cv2.drawKeypoints(source, reduced_keypoints, None, color=(0, 255, 0))


plt.figure(figsize=(10, 10))
plt.imshow(image_with_keypoints)
plt.title("Image with Reduced Keypoints")
plt.show()