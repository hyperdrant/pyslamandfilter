### Imports ### 
from scipy.spatial import KDTree
import numpy as np

### Parameters ### 
radius = 2 # Radius for filtering the features based on positions

### Position Filter ### 
def prefilter_pos(keypoints, descriptors):
    """
    Filters keypoints and corresponding descriptors based on spatial proximity, 
    ensuring that only the most relevant keypoints within a certain radius are retained.

    Parameters:
    - keypoints: List of cv2.KeyPoint objects.
    - descriptors: Numpy array of corresponding feature descriptors for the keypoints.

    Returns:
    - filtered_keypoints: List of cv2.KeyPoint objects.
    - filtered_descriptors: Numpy array of filtered descriptors.
    """
    # Convert keypoint coordinates to a numpy array for efficient spatial querying.
    kp_pts = np.array([kp.pt for kp in keypoints])
    
    # Build a KDTree for fast querying of neighboring keypoints.
    kdtree = KDTree(kp_pts)
    
    filtered_keypoints = []
    filtered_descriptors = []
    used_indices = set()  # Track indices of keypoints that have already been processed.
    
    for i, kp in enumerate(keypoints):
        if i in used_indices:
            continue  # Skip keypoints that have already been considered.
        
        # Add the current keypoint and its descriptor to the final list.
        filtered_keypoints.append(kp)
        filtered_descriptors.append(descriptors[i])
        used_indices.add(i)
        
        # Find all keypoints within the defined radius of the current keypoint.
        indices = kdtree.query_ball_point(kp.pt, radius)
        
        # Mark all nearby keypoints as used.
        for idx in indices:
            used_indices.add(idx)

    # Convert the list of final descriptors back to a numpy array.
    filtered_descriptors = np.array(filtered_descriptors)
    
    return filtered_keypoints, filtered_descriptors


### Observability Filter ### 
def prefilter_obs(feature_dict): # Remove features with obs = 1

    """
    Filters out features from the feature_dict that have an observability of 1,
    and renumbers the remaining features starting from 0.

    Parameters:
    - feature_dict (dict): Dictionary containing feature data.

    Returns:
    - filtered_feature_dict (dict): Filtered and renumbered feature dictionary with only features that have an observability greater than 1.
    """

    # Create a new dictionary to store the filtered and renumbered features
    filtered_feature_dict = {}

    # Initialize the new feature ID counter
    new_feature_id = 0

    # Iterate through the feature_dict
    for feature_id, feature_data in feature_dict.items():
        if feature_data["observability"] > 1:
            # Assign a new ID starting from 0 to the filtered features
            filtered_feature_dict[new_feature_id] = feature_data
            new_feature_id += 1

    # Get the number of features remaining after filtering
    num_features = len(filtered_feature_dict)

    # Print the number of remaining features
    # print(f"[INFO] Remaining features after prefiltering: {num_features}")

    return filtered_feature_dict
