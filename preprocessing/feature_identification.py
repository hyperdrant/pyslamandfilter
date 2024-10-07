### Imports ### 
import cv2
import numpy as np

from preprocessing.feature_matching import match_featuresBF
# Globale Variable zur Verfolgung der aktuellen Feature-ID
global_feature_id = 0

# Create Mapping
def create_mapping(keypoints):
    """
    Erstellt ein Mapping von den KeyPoint-Objekten im Bild zu ihren lokalen IDs.

    Parameters:
    - keypoints: Liste der cv2.KeyPoint-Objekte des aktuellen Bildes (wie von OpenCV extrahiert).

    Returns:
    - mapping: Dictionary mit dem KeyPoint-Objekt als Schlüssel und der lokalen ID als Wert.
    """
    mapping = {}
    
    for i, keypoint in enumerate(keypoints):
        mapping[keypoint] = i  # Verwende das KeyPoint-Objekt als Schlüssel und die lokale ID als Wert
    return mapping


# Function to assign an ID to an unmatched feature
def assign_ID_unmatched(unmatched_features, mapping, start_id, features_with_ids):
    """
    Assigns global feature IDs to unmatched keypoints in an image and stores them in a dictionary.

    Parameters:
    - unmatched_features: List of unmatched keypoints (KeyPoint objects) that need a new ID.
    - mapping: Dictionary that maps KeyPoint objects to their original index positions in the image.
    - start_id: Starting value for the global feature ID (not directly used here, as global_feature_id is global).
    - features_with_ids: Dictionary that stores the global feature ID for each KeyPoint (key: KeyPoint ID, value: global feature ID).

    Returns:
    - features_with_ids: Updated dictionary that now includes the global feature IDs of the unmatched keypoints.
    """
    global global_feature_id  # Use the global variable global_feature_id to assign IDs sequentially
    
    for unmatched_idx, keypoint in enumerate(unmatched_features):
        # Use the KeyPoint object as the key in the mapping to retrieve its original index
        original_idx = mapping[keypoint]  
        
        # Assign the current global feature ID to the corresponding KeyPoint in the image
        features_with_ids[original_idx] = global_feature_id
        
        # Increment the global feature ID for the next unmatched KeyPoint
        global_feature_id += 1

    return features_with_ids


# Function to assign an ID to a matched feature
def assign_ID_matched(inlier_matches, features_with_ids_prev, features_with_ids_current, mapping, matched_keypoints):
    """
    Assigns global feature IDs to matched keypoints in the current image based on their matches in a previous image.

    Parameters:
    - inlier_matches: List of inlier matches between two images, which provides the query and train keypoint indices.
    - features_with_ids_prev: Dictionary containing the global feature IDs from the previous image (keys: Keypoint IDs, values: global feature IDs).
    - features_with_ids_current: Dictionary to store the global feature IDs for the current image (keys: Keypoint IDs, values: global feature IDs).
    - mapping: Dictionary mapping KeyPoint objects to their original index positions in the current image.
    - matched_keypoints: List of KeyPoint objects in the current image that were matched with keypoints in the previous image.

    Returns:
    - features_with_ids_current: Updated dictionary containing the global feature IDs for the matched keypoints in the current image.
    """

    for matched_idx, keypoint in enumerate(matched_keypoints):
        # Use the KeyPoint object as the key in the mapping to retrieve its original index in the current image
        original_idx = mapping[keypoint]  

        
        # Retrieve the global feature ID from the previous image using the query ID from the match
        query_id = inlier_matches[matched_idx].queryIdx  # Keypoint ID in the query image (previous image)
        ID = features_with_ids_prev.get(query_id, None)

        # Debugging: Check if the global feature ID was correctly retrieved
        if ID is None:
            print(f"Warning: No global feature ID found for query ID {query_id}")
            continue  # Skip this case if no global feature ID was found
        
        # Debugging: Check if the matched IDs align correctly between the two images
        if original_idx != inlier_matches[matched_idx].trainIdx:
            print(f"Mismatch in mapping for matched_idx {matched_idx}: expected {inlier_matches[matched_idx].trainIdx}, got {original_idx}")
            print(f"Keypoint (Query ID: {query_id}, Train ID: {inlier_matches[matched_idx].trainIdx})")
            print(f"Feature ID (Previous: {ID}, Current: {features_with_ids_current.get(original_idx)})")

        # Assign the global feature ID from the previous image to the corresponding keypoint in the current image
        features_with_ids_current[original_idx] = ID


    return features_with_ids_current


def find_unmatched_keypoints(keypoints, descriptors, inlier_matches):
    """
    Finds the keypoints and descriptors in the current image that were not matched with the previous image.

    Parameters:
    - keypoints: List of keypoints in the current image (as extracted by OpenCV).
    - descriptors: Numpy array of descriptors in the current image (as extracted by OpenCV).
    - inlier_matches: List of inlier matches between the previous and the current image.

    Returns:
    - unmatched_keypoints: List of keypoints that were not matched (in OpenCV format).
    - unmatched_descriptors: Numpy array of descriptors that were not matched (in OpenCV format).
    """
    # Extract the IDs of the keypoints in the current image that were matched (train IDs)
    matched_train_ids = set([match.trainIdx for match in inlier_matches])
    
    # Initialize lists to store unmatched keypoints and descriptors
    unmatched_keypoints = []
    unmatched_descriptors = []

    # Iterate through all keypoints in the current image
    for i, keypoint in enumerate(keypoints):
        # If the keypoint ID is not in the set of matched IDs, it is considered unmatched
        if i not in matched_train_ids:  
            unmatched_keypoints.append(keypoint)
            unmatched_descriptors.append(descriptors[i])

    # Convert the list of unmatched descriptors back into a numpy array
    unmatched_descriptors = np.array(unmatched_descriptors)
    
    return unmatched_keypoints, unmatched_descriptors


def find_matched_keypoints(keypoints, inlier_matches):
    """
    Extracts the keypoints in the current image that were matched with the previous image.

    Parameters:
    - keypoints: List of keypoints in the current image (as extracted by OpenCV).
    - inlier_matches: List of inlier matches between the previous and the current image.

    Returns:
    - matched_keypoints: List of keypoints that were matched (in OpenCV format).
    """
    # Initialize a list to store the matched keypoints
    matched_keypoints = []

    # Iterate through all inlier matches
    for match in inlier_matches:
        train_id = match.trainIdx  # Get the keypoint ID in the current image (train ID)
        matched_keypoints.append(keypoints[train_id])  # Add the matched keypoint to the list
    
    return matched_keypoints



def remap_inlier_matches(inlier_matches, mapping, unmatched_keypoints):
    """
    Remaps the trainIdx in the inlier_matches using the provided mapping and the keypoints.

    Parameters:
    - inlier_matches: List of inlier matches where the trainIdx needs to be remapped.
    - mapping: Dictionary that maps keypoints to their original index positions.
    - unmatched_keypoints: List of keypoints that appear in the inlier_matches.

    Returns:
    - remapped_matches: New list of matches with the remapped trainIdx.
    """
    # Initialize a list to store the remapped matches
    remapped_matches = []

    # Iterate through each match in the inlier_matches
    for match in inlier_matches:
        query_id = match.queryIdx  # Extract the query ID from the match (keypoint ID in the previous image)
        train_id = match.trainIdx  # Extract the train ID from the match (keypoint ID in the current image)

        # Use the train_id to get the corresponding keypoint in unmatched_keypoints
        keypoint = unmatched_keypoints[train_id]
        
        # Use the mapping to remap the keypoint to its original index position
        remapped_train_id = mapping.get(keypoint, None)

        if remapped_train_id is not None:
            # Create a new DMatch object with the remapped trainIdx
            remapped_match = cv2.DMatch(_distance=match.distance, _imgIdx=match.imgIdx, 
                                        _queryIdx=query_id, _trainIdx=remapped_train_id)
            remapped_matches.append(remapped_match)  # Add the remapped match to the list
        else:
            # Print a warning if no remapped train ID was found for the given train ID
            print(f"Warning: No remapped train ID found for train ID {train_id}")

    return remapped_matches  # Return the list of remapped matches



def feature_ID(current_keypoints, current_descriptors, 
               prev_keypoints1=None, prev_descriptors1=None, features_IDs1=None,
               prev_keypoints2=None, prev_descriptors2=None, features_IDs2=None,
               prev_keypoints3=None, prev_descriptors3=None, features_IDs3=None,
               ransac_params_prev=(25, 2000, 0.99), ransac_params_prev2=(25, 2000, 0.99), ransac_params_prev3=(25, 2000, 0.99),
               num_images_to_match=2):
    """
    Assigns global feature IDs to the keypoints of the current image by matching them with the keypoints
    of the previous images. The function supports matching with up to three previous images.

    Parameters:
    - current_image: The current image being processed.
    - current_keypoints: List of keypoints in the current image.
    - current_descriptors: Descriptors corresponding to the current keypoints.
    - prev_keypoints1, prev_keypoints2, prev_keypoints3: Keypoints from the previous images.
    - prev_descriptors1, prev_descriptors2, prev_descriptors3: Descriptors from the previous images.
    - features_IDs1, features_IDs2, features_IDs3: Feature ID dictionaries from the previous images.
    - ransac_params_prev, ransac_params_prev2, ransac_params_prev3: Tuples containing the RANSAC parameters 
      (ransacReprojThreshold, maxIters, confidence) for matching with the previous images.
    - num_images_to_match: Number of previous images to match with (1, 2, or 3; default is 2).

    Returns:
    - features_IDs_current: Dictionary containing the global feature IDs for the keypoints in the current image.
    - total_matches: Total number of matches found with the previous images.
    """
    global global_feature_id

    features_IDs_current = {}  # Create a dictionary to store the feature IDs for the current image

    keypoints_with_zero = [(kp, 0) for kp in current_keypoints] # Create and initialize a list to store distance score

    mapping_current = create_mapping(current_keypoints)  # Create a mapping for the keypoints in the current image
    total_matches = 0  # Initialize the total matches counter

    # Step 1: Match with the most recent previous image
    if num_images_to_match >= 1 and prev_keypoints1 is not None and prev_descriptors1 is not None:
        ransac_thresh, max_iters, confidence = ransac_params_prev
        inlier_matches_1, distance_score_1 = match_featuresBF(prev_keypoints1, prev_descriptors1, current_keypoints, current_descriptors, 
                                            ransac_thresh, max_iters, confidence)
   
        keypoints_with_zero = [(kp, distance_score_1.get(kp, 0)) for kp, val in keypoints_with_zero]
        total_matches += len(inlier_matches_1)  # Add the number of matches to the total
        matched_keypoints_1 = find_matched_keypoints(current_keypoints, inlier_matches_1)
        features_IDs_current = assign_ID_matched(inlier_matches_1, features_IDs1, features_IDs_current, 
                                                 mapping_current, matched_keypoints_1)

        # Find unmatched keypoints and descriptors
        unmatched_keypoints, unmatched_descriptors = find_unmatched_keypoints(current_keypoints, current_descriptors, inlier_matches_1)
    else:
        unmatched_keypoints = current_keypoints
        unmatched_descriptors = current_descriptors

    # Step 2: Match with the second most recent previous image (if available)
    if num_images_to_match >= 2 and prev_keypoints2 is not None and prev_descriptors2 is not None:
        ransac_thresh, max_iters, confidence = ransac_params_prev2
        inlier_matches_2, distance_score_2 = match_featuresBF(prev_keypoints2, prev_descriptors2, unmatched_keypoints, unmatched_descriptors, 
                                            ransac_thresh, max_iters, confidence)
        
        # keypoints_with_zero = [(kp, distance_score_2.get(kp, 0)) for kp, val in keypoints_with_zero]
        total_matches += len(inlier_matches_2)  # Add the number of matches to the total
        inlier_matches_2_c = remap_inlier_matches(inlier_matches_2, mapping_current, unmatched_keypoints)
        matched_keypoints_2 = find_matched_keypoints(unmatched_keypoints, inlier_matches_2)
        features_IDs_current = assign_ID_matched(inlier_matches_2_c, features_IDs2, features_IDs_current, 
                                                 mapping_current, matched_keypoints_2)

        # Update unmatched keypoints and descriptors
        unmatched_keypoints, unmatched_descriptors = find_unmatched_keypoints(unmatched_keypoints, unmatched_descriptors, inlier_matches_2)

    # Step 3: Match with the third most recent previous image (if available and required)
    if num_images_to_match >= 3 and prev_keypoints3 is not None and prev_descriptors3 is not None:
        ransac_thresh, max_iters, confidence = ransac_params_prev3
        inlier_matches_3, distance_score_3 = match_featuresBF(prev_keypoints3, prev_descriptors3, unmatched_keypoints, unmatched_descriptors, 
                                            ransac_thresh, max_iters, confidence)
       
        # keypoints_with_zero = [(kp, distance_score_3.get(kp, 0)) for kp, val in keypoints_with_zero]
        total_matches += len(inlier_matches_3)  # Add the number of matches to the total
        inlier_matches_3_c = remap_inlier_matches(inlier_matches_3, mapping_current, unmatched_keypoints)
        matched_keypoints_3 = find_matched_keypoints(unmatched_keypoints, inlier_matches_3)
        features_IDs_current = assign_ID_matched(inlier_matches_3_c, features_IDs3, features_IDs_current, 
                                                 mapping_current, matched_keypoints_3)

        # Update unmatched keypoints and descriptors
        unmatched_keypoints, unmatched_descriptors = find_unmatched_keypoints(unmatched_keypoints, unmatched_descriptors, inlier_matches_3)

    # Step 4: Assign new IDs to the remaining unmatched keypoints
    features_IDs_current = assign_ID_unmatched(unmatched_keypoints, mapping_current, global_feature_id, features_IDs_current)
    try:
        distance_score_1
    except NameError:
        distance_score_1 = {}

    try:
        distance_score_2
    except NameError:
        distance_score_2 = {}

    try:
        distance_score_3
    except NameError:
        distance_score_3 = {}

    # merge the dict for distance score
    merged_dict = {**distance_score_1, **distance_score_2, **distance_score_3}
    updated_keypoints = [(kp, merged_dict.get(kp, val)) for kp, val in keypoints_with_zero]

    # Sort features_IDs_current by key
    features_IDs_current = dict(sorted(features_IDs_current.items()))

    return features_IDs_current, total_matches, updated_keypoints