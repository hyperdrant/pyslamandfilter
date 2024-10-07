### Imports ### 
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from python_orb_slam3 import ORBExtractor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src')) # System path

from preprocessing.feature_extraction import extract_features # Feature Extractor
from preprocessing.feature_matching import match_featuresNN 
from selection.feature_prefiltering import prefilter_pos
# Pfad zum Ordner, der die Bilder enthält
folder_path = '/home/q633974/Desktop/Semester_thesis/Feature_Filter_v3/data/input/KITTI/Residential/2011_09_26_drive_0039_sync/2011_09_26/2011_09_26_drive_0039_sync/image_02/test'

# Liste der Bildpfade im Ordner
img_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
img_paths.sort()

# Liste zum Speichern der geladenen Bilder
images = []

# Schleife zum Laden der Bilder
for img_path in img_paths:
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Lade das Bild in Graustufen
    images.append(image)  # Füge das geladene Bild der Liste hinzu

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
def assign_ID_unmatched(unmatched_features, mapping, start_id, features_with_ids, name):
    """
    Assigns global feature IDs to unmatched keypoints in an image and stores them in a dictionary.

    Parameters:
    - unmatched_features: List of unmatched keypoints (KeyPoint objects) that need a new ID.
    - mapping: Dictionary that maps KeyPoint objects to their original index positions in the image.
    - start_id: Starting value for the global feature ID (not directly used here, as global_feature_id is global).
    - features_with_ids: Dictionary that stores the global feature ID for each KeyPoint (key: KeyPoint ID, value: global feature ID).
    - name: Name of the current image or processing unit, used for labeling the output.

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

    # Print the length of the features_with_ids dictionary after assignment
    print(f"Length of {name}: {len(features_with_ids)} after assigning matches")

    return features_with_ids


# Function to assign an ID to a matched feature
def assign_ID_matched(inlier_matches, features_with_ids_prev, features_with_ids_current, mapping, matched_keypoints, name):
    """
    Assigns global feature IDs to matched keypoints in the current image based on their matches in a previous image.

    Parameters:
    - inlier_matches: List of inlier matches between two images, which provides the query and train keypoint indices.
    - features_with_ids_prev: Dictionary containing the global feature IDs from the previous image (keys: Keypoint IDs, values: global feature IDs).
    - features_with_ids_current: Dictionary to store the global feature IDs for the current image (keys: Keypoint IDs, values: global feature IDs).
    - mapping: Dictionary mapping KeyPoint objects to their original index positions in the current image.
    - matched_keypoints: List of KeyPoint objects in the current image that were matched with keypoints in the previous image.
    - name: Name of the current processing step or image, used for labeling the output.

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
    
    # Print the length of the features_with_ids_current dictionary after assignment
    print(f"Length of {name}: {len(features_with_ids_current)} after assigning matches")
    
    return features_with_ids_current



# Feature matching
def match_features(prev_keypoints, prev_descriptors, keypoints, descriptors, ransacReprojThreshold, maxIters, confidence, prev_keypoints_name='Keypoints 1', keypoints_name='Keypoints 2'):
    """
    Matcht die Features zwischen zwei Sätzen von Keypoints und Deskriptoren und gibt die Inlier-Matches zurück.

    Parameters:
    - prev_keypoints: Liste der Keypoints aus dem vorherigen Bild (Query).
    - prev_descriptors: Deskriptoren aus dem vorherigen Bild (Query).
    - keypoints: Liste der Keypoints aus dem aktuellen Bild (Train).
    - descriptors: Deskriptoren aus dem aktuellen Bild (Train).
    - ransacReprojThreshold: Threshold für RANSAC.
    - maxIters: Maximale Anzahl von Iterationen für RANSAC.
    - confidence: Konfidenz für die Homographie-Schätzung.
    - prev_keypoints_name: Name der Keypoints aus dem vorherigen Bild (wird in der Ausgabe verwendet).
    - keypoints_name: Name der Keypoints aus dem aktuellen Bild (wird in der Ausgabe verwendet).

    Returns:
    - inlier_matches: Liste der Inlier-Matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # BFMatcher with Hamming distance and cross-check

    matches = bf.match(prev_descriptors, descriptors)  # Match descriptors
    
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance

    # Initialization of arrays to store keypoints coordinates
    points_prev = np.zeros((len(matches), 2), dtype=np.float32)
    points_current = np.zeros((len(matches), 2), dtype=np.float32)

    # Extract coordinates
    for i, match in enumerate(matches):
        points_prev[i, :] = prev_keypoints[match.queryIdx].pt
        points_current[i, :] = keypoints[match.trainIdx].pt

    # Find homography using MAGSAC++
    homography, mask = cv2.findHomography(points_prev, points_current, cv2.USAC_MAGSAC, ransacReprojThreshold, maxIters=maxIters, confidence=confidence)

    # Select inlier matches
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    
    # Print number of matches between the given keypoints
    print(f"Number of matches between {prev_keypoints_name} and {keypoints_name}: {len(inlier_matches)}")
    
    return inlier_matches


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
    
    # Print the number of unmatched keypoints for debugging purposes
    print("Number of unmatched keypoints: ", len(unmatched_descriptors))
    
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





### Image 0 ### 
print("### IMAGE 0 ###")
features_IDs0 = {} # Erstelle ein Dictionary zum Speichern der Feature-IDs

# Extrahiere Keypoints und Deskriptoren für das nullte Bild
keypoints0, descriptors0 = extract_features(images[0])
keypoints0, descriptors0 = prefilter_pos(keypoints0, descriptors0)
# Erstelle ein Mapping der Keypoints für Bild 0
mapping0 = create_mapping(keypoints0)

# Weise den Keypoints in Bild 0 eindeutige Feature-IDs zu
# Da Bild 0 das erste Bild ist, werden alle Keypoints als ungematcht betrachtet und erhalten neue IDs
features_IDs0 = assign_ID_unmatched(keypoints0, mapping0, global_feature_id, features_IDs0, name="features_with_ids0")


### Image 1 ###
print("### IMAGE 1 ###")
features_IDs1 = {} # Erstelle ein Dictionary zum Speichern der Feature-IDs

# Extrahiere Keypoints und Deskriptoren für das erste Bild
keypoints1, descriptors1 = extract_features(images[1])
keypoints1, descriptors1 = prefilter_pos(keypoints1, descriptors1)
# Erstelle ein Mapping der Keypoints für Bild 1
mapping1 = create_mapping(keypoints1)

# Matche die Keypoints von Bild 1 mit den Keypoints von Bild 0
inlier_matches_0_1 = match_features(keypoints0, descriptors0, keypoints1, descriptors1, 25, 2000, 0.99, prev_keypoints_name='Keypoints 0', keypoints_name='Keypoints 1')

# Finde die Keypoints in Bild 1, die mit Bild 0 gematched wurden
matched_keypoints1 = find_matched_keypoints(keypoints1, inlier_matches_0_1)

# Weisen Sie IDs den gematchten Keypoints in Bild 1 zu, basierend auf den IDs in Bild 0
features_IDs1 = assign_ID_matched(inlier_matches_0_1, features_IDs0, features_IDs1, mapping1, matched_keypoints1, name="features_with_ids1")

# Finde die Keypoints in Bild 1, die nicht mit Bild 0 gematched wurden
unmatched_keypoints1, unmatched_descriptors1 = find_unmatched_keypoints(keypoints1, descriptors1, inlier_matches_0_1)

# Weisen Sie IDs den verbleibenden ungematchten Keypoints in Bild 1 zu
features_IDs1 = assign_ID_unmatched(unmatched_keypoints1, mapping1, global_feature_id, features_IDs1, name="features_with_ids1")



### Image 2 ###
print("### IMAGE 2 ###")
features_IDs2 = {} # Erstelle ein Dictionary zum Speichern der Feature-IDs

# Extrahiere Keypoints und Deskriptoren für Bild 2
keypoints2, descriptors2 = extract_features(images[2])
keypoints2, descriptors2 = prefilter_pos(keypoints2, descriptors2)
# Erstelle ein Mapping der Keypoints für Bild 2
mapping2 = create_mapping(keypoints2)

# Matche die Keypoints von Bild 2 mit den Keypoints von Bild 1
inlier_matches_1_2 = match_features(keypoints1, descriptors1, keypoints2, descriptors2, 25, 2000, 0.99, prev_keypoints_name='Keypoints 1', keypoints_name='Keypoints 2')

# Finde die Keypoints in Bild 2, die mit Bild 1 gematched wurden
matched_keypoints1_2 = find_matched_keypoints(keypoints2, inlier_matches_1_2)

# Weisen Sie IDs den gematchten Keypoints in Bild 2 zu, basierend auf den IDs in Bild 1
features_IDs2 = assign_ID_matched(inlier_matches_1_2, features_IDs1, features_IDs2, mapping2, matched_keypoints1_2, name="features_with_ids2")

# Finde die Keypoints in Bild 2, die nicht mit Bild 1 gematched wurden
unmatched_keypoints2_1, unmatched_descriptors2_1 = find_unmatched_keypoints(keypoints2, descriptors2, inlier_matches_1_2)

# Matche die ungematchten Keypoints von Bild 2 mit den Keypoints von Bild 0
inlier_matches_0_2 = match_features(keypoints0, descriptors0, unmatched_keypoints2_1, unmatched_descriptors2_1, 25, 2000, 0.99, prev_keypoints_name='Keypoints 0', keypoints_name='Keypoints 2')
inlier_matches_0_2_c = remap_inlier_matches(inlier_matches_0_2, mapping2, unmatched_keypoints2_1)

# Finde die Keypoints in den ungematchten Keypoints von Bild 2, die mit Bild 0 gematched wurden
matched_keypoints0_2 = find_matched_keypoints(unmatched_keypoints2_1, inlier_matches_0_2)

# Weisen Sie IDs den gematchten Keypoints in Bild 2 zu, basierend auf den IDs in Bild 0
features_IDs2 = assign_ID_matched(inlier_matches_0_2_c, features_IDs0, features_IDs2, mapping2, matched_keypoints0_2, name="features_with_ids2")

# Finde die verbleibenden ungematchten Keypoints in Bild 2
unmatched_keypoints2_2, unmatched_descriptors2_2 = find_unmatched_keypoints(unmatched_keypoints2_1, unmatched_descriptors2_1, inlier_matches_0_2)

# Weisen Sie IDs den verbleibenden ungematchten Keypoints in Bild 2 zu
features_IDs2 = assign_ID_unmatched(unmatched_keypoints2_2, mapping2, global_feature_id, features_IDs2, name="features_with_ids2")




### Image 3 ###
print("### IMAGE 3 ###")
features_IDs3 = {} # Erstelle ein Dictionary zum Speichern der Feature-IDs

# Extrahiere Keypoints und Deskriptoren für Bild 3
keypoints3, descriptors3 = extract_features(images[3])
keypoints3, descriptors3 = prefilter_pos(keypoints3, descriptors3)
# Erstelle ein Mapping der Keypoints für Bild 3
mapping3 = create_mapping(keypoints3)

# Matche die Keypoints von Bild 3 mit den Keypoints von Bild 2
inlier_matches_2_3 = match_features(keypoints2, descriptors2, keypoints3, descriptors3, 25, 2000, 0.99, prev_keypoints_name='Keypoints 2', keypoints_name='Keypoints 3')

# Finde die Keypoints in Bild 3, die mit Bild 2 gematched wurden
matched_keypoints2_3 = find_matched_keypoints(keypoints3, inlier_matches_2_3)

# Weisen Sie IDs den gematchten Keypoints in Bild 3 zu, basierend auf den IDs in Bild 2
features_IDs3 = assign_ID_matched(inlier_matches_2_3, features_IDs2, features_IDs3, mapping3, matched_keypoints2_3, name="features_with_ids3")

# Finde die Keypoints in Bild 3, die nicht mit Bild 2 gematched wurden
unmatched_keypoints3_1, unmatched_descriptors3_1 = find_unmatched_keypoints(keypoints3, descriptors3, inlier_matches_2_3)

# Matche die ungematchten Keypoints von Bild 3 mit den Keypoints von Bild 1
inlier_matches_1_3 = match_features(keypoints1, descriptors1, unmatched_keypoints3_1, unmatched_descriptors3_1, 25, 2000, 0.99, prev_keypoints_name='Keypoints 1', keypoints_name='Keypoints 3')
inlier_matches_1_3_c = remap_inlier_matches(inlier_matches_1_3, mapping3, unmatched_keypoints3_1)

# Finde die Keypoints in den ungematchten Keypoints von Bild 3, die mit Bild 1 gematched wurden
matched_keypoints1_3 = find_matched_keypoints(unmatched_keypoints3_1, inlier_matches_1_3)

# Weisen Sie IDs den gematchten Keypoints in Bild 3 zu, basierend auf den IDs in Bild 1
features_IDs3 = assign_ID_matched(inlier_matches_1_3_c, features_IDs1, features_IDs3, mapping3, matched_keypoints1_3, name="features_with_ids3")

# Finde die verbleibenden ungematchten Keypoints in Bild 3
unmatched_keypoints3_2, unmatched_descriptors3_2 = find_unmatched_keypoints(unmatched_keypoints3_1, unmatched_descriptors3_1, inlier_matches_1_3)

# Matche die verbleibenden ungematchten Keypoints von Bild 3 mit den Keypoints von Bild 0
inlier_matches_0_3 = match_features(keypoints0, descriptors0, unmatched_keypoints3_2, unmatched_descriptors3_2, 25, 2000, 0.99, prev_keypoints_name='Keypoints 0', keypoints_name='Keypoints 3')
inlier_matches_0_3_c = remap_inlier_matches(inlier_matches_0_3, mapping3, unmatched_keypoints3_2)

# Finde die Keypoints in den ungematchten Keypoints von Bild 3, die mit Bild 0 gematched wurden
matched_keypoints0_3 = find_matched_keypoints(unmatched_keypoints3_2, inlier_matches_0_3)

# Weisen Sie IDs den gematchten Keypoints in Bild 3 zu, basierend auf den IDs in Bild 0
features_IDs3 = assign_ID_matched(inlier_matches_0_3_c, features_IDs0, features_IDs3, mapping3, matched_keypoints0_3, name="features_with_ids3")

# Finde die verbleibenden ungematchten Keypoints in Bild 3
unmatched_keypoints3_3, unmatched_descriptors3_3 = find_unmatched_keypoints(unmatched_keypoints3_2, unmatched_descriptors3_2, inlier_matches_0_3)

# Weisen Sie IDs den verbleibenden ungematchten Keypoints in Bild 3 zu
features_IDs3 = assign_ID_unmatched(unmatched_keypoints3_3, mapping3, global_feature_id, features_IDs3, name="features_with_ids3")





### Image 4 ###
print("### IMAGE 4 ###")
features_IDs4 = {} # Create feature storage

# Extrahiere Keypoints und Deskriptoren für Bild 4
keypoints4, descriptors4 = extract_features(images[4])
keypoints4, descriptors4 = prefilter_pos(keypoints4, descriptors4)
# Erstelle ein Mapping der Keypoints
mapping4 = create_mapping(keypoints4)

# Matching mit Bild 3
inlier_matches_3_4 = match_features(keypoints3, descriptors3, keypoints4, descriptors4, 25, 5000, 0.99, prev_keypoints_name='Keypoints 3', keypoints_name='Keypoints 4') # matching

# Finde die gematchten Keypoints
matched_keypoints3_4 = find_matched_keypoints(keypoints4, inlier_matches_3_4)

# Weisen Sie IDs den gematchten Keypoints zu
features_IDs4 = assign_ID_matched(inlier_matches_3_4, features_IDs3, features_IDs4, mapping4, matched_keypoints3_4, name="features_with_ids4")

# Finde die ungematchten Keypoints (zwischen Bild 3 und Bild 4)
unmatched_keypoints4_1, unmatched_descriptors4_1 = find_unmatched_keypoints(keypoints4, descriptors4, inlier_matches_3_4)

# Matching der ungematchten Keypoints mit Bild 2
inlier_matches_2_4 = match_features(keypoints2, descriptors2, unmatched_keypoints4_1, unmatched_descriptors4_1, 25, 5000, 0.99, prev_keypoints_name='Keypoints 2', keypoints_name='Keypoints 4')
inlier_matches_2_4_c = remap_inlier_matches(inlier_matches_2_4, mapping4, unmatched_keypoints4_1)

# Finde die gematchten Keypoints (zwischen Bild 2 und den ungematchten Keypoints von Bild 4)
matched_keypoints2_4 = find_matched_keypoints(unmatched_keypoints4_1, inlier_matches_2_4)

# Weisen Sie IDs den gematchten Keypoints zu
features_IDs4 = assign_ID_matched(inlier_matches_2_4_c, features_IDs2, features_IDs4, mapping4, matched_keypoints2_4, name="features_with_ids4")

# Finde die ungematchten Keypoints (zwischen Bild 2 und den ungematchten Keypoints von Bild 4)
unmatched_keypoints4_2, unmatched_descriptors4_2 = find_unmatched_keypoints(unmatched_keypoints4_1, unmatched_descriptors4_1, inlier_matches_2_4)

# Matching der ungematchten Keypoints mit Bild 1
inlier_matches_1_4 = match_features(keypoints1, descriptors1, unmatched_keypoints4_2, unmatched_descriptors4_2, 8, 5000, 0.99, prev_keypoints_name='Keypoints 1', keypoints_name='Keypoints 4')
inlier_matches_1_4_c = remap_inlier_matches(inlier_matches_1_4, mapping4, unmatched_keypoints4_2)

# Finde die gematchten Keypoints (zwischen Bild 1 und den ungematchten Keypoints von Bild 4)
matched_keypoints1_4 = find_matched_keypoints(unmatched_keypoints4_2, inlier_matches_1_4)

# Weisen Sie IDs den gematchten Keypoints zu
features_IDs4 = assign_ID_matched(inlier_matches_1_4_c, features_IDs1, features_IDs4, mapping4, matched_keypoints1_4, name="features_with_ids4")

# Finde die ungematchten Keypoints (zwischen Bild 1 und den ungematchten Keypoints von Bild 4)
unmatched_keypoints4_3, unmatched_descriptors4_3 = find_unmatched_keypoints(unmatched_keypoints4_2, unmatched_descriptors4_2, inlier_matches_1_4)

# Matching der ungematchten Keypoints mit Bild 0
inlier_matches_0_4 = match_features(keypoints0, descriptors0, unmatched_keypoints4_3, unmatched_descriptors4_3, 8, 5000, 0.99, prev_keypoints_name='Keypoints 0', keypoints_name='Keypoints 4')
inlier_matches_0_4_c = remap_inlier_matches(inlier_matches_0_4, mapping4, unmatched_keypoints4_3)

# Finde die gematchten Keypoints (zwischen Bild 0 und den ungematchten Keypoints von Bild 4)
matched_keypoints0_4 = find_matched_keypoints(unmatched_keypoints4_3, inlier_matches_0_4)

# Weisen Sie IDs den gematchten Keypoints zu
features_IDs4 = assign_ID_matched(inlier_matches_0_4_c, features_IDs0, features_IDs4, mapping4, matched_keypoints0_4, name="features_with_ids4")

# Finde die ungematchten Keypoints (zwischen Bild 0 und den ungematchten Keypoints von Bild 4)
unmatched_keypoints4_4, unmatched_descriptors4_4 = find_unmatched_keypoints(unmatched_keypoints4_3, unmatched_descriptors4_3, inlier_matches_0_4)

# Weisen Sie IDs den verbleibenden ungematchten Keypoints zu
features_IDs4 = assign_ID_unmatched(unmatched_keypoints4_4, mapping4, global_feature_id, features_IDs4, name="features_with_ids4")


























def visualize_matches(image0, image2, keypoints0, keypoints2, inlier_matches, img0_idx, img2_idx):
    """
    Visualisiert die Inlier-Matches zwischen zwei Bildern und passt den Titel dynamisch an.

    Parameters:
    - image0: Erstes Bild (z.B. Bild 0).
    - image2: Zweites Bild (z.B. Bild 2).
    - keypoints0: Keypoints des ersten Bildes.
    - keypoints2: Keypoints des zweiten Bildes.
    - inlier_matches: Liste der Inlier-Matches zwischen den beiden Bildern.
    - img0_idx: Index des ersten Bildes.
    - img2_idx: Index des zweiten Bildes.
    """

    # Zeichne die Matches zwischen den beiden Bildern
    img_matches = cv2.drawMatches(image0, keypoints0, image2, keypoints2, inlier_matches, None)

    # Zeige das Bild mit den Matches an
    plt.figure(figsize=(15, 10))
    plt.imshow(img_matches)
    plt.title(f'Inlier Matches between Image {img0_idx} and Image {img2_idx} ({len(inlier_matches)} matches)')
    plt.axis('off')
    plt.show()

# Beispiel-Aufruf der Funktion
#visualize_matches(images[0], images[4], keypoints0, keypoints4, inlier_matches_0_4_c, img0_idx=0, img2_idx=4)






def plot_common_features(image1, image2, keypoints1, keypoints2, features_IDs0, features_with_ids1, inlier_matches, title):
    # Zähler für die Anzahl der gezeichneten Punkte
    count_image1 = 0
    count_image2 = 0
    
    plt.figure(figsize=(15, 10))
    
    # Plot Bild 0
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    for match in inlier_matches:
        key0 = match.queryIdx
        key1 = match.trainIdx
        feature_id0 = features_IDs0.get(key0)
        feature_id1 = features_with_ids1.get(key1)
        
        if feature_id0 == feature_id1:
            # Zeichne den Keypoint in Bild 0
            x0, y0 = keypoints1[key0].pt
            plt.scatter(x0, y0, c='r', s=40, marker='x')
            plt.text(x0 + 5, y0 + 5, str(feature_id0), color='yellow', fontsize=12)  # Zeigt die Feature-ID an
            count_image1 += 1  # Erhöhe den Zähler für Bild 0
    plt.title(f"Common Features in Image 0")
    plt.axis('off')
    
    # Plot Bild 1
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    for match in inlier_matches:
        key0 = match.queryIdx
        key1 = match.trainIdx
        feature_id0 = features_IDs0.get(key0)
        feature_id1 = features_with_ids1.get(key1)
        
        if feature_id0 == feature_id1:
            # Zeichne den Keypoint in Bild 1
            x1, y1 = keypoints2[key1].pt
            plt.scatter(x1, y1, c='r', s=40, marker='x')
            plt.text(x1 + 5, y1 + 5, str(feature_id1), color='yellow', fontsize=12)  # Zeigt die Feature-ID an
            count_image2 += 1  # Erhöhe den Zähler für Bild 1
    plt.title(f"Common Features in Image 1")
    plt.axis('off')
    
# Beispiel-Aufruf der Funktion
#plot_common_features(images[0], images[4], keypoints0, keypoints4, features_IDs0, features_IDs4, inlier_matches_0_4_c, title="Common Features between Image 0 and Image 2")



def print_inlier_match_ids(inlier_matches):

    for match in inlier_matches:
        query_id = match.queryIdx  # ID des Keypoints im Query-Bild (Bild 0)
        train_id = match.trainIdx  # ID des Keypoints im Train-Bild (Bild 1)
        print(f"Query ID: {query_id}, Train ID: {train_id}")

# Beispiel-Aufruf der Funktion
#print_inlier_match_ids(inlier_matches_0_2_c)


def print_features_with_ids(features_with_ids):
    """
    Gibt alle Keys und zugehörigen Einträge in einem Dictionary aus.

    Parameters:
    - features_with_ids: Dictionary, das Keypoint-IDs als Schlüssel und die Feature-IDs als Werte enthält.
    """
    for key, value in features_with_ids.items():
        print(f"Keypoint ID: {key}, Feature ID: {value}")

# Beispiel: Alle Keys und Entries in features_IDs1 anzeigen
#print_features_with_ids(features_IDs2)


def verify_matches(inlier_matches, features_IDs0, features_IDs1):

    for match in inlier_matches:
        query_id = match.queryIdx  # ID des Keypoints im Query-Bild (Bild 0)
        train_id = match.trainIdx  # ID des Keypoints im Train-Bild (Bild 1)

        # Hole die Feature-IDs aus den Dictionaries
        feature_id0 = features_IDs0.get(query_id, None)
        feature_id1 = features_IDs1.get(train_id, None)

        # Überprüfe, ob die Feature-IDs übereinstimmen
        if feature_id0 == feature_id1:
            print(f"Match (Query ID: {query_id}, Train ID: {train_id}): True")
        else:
            print(f"Match (Query ID: {query_id}, Train ID: {train_id}): False")

# Beispiel-Aufruf der Funktion
#verify_matches(inlier_matches_3_4, features_IDs2, features_IDs4)



""" 
def get_keypoint_ids(keypoints):

    keypoint_ids = list(range(len(keypoints)))
    return keypoint_ids

# Beispiel-Aufruf der Funktion
keypoint_ids = get_keypoint_ids(matched_keypoints0_2)

# Ausgabe aller Keypoint-IDs
for keypoint_id in keypoint_ids:
    print(f"Keypoint ID: {keypoint_id}") """




def plot_keypoints_from_ids(image1, image2, keypoints1, keypoints2, keypoint_id0, keypoint_id1):

    plt.figure(figsize=(15, 10))
    
    # Plot für Bild 0
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    x0, y0 = keypoints1[keypoint_id0].pt
    plt.scatter(x0, y0, c='r', s=100, marker='o')
    plt.text(x0 + 5, y0 + 5, f"ID: {keypoint_id0}", color='yellow', fontsize=12)  # Zeigt die Keypoint-ID an
    plt.title(f"Keypoint {keypoint_id0} in Image 0")
    plt.axis('off')
    
    # Plot für Bild 1
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    x1, y1 = keypoints2[keypoint_id1].pt
    plt.scatter(x1, y1, c='r', s=100, marker='o')
    plt.text(x1 + 5, y1 + 5, f"ID: {keypoint_id1}", color='yellow', fontsize=12)  # Zeigt die Keypoint-ID an
    plt.title(f"Keypoint {keypoint_id1} in Image 1")
    plt.axis('off')
    
    plt.show()

# Beispiel: Vorgabe der Keypoint-IDs aus den Dictionaries
keypoint_id_in_image0 = 73  # Beispiel Keypoint-ID aus features_with_ids0
keypoint_id_in_image1 = 48  # Beispiel Keypoint-ID aus features_with_ids1

# Aufruf der Funktion
#plot_keypoints_from_ids(images[0], images[2], keypoints0, keypoints2, keypoint_id_in_image0, keypoint_id_in_image1)




""" # Visualisierungen für Bild 1
visualize_matches(images[0], images[1], keypoints0, keypoints1, inlier_matches_0_1, img0_idx=0, img2_idx=1)
plot_common_features(images[0], images[1], keypoints0, keypoints1, features_IDs0, features_IDs1, inlier_matches_0_1, title="Common Features between Image 0 and Image 1")

# Visualisierungen für Bild 2
visualize_matches(images[1], images[2], keypoints1, keypoints2, inlier_matches_1_2, img0_idx=1, img2_idx=2)
plot_common_features(images[1], images[2], keypoints1, keypoints2, features_IDs1, features_IDs2, inlier_matches_1_2, title="Common Features between Image 1 and Image 2")

visualize_matches(images[0], images[2], keypoints0, keypoints2, inlier_matches_0_2_c, img0_idx=0, img2_idx=2)
plot_common_features(images[0], images[2], keypoints0, keypoints2, features_IDs0, features_IDs2, inlier_matches_0_2_c, title="Common Features between Image 0 and Image 2")

# Visualisierungen für Bild 3
visualize_matches(images[2], images[3], keypoints2, keypoints3, inlier_matches_2_3, img0_idx=2, img2_idx=3)
plot_common_features(images[2], images[3], keypoints2, keypoints3, features_IDs2, features_IDs3, inlier_matches_2_3, title="Common Features between Image 2 and Image 3")

visualize_matches(images[1], images[3], keypoints1, keypoints3, inlier_matches_1_3_c, img0_idx=1, img2_idx=3)
plot_common_features(images[1], images[3], keypoints1, keypoints3, features_IDs1, features_IDs3, inlier_matches_1_3_c, title="Common Features between Image 1 and Image 3")

visualize_matches(images[0], images[3], keypoints0, keypoints3, inlier_matches_0_3_c, img0_idx=0, img2_idx=3)
plot_common_features(images[0], images[3], keypoints0, keypoints3, features_IDs0, features_IDs3, inlier_matches_0_3_c, title="Common Features between Image 0 and Image 3") """

# Visualisierungen für Bild 4
visualize_matches(images[3], images[4], keypoints3, keypoints4, inlier_matches_3_4, img0_idx=3, img2_idx=4)
plot_common_features(images[3], images[4], keypoints3, keypoints4, features_IDs3, features_IDs4, inlier_matches_3_4, title="Common Features between Image 3 and Image 4")

visualize_matches(images[2], images[4], keypoints2, keypoints4, inlier_matches_2_4_c, img0_idx=2, img2_idx=4)
plot_common_features(images[2], images[4], keypoints2, keypoints4, features_IDs2, features_IDs4, inlier_matches_2_4_c, title="Common Features between Image 2 and Image 4")

visualize_matches(images[1], images[4], keypoints1, keypoints4, inlier_matches_1_4_c, img0_idx=1, img2_idx=4)
plot_common_features(images[1], images[4], keypoints1, keypoints4, features_IDs1, features_IDs4, inlier_matches_1_4_c, title="Common Features between Image 1 and Image 4")
