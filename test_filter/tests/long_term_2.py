import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import deque
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.feature_matching import match_features
from preprocessing.feature_extraction import extract_features
def get_image_paths(folder):
    image_paths = [os.path.join(folder, fname) for fname in sorted(os.listdir(folder)) if fname.endswith('.jpg')]
    return image_paths

def get_unmatched_keypoints_descriptors(keypoints, descriptors, matches):
    matched_idx = [match.queryIdx for match in matches]
    unmatched_keypoints = [kp for i, kp in enumerate(keypoints) if i not in matched_idx]
    unmatched_descriptors = np.array([descriptors[i] for i in range(len(descriptors)) if i not in matched_idx])
    return unmatched_keypoints, unmatched_descriptors

def process_images(folder, display_output=0, print_output=0):
    img_paths = get_image_paths(folder)
    
    # ORB-Detektor initialisieren
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Puffer zum Speichern der Keypoints und Deskriptoren der letzten fünf Bilder
    keypoints_buffer = deque(maxlen=5)
    descriptors_buffer = deque(maxlen=5)
    
    start_time = time.time()  # Startzeit messen
    
    for idx in range(len(img_paths)):
        current_image_path = img_paths[idx]
        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        keypoints_current, descriptors_current = extract_features(current_image)
        
        if idx >= 5:
            if print_output:
                print(f'Processing image {idx}: {os.path.basename(current_image_path)}')
            
            remaining_keypoints = keypoints_current
            remaining_descriptors = descriptors_current
            
            for offset in range(1, 6):
                previous_keypoints = keypoints_buffer[-offset]
                previous_descriptors = descriptors_buffer[-offset]
                
                inlier_matches = match_features(remaining_keypoints, remaining_descriptors, previous_keypoints, previous_descriptors)
                
                # Berechne verbleibende Keypoints und Deskriptoren basierend auf dem aktuellen Match
                new_remaining_keypoints, new_remaining_descriptors = get_unmatched_keypoints_descriptors(remaining_keypoints, remaining_descriptors, inlier_matches)
                
                if print_output:
                    print(f'Matching with image {idx - offset}: {os.path.basename(img_paths[idx - offset])}')
                    print(f'Number of inlier matches: {len(inlier_matches)}')
                    print(f'Number of unmatched features: {len(new_remaining_descriptors)}')
                
                if display_output:
                    img_matches = cv2.drawMatches(current_image, remaining_keypoints, cv2.imread(img_paths[idx - offset], cv2.IMREAD_GRAYSCALE), previous_keypoints, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    plt.figure(figsize=(20, 10))
                    plt.imshow(img_matches)
                    plt.title(f'Inlier Matches between Image {idx} and Image {idx - offset}')
                    plt.axis('off')
                    plt.show()
                
                # Aktualisiere die verbleibenden Keypoints und Deskriptoren
                remaining_keypoints, remaining_descriptors = new_remaining_keypoints, new_remaining_descriptors
                
                # Beende die Schleife, wenn keine nicht gematchten Deskriptoren mehr übrig sind
                if len(remaining_descriptors) == 0:
                    break
        
        # Füge die aktuellen Keypoints und Deskriptoren in den Puffer ein
        keypoints_buffer.append(keypoints_current)
        descriptors_buffer.append(descriptors_current)
    
    end_time = time.time()  # Endzeit messen
    processing_time = end_time - start_time
    return processing_time

# Beispielpfad zum Ordner mit Bildern einer Fahrtsequenz
image_folder_path = '/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/tets'

# Variablen zur Steuerung der Ausgabe
display_output = 1  # 0 für keine grafische Ausgabe, 1 für grafische Ausgabe
print_output = 1    # 0 für keine Konsolenausgabe, 1 für Konsolenausgabe

# Bilder verarbeiten und Zeit messen
processing_time = process_images(image_folder_path, display_output, print_output)
print(f'Total processing time: {round(processing_time,3)} seconds')