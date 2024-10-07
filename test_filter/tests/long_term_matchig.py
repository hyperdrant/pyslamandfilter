import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

# Fügen Sie den Pfad zu Ihrem Modul hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.feature_matching import match_features
from preprocessing.feature_extraction import extract_features

# Funktion zum Extrahieren der nicht gematchten Keypoints und Deskriptoren
def get_unmatched_keypoints_descriptors(keypoints, descriptors, matches):
    matched_idx = [match.queryIdx for match in matches]
    unmatched_keypoints = [kp for i, kp in enumerate(keypoints) if i not in matched_idx]
    unmatched_descriptors = np.array([descriptors[i] for i in range(len(descriptors)) if i not in matched_idx])
    return unmatched_keypoints, unmatched_descriptors

# Funktion zum Lesen aller Bildpfade aus einem Ordner
def get_image_paths_from_folder(folder_path):
    return sorted([os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(('.png', '.jpg', '.jpeg'))])

# Hauptfunktion
def main(image_folder, ransac_params_prev, ransac_params_prev2, ransac_params_prev3, ransac_params_prev4, ransac_params_prev5):
    img_paths = get_image_paths_from_folder(image_folder)
    
    # Iteriere über die Bilder
    for i in range(len(img_paths)):
        # Aktuelles Bild laden
        current_image = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
        keypoints_current, descriptors_current = extract_features(current_image)
        
        unmatched_keypoints = keypoints_current
        unmatched_descriptors = descriptors_current
        
        # Match mit den letzten 5 (oder weniger) Bildern
        for j in range(1, 6):
            if i - j >= 0:
                previous_image = cv2.imread(img_paths[i - j], cv2.IMREAD_GRAYSCALE)
                keypoints_previous, descriptors_previous = extract_features(previous_image)
                
                # Setzen der RANSAC-Parameter für den jeweiligen Vergleich
                if j == 1:
                    ransac_params = ransac_params_prev
                elif j == 2:
                    ransac_params = ransac_params_prev2
                elif j == 3:
                    ransac_params = ransac_params_prev3
                elif j == 4:
                    ransac_params = ransac_params_prev4
                elif j == 5:
                    ransac_params = ransac_params_prev5
                
                # Features matchen
                ransac_thresh, max_iters, confidence = ransac_params
                inlier_matches = match_features(keypoints_current, descriptors_current, keypoints_previous, descriptors_previous, 
                                                ransac_thresh, max_iters, confidence)
                
                print(f'Number of inlier matches between image {i} and image {i-j}: {len(inlier_matches)}')
                
                img_matches = cv2.drawMatches(current_image, keypoints_current, previous_image, keypoints_previous, inlier_matches, None)
                plt.figure(figsize=(20, 10))
                plt.imshow(img_matches)
                plt.title(f'Inlier Matches between Image {i} and Image {i-j}')
                plt.axis('off')
                plt.show()
                
                # Nicht gematchte Keypoints und Deskriptoren extrahieren
                unmatched_keypoints, unmatched_descriptors = get_unmatched_keypoints_descriptors(unmatched_keypoints, unmatched_descriptors, inlier_matches)
                
                # Falls keine nicht gematchten Deskriptoren mehr vorhanden sind, beende die Schleife
                if len(unmatched_descriptors) == 0:
                    break

if __name__ == "__main__":
    image_folder = "/Users/alexanderwitt/Desktop/Semesterarbeit/Code/Image_Preprocessing/Dataset/Output/erding_3_cropped"
    
    # Setzen Sie die RANSAC-Parameter für die letzten 5 (oder weniger) Bildvergleiche
    ransac_params_prev = (25, 2000, 0.99)
    ransac_params_prev2 = (25, 2000, 0.99)
    ransac_params_prev3 = (25, 2000, 0.99)
    ransac_params_prev4 = (25, 2000, 0.99)
    ransac_params_prev5 = (25, 2000, 0.99)
    
    main(image_folder, ransac_params_prev, ransac_params_prev2, ransac_params_prev3, ransac_params_prev4, ransac_params_prev5)