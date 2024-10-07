import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from python_orb_slam3 import ORBExtractor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.feature_matching import match_features
from preprocessing.feature_extraction import extract_features

def test_match_orb_features(img1_path, img2_path):
    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    
    
    # Find the keypoints and descriptors with ORB
    keypoints1, descriptors1 = extract_features(img1)
    keypoints2, descriptors2 = extract_features(img2)
    
    # Match features using the provided function
    inlier_matches = match_features(keypoints1, descriptors1, keypoints2, descriptors2)
    
    # Print the number of matches
    print(f'Number of inlier matches: {len(inlier_matches)}')
    
    # Draw inlier matches
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Display the matches
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title('Inlier Matches')
    plt.axis('off')
    plt.show()

# Example usage:
test_match_orb_features('/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/tets/image_0000.jpg', '/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/tets/image_0001.jpg')