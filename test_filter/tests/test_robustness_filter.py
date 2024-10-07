import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.feature_extraction import extract_features
from preprocessing.feature_matching import match_features
from attributes.robustness_attribute import adjust_brightness_contrast  # Assuming these functions are in this module
from preprocessing.image_classification import classify_contrast
from preprocessing.image_preprocessing import preprocess_image

def visualize_images(images, titles, figsize=(25, 15)):
    plt.figure(figsize=figsize)
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, (len(images) + 1) // 2, i + 1)  # Adjusted to handle odd number of images
        if len(image.shape) == 2:  # grayscale image
            plt.imshow(image, cmap='gray')
        else:  # color image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.show()

def draw_keypoints(image, keypoints):
    return cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

def save_image_with_keypoints(image, keypoints, file_path):
    image_with_keypoints = draw_keypoints(image, keypoints)
    cv2.imwrite(file_path, image_with_keypoints)

def test_feature_matching_with_robustness():
    original_img_path = '/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/weather_1_rgb_cutted.png'
    original_img = cv2.imread(original_img_path)

    image_class = 2
    print(image_class)
    preprocessed_img = preprocess_image(original_img_path, image_class)  # Keep the preprocessed image in color

    # Transform images
    bright_img = adjust_brightness_contrast(preprocessed_img, alpha=1.4, beta=45)
    dark_img = adjust_brightness_contrast(preprocessed_img, alpha=0.6, beta=-25)
    high_contrast_img = adjust_brightness_contrast(preprocessed_img, alpha=2.0, beta=0)
    low_contrast_img = adjust_brightness_contrast(preprocessed_img, alpha=0.35, beta=0)

    # Extract features
    keypoints, descriptors = extract_features(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY))
    bright_kp, bright_desc = extract_features(cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY))
    dark_kp, dark_desc = extract_features(cv2.cvtColor(dark_img, cv2.COLOR_BGR2GRAY))
    high_contrast_kp, high_contrast_desc = extract_features(cv2.cvtColor(high_contrast_img, cv2.COLOR_BGR2GRAY))
    low_contrast_kp, low_contrast_desc = extract_features(cv2.cvtColor(low_contrast_img, cv2.COLOR_BGR2GRAY))

    # Match features
    inlier_matches_bright = match_features(keypoints, descriptors, bright_kp, bright_desc)
    inlier_matches_dark = match_features(keypoints, descriptors, dark_kp, dark_desc)
    inlier_matches_high_contrast = match_features(keypoints, descriptors, high_contrast_kp, high_contrast_desc)
    inlier_matches_low_contrast = match_features(keypoints, descriptors, low_contrast_kp, low_contrast_desc)

    print(f"Number of inlier matches with bright image: {len(inlier_matches_bright)}")
    print(f"Number of inlier matches with dark image: {len(inlier_matches_dark)}")
    print(f"Number of inlier matches with high contrast image: {len(inlier_matches_high_contrast)}")
    print(f"Number of inlier matches with low contrast image: {len(inlier_matches_low_contrast)}")

    # Visualize images
    images = [original_img, preprocessed_img, bright_img, dark_img, high_contrast_img, low_contrast_img]
    titles = ['Original Image', 'Preprocessed Image', 'Bright Image', 'Dark Image', 'High Contrast Image', 'Low Contrast Image']
    visualize_images(images, titles)

    # Draw keypoints on RGB images and save them
    save_image_with_keypoints(bright_img, bright_kp, os.path.join(os.path.dirname(original_img_path), 'bright_img_with_keypoints.png'))
    save_image_with_keypoints(dark_img, dark_kp, os.path.join(os.path.dirname(original_img_path), 'dark_img_with_keypoints.png'))
    save_image_with_keypoints(high_contrast_img, high_contrast_kp, os.path.join(os.path.dirname(original_img_path), 'high_contrast_img_with_keypoints.png'))
    save_image_with_keypoints(low_contrast_img, low_contrast_kp, os.path.join(os.path.dirname(original_img_path), 'low_contrast_img_with_keypoints.png'))

    # Visualize matches
    img_matches_bright = cv2.drawMatches(preprocessed_img, keypoints, bright_img, bright_kp, inlier_matches_bright, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_dark = cv2.drawMatches(preprocessed_img, keypoints, dark_img, dark_kp, inlier_matches_dark, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_high_contrast = cv2.drawMatches(preprocessed_img, keypoints, high_contrast_img, high_contrast_kp, inlier_matches_high_contrast, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_low_contrast = cv2.drawMatches(preprocessed_img, keypoints, low_contrast_img, low_contrast_kp, inlier_matches_low_contrast, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    match_images = [img_matches_bright, img_matches_dark, img_matches_high_contrast, img_matches_low_contrast]
    match_titles = ['Matches with Bright Image', 'Matches with Dark Image', 'Matches with High Contrast Image', 'Matches with Low Contrast Image']
    visualize_images(match_images, match_titles, figsize=(25, 15))

    # Calculate robustness scores
    robustness_scores = [0] * len(keypoints)
    for match in inlier_matches_bright:
        robustness_scores[match.queryIdx] += 1
    for match in inlier_matches_dark:
        robustness_scores[match.queryIdx] += 1
    for match in inlier_matches_high_contrast:
        robustness_scores[match.queryIdx] += 1
    for match in inlier_matches_low_contrast:
        robustness_scores[match.queryIdx] += 1

    # Draw keypoints with robustness
    preprocessed_with_keypoints = preprocessed_img.copy()
    colors = {
        0: (0, 0, 255),    # red
        1: (0, 165, 255),  # orange
        2: (255, 0, 255),  # magenta
        3: (255, 0, 0),    # blue
        4: (0, 255, 0)     # green
    }

    for i, kp in enumerate(keypoints):
        color = colors.get(robustness_scores[i], (0, 0, 0))  # Default black if not in colors
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(preprocessed_with_keypoints, (x, y), 3, color, -1)

    # Create a legend
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(preprocessed_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Preprocessed Image with Keypoints')
    plt.axis('off')

    # Use normalized RGBA colors for the legend
    rgba_colors = {
        0: (1.0, 0, 0, 1.0),    # red
        1: (1.0, 0.647, 0, 1.0),  # orange
        2: (1.0, 0, 1.0, 1.0),  # magenta
        3: (0, 0, 1.0, 1.0),    # blue
        4: (0, 1.0, 0, 1.0)     # green
    }

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=rgba_colors[i], markersize=10) for i in range(5)]
    labels = ['Robustness 0 (Red)', 'Robustness 1 (Orange)', 'Robustness 2 (Magenta)', 'Robustness 3 (Blue)', 'Robustness 4 (Green)']
    plt.legend(handles, labels, loc='upper right')

    plt.show()

if __name__ == "__main__":
    test_feature_matching_with_robustness()