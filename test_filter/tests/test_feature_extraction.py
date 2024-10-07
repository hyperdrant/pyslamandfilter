import os
import sys
import cv2
import matplotlib.pyplot as plt

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.feature_extraction import extract_features


def display_keypoints(image, keypoints):
    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    
    # Convert the image to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
    
    # Display the image with keypoints
    plt.imshow(image_rgb)
    plt.title("ORB Keypoints")
    plt.axis('off')
    plt.draw()
    plt.pause(0.01)  # Display each image for 2 seconds
    plt.clf()  # Clear the current figure for the next image

def main():
    input_dir = "/Users/alexanderwitt/Desktop/Semesterarbeit/Code/Image_Preprocessing/Dataset/Output/erding_4_2"
    images = sorted(os.listdir(input_dir))

    for image_name in images:
        img_path = os.path.join(input_dir, image_name)

        # Load the image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot open or read the image file at {img_path}")

        # Extract ORB features
        keypoints, descriptors = extract_features(image)

        # Display the image with keypoints
        display_keypoints(image, keypoints)

        # Print the number of keypoints and descriptor size for debugging
        print(f"Processed image: {image_name}")
        print(f"Number of keypoints: {len(keypoints)}")
        if descriptors is not None:
            print(f"Descriptor shape: {descriptors.shape}")

    plt.close()

if __name__ == "__main__":
    main()