### Imports ###
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import cv2

### Parameters ###
# Define the class labels globally in the visualization module (defined by cityscapes: [5] https://www.cityscapes-dataset.com/)
class_labels = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

### Create and Save Plots ### 
def save_plots(feature_dict, image_contrast_distribution, output_dir):
    """
    Creates and saves plots to visualize the distribution of different feature attributes.

    Parameters:
    - feature_dict (dict): A dictionary where keys are feature IDs and values are dictionaries containing feature attributes.
    - output_dir (str): The directory where the plots will be saved.

    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize counters and lists for feature attributes
    class_count = {label: 0 for label in class_labels}  # Counts for each class label
    depth_values = []  # List to store average depth values
    observability_values = []  # List to store observability values
    std_values = []  # List to store standard deviation values
    robustness_values = []  # List to store robustness values

    # Process each feature to accumulate statistics
    for feature_id, attributes in feature_dict.items():
        # Calculate the most common class for the feature
        semantic_classes = attributes["semantic_class"]
        if semantic_classes:
            most_common_class = max(set(semantic_classes), key=semantic_classes.count)
            class_label = class_labels[most_common_class]
            class_count[class_label] += 1
        
        # Calculate the average depth for the feature
        if attributes["depth"]:
            average_depth = np.mean(attributes["depth"])
            depth_values.append(average_depth)
        
        # Extract observability, standard deviation, and robustness for the feature
        observability_values.append(attributes.get("observability", 0))
        
        if isinstance(attributes.get("std", []), list):
            std_values.extend(attributes["std"])
        elif isinstance(attributes.get("std", []), (int, float, np.float64)):
            std_values.append(attributes["std"])
        
        if attributes.get("robustness", []):
            average_robustness = round(np.mean(attributes["robustness"]))
            robustness_values.append(average_robustness)

    # Function to add value labels to bars
    def add_value_labels(ax):
        for rect in ax.patches:
            height = rect.get_height()
            ax.annotate('{}'.format(int(height)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # 1) Class distribution
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    bars = ax.bar(class_count.keys(), class_count.values())
    add_value_labels(ax)
    plt.xlabel('Feature Classes')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Classes')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

    # 2) Depth Distribution
    plt.figure(figsize=(10, 5))
    bins = np.arange(0, 256, 25)
    counts, _, _ = plt.hist(depth_values, bins=bins, edgecolor='k')
    for count, bin_ in zip(counts, bins):
        plt.text(bin_ + 12.5, count, str(int(count)), ha='center', va='bottom')
    plt.xlabel('Depth')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Depth')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "depth_distribution.png"))
    plt.close()

    # 3) Observability Distribution
    observability_bins = np.arange(0, max(observability_values) + 50, 50)  # Set bin interval to 50
    observability_binned_counts = np.zeros(len(observability_bins) - 1)

    for obs_value in observability_values:
        bin_index = np.digitize(obs_value, observability_bins) - 1
        if bin_index < len(observability_binned_counts):
            observability_binned_counts[bin_index] += 1

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    bars = ax.bar(range(len(observability_binned_counts)), observability_binned_counts, tick_label=[f"{observability_bins[i]}-{observability_bins[i+1]-1}" for i in range(len(observability_bins)-1)], width=0.7)
    add_value_labels(ax)
    plt.xlabel('Observability')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Observability')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels and align them to the right
    plt.tight_layout(pad=4)  # Increase padding to avoid label overlap
    plt.savefig(os.path.join(output_dir, "observability_distribution.png"))
    plt.close()



    # 4) Standard Deviation Distribution
    plt.figure(figsize=(10, 5))
    bins = np.arange(0, max(std_values, default=0) + 5, 5)
    counts, _, _ = plt.hist(std_values, bins=bins, edgecolor='k')
    for count, bin_ in zip(counts, bins):
        plt.text(bin_ + 2.5, count, str(int(count)), ha='center', va='bottom')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Standard Deviation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "std_distribution.png"))
    plt.close()

    # 5) Robustness Distribution
    plt.figure(figsize=(10, 5))
    bins = np.arange(-0.5, 5, 1)
    counts, _, _ = plt.hist(robustness_values, bins=bins, edgecolor='k', align='mid')
    for count, bin_ in zip(counts, bins[:-1]):
        plt.text(bin_ + 0.5, count, str(int(count)), ha='center', va='bottom')
    plt.xlabel('Robustness (0-4)')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Robustness')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "robustness_distribution.png"))
    plt.close()

    # 6) Class-Observability Heatmap
    heatmap_matrix = np.zeros((len(class_labels), len(observability_bins) - 1))

    for feature_id, attributes in feature_dict.items():
        observability = attributes.get("observability", 0)
        bin_index = np.digitize(observability, observability_bins) - 1  # Adjust bin index to be 1-based
        semantic_classes = attributes["semantic_class"]
        if semantic_classes:
            most_common_class = max(set(semantic_classes), key=semantic_classes.count)
            if bin_index < heatmap_matrix.shape[1]:  # Avoid index out of bounds
                heatmap_matrix[most_common_class, bin_index] += 1

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_matrix, xticklabels=[f"{observability_bins[i]}-{observability_bins[i+1]-1}" for i in range(len(observability_bins)-1)], yticklabels=class_labels, cmap="YlGnBu", annot=True, fmt='g')
    plt.xlabel('Observability')
    plt.ylabel('Feature Classes')
    plt.title('Heatmap of Feature Classes and Observability')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_observability_heatmap.png"))
    plt.close()



    # 7) Image Contrast Distribution
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    contrast_labels = list(image_contrast_distribution.keys())
    contrast_values = list(image_contrast_distribution.values())
    bars = ax.bar(contrast_labels, contrast_values)
    add_value_labels(ax)
    plt.xlabel('Image Contrast Type')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Image Contrast Types')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "image_contrast_distribution.png"))
    plt.close()

### Visulization ### 
def visualization(  # Visualization during running the algorithm
    original_img_path, preprocessed_img, image_contrast, keypoints, inlier_matches, prev_keypoints, 
    prev_img_path, image_name, num_keypoints, num_matches, recognized_features, new_features, seg_mask, depth_map
):
    """
    Source: /

    Visualizes the image processing pipeline.

    Parameters:
    - original_img_path (str): Path to the original image.
    - preprocessed_img (numpy array): The preprocessed image.
    - image_contrast (int): Contrast classification of the image (0: Normal, 1: Bright, 2: Dimmed).
    - keypoints (list): Extracted keypoints from the image.
    - inlier_matches (list): Inlier matches with keypoints from the previous image.
    - prev_keypoints (list): Keypoints from the previous image.
    - prev_img_path (str): Path to the previous image.
    - image_name (str): Name of the current image.
    - num_keypoints (int): Number of extracted keypoints.
    - num_matches (int): Number of matches with keypoints from the previous image.
    - recognized_features (int): Number of recognized features.
    - new_features (int): Number of new features.
    - seg_mask (numpy array): Segmentation mask for the image.
    - depth_map (numpy array): Depth map for the image.

    Returns:
    - None
    """

    # Define labels for the contrast levels
    contrast_labels = ["Normal", "Bright", "Dimmed"]
    contrast_label = contrast_labels[image_contrast]  # Get the label for the current image's contrast level

    # Read the original image
    original_img = cv2.imread(original_img_path)
    # Draw keypoints on the preprocessed image
    img_with_keypoints = cv2.drawKeypoints(preprocessed_img, keypoints, None, color=(0, 255, 0))

    plt.ion()  # Enable interactive mode for the plot
    plt.figure(figsize=(12, 8))  # Create a new figure with specified size

    # Original image subplot
    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
    plt.title('Original Image', fontsize=10)
    plt.axis('off')  # Turn off axis

    # Preprocessed image with contrast label
    plt.subplot(3, 2, 2)
    plt.imshow(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Preprocessed Image - Contrast: {contrast_label}', fontsize=10)
    plt.axis('off')

    # Extracted features (keypoints) subplot
    plt.subplot(3, 2, 3)
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Extracted Features', fontsize=10)
    plt.axis('off')

    # Matches with the previous image (only inliers) subplot
    plt.subplot(3, 2, 4)
    if prev_keypoints is not None and prev_img_path is not None:
        prev_img = cv2.imread(prev_img_path)  # Read the previous image
        img_matches = cv2.drawMatches(prev_img, prev_keypoints, preprocessed_img, keypoints, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(np.zeros_like(original_img))  # Display an empty image if no previous image
    plt.title('Inlier Matches with Previous Image', fontsize=10)
    plt.axis('off')

    # Segmentation mask overlay subplot
    plt.subplot(3, 2, 5)
    if seg_mask is not None:
        cmap = plt.get_cmap('tab20', np.max(seg_mask) + 1)  # Get a colormap
        seg_img = cmap(seg_mask)[:, :, :3]  # Apply colormap and ignore alpha channel
        seg_overlay = cv2.addWeighted(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB), 0.7, (seg_img * 255).astype(np.uint8), 0.3, 0)
        plt.imshow(seg_overlay)
        plt.title('Segmentation Mask', fontsize=10)
        plt.axis('off')

        # Create the legend with smaller markers
        legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i)[:3], markersize=5, label=class_labels[i]) for i in range(len(class_labels))]
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=5, markerscale=0.5)
    else:
        plt.imshow(np.zeros_like(original_img))  # Display an empty image if no segmentation mask
        plt.axis('off')

    # Depth map subplot
    plt.subplot(3, 2, 6)
    if depth_map is not None and isinstance(depth_map, np.ndarray) and depth_map.dtype != np.dtype('O'):
        plt.imshow(depth_map, cmap='plasma')
        plt.title('Depth Map', fontsize=10)
        plt.axis('off')

        # Add colorbar for depth map
        cbar = plt.colorbar()
        cbar.set_label('Depth Value', fontsize=7)
        cbar.ax.tick_params(labelsize=5)
    else:
        plt.imshow(np.zeros_like(original_img))  # Display an empty image if no depth map
        plt.axis('off')

    # Additional information below the plots
    plt.figtext(0.5, 0.01, f"Current Image: {image_name}, Extracted Features: {num_keypoints}, Matches: {num_matches}, Storage Matches: {recognized_features}, New Features: {new_features}", wrap=True, horizontalalignment='center', fontsize=10)

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.draw()  # Draw the plot
    plt.pause(0.1)  # Pause to allow the plot to be updated

    # Close the plot window before the next image is processed
    plt.close()









