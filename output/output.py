### Imports ###
import os
import csv
import cv2
from tqdm import tqdm
from datetime import datetime
import sys 
import json
import numpy as np

# Set the field size limit to a large value that is safe for most systems
csv.field_size_limit(10**7)

### Parameters ###
# Define the class labels globally in the visualization module (defined by cityscapes: [5] https://www.cityscapes-dataset.com/)
class_labels = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

### Progress Bar ###
def initialize_progress_bar(total_images):
    """
    Source: / 

    Initializes a progress bar for tracking the processing of images.

    Parameters:
    - total_images (int): The total number of images to be processed.

    Returns:
    - tqdm: An instance of tqdm progress bar configured for the given total number of images.
    """
    return tqdm(total=total_images, desc="Processing images", unit="image")

def update_progress_bar(pbar, image_name, num_keypoints, num_matches, recognized_features, new_features):
    """
    Source: / 

    Updates the progress bar with the current processing status of an image.

    Parameters:
    - pbar (tqdm): The tqdm progress bar instance.
    - image_name (str): The name of the current image being processed.
    - num_keypoints (int): The number of keypoints extracted from the current image.
    - num_matches (int): The number of matches with keypoints from the previous image.
    - recognized_features (int): The number of recognized features.
    - new_features (int): The number of new features detected.

    Returns:
    - None
    """
    pbar.set_postfix({
        "Current Image": image_name,           # Name of the current image
        "Extracted Features": num_keypoints,   # Number of keypoints extracted from the current image
        "Matches": num_matches,                # Number of matches with the previous image
        "Storage Matches": recognized_features,# Number of recognized features
        "New Features": new_features           # Number of new features detected
    })
    pbar.update(1)

### Create and Save Output ###
def save_output(feature_dict, scores, selected_features, weights, class_scores, percentage, N, output_dir, input_dir):
    """
    Saves outputs related to feature extraction and scoring.

    Parameters:
    - feature_dict (dict): Dictionary containing feature details.
    - scores (list): List of tuples containing feature IDs and their scores.
    - selected_features (list): List of selected features with their details.
    - output_dir (str): Directory where the output files will be saved.
    - input_dir (str): Directory containing the input images.

    Returns:
    - None
    """
    print("Saving Outputs...")
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save feature scores to a text file
    scores_file = os.path.join(output_dir, "feature_scores.txt")
    with open(scores_file, "w") as f:
        for idx, (feature_id, score_details) in enumerate(scores, 1):
            f.write(f"{idx}. Feature ID: {feature_id}\n")
            f.write(f"  Total Weighted Feature Score: {score_details['total_feature_score']}\n")
            f.write(f"  Normalized Corner Score: {score_details['normalized_corner_score']}\n")
            f.write(f"  Normalized Class Score: {score_details['normalized_class_score']}\n")
            f.write(f"  Normalized Depth Score : {score_details['normalized_depth']}\n")
            f.write(f"  Normalized Observability Score: {score_details['normalized_observability']}\n")
            f.write(f"  Normalized Entropy Score: {score_details['normalized_entropy']}\n")
            f.write(f"  Normalized Geometric Score: {score_details['normalized_geometric']}\n")
            f.write(f"  Normalized Displacement Score: {score_details['normalized_displacement']}\n")
            f.write(f"  Normalized Angle Score: {score_details['normalized_angle']}\n")
            normalized_std = score_details['normalized_std']
            if normalized_std is not None:
                f.write(f"  Normalized std Score: {round(1 - normalized_std,3)}\n")
            else:
                f.write(f"  Normalized std Score: None\n")
            f.write(f"  Normalized Contrast Score: {score_details['normalized_contrast_score']}\n")
            f.write(f"  Normalized Robustness Score: {score_details['normalized_robustness_score']}\n")
            f.write("\n")
    print(f"Feature scores saved to {scores_file}")

    # Save selected features to a CSV file
    selected_file = os.path.join(output_dir, "selected_features.csv")
    with open(selected_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Feature ID", "Keypoints", "Descriptors", "Image Names"])

        for idx, feature in enumerate(selected_features, 1):
            writer.writerow([
                idx,
                feature['feature_id'],
                feature['keypoints'],
                feature['descriptors'],
                feature['image_names']
            ])
    print(f"Selected features CSV saved to {selected_file}")

    # Save all features to a text file
    all_features_file = os.path.join(output_dir, "features.txt")
    with open(all_features_file, "w") as file:
        for feature_id, feature_data in feature_dict.items():
            file.write(f"Feature ID: {feature_id}\n")
            file.write(f"  Positions: {feature_data['positions']}\n")
            file.write(f"  Descriptors: {feature_data['descriptors']}\n")
            file.write(f"  Semantic Class: {feature_data['semantic_class']}\n")
            file.write(f"  Confidence: {feature_data['confidence']}\n")
            file.write(f"  Depth: {feature_data['depth']}\n")
            file.write(f"  Observability: {feature_data['observability']}\n")
            file.write(f"  Corner Score: {feature_data['corner_scores']}\n")
            file.write(f"  Standard Deviation: {feature_data['std']}\n")
            file.write(f"  Contrast: {feature_data['contrast']}\n")
            file.write(f"  Image Names: {feature_data['image_names']}\n")
            file.write(f"  Robustness: {feature_data['robustness']}\n")
            file.write(f"  Entropy: {feature_data['entropy']}\n")
            file.write(f"  Geometric: {feature_data['geometric']}\n")
            file.write(f"  Displacement: {feature_data['displacement']}\n")
            file.write(f"  Angle: {feature_data['angle']}\n")
            file.write("\n")

    print(f"All features txt saved to {all_features_file}")

    # for feature_id, feature in feature_dict.items():
    #     feature['image_names'] = [int(name.lstrip('0').rstrip('.png')) if name.lstrip('0').rstrip('.png') else 0 for name in feature['image_names']]

    # # 定义插值函数，保留序号
    # def interpolate_data(entry, index):
    
    #     positions = np.array(entry['positions'])
    #     image_names = np.array(entry['image_names'])

    #     # 找出 image_names 中的整数插值点
    #     min_image = min(image_names)
    #     max_image = max(image_names)
    #     full_range = np.arange(min_image, max_image + 1)

    #     # 进行线性插值
    #     interpolated_positions = np.zeros((len(full_range), 2))
    #     for i in range(2):  # 对 x 和 y 坐标分别插值
    #         interpolated_positions[:, i] = np.interp(full_range, image_names, positions[:, i])

    #     # 返回插值后的数据，并保留原始的序号
    #     return {
    #         'index': index,
    #         'positions': interpolated_positions.tolist(),
    #         'image_names': full_range.tolist()
    #     }

    # interpolared_features = []
    # # 对每条数据进行插值，并保留序号
    # for feature_id, feature in feature_dict.items():
        
    #     data_list = interpolate_data(feature, feature_id) 
    #     interpolared_features.append(data_list)
    
    # # initial a dict
    # frame_data = {}
    
    
    # for feature_id, feature in feature_dict.items():
    #     # go through all data
    #     for i, img_name in enumerate(feature['image_names']):
            
    #         if img_name not in frame_data:
    #             frame_data[img_name] = {
    #                 "Positions_cur": [],
    #                 "Positions_ref": [],
    #                 "Feature IDs": []
    #             }
    #         # 将该特征点在当前帧的位置信息添加到对应的字典项中
    #         # if feature['positions'][i+1]存在，则赋值这一对
    #         frame_data[img_name]["Positions"].append(list(feature['positions'][i]))
    #         frame_data[img_name]["Descriptors"].append(feature['descriptors'][i].tolist())
    #         frame_data[img_name]["Feature IDs"].append(feature_id)
    
    # # 定义保存数据到json文件的函数
    # def save_to_json(data, file_path):
    #     with open(file_path, 'w') as f:
    #         json.dump(data, f, indent=4)
    
    # # 保存frame_data到json文件
    # save_to_json(frame_data, 'frame_positions.json')

    # Additional information
    output_file = os.path.join(output_dir, "info.txt")
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write additional information to the text file
    with open(output_file, "w") as f:
        total_features = len(feature_dict)
        total_extracted_features = sum(len(feature['keypoints']) for feature in feature_dict.values())
        f.write(f"Date and Time: {current_datetime}\n")
        f.write(f"Total Extracted Features: {total_extracted_features}\n")
        f.write(f"Total Unique Features: {total_features}\n")
        f.write(f"Matches: {total_extracted_features - total_features}\n")
        f.write(f"Percentage: {percentage}%\n")       
        f.write(f"Number of selected best features: {N}\n")  

        f.write("\nWeights:\n")
        for key, value in weights.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nClass Scores:\n")
        for key, value in class_scores.items():
            class_name = class_labels[key]
            f.write(f"  {class_name} ({key}): {value}\n")

    print(f"Info saved to {output_file}")

    # Create and save the feature video
    image_folder = input_dir
    video_output_path = os.path.join(output_dir, "selected_features_visualization.mp4")
    create_feature_video(image_folder, selected_file, video_output_path)

def create_feature_video(image_folder, csv_file, video_output_path):
    """
    Source: / 

    Creates a video visualizing the selected features in each image based on the provided CSV file.

    Parameters:
    - image_folder (str): Path to the folder containing the images.
    - csv_file (str): Path to the CSV file containing the selected features.
    - video_output_path (str): Path to save the output video.

    Returns:
    - None
    """
    
    # Load the selected features from the CSV file
    selected_features = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            keypoints = eval(row["Keypoints"])  # Convert string representation of list to actual list
            image_names = eval(row["Image Names"])
            selected_features.append({
                "feature_id": row["Feature ID"],
                "keypoints": keypoints,
                "image_names": image_names
            })

    # Define the video writer
    frame_width, frame_height = None, None
    video_writer = None

    # Create the images folder within the output directory
    images_output_folder = os.path.join(os.path.dirname(video_output_path), "images")
    if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)

    # Iterate through each image in the folder
    for image_name in sorted(os.listdir(image_folder)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)

            if frame_width is None or frame_height is None:
                frame_height, frame_width, _ = image.shape
                video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (frame_width, frame_height))

            # Count the number of features in this image
            feature_count = 0
            for feature in selected_features:
                if image_name in feature["image_names"]:
                    index = feature["image_names"].index(image_name)
                    kp = feature["keypoints"][index]
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle for the keypoint
                    cv2.putText(image, f"ID: {feature['feature_id']}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    feature_count += 1

            # Display the image name and feature count on the image
            cv2.putText(image, f"Image: {image_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Features: {feature_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Save the processed image to the images output folder
            output_image_path = os.path.join(images_output_folder, image_name)
            cv2.imwrite(output_image_path, image)

            # Write the frame to the video
            video_writer.write(image)

    # Release the video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {video_output_path}")
        print(f"Images saved to {images_output_folder}")