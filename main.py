######################################################################################################################################################################################################################################################################################################################
#                                                                              Imports                                                                          
######################################################################################################################################################################################################################################################################################################################
import os
# import json
from preprocessing.image_classification import classify_contrast, classify_weather
from preprocessing.image_preprocessing import preprocess_image
from preprocessing.semantic_segmentation import segment_image
from preprocessing.feature_extraction import extract_features
from preprocessing.feature_identification import feature_ID
from attributes.entropy_attribute import estimate_entropy
from attributes.geometric_attribute import geometric_attribute
from attributes.semantic_attribute import semantic_attribute
from attributes.depth_attribute import depth_attribute
from attributes.observability_attribute import observability_attribute
from attributes.stochastic_filter import stochastic_filter
from attributes.robustness_attribute import robustness_attribute
from attributes.trajectroy_attribute import calculate_trajectory_scores
from selection.feature_storage import FeatureStorage
from output.visualization import save_plots, visualization
from selection.feature_last_frames import preserve_last_frames
from selection.feature_prefiltering import prefilter_pos, prefilter_obs
from selection.feature_scoring import calculate_scores
from selection.feature_selector import select_features
from output.output import initialize_progress_bar, update_progress_bar, save_output
######################################################################################################################################################################################################################################################################################################################

def main(input_dir, output_dir):
    ### Load images ###
    images = sorted(os.listdir(input_dir)) # Sorted list of image names in input directory
    total_images = len(images) # Number of images in the directory 
    print("Found:", total_images, "images") 

    if total_images < 2: # Check number of images in the folder
        print("Error: at least 2 images required") 
        exit()

    ### Initialization ###
    feature_storage = FeatureStorage() # Initialize instace of FeatureStorage (store extracted keypoints)
    progress_bar = initialize_progress_bar(total_images) # Initialize progress bar

    prev_keypoints1, prev_descriptors1, features_IDs1 = None, None, None # Initialize variables to store keypoints, descriptors, and feature IDs for the previous three images
    prev_keypoints2, prev_descriptors2, features_IDs2 = None, None, None # Initialize variables to store keypoints, descriptors, and feature IDs for the previous three images
    prev_keypoints3, prev_descriptors3, features_IDs3 = None, None, None # Initialize variables to store keypoints, descriptors, and feature IDs for the previous three images

    newest_data = {}

    for image_name in images: # Go through all images
        img_path = os.path.join(input_dir, image_name)
        # newest_data = {}
        ##############################################################################################################################################################################################################################################################################################################
        #                                                                          Preprocessing                                                                        
        ##############################################################################################################################################################################################################################################################################################################
        ### Image Classification ###
        image_contrast, image_contrast_distribution = classify_contrast(img_path) # 0: Normal, 1: Bright, 2: Dimmed
        image_weather = 0 # 0: Cloudy, 1: Sunny, 2: Rainy, 3: Snowy, 4: Foggy (currently not implemented since DL-approach not sufficient)

        ### Image Preprocessing ###
        preprocessed_image = preprocess_image(img_path, image_contrast) # Apply contrast enhancement (for 1, 2) and sharpen image

        ### Semantic Segmentation ###
        seg_mask, conf_mask = segment_image(preprocessed_image) 

        ### Feature Extraction ###
        keypoints, descriptors = extract_features(preprocessed_image)

        ### Feature Prefilter
        keypoints, descriptors = prefilter_pos(keypoints, descriptors)
        num_keypoints = len(keypoints) # Number of extracted keypoints from image after prefiltering

        ### Feature Identification ###
        features_IDs_current, total_matches, feature_distance_score = feature_ID(keypoints, descriptors,
                                          prev_keypoints1, prev_descriptors1, features_IDs1,
                                          prev_keypoints2, prev_descriptors2, features_IDs2,
                                          prev_keypoints3, prev_descriptors3, features_IDs3,
                                          ransac_params_prev=(25, 5000, 0.99), 
                                          ransac_params_prev2=(25, 5000, 0.99), 
                                          ransac_params_prev3=(8, 5000, 0.99),
                                          num_images_to_match=2)
       
        # Update the previous keypoints, descriptors, and feature IDs for the next iteration
        prev_keypoints3, prev_descriptors3, features_IDs3 = prev_keypoints2, prev_descriptors2, features_IDs2
        prev_keypoints2, prev_descriptors2, features_IDs2 = prev_keypoints1, prev_descriptors1, features_IDs1
        prev_keypoints1, prev_descriptors1, features_IDs1 = keypoints, descriptors, features_IDs_current
        ##############################################################################################################################################################################################################################################################################################################

        ##############################################################################################################################################################################################################################################################################################################
        #                                                                       Attribute Extraction                                                                    
        ##############################################################################################################################################################################################################################################################################################################
        ### new!!!!!!!!!!   Entropy Attribute Extraction ###
        feature_entropy = estimate_entropy(keypoints, preprocessed_image) 

        ### new!!!!!!!!!!!  Find Local Maxima (NMS) ###
        feature_geometric = geometric_attribute(keypoints, radius=20)

        ### Semantic Attribute Extraction ###
        feature_segmentation, feature_confidence = semantic_attribute(keypoints, seg_mask, conf_mask)
        
        ### Depth Attribute Extraction ###
        feature_depth, depth_map = depth_attribute(keypoints, preprocessed_image)
  
        ### Robustness Attribute Extraction ###
        feature_robustness = robustness_attribute(preprocessed_image, keypoints, descriptors)

        ### Feature Storage Update ###
        num_matched_features, num_new_features = feature_storage.update_feature_dict(features_IDs_current, keypoints, descriptors, image_contrast, feature_entropy, feature_geometric, feature_distance_score, feature_segmentation, feature_confidence, feature_depth, feature_robustness, image_name)

        # frame_id = image_name.rstrip('.png').lstrip('0')

        # # 如果所有数字都是0，返回 '0'
        # if not frame_id:
        #     frame_id = 0
        # else:
        #     frame_id = int(frame_id)
        # 提取数字部分并将其转换为整数
        image_id_str = image_name.split('.')[0]
        image_id = int(image_id_str)  # 将数字部分转换为整数
        update_progress_bar(progress_bar, image_name, num_keypoints, total_matches, num_matched_features, num_new_features) # Update progress bar

        # 判断如果 image_name 不是 0000000000.png
        if image_id != 0:
            #print(image_name)
            ### Preserve only the data of last n frames ###
            newest_data[str(image_name)] = preserve_last_frames(feature_storage.feature_dict, image_name, number_of_frames=5)
            # print(newest_data)
            ### Update Progress Bar ###
            #update_progress_bar(progress_bar, image_name, num_keypoints, total_matches, num_matched_features, num_new_features) # Update progress bar

            # progress_bar.close() # Close progress bar
            # print(newest_data.keys())
            ### Observability Atribute Extraction ###
            newest_data[str(image_name)] = observability_attribute(newest_data[str(image_name)]) # Update feature_dict with observability attribute
        
            ### Stochastic Attribute Extraction ###
            newest_data[str(image_name)] = stochastic_filter(newest_data[str(image_name)]) # Update feature_dict with stochastic attribute
            ##############################################################################################################################################################################################################################################################################################################

            ##############################################################################################################################################################################################################################################################################################################
            #                                                                  Scoring and Feature Selection                                                                
            ##############################################################################################################################################################################################################################################################################################################
            ### new!!!!!!!!!  Calculate Trajectory Score ###
            newest_data[str(image_name)] = calculate_trajectory_scores(newest_data[str(image_name)])

            ### Prefiltering ###
            newest_data[str(image_name)] = prefilter_obs(newest_data[str(image_name)]) # Prefilter feature_dict (remove features with obs = 1)
            # print(feature_storage.feature_dict)
            ### Calculate Feature Scores ###
            feature_scores, weights, class_scores = calculate_scores(newest_data[str(image_name)]) # Calculate score for every feature in the store
            
            ### Select Best Features ###
            selected_features, percentage, N, selected_feature_details = select_features(feature_scores, newest_data[str(image_name)], image_name) # Dictionary with the best N features
            # print(len(selected_features['ref_features'][0]))
            # print(len(selected_features['current_features'][0]))
    progress_bar.close() # Close progress bar
    ##############################################################################################################################################################################################################################################################################################################

    ##############################################################################################################################################################################################################################################################################################################
    #                                                                Output and Visualization                                                                      
    ##############################################################################################################################################################################################################################################################################################################
    save_plots(newest_data['0000000010.png'], image_contrast_distribution, output_dir) # Create plots
    # print(feature_storage.feature_dict.items())
    save_output(newest_data['0000000010.png'], feature_scores, selected_feature_details, weights, class_scores, percentage, N, output_dir, input_dir) # Save files
    ##############################################################################################################################################################################################################################################################################################################


if __name__ == "__main__":
    input_dir = "/home/q661086/test/data/123"
    output_dir = "/home/q661086/test/data/output"
    main(input_dir, output_dir)
    