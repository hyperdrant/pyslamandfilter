#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

### Import of filter ###
import os
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

import numpy as np
import cv2
import math
import time 
import platform 

from config import Config

from visual_odometry import VisualOdometry
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory

#from mplot3d import Mplot3d
#from mplot2d import Mplot2d
from mplot_thread import Mplot2d, Mplot3d

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import FeatureMatcherTypes

from feature_tracker_configs import FeatureTrackerConfigs

from rerun_interface import Rerun



kUseRerun = True
# check rerun does not have issues 
if kUseRerun and not Rerun.is_ok():
    kUseRerun = False
    
"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""
kUsePangolin = True  
if platform.system() == 'Darwin':
    kUsePangolin = True # Under mac force pangolin to be used since Mplot3d() has some reliability issues
                
if kUsePangolin:
    from viewer3D import Viewer3D


if __name__ == "__main__":

    config = Config()
    
    dataset = dataset_factory(config.dataset_settings)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                        config.DistCoef, config.cam_settings['Camera.fps'])


    num_features=2000  # how many features do you want to detect and track?

    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT, LIGHTGLUE, XFEAT, XFEAT_XFEAT, LOFTR
    tracker_config = FeatureTrackerConfigs.ORB
    tracker_config['num_features'] = num_features
   
    feature_tracker = feature_tracker_factory(**tracker_config)

    # create visual odometry object 
    vo = VisualOdometry(cam, groundtruth, feature_tracker)

    is_draw_traj_img = True
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5*traj_img_size)
    draw_scale = 1

    is_draw_3d = True
    
    is_draw_with_rerun = kUseRerun
    if is_draw_with_rerun:
        Rerun.init_vo()
    else: 
        if kUsePangolin:
            viewer3D = Viewer3D()
        else:
            plt3d = Mplot3d(title='3D trajectory')

    is_draw_err = True 
    err_plt = Mplot2d(xlabel='img id', ylabel='m',title='error')
    
    is_draw_matched_points = True 
    matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')

    ### Initialization ###
    feature_storage = FeatureStorage() # Initialize instace of FeatureStorage (store extracted keypoints)
    # progress_bar = initialize_progress_bar(total_images) # Initialize progress bar

    prev_keypoints1, prev_descriptors1, features_IDs1 = None, None, None # Initialize variables to store keypoints, descriptors, and feature IDs for the previous three images
    prev_keypoints2, prev_descriptors2, features_IDs2 = None, None, None # Initialize variables to store keypoints, descriptors, and feature IDs for the previous three images
    prev_keypoints3, prev_descriptors3, features_IDs3 = None, None, None # Initialize variables to store keypoints, descriptors, and feature IDs for the previous three images
    newest_data = {}
    matched_ref_features = None
    matched_cur_features = None
    matched_cur_descriptors = None
    img_id = 0
    while dataset.isOk():

        img = dataset.getImage(img_id)
        image_name = str(img_id).zfill(6) + '.png'
        if img is not None:
            
            image_contrast, image_contrast_distribution = classify_contrast(img) # 0: Normal, 1: Bright, 2: Dimmed
            image_weather = 0 # 0: Cloudy, 1: Sunny, 2: Rainy, 3: Snowy, 4: Foggy (currently not implemented since DL-approach not sufficient)

            ### Image Preprocessing ###
            preprocessed_image = preprocess_image(img, image_contrast) # Apply contrast enhancement (for 1, 2) and sharpen image

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
            # update_progress_bar(progress_bar, image_name, num_keypoints, total_matches, num_matched_features, num_new_features) # Update progress bar

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
                print("我tm处理了一张图片")

                matched_ref_features = np.array(selected_features['ref_features'][0])
                matched_cur_features = np.array(selected_features['current_features'][0])
                matched_cur_descriptors = selected_features['current_features'][1]

            vo.track(img, img_id, matched_ref_features, matched_cur_features, matched_cur_descriptors)  # main VO function 

            if(img_id > 2):	       # start drawing from the third image (when everything is initialized and flows in a normal way)

                x, y, z = vo.traj3d_est[-1]
                x_true, y_true, z_true = vo.traj3d_gt[-1]

                if is_draw_traj_img:      # draw 2D trajectory (on the plane xz)
                    draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
                    true_x, true_y = int(draw_scale*x_true) + half_traj_img_size, half_traj_img_size - int(draw_scale*z_true)
                    cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
                    cv2.circle(traj_img, (true_x, true_y), 1,(0, 0, 255), 1)  # groundtruth in red
                    # write text on traj_img
                    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                    cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                    # show 		

                    if is_draw_with_rerun:
                        Rerun.log_img_seq('trajectory_img/2d', img_id, traj_img)
                    else:
                        cv2.imshow('Trajectory', traj_img)


                if is_draw_with_rerun:                                        
                    Rerun.log_2d_seq_scalar('trajectory_error/err_x', img_id, math.fabs(x_true-x))
                    Rerun.log_2d_seq_scalar('trajectory_error/err_y', img_id, math.fabs(y_true-y))
                    Rerun.log_2d_seq_scalar('trajectory_error/err_z', img_id, math.fabs(z_true-z))
                    
                    Rerun.log_2d_seq_scalar('trajectory_stats/num_matches', img_id, vo.num_matched_kps)
                    Rerun.log_2d_seq_scalar('trajectory_stats/num_inliers', img_id, vo.num_inliers)
                    
                    Rerun.log_3d_camera_img_seq(img_id, vo.draw_img, cam, vo.poses[-1])
                    Rerun.log_3d_trajectory(img_id, vo.traj3d_est, 'estimated', color=[0,0,255])
                    Rerun.log_3d_trajectory(img_id, vo.traj3d_gt, 'ground_truth', color=[255,0,0])     
                else:
                    if is_draw_3d:           # draw 3d trajectory 
                        if kUsePangolin:
                            viewer3D.draw_vo(vo)   
                        else:
                            plt3d.drawTraj(vo.traj3d_gt,'ground truth',color='r',marker='.')
                            plt3d.drawTraj(vo.traj3d_est,'estimated',color='g',marker='.')
                            plt3d.refresh()

                    if is_draw_err:         # draw error signals 
                        errx = [img_id, math.fabs(x_true-x)]
                        erry = [img_id, math.fabs(y_true-y)]
                        errz = [img_id, math.fabs(z_true-z)] 
                        err_plt.draw(errx,'err_x',color='g')
                        err_plt.draw(erry,'err_y',color='b')
                        err_plt.draw(errz,'err_z',color='r')
                        err_plt.refresh()    

                    if is_draw_matched_points:
                        matched_kps_signal = [img_id, vo.num_matched_kps]
                        inliers_signal = [img_id, vo.num_inliers]                    
                        matched_points_plt.draw(matched_kps_signal,'# matches',color='b')
                        matched_points_plt.draw(inliers_signal,'# inliers',color='g')                    
                        matched_points_plt.refresh()                                   
                    
            # draw camera image 
            if not is_draw_with_rerun:
                cv2.imshow('Camera', vo.draw_img)				

        # press 'q' to exit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_id += 1

    #print('press a key in order to exit...')
    #cv2.waitKey(0)

    if is_draw_traj_img:
        print('saving map.png')
        cv2.imwrite('map.png', traj_img)
    if is_draw_3d:
        if not kUsePangolin:
            plt3d.quit()
        else: 
            viewer3D.quit()
    if is_draw_err:
        err_plt.quit()
    if is_draw_matched_points is not None:
        matched_points_plt.quit()
                
    cv2.destroyAllWindows()
