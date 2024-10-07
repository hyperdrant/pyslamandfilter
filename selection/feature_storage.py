### Feature Storage ###
class FeatureStorage: # Class to store and update the attributes of the extracted features
    def __init__(self):
        """
        Source: /

        Initializes a new instance of the FeatureStorage class.
        
        Attributes:
        - feature_dict (dict): Dictionary to store features with their attributes.
        - feature_id_counter (int): Counter to keep track of the feature IDs.
        - mapping_dict (dict): Dictionary to map descriptors to feature IDs (required to decide whether a extracted feature already exists).

        Returns: /
        """

        self.feature_dict = {}
        self.feature_dict_last_frames = {}
        self.feature_id_counter = 0 # Feature ID starts at zero
        self.mapping_dict = {}  

    ### Update Features ###
    def update_feature(self, feature_id, contrast, keypoint, descriptor, entropy, geometric, vanishing_point, semantic_class, confidence, depth, robustness, image_name):
        """
        Source: /

        Updates the feature dictionary with new feature data.

        Parameters:
        - feature_id (int): Unique identifier for the feature.
        - contrast (int): Contrast attribute of the feature.
        - keypoint (cv2.KeyPoint): Keypoint object.
        - descriptor (ndarray): Descriptor corresponding to the keypoint.
        - entropy (float): Entropy attribute of the feature.
        - geometric (float): Score accroding to distance to local resnponse-maxima.
        - vanishing_point (float): Score according to distance to vanishing point
        - semantic_class (int): Semantic attribute of the feature.
        - depth (float): Depth attribute of the feature.
        - robustness (float): Robustness attribute of the feature.
        - image_name (str): Name of the image from which the feature was extracted.

        Returns:
        - is_new_feature (bool): True if the feature was newly created, False if it already existed.
        """
        new_feature = False

        # If the feature_id of the extracted feature does not exist in the feature_dict --> new entry
        if feature_id not in self.feature_dict:
            self.feature_dict[feature_id] = {
                "keypoints": [], # List to store keypoint objects
                "positions": [], # List to store keypoint positions
                "descriptors": [], # List to store descriptors
                "entropy": [], # List to store entropy attributes
                "geometric": [], # List to store geometric attributes
                "vanishing_point": [], # List to store vanishing point attributes
                "semantic_class": [], # List to store semantic attributes
                "confidence": [],  # List to store confidence values
                "depth": [], # List to store depth attributes
                "observability": [], # List to store observability attributes (calculated later)
                "corner_scores": [], # List to store corner score attributes (response values)                 
                "std": [],  # List to store standard deviation attributes (calculated later)
                "robustness": [], # List to store robustness attrbiutes
                "contrast": [],  # List to store contrast of the images the feature has been extracted from                    
                "image_names": [], # List to store image names the feature has been extracted from  
                "displacement": [],         
                "angle": []
            }
            new_feature = True
        
        # Append the extracted attributes to the lists in the feature dictionary
        if not self.feature_dict[feature_id]["image_names"]:
            self.feature_dict[feature_id]["keypoints"].append(keypoint)
            self.feature_dict[feature_id]["positions"].append((round(keypoint.pt[0],3), round(keypoint.pt[1],3))) # Directly calculated in the storage, no separate function to lower computations
            self.feature_dict[feature_id]["descriptors"].append(descriptor)
            self.feature_dict[feature_id]["entropy"].append(entropy)
            self.feature_dict[feature_id]["geometric"].append(geometric)
            self.feature_dict[feature_id]["vanishing_point"].append(vanishing_point)
            self.feature_dict[feature_id]["semantic_class"].append(semantic_class)
            self.feature_dict[feature_id]["confidence"].append(confidence) 
            self.feature_dict[feature_id]["depth"].append(depth)
            self.feature_dict[feature_id]["corner_scores"].append(keypoint.response) # Directly calculated in the storage, no separate function to lower computations
            self.feature_dict[feature_id]["robustness"].append(robustness)
            self.feature_dict[feature_id]["contrast"].append(contrast)
            self.feature_dict[feature_id]["image_names"].append(image_name)
        elif self.feature_dict[feature_id]["image_names"][-1] != image_name:
            self.feature_dict[feature_id]["keypoints"].append(keypoint)
            self.feature_dict[feature_id]["positions"].append((round(keypoint.pt[0],3), round(keypoint.pt[1],3))) # Directly calculated in the storage, no separate function to lower computations
            self.feature_dict[feature_id]["descriptors"].append(descriptor)
            self.feature_dict[feature_id]["entropy"].append(entropy)
            self.feature_dict[feature_id]["geometric"].append(geometric)
            self.feature_dict[feature_id]["vanishing_point"].append(vanishing_point)
            self.feature_dict[feature_id]["semantic_class"].append(semantic_class)
            self.feature_dict[feature_id]["confidence"].append(confidence) 
            self.feature_dict[feature_id]["depth"].append(depth)
            self.feature_dict[feature_id]["corner_scores"].append(keypoint.response) # Directly calculated in the storage, no separate function to lower computations
            self.feature_dict[feature_id]["robustness"].append(robustness)
            self.feature_dict[feature_id]["contrast"].append(contrast)
            self.feature_dict[feature_id]["image_names"].append(image_name)
        return new_feature

    ### Update Feature Dictionary ###
    def update_feature_dict(self, features_IDs_current, keypoints, descriptors, image_contrast, feature_entropy, feature_geometric, feature_distance_score, feature_segmentation, feature_confidence, feature_depth, feature_robustness, image_name):
        """
        Updates the feature dictionary with new keypoints and their attributes from the current image.

        Parameters:
        - features_IDs_current (dict): Dictionary mapping current keypoints to global feature IDs.
        - keypoints (list): List of cv2.KeyPoint objects from the current image.
        - descriptors (ndarray): Descriptors corresponding to the keypoints.
        - image_contrast (int): Contrast attribute of the current image.
        - feature_segmentation (list): List of tuples containing keypoints and their semantic class attributes.
        - feature_confidence (list): List of tuples containing keypoints and their confidence values.
        - feature_depth (list): List of tuples containing keypoints and their depth attributes.
        - feature_robustness (list): List of tuples containing keypoints and their robustness attributes.
        - image_name (str): Name of the current image.

        Returns:
        - num_matched_features (int): Number of features that were matched with existing features in the storage.
        - num_new_features (int): Number of new features that were added to the storage.
        """
        num_matched_features = 0  # Counter for matched features
        num_new_features = 0  # Counter for new features

        for i, (kp, descriptor) in enumerate(zip(keypoints, descriptors)):  # i = keypoint index in the current image
            # Use the global feature ID directly from features_IDs_current
            if i in features_IDs_current:
                feature_id = features_IDs_current[i]
            else:
                print("[ERROR] Keypoint ID not found")
            # Extract additional attributes for the keypoint
            entropy = next((entropy_value for (k, entropy_value) in feature_entropy if k == kp), None)
            geometric = next((feature_geometric for (k, feature_geometric) in feature_geometric if k == kp), None)
            vanishing_point = next((feature_distance_score for (k, feature_distance_score) in feature_distance_score if k == kp), None)
            semantic_class = next((class_label for (k, class_label) in feature_segmentation if k == kp), None)
            confidence = next((conf_value for (k, conf_value) in feature_confidence if k == kp), None)
            depth = next((depth_value for (k, depth_value) in feature_depth if k == kp), None)
            robustness = next((robustness_score for (k, robustness_score) in feature_robustness if k == kp), None)
            
            # def process_filename(filename):
            #     # 去掉文件名的前导零和后缀 ".png"
            #     number_str = filename.rstrip('.png').lstrip('0')
                
            #     # 如果所有数字都是0，返回 '0'
            #     if not number_str:
            #         return '0'
                
            #     return number_str
            
            # image_name = process_filename(image_name)

            # Update the feature storage with the new attributes
            new_feature = self.update_feature(feature_id, image_contrast, kp, descriptor, entropy, geometric, vanishing_point, semantic_class, confidence, depth, robustness, image_name)

            
            if new_feature:
                num_new_features += 1  # Increment new features counter
            else:
                num_matched_features += 1  # Increment matched features counter
            
        return num_matched_features, num_new_features
