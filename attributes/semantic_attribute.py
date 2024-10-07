### Imports ###
import numpy as np

### Semantic Attribute Extraction ###
def semantic_attribute(keypoints, seg_mask, conf_mask):
    """
    Source: /

    Determines the semantic class and confidence for each extracted keypoint based on the segmentation mask.
    
    Parameters:
    - keypoints: List of cv2.KeyPoint objects
    - seg_mask: Segmentation mask (2D numpy array) with class indices
    - conf_mask: Confidence mask (2D numpy array) with confidence values corresponding to the segmentation mask
    
    Returns:
    - feature_segmentation: List of tuples, each containing a keypoint and its corresponding class index
    - feature_confidence: List of tuples, each containing a keypoint and its corresponding confidence value
    """

    feature_segmentation = []
    feature_confidence = []   

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1]) # Extract keypoints x / y coordinates (convert to int for seg_mask)
        class_index = seg_mask[y, x] # Determine keypoint class based on position in seg_mask
        confidence = conf_mask[y, x]  # Determine keypoint confidence based on position in conf_mask
        feature_segmentation.append((kp, class_index)) # Append keypoint + the corresponding class to the list (required for feature storage)
        feature_confidence.append((kp, round(confidence, 2)))   # Append keypoint and confidence to the list
    
    return feature_segmentation, feature_confidence