### Imports ### 
import cv2
import numpy as np
from matplotlib import pyplot as plt 
from python_orb_slam3 import ORBExtractor
 
def geometric_attribute(keypoints, radius=20):
    if not keypoints:
        return [], []
 
    # Sort keypoints according to their response
    idxs = np.argsort([-kp.response for kp in keypoints])
    
    # Create mask to keep selected features
    suppressed = np.zeros(len(keypoints), dtype=bool)
    
    # Initialize list to store final scores for each keypoint
    scores = [0.0] * len(keypoints)
    
    for i in idxs:
        if suppressed[i]:
            continue
        
        # Assign a full score of 1.0 to the keypoints that are not suppressed
        scores[i] = 1.0

        # Traverse through the remaining keypoints and suppress nearby ones
        for j in idxs[i + 1:]:
            if suppressed[j]:
                continue
            
            # Calculate distance between two features
            dist = np.sqrt((keypoints[i].pt[0] - keypoints[j].pt[0])**2 +
                           (keypoints[i].pt[1] - keypoints[j].pt[1])**2)
            
            # Suppress the feature if it is too close
            if dist < radius:
                suppressed[j] = True
                
                # Assign a score to the suppressed feature based on distance
                # The score decreases as distance decreases, using a simple linear relation
                scores[j] = 1.0 - (dist / radius) if dist < radius else 0.0
    
    # Combine keypoints with their corresponding scores
    keypoints_with_scores = [(keypoints[i], scores[i]) for i in range(len(keypoints))]
    
    return keypoints_with_scores
 
# # Load image

# image = cv2.imread('/home/q661086/create_covariance/59.png', 0)
 
# # ORB feature extraction

# orb = ORBExtractor(n_features=1000)

# keypoints, descriptors = orb.detectAndCompute(image, None)
 
# # Apply Non-Maximum Suppression (NMS)



# filtered_descriptors = geometric_attribute(keypoints)
 

# print(filtered_descriptors)