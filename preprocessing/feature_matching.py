### Imports ###
import cv2
import numpy as np
# from preprocessing.vanishing_point_attribute import find_vanishing_point, long_filter_matches, short_filter_matches, distance_to_vanishing_point
from collections import Counter
# import numpy as np
# from sklearn.cluster import DBSCAN
# from collections import Counter
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import matplotlib
# import cv2
# import numpy as np
from sklearn.cluster import DBSCAN
# from python_orb_slam3 import ORBExtractor


def filter_matches(matches, keypoints1, keypoints2, min_distance):
    filtered_matches = []
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        distance = calculate_distance(pt1, pt2)
        if distance > min_distance:
            filtered_matches.append(match)
    return filtered_matches

def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def compute_line(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    a = y2 - y1
    b = x1 - x2
    c = -(a * x1 + b * y1)
    return a, b, c


def calculate_intersections_batch(lines):
    num_lines = len(lines)
    intersections = []

    for i in range(num_lines):
        for j in range(i + 1, num_lines):
            a1, b1, c1 = lines[i]
            a2, b2, c2 = lines[j]
            
            A = np.array([[a1, b1], [a2, b2]])
            B = np.array([-c1, -c2])

        
            if np.linalg.det(A) != 0:
            
                intersection = np.linalg.solve(A, B)
                intersections.append(intersection)

    return np.array(intersections)

def find_vanishing_point(keypoints1, keypoints2, matches):
    lines = []
    for match in matches:        
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt

        line = compute_line(pt1, pt2)
        lines.append(line)

    intersections = calculate_intersections_batch(lines)

    int_list1 = [[int(item) for item in sublist] for sublist in intersections]

    int_list = np.array(int_list1)
        

    # DBSCAN clustering
    dbscan = DBSCAN(eps=5, min_samples=2)  # parameters
    try:
        # try DBSCAN clustering
        labels = dbscan.fit_predict(int_list)
    except Exception as e:
        mean_point = None
        return mean_point

    # numbers of clusters
    label_counts = Counter(labels)

    # select 5 biggest clusters except noise
    filtered_counts = {label: count for label, count in label_counts.items() if label != -1}
    most_common_labels = [label for label, _ in sorted(filtered_counts.items(), key=lambda item: item[1], reverse=True)[:1]]


    # calculate mean point of the biggest cluster
    if most_common_labels:
        max_label = most_common_labels[0]
        largest_cluster_points = int_list[labels == max_label]
        mean_point = np.mean(largest_cluster_points, axis=0)

    else:
        mean_point = None
    
    return mean_point

def long_filter_matches(matches, keypoints1, keypoints2, min_distance):
    filtered_matches = []
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        distance = calculate_distance(pt1, pt2)
        if distance > min_distance:
            filtered_matches.append(match)
    return filtered_matches

def short_filter_matches(matches, keypoints1, keypoints2, max_distance):
    score = {}
    
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        distance = calculate_distance(pt1, pt2)
        
        # scoring
        if distance == max_distance:
            match_score = 0.5
        else:
            
            match_score = 0.5 + (max_distance - distance) * 0.1

        # restrict scores from 0 to 1
        match_score = max(0, min(match_score, 1))
        
        # add score to list
        score[keypoints2[match.trainIdx]] = match_score
    
    return score


def distance_to_vanishing_point(matches, keypoints1, keypoints2, vanishing_point, threshold_distance):
  
    distance_score = {}

    for match in matches:        
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt

        line = compute_line(pt1, pt2)
 
        a, b, c = line[:3]
        
        denominator = np.sqrt(a**2 + b**2)
        if denominator == 0:
            distance_score[keypoints2[match.trainIdx]] = 0
            continue

        distance = abs(a * vanishing_point[0] + b * vanishing_point[1] + c) / denominator
 
        # scoring
        if distance == threshold_distance:
            match_score = 0.5
        else:
            
            match_score = 0.5 + (threshold_distance - distance) * 0.01

        # restrict scores from 0 to 1
        match_score = max(0, min(match_score, 1))
        
        # add score to list
        distance_score[keypoints2[match.trainIdx]] = match_score
        # distance_score.append((keypoints2[match.trainIdx], match_score))

    return distance_score
### Paramaters (defined in main.py) ### 
#ransacReprojThreshold = 25  # Reprojection threshold 
#maxIters = 2000 # maxIters
#confidence = 0.99 # Confidence level

### Feature Matching ### 
def match_featuresBF(prev_keypoints, prev_descriptors, keypoints, descriptors, ransacReprojThreshold, maxIters, confidence):
    """
    Source: [9] https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html, [10] https://openaccess.thecvf.com/content_CVPR_2020/papers/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.pdf, [11] https://docs.opencv.org/4.x/de/d3e/tutorial_usac.html, [12] #https://opencv.org/blog/evaluating-opencvs-new-ransacs/ 

    Matches features between two sets of keypoints and descriptors.

    Arguments:
    prev_keypoints (list): List of keypoints from the previous image.
    prev_descriptors (ndarray): Descriptors corresponding to the previous keypoints.
    keypoints (list): List of keypoints from the current image.
    descriptors (ndarray): Descriptors corresponding to the current keypoints.

    Returns:
    inlier_matches (list): List of inlier matches after applying homography and MAGSAC++.
    """
    min_distance = 10
    max_distance = 6

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # BFMatcher with Hamming distance and cross-check

    matches = bf.match(prev_descriptors, descriptors) # Match descriptors
    
    matches = sorted(matches, key=lambda x: x.distance) # Sort matches by distance

    # Initialization of arrays to store keypoints coordinates
    points_prev = np.zeros((len(matches), 2), dtype=np.float32)
    points_current = np.zeros((len(matches), 2), dtype=np.float32)

    # Extract coordinates
    for i, match in enumerate(matches):
        points_prev[i, :] = prev_keypoints[match.queryIdx].pt
        points_current[i, :] = keypoints[match.trainIdx].pt

    # Find homography using MAGSAC++
    homography, mask = cv2.findHomography(points_prev, points_current, cv2.USAC_MAGSAC, ransacReprojThreshold, maxIters=maxIters, confidence=confidence) # [9, 10, 11, 12]

    # Select inlier matches
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    
    # keep the matchings that are long enough
    long_matches = long_filter_matches(inlier_matches, prev_keypoints, keypoints, min_distance)

    vanishing_points = []
    
    # find vanishing point
    vanishing_points = find_vanishing_point(prev_keypoints, keypoints, long_matches)
    if (len(long_matches)) < 20:      
        distance_score = short_filter_matches(inlier_matches, prev_keypoints, keypoints, max_distance)
    else:        
        # keep the matchings that are not far away from vanishing point
        distance_score = distance_to_vanishing_point(inlier_matches, prev_keypoints, keypoints, vanishing_points, threshold_distance=50)


    return inlier_matches, distance_score



def match_featuresNN(prev_descriptors, descriptors):

    # BFMatcher with default params
    bf = cv2.BFMatcher()    
    matches = bf.knnMatch(prev_descriptors,descriptors,k=2)
 
    # Apply ratio test
    inlier_matches = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            inlier_matches.append([m])
 
    return inlier_matches
