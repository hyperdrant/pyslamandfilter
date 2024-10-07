import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from python_orb_slam3 import ORBExtractor


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
    score = []
    
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        distance = calculate_distance(pt1, pt2)
        
        # 根据距离计算分数
        if distance == max_distance:
            match_score = 0.5
        else:
            # 差值对分数的影响，每小于或大于 max_distance 1，分数变化 0.1
            match_score = 0.5 + (max_distance - distance) * 0.1

        # 限制分数在0到1之间
        match_score = max(0, min(match_score, 1))
        
        # 添加分数到 score 列表
        score.append(match_score)
    
    return score
# def short_filter_matches(matches, keypoints1, keypoints2, max_distance):
#     filtered_matches = []
#     score = []
#     for match in matches:
#         pt1 = keypoints1[match.queryIdx].pt
#         pt2 = keypoints2[match.trainIdx].pt
#         distance = calculate_distance(pt1, pt2)
#         if distance < max_distance:
#             filtered_matches.append(match)
#     return filtered_matches

def distance_to_vanishing_point(matches, keypoints1, keypoints2, vanishing_point, threshold_distance):
  
    distance_score = []

    for match in matches:        
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt

        line = compute_line(pt1, pt2)
 
        a, b, c = line[:3]
        
        denominator = np.sqrt(a**2 + b**2)
        if denominator == 0:
            continue

        distance = abs(a * vanishing_point[0] + b * vanishing_point[1] + c) / denominator
        # 根据距离计算分数
        if distance == threshold_distance:
            match_score = 0.5
        else:
            # 差值对分数的影响，每小于或大于 max_distance 1，分数变化 0.1
            match_score = 0.5 + (threshold_distance - distance) * 0.001

        # 限制分数在0到1之间
        match_score = max(0, min(match_score, 1))
        
        # 添加分数到 score 列表
        distance_score.append(match_score)

    return distance_score