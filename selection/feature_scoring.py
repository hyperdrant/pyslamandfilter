### Imports ### 
import numpy as np
from collections import defaultdict

### Parameters ###
weights = { # Weights of the scoring function -> Need to be adapted
    "corner_score": 0.7,
    "semantic_class": 1.5,  
    "depth": 0.5,
    "entropy": 0.4,
    "geometric": 0.3,
    "vanishing_point": 0.5,
    "displacement": 0.6,
    "angle": 0.6,
    "observability": 1.4,
    "std_deviation": 0.6,
    "contrast": 0.3,
    "robustness": 0.4
    }

class_scores = { # Weights of the different object classes -> Need to be adapted
    0: 0.5,  # road
    1: 0.6,  # sidewalk
    2: 0.9,  # building
    3: 0.8,  # wall
    4: 0.6,  # fence
    5: 0.7,  # pole
    6: 0.8,  # traffic light
    7: 1.0,  # traffic sign
    8: 0.2,  # vegetation
    9: 0.3,  # terrain
    10: 0.0,  # sky
    11: 0.0,  # person
    12: 0.0,  # rider
    13: 0.0,  # car
    14: 0.0,  # truck
    15: 0.0,  # bus
    16: 0.0,  # train
    17: 0.0,  # motorcycle
    18: 0.0,  # bicycle
    }

contrast_weights = {0: 1.0, 1: 0.5, 2: 0.5} # Weights for the contrast (0: Normal, 1: Bright, 2: Dimmed)

### Corner Score Average ###
def calc_corner_score_average(scores):
    """
    Source: /

    Calculates the average corner score from a list of corner scores.

    Parameters:
    - scores (list): A list of numerical corner scores.

    Returns:
    - float: The average corner score if the list is not empty (-> otherwise: 0).
    """

    return np.mean(scores) if scores else 0

### Depth Average Calculation ###
def calc_depth_average(depths):
    """
    Source: /

    Calculates the average depth from a list of depths.

    Parameters:
    - scores (list): A list of numerical depths.

    Returns:
    - float: The average depth if the list is not empty (-> otherwise: 0).
    """

    return np.mean(depths) if depths else 0

### Entropy Average Calculation ###
def calc_entropy_average(entropies):
    """
    Source: /

    Calculates the average entropy from a list of entropies.

    Parameters:
    - scores (list): A list of numerical entropies.

    Returns:
    - float: The average entropy if the list is not empty (-> otherwise: 0).
    """

    return np.mean(entropies) if entropies else 0

### Geometric score Average Calculation ###
def calc_geometric_average(geometrics):
    """
    Source: /

    Calculates the average geometric score from a list of geometric scores.

    Parameters:
    - scores (list): A list of numerical geometric scores.

    Returns:
    - float: The average geometric score if the list is not empty (-> otherwise: 0).
    """

    return np.mean(geometrics) if geometrics else 0

### Object Class Calculation ###
def calc_object_class(semantic_classes, confidences):
    """
    Determines the most frequent semantic class using a confidence-weighted sum.
    
    Parameters:
    - semantic_classes (list): A list of semantic class labels.
    - confidences (list): A list of confidence values corresponding to the semantic classes.

    Returns:
    - weighted_class_score (float): The final weighted score for the assigned class.
    """
    weighted_sums = defaultdict(float)
    
    # Calculate the sum of confidences for each class, rounding to two decimal places
    for cls, conf in zip(semantic_classes, confidences):
        weighted_sums[cls] += conf

    
    # Determine the class with the highest confidence-weighted sum
    assigned_class = max(weighted_sums, key=weighted_sums.get)
    
    # Calculate the average confidence for the assigned class, rounding to two decimal places
    assigned_confidences = [round(conf, 2) for cls, conf in zip(semantic_classes, confidences) if cls == assigned_class]
    average_confidence = round(sum(assigned_confidences) / len(assigned_confidences), 2)
    
    # Adjust class score if average confidence is less than 0.8
    adjusted_class_score = class_scores.get(assigned_class, 0.0)
    if average_confidence < 0.9:
        adjusted_class_score *= 0.5
    
    # Calculate the final weight using the (possibly adjusted) class weight, rounding to two decimal places
    weighted_class_score = round(average_confidence * adjusted_class_score, 2)
    
    return weighted_class_score


### Depth Normalization ###
def normalize_depth(depth_avg, min_depth=0, max_depth=255): # Sepearate since value range is fixed (other attributes: range dynamic)
    """
    Source: / 

    Normalizes a depth value to a range between 0 and 1 based on minimum (0) and maximum (255) depth values.

    Parameters:
    - depth_avg (float): The average depth value to be normalized.
    - min_depth (float): The minimum possible depth value (default is 0).
    - max_depth (float): The maximum possible depth value (default is 255).

    Returns:
    - float: The normalized depth value if max_depth is greater than min_depth; otherwise, returns the original depth value.
    """

    return (depth_avg - min_depth) / (max_depth - min_depth) if max_depth > min_depth else depth_avg

### Entropy Normalization ###
def normalize_entropy(entropy_avg, min_entropy=0, max_entropy=8): # Sepearate since value range is fixed (other attributes: range dynamic)
    """
    Source: / 

    Normalizes a entropy value to a range between 0 and 8 based on minimum (0) and maximum (8) entropy values.

    Parameters:
    - entropy_avg (float): The average entropy value to be normalized.
    - min_entropy (float): The minimum possible entropy value (default is 0).
    - max_entropy (float): The maximum possible entropy value (default is 8).

    Returns:
    - float: The normalized entropy value if max_entropy is greater than min_entropy; otherwise, returns the original entropy value.
    """

    return (entropy_avg - min_entropy) / (max_entropy - min_entropy) if max_entropy > min_entropy else entropy_avg

### Displacement Normalization ###
def normalize_displacement(displacement_all): # Sepearate since value range is fixed (other attributes: range dynamic)
    """
    Source: / 

    Normalizes a displacement value to a range between 0 and 1 based on minimum (0) and maximum (1) displacement values.

    Parameters:
    - displacement_avg (float): The average displacement value to be normalized.

    Returns:
    - float: The normalized displacement value.
    """
    displacement_avg = np.mean(displacement_all)
    if displacement_avg > 1:
        normalized_displacement = 0
    else: 
        normalized_displacement = 1 - displacement_avg

    return normalized_displacement

### Angle Normalization ###
def normalize_angle(angle_all, min_angle=0, max_angle=np.pi): # Sepearate since value range is fixed (other attributes: range dynamic)
    """
    Source: / 

    Normalizes a angle value to a range between 0 and 1 based on minimum (0) and maximum (pi) angle values.

    Parameters:
    - angle_avg (float): The average angle value to be normalized.
    - min_angle (float): The minimum possible angle value (default is 0).
    - max_angle (float): The maximum possible angle value (default is 8).

    Returns:
    - float: The normalized angle value if max_angle is greater than min_angle; otherwise, returns the original angle value.
    """
    angle_avg = np.mean(angle_all)
    return (angle_avg - min_angle) / (max_angle - min_angle) if max_angle > min_angle else angle_avg

### Vanishing Point Score Normalization ###
def normalize_vanishing_point(vanishing_point_score, min_score=0, max_score=1): # Sepearate since value range is fixed (other attributes: range dynamic)
    """
    Source: / 

    Normalizes a vanishing point score value to a range between 0 and 1 based on minimum (0) and maximum (1) vanishig point score values.

    Parameters:
    - vanishing_point_score (float): The average vanishing point score value to be normalized.
    - min_score (float): The minimum possible vanishig point score value (default is 0).
    - max_score (float): The maximum possible vanishig point score value (default is 8).

    Returns:
    - float: The normalized vanishing point score value if max_score is greater than min_score; otherwise, returns the original vanishig point score nvalue.
    """
    vanishing_point_score_avg = np.mean(vanishing_point_score)
    return (vanishing_point_score_avg - min_score) / (max_score - min_score) if max_score > min_score else vanishing_point_score_avg

### General Normalization ###
def normalize_min_max(values): # Used for all attributes except depth
    """
    Source: /

    Normalizes a list of values to a range between 0 and 1 using min-max normalization.

    Parameters:
    - values (list): A list of numerical values to be normalized.

    Returns:
    - list: A list of normalized values.
    """

    min_val = np.min(values)
    max_val = np.max(values)
    return [(val - min_val) / (max_val - min_val) if max_val > min_val else val for val in values]

### Contrast Score Calculation ###
def calc_contrast_score(contrasts):
    """
    Source: / 

    Calculates a weighted average contrast score from a list of contrast values.

    Parameters:
    - contrasts (list): A list of contrast values (int).

    Returns:
    - float: The weighted average contrast score if the list is not empty (-> otherwise: 0).
    """

    total_weight = sum(contrast_weights[c] for c in contrasts)
    return total_weight / len(contrasts) if contrasts else 0

### Feature Score Calculation ###
def calculate_feature_score(normalized_observability, normalized_corner_score, normalized_std, normalized_contrast_score, normalized_robustness_score, normalized_class_score, normalized_depth, normalized_entropy, normalized_geometric, normalized_displacement, normalized_angle, vanishing_point):
    """
    Source: / 

    Calculates the overall feature score for a given feature based on normalized attributes.

    Parameters:
    - normalized_observability (float): Normalized observability score.
    - normalized_corner_score (float): Normalized corner score.
    - normalized_std (float): Normalized standard deviation of descriptors.
    - normalized_contrast_score (float): Normalized contrast score.
    - normalized_robustness_score (float): Normalized robustness score.
    - normalized_class_score (float): Normalized class score.
    - normalized_depth (float): Normalized depth value.
    - normalized_entropy (float): Normalized entropy value.
    - normalized_geometric (float): Normalized geometric value.
    - normalized_displacement (float): Normalized displacement value.
    - normalized_angle (float): Normalized angle value.
    - vanishing_point (float): Vanishing_point value.
    Returns:
    - dict: A dictionary with the total calculated scores and the individual attribute scores.
    """

    # Total score = sum of indivdiual attribute scores mutliplied by their weights
    total_score = (
        weights["corner_score"] * normalized_corner_score +          # Weight for corner attribute score
        weights["semantic_class"] * normalized_class_score +         # Weight for semantic attribute score
        weights["depth"] * normalized_depth +                        # Weight for depth attribute score
        weights["entropy"] * normalized_entropy +                    # Weight for entropy attribute score
        weights["geometric"] * normalized_geometric +                # Weight for geometric attribute score
        weights["vanishing_point"] * vanishing_point +               # Weight for vanishing point attribute score
        weights["displacement"] * normalized_displacement +          # Weight for displacement attribute score
        weights["angle"] * normalized_angle +                        # Weight for angle attribute score
        weights["observability"] * normalized_observability +        # Weight for observability attribute score
        weights["contrast"] * normalized_contrast_score +            # Weight for contrast attribute score
        weights["robustness"] * normalized_robustness_score          # Weight for robustness attribute score
    )

    # Add std attribute score in the total score if it exists (more than one descriptor)
    if normalized_std is not None:
        total_score += weights["std_deviation"] * (1 - normalized_std)  # Lower std deviation should result in higher score

    # Return a dictionary containing the total score and individual normalized attribute scores (rounded to three decimal places)
    return {
        "total_feature_score": round(total_score, 3),
        "normalized_corner_score": round(normalized_corner_score, 3),
        "normalized_class_score": round(normalized_class_score, 3),
        "normalized_depth": round(normalized_depth, 3),
        "normalized_entropy": round(normalized_depth, 3),
        "normalized_geometric": round(normalized_geometric, 3),
        "normalized_vanishing_point_score": round(vanishing_point, 3),
        "normalized_displacement": round(normalized_displacement, 3),
        "normalized_angle": round(normalized_angle, 3),
        "normalized_observability": round(normalized_observability, 3),
        "normalized_std": round(normalized_std, 3) if normalized_std is not None else None,
        "normalized_contrast_score": round(normalized_contrast_score, 3),
        "normalized_robustness_score": round(normalized_robustness_score, 3)
    }

### General Score Calculation ###
def calculate_scores(feature_dict):
    """
    Source: / 

    Calculates the scores for all features in the feature dictionary.

    Parameters:
    - feature_dict (dict): A dictionary (keys are feature IDs), values are dictionaries containing feature attributes.

    Returns:
    - list: A list of tuples where each tuple contains a feature ID and its score details, sorted by the total feature score in descending order.
    """
    
    # print("Calculating Feature Scores...")
    scores = [] # Initialize empty list to store scores
    
    # Normalize corner attribute score
    all_corner_scores_values = [calc_corner_score_average(feature["corner_scores"]) for feature in feature_dict.values()]
    normalized_corner_scores_values = normalize_min_max(all_corner_scores_values)
    
    # Normalize observability attribute score
    all_observability_values = [feature["observability"] for feature in feature_dict.values()]
    normalized_observability_values = normalize_min_max(all_observability_values)
    
    # Normalize std attribute score
    all_std_values = [np.mean(feature["std"]) if feature["std"] else None for feature in feature_dict.values()]
    all_std_values = [val for val in all_std_values if val is not None]  # Remove None values
    normalized_std_values = normalize_min_max(all_std_values)

    # Normalize contrast attribute score
    all_contrast_values = [calc_contrast_score(feature["contrast"]) for feature in feature_dict.values()]
    normalized_contrast_values = normalize_min_max(all_contrast_values)

    # Normalize robustness attribute score
    all_robustness_values = [np.mean(feature["robustness"]) if feature["robustness"] else 0 for feature in feature_dict.values()]
    normalized_robustness_values = normalize_min_max(all_robustness_values)

    # Iterate over all features in the storage
    for i, (feature_id, feature) in enumerate(feature_dict.items()):
        # Normalized scores (current feature)
        normalized_corner_score = normalized_corner_scores_values[i]
        normalized_observability = normalized_observability_values[i]
        normalized_contrast_score = normalized_contrast_values[i]
        normalized_robustness_score = normalized_robustness_values[i]

        # Normalized std attribute score if it exists (more than one descriptor)
        if feature["std"]:
            normalized_std = normalized_std_values.pop(0)
        else:
            normalized_std = None

        # Rest of the attribute scores -> In contrast to the other attributes, these attributes of the individual feature do not depend on the other features in the storage (normalisation) 
        normalized_class_score = calc_object_class(feature["semantic_class"], feature["confidence"]) # Determine object class + weight
        depth_avg = calc_depth_average(feature["depth"])
        entropy_avg = calc_entropy_average(feature["entropy"])
        normalized_depth_score = normalize_depth(depth_avg)
        normalized_entropy_score = normalize_entropy(entropy_avg)
        geometric_avg = calc_geometric_average(feature["geometric"])
        normalized_displacement = normalize_displacement(feature["displacement"])
        normalized_angle = normalize_angle(feature["angle"])
        normalized_vanishing_point = normalize_vanishing_point(feature["vanishing_point"])
        
        # Calculate the total feature score
        score_details = calculate_feature_score(
            normalized_observability, normalized_corner_score, normalized_std, 
            normalized_contrast_score, normalized_robustness_score, normalized_class_score, normalized_depth_score, normalized_entropy_score, 
            geometric_avg, normalized_displacement, normalized_angle, normalized_vanishing_point
        )
        # Append feature_id and scores
        scores.append((feature_id, score_details))

    # Sort scores in descending order (based on: total_feature_score)
    scores = sorted(scores, key=lambda x: x[1]["total_feature_score"], reverse=True)
    
    return scores, weights, class_scores
