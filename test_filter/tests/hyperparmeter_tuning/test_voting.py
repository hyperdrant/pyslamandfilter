from collections import defaultdict

def calc_object_class(semantic_classes, confidences):
    """
    Determines the most frequent semantic class using a confidence-weighted sum.
    
    Parameters:
    - semantic_classes (list): A list of semantic class labels.
    - confidences (list): A list of confidence values corresponding to the semantic classes.

    Returns:
    - assigned_class (int): The class with the highest confidence-weighted sum.
    - weighted_class_score (float): The final weighted score for the assigned class.
    """
    weighted_sums = defaultdict(float)
    
    # Calculate the sum of confidences for each class, rounding to two decimal places
    for cls, conf in zip(semantic_classes, confidences):
        weighted_sums[cls] += conf
        print(f"Adding confidence {round(conf, 2)} to class {cls}: Current sum = {round(weighted_sums[cls], 2)}")
    
    # Print the weighted sums for each class, rounded to two decimal places
    print("Weighted sums:", {k: round(v, 2) for k, v in weighted_sums.items()})
    
    # Determine the class with the highest confidence-weighted sum
    assigned_class = max(weighted_sums, key=weighted_sums.get)
    print(f"Assigned class based on highest sum: {assigned_class}")
    
    # Calculate the average confidence for the assigned class, rounding to two decimal places
    assigned_confidences = [round(conf, 2) for cls, conf in zip(semantic_classes, confidences) if cls == assigned_class]
    average_confidence = round(sum(assigned_confidences) / len(assigned_confidences), 2)
    print(f"Average confidence for assigned class {assigned_class}: {average_confidence}")
    
    # Adjust class score if average confidence is less than 0.8
    adjusted_class_score = class_scores.get(assigned_class, 0.0)
    if average_confidence < 0.8:
        adjusted_class_score *= 0.5
        print(f"Class score for class {assigned_class} reduced by 50% due to low confidence: New class score = {adjusted_class_score}")
    
    # Calculate the final weight using the (possibly adjusted) class weight, rounding to two decimal places
    weighted_class_score = round(average_confidence * adjusted_class_score, 2)
    print(f"Final weighted class score: {weighted_class_score}")
    
    return assigned_class, weighted_class_score

# Define class scores
class_scores = {
    0: 0.5,  # road
    1: 0.6,  # sidewalk
    2: 0.9,  # building
    3: 0.8,  # wall
    4: 0.6,  # fence
    5: 0.7,  # pole
    6: 0.8,  # traffic light
    7: 0.9,  # traffic sign
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

# Test Case Example
semantic_classes = [2, 8]  # Example classes
confidences = [0.5, 0.4]  # Example confidences

print("Test Case:")
result_class, final_weight = calc_object_class(semantic_classes, confidences)
print(f"Result: Assigned class = {result_class}, Final weight = {final_weight}")