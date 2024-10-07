def observability_attribute(feature_dict):
    """
    Source: / 

    Calculates the observability attribute for each feature in the feature dictionary.

    Parameters:
    - feature_dict (dict): A dictionary where keys are feature IDs and values are dictionaries containing feature attributes.

    Returns:
    - feature_dict (dict): The updated feature dictionary with the observability attribute added for each feature.
    """
    
    for feature_id, attributes in feature_dict.items():
        observability_count = len(attributes["contrast"]) # Based on the number of the attribute "contrast" of each feature (could use other attributes too)
        feature_dict[feature_id]["observability"] = observability_count 
    return feature_dict