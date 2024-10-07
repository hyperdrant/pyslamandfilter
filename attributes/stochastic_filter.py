### Imports ###
import numpy as np

### Stochastic Attribute Extraction ### 
def stochastic_filter(feature_dict):
    """
    Source: /

    Computes the standard deviation of descriptors for each feature in the feature dictionary and updates the feature attributes.

    Parameters:
    - feature_dict (dict): A dictionary where keys are feature IDs containing feature attributes.

    Returns:
    - feature_dict (dict): The updated feature dictionary with the standard deviation attribute added for each feature.
    """

    for feature_id, attributes in feature_dict.items():
        descriptors = attributes.get("descriptors", [])
        
        if len(descriptors) > 1: # Check if there is more than one descriptor to calculate std
            descriptors = np.array(descriptors) # Convert to numpy array to calculate std
            
            std_descriptor = np.std(descriptors, axis=0) # Calculate std of the descriptors for each "dimension" 
            
            avg_std_deviation = np.mean(std_descriptor) # Take average std 

            avg_std_deviation = round(avg_std_deviation, 1) # Round std to one decimal place
        else:
            avg_std_deviation = [] # If only one descriptor -> std can not be calculated -> empty []
        
        attributes["std"] = avg_std_deviation # Update the feature attributes with calculated std
    
    return feature_dict

