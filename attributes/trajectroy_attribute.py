### Imports ### 
import numpy as np

# def compute_displacements(trajectory):
#     displacements = []
#     for i in range(1, len(trajectory)):
#         displacement = np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1]))
#         displacements.append(displacement)
#     return displacements

def compute_displacements(trajectory):
    displacements = []
    for i in range(1, len(trajectory)):
        try:
            # 计算位移
            displacement = np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1]))
        except (FloatingPointError, ValueError, ZeroDivisionError):
            # 如果发生错误，位移设为100
            displacement = 100
        displacements.append(displacement)
    return displacements


def compute_relative_displacement_change(displacements):
    changes = []
    for i in range(1, len(displacements)):
        change = abs(displacements[i] - displacements[i-1])
        changes.append(change)
    return changes

# def compute_direction_changes(trajectory):
#     direction_changes = []
#     for i in range(2, len(trajectory)):
#         vec1 = np.array(trajectory[i-1]) - np.array(trajectory[i-2])
#         vec2 = np.array(trajectory[i]) - np.array(trajectory[i-1])
#         angle_change = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
#         direction_changes.append(angle_change)
#     return direction_changes

def compute_direction_changes(trajectory):
    direction_changes = []
    for i in range(2, len(trajectory)):
        vec1 = np.array(trajectory[i-1]) - np.array(trajectory[i-2])
        vec2 = np.array(trajectory[i]) - np.array(trajectory[i-1])
        
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            angle_change = 3  
        else:
            
            dot_product = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
            dot_product = np.clip(dot_product, -1.0, 1.0)  
            try:
                angle_change = np.arccos(dot_product)
            except ValueError:
                angle_change = 3  
        direction_changes.append(angle_change)
    
    return direction_changes


def calculate_trajectory_scores(data):

    for key, info in data.items():

        trajectory = info["positions"]
        displacements = compute_displacements(trajectory)
        avg_displacement = np.mean(displacements)

        if len(displacements) < 3:

            info['displacement'].append(0)
            info['angle'].append(0)

            continue  

        displacements_changes = compute_relative_displacement_change(displacements)
        # try:
        #     displacements_changes = np.mean(displacements_changes)
        #     displacements_score = displacements_changes / avg_displacement
        #     if np.isnan(displacements_score):
        #         displacements_score = 0
        # except:
        #     displacements_score = 0

        try:
           
            displacements_changes = np.mean(displacements_changes)
            
           
            if not np.isfinite(displacements_changes) or avg_displacement == 0:
                displacements_score = 100
            else:
                
                displacements_score = displacements_changes / avg_displacement
            
            
            if np.isnan(displacements_score):
                displacements_score = 100
        
        except Exception as e:
             
            displacements_score = 100

        direction_changes = compute_direction_changes(trajectory)
        try:
            direction_score = np.mean(direction_changes)
            if np.isnan(direction_score):
                direction_score = 0
        except:
            direction_score = 0
        

        info['displacement'].append(displacements_score)
        info['angle'].append(direction_score)

    return data









