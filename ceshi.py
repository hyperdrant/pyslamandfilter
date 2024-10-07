import json
import numpy as np

file_path = 'frame_positions.json'

with open(file_path, 'r') as f:
    data = json.load(f)

positions_list = []


for image_name, kp_data in data.items():
    kps = kp_data['Descriptors']
    
    kps_output = np.array(kps)

    positions_list.append(kps_output)

print(positions_list[1])