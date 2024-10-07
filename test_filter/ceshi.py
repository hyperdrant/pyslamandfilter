import numpy as np, array
import os
# 假设您的数据保存在名为 data_dict 的变量中
data_dict = {
    1319:{
  'Positions': [(697.0, 154.0), (695.0, 155.0), (693.0, 153.0), (690.0, 151.0)],
  'Descriptors': [np.array([ 63,  95, 118,  78,  53,  54, 222, 240,  85,  78, 255, 230, 107,
       104, 183, 181, 126, 224, 198, 238, 237, 163, 109, 238,  50, 255,
       128, 131, 127, 255, 106,  85]), np.array([ 63,  95, 118,  78,  53,  54,  94, 248, 117,  14, 255, 230, 107,
       104, 183, 181, 126, 226, 134, 238, 236, 167, 109, 239, 178, 255,
       128, 131, 103, 255, 110, 197]), np.array([ 63,  95, 118,  78, 127, 183, 222, 120, 101, 206, 253, 175, 106,
       108, 183, 181, 126, 226, 198, 238, 236, 163, 108, 239,  50, 253,
         0, 155, 127, 255, 110, 213]), np.array([ 63,  95, 118,  78, 119, 183, 222, 124, 101, 206, 253, 175, 107,
       108, 183, 181, 126, 226, 198, 238, 236, 163,  44, 239,  50, 255,
         0, 147, 111, 255, 110, 221])],
  'Semantic Class': [8, 8, 10, 10],
  'Confidence': [0.42, 0.45, 0.51, 0.52],
  'Depth': [4, 4, 3, 3],
  'Observability': 4,
  'Corner Score': [153.0, 152.0, 152.0, 156.0],
  'Standard Deviation': 18.7,
  'Contrast': [0, 0, 0, 0],
  'image_names': ['6','7', '9', '10'],
  'Robustness': [3.0, 4.0, 3.0, 2.0],
  'Entropy': [6.7521908292410835, 6.741145256380516, 6.620488105054848, 6.729058368484813],
  'Geometric': [1.0, 1.0, 1.0, 1.0]
},
1410:
  {'Positions': [(1045.095, 230.17), (1084.493, 236.39), (1132.186, 244.685), (1186.929, 251.32)],
  'Descriptors': [np.array([ 48, 129,  95, 239,  22,  66, 111, 121,  84,   0, 243, 226, 115,
       119, 168, 112, 122, 156,   6, 234,  73, 147,  45,  31, 177, 239,
       163, 132,  53, 170, 100,  33]), np.array([145, 157,  91, 238,  22, 194,  75, 248,  20,  64, 219, 226,  87,
        87, 168,  97, 118, 220,  22, 234, 105, 153,  57, 191, 240, 238,
       163,  68,  60, 139,  96,  57]), np.array([ 21, 149,  91, 233,  52, 135, 106, 252,  20,   9, 219,  70,  85,
       243, 168,  96, 112, 136,  30, 234, 121, 145,  25, 159, 184, 237,
       162, 196,  60, 138, 240,  49]), np.array([ 17, 149,  95, 232,  22, 198,  75, 248,  36,  72, 218,  70,  83,
        87, 184,  96, 112, 196,  30, 238, 105, 153,  25,  63, 241, 173,
       162, 196,  60, 139, 112,  56])],
  'Semantic Class': [8, 8, 8, 8],
  'Confidence': [0.92, 0.98, 0.96, 0.95],
  'Depth': [84, 92, 103, 110],
  'Observability': 4,
  'Corner Score': [50.0, 44.0, 38.0, 38.0],
  'Standard Deviation': 24.7,
  'Contrast': [0, 0, 0, 0],
  'image_names': ['7', '8', '9', '10'],
  'Robustness': [3.0, 2.0, 2.0, 3.0],
  'Entropy': [6.968842204976724, 6.89801088849259, 6.814729014208125, 6.899569331555351],
  'Geometric': [0.021457130985747974, 0.8991960143137552, 0.8030908975055184, 0.8015148459286826]}
}
    

# image_id = 10
# valid_image_names = [str(image_id - i) for i in range(2)]  # ['10', '9', '8', '7', '6']

# filtered_data = {}

image_id = 10

number_of_frames = 3
filtered_data = {}
def preserve_last_frames(data_dict,image_id,number_of_frames):
    valid_image_names = [str(image_id - i) for i in range(number_of_frames)]  # ['10', '9', '8', '7', '6']
    required_images = {str(image_id), str(image_id - 1)}  # {'10', '9'}
    for key, data_item in data_dict.items():
        image_names = data_item['image_names']
        # 检查是否包含 image_id 和 image_id - 1
        if required_images.issubset(image_names):
            # 创建保留的索引列表
            indices_to_keep = [i for i, name in enumerate(image_names) if name in valid_image_names]
            if indices_to_keep:
                # 过滤 image_names
                filtered_image_names = [data_item['image_names'][i] for i in indices_to_keep]
                # 过滤其他字段
                filtered_data_item = {}
                for field in data_item:
                    value = data_item[field]
                    if isinstance(value, list) and len(value) == len(image_names):
                        filtered_data_item[field] = [value[i] for i in indices_to_keep]
                    else:
                        filtered_data_item[field] = value
                filtered_data_item['image_names'] = filtered_image_names
                # 过滤后再次检查
                if required_images.issubset(filtered_data_item['image_names']):
                    filtered_data[key] = filtered_data_item
            else:
                continue
        else:
            continue
    return filtered_data
filtered = preserve_last_frames(data_dict,image_id,number_of_frames )
print(filtered)
# 现在，filtered_data 包含了按照要求筛选后的数据


