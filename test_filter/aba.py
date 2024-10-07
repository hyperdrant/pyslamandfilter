import numpy as np, array


data = [{'feature_id': 65, 'keypoints': [(280.0, 33.0), (266.0, 28.0), (251.0, 24.0)], 'descriptors': [np.array([  9, 180, 186, 113, 183, 175, 243,  87, 126,  11, 127,  84, 159,
       185,   8,  48, 179,  10,  66,  24, 147, 184,  17,  23, 237, 207,
        52,  64, 200,  66, 227, 131]), np.array([  9, 180, 154, 121, 225, 189, 177,  83, 126,  11, 105,  84, 158,
       178,  70,  56, 183,  43,  66,  24, 151, 248, 178,  83,  79, 143,
        50,  78, 200,  64, 235, 131]), np.array([  5, 180, 154,  89, 193, 188, 240,  91, 122,  27, 109,  92, 158,
       186,  76,  57, 191,  11, 114,  24, 151, 126, 178,  80,  79, 155,
        82, 110, 201,  64, 235, 135])], 'image_names': ['0000000000.png', '0000000001.png', '0000000002.png']}, {'feature_id': 136, 'keypoints': [(328.458, 149.299), (319.334, 149.299), (306.063, 151.788)], 'descriptors': [np.array([ 84,  37, 152, 182, 155, 104,  22, 143, 178, 179, 139,  25, 198,
       247,  73,  90, 209, 215, 233,  16,  26,  67, 203,  34, 249, 231,
        77,  63, 160, 115, 132,   2]), np.array([ 84, 129, 148, 182, 137,  96,  23, 141, 170, 183, 139,  17, 198,
       226,  72,  74, 193,  87, 232, 145,  30,  73, 203,  32, 249, 166,
       223,  93, 128, 115, 165,   2]), np.array([ 84,   5, 156, 182, 137,  96,  22, 141, 162, 179, 139,  25, 198,
       247,  73,  90, 193, 215, 232,  16,  24,  75, 203,  32, 249, 231,
        95, 127, 128, 115, 165,   2])], 'image_names': ['0000000000.png', '0000000001.png', '0000000002.png']}]

# 首先，针对每个特征，只保留最后两张图片的信息
for feature in data:
    num_images = len(feature['image_names'])
    if num_images > 2:
        # 只保留最后两项
        feature['image_names'] = feature['image_names'][-2:]
        feature['keypoints'] = feature['keypoints'][-2:]
        feature['descriptors'] = feature['descriptors'][-2:]

# 创建一个从图片名到特征列表的映射
image_to_keypoints = {}
image_to_descriptors = {}

for feature in data:
    image_names = feature['image_names']
    keypoints = feature['keypoints']
    descriptors = feature['descriptors']

    for idx, img_name in enumerate(image_names):
        kp = keypoints[idx]
        desc = descriptors[idx]

        if img_name not in image_to_keypoints:
            image_to_keypoints[img_name] = []
            image_to_descriptors[img_name] = []

        image_to_keypoints[img_name].append(kp)
        image_to_descriptors[img_name].append(desc)

# 定义参考帧和当前帧
ref_frame = '0000000001.png'
current_frame = '0000000002.png'

# 获取参考帧和当前帧中的关键点和描述子
ref_keypoints = image_to_keypoints.get(ref_frame, [])
ref_descriptors = image_to_descriptors.get(ref_frame, [])
current_keypoints = image_to_keypoints.get(current_frame, [])
current_descriptors = image_to_descriptors.get(current_frame, [])

# 合并参考帧和当前帧的特征到一个数据输出
combined_data = {
    'ref_frame': ref_frame,
    'current_frame': current_frame,
    'ref_features': [ref_keypoints, ref_descriptors],
    'current_features': [current_keypoints, current_descriptors]
}

# 输出最终的合并数据
print("合并后的数据输出：")
print(combined_data)