### Parameters ###
percentage = 10  # The percentage of top features to select (e.g., 1% of the total unique features)

### Feature Selection ###
def select_features(feature_scores, feature_dict, image_name):
    """
    Source: /

    Selects the top features based on their scores and collects their details. The number of features
    selected is a percentage of the total number of features in the feature_dict.

    Parameters:
    - feature_scores (list): A list of tuples where each tuple contains a feature ID and its score details.
    - feature_dict (dict): A dictionary (keys = feature ID)
    - percentage (float): The percentage of top features to select.

    Returns:
    - selected_feature_details (list): A list of dictionaries, each containing details of a selected feature.
    """
    image_id_str = image_name.split('.')[0]  # 提取文件名中的数字部分
    image_id = int(image_id_str)  # 将数字部分转换为整数

    original_length = len(image_id_str)  # 获取数字部分的长度

    current_frame = image_name
    ref_frame = f"{str(image_id - 1).zfill(original_length)}.png"

    # print("Selecting Features...")
    # Calculate the number of features to select
    total_features = len(feature_dict)
    N = max(1, int(total_features * (percentage / 100)))  # Ensure at least one feature is selected

    # Select the top N features
    selected_features = feature_scores[:N]

    # Collect the feature details
    selected_feature_details = []
    for feature_id, score_details in selected_features:
        feature = feature_dict[feature_id]
        keypoints = feature["positions"]
        image_names = feature["image_names"]
        descriptors = feature["descriptors"]
        selected_feature_details.append({
            "feature_id": feature_id,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "image_names": image_names
        })


    # 首先，针对每个特征，只保留最后两张图片的信息
    for feature in selected_feature_details:
        num_images = len(feature['image_names'])
        if num_images > 2:
            # 只保留最后两项
            feature['image_names'] = feature['image_names'][-2:]
            feature['keypoints'] = feature['keypoints'][-2:]
            feature['descriptors'] = feature['descriptors'][-2:]

    # 创建一个从图片名到特征列表的映射
    image_to_keypoints = {}
    image_to_descriptors = {}

    for feature in selected_feature_details:
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

    # # 定义参考帧和当前帧
    # ref_frame = '0000000001.png'
    # current_frame = '0000000002.png'

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
    return combined_data, percentage, N, selected_feature_details

