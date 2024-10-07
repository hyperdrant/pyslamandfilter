import os
import cv2
import matplotlib.pyplot as plt

def load_feature_storage(txt_path):
    feature_storage = {}
    with open(txt_path, "r") as f:
        lines = f.readlines()
        current_feature_id = None
        for line in lines:
            line = line.strip()
            if line.startswith("Feature ID:"):
                current_feature_id = int(line.split(":")[1].strip())
                feature_storage[current_feature_id] = {
                    "positions": [],
                    "image_names": []
                }
            elif current_feature_id is not None:
                if ":" in line:
                    attr_name, attr_values = line.split(":")
                    attr_name = attr_name.strip()
                    if attr_name in ["positions", "image_names"]:
                        feature_storage[current_feature_id][attr_name] = eval(attr_values.strip())
    return feature_storage

def highlight_feature_in_images(feature_storage, feature_id, images_dir):
    if feature_id not in feature_storage:
        print(f"Feature ID {feature_id} not found in the feature storage.")
        return

    feature_data = feature_storage[feature_id]
    image_names = feature_data["image_names"]
    positions = feature_data["positions"]

    for image_name, position in zip(image_names, positions):
        img_path = os.path.join(images_dir, image_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Could not read image: {img_path}")
            continue

        # Draw the keypoint position
        x, y = int(position[0]), int(position[1])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red color for the feature

        # Display the image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Feature ID: {feature_id} in {image_name}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Beispielaufruf
    feature_storage_path = "/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/output_features/features.txt"
    images_dir = "/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/input_images/erding_3_cropped"
    feature_id = 1237713  # Beispiel Feature ID


    feature_storage = load_feature_storage(feature_storage_path)
    highlight_feature_in_images(feature_storage, feature_id, images_dir)

