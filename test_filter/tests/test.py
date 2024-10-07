from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn as nn
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Source: [4] https://huggingface.co/nvidia/segformer-b4-finetuned-cityscapes-1024-1024
feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
if device.type == 'cuda':
    print("Semantic Segmentation uses", torch.cuda.get_device_name(0))
model.to(device)

# Source: [5] https://www.cityscapes-dataset.com/
class_labels = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

def segment_image(image_np):
    """
    Source: [4] https://huggingface.co/nvidia/segformer-b4-finetuned-cityscapes-1024-1024, [6] https://doi.org/10.48550/arXiv.2105.15203

    Segments the input image using a NVIDIA Segformer pretrained on Cityscapes.

    Arguments:
    image_np (ndarray): Input image in BGR format as a numpy array.

    Returns:
    pred_seg (ndarray): Predicted segmentation mask as a numpy array.
    class_labels (list): List of class labels corresponding to the segmentation mask.
    """

    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)) # Convert numpy array (BGR) to PIL Image (RGB)
    
    inputs = feature_extractor(images=image_pil, return_tensors="pt") # Preprocess the image

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # Upsample logits to the size of the original image
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=(image_pil.height, image_pil.width),
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy() # Get the predicted segmentation mask
    
    return pred_seg


def draw_bounding_boxes_and_extract_orb(image_np, segmentation_mask, class_id, color=(0, 255, 0)):
    """
    Draws bounding boxes around the specified class in the segmentation mask and extracts ORB features.

    Arguments:
    image_np (ndarray): The original image as a numpy array.
    segmentation_mask (ndarray): The segmentation mask as a numpy array.
    class_id (int): The class id to draw bounding boxes around.
    color (tuple): The color of the bounding boxes.

    Returns:
    image_with_boxes (ndarray): The image with bounding boxes drawn.
    keypoints_list (list): List of keypoints for each detected object.
    descriptors_list (list): List of descriptors for each detected object.
    """
    image_with_boxes = image_np.copy()
    contours, _ = cv2.findContours((segmentation_mask == class_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    orb = cv2.ORB_create(edgeThreshold = 10)
    keypoints_list = []
    descriptors_list = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), color, 2)
        
        roi = image_np[y:y+h, x:x+w]
        keypoints, descriptors = orb.detectAndCompute(roi, None)
        keypoints = [cv2.KeyPoint(kp.pt[0] + x, kp.pt[1] + y, kp.size) for kp in keypoints]
        
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    
    return image_with_boxes, keypoints_list, descriptors_list

def plot_image_with_orb_features(image_path):
    """
    Plots the image with bounding boxes around traffic signs and extracts ORB features.

    Arguments:
    image_path (str): Path to the input image.
    """
    image_np = cv2.imread(image_path)
    segmentation_mask = segment_image(image_np)
    
    traffic_sign_class_id = class_labels.index("traffic sign")
    image_with_boxes, keypoints_list, descriptors_list = draw_bounding_boxes_and_extract_orb(image_np, segmentation_mask, traffic_sign_class_id)
    
    for keypoints in keypoints_list:
        image_with_boxes = cv2.drawKeypoints(image_with_boxes, keypoints, None, color=(255, 0, 0), flags=0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


    # Replace 'path_to_image.jpg' with the actual image path
plot_image_with_orb_features('/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/data/input_images/test/office_spring_winter/4_1.png')