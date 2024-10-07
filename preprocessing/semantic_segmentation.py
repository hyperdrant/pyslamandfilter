### Imports ###
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn as nn
import torch
from PIL import Image
import cv2
import torch.nn.functional as F

### Load Models ### 
# Source: [4] https://huggingface.co/nvidia/segformer-b4-finetuned-cityscapes-1024-1024
feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")

### CPU / GPU Config ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
if device.type == 'cuda':
    print("Semantic Segmentation uses", torch.cuda.get_device_name(0))
model.to(device)

### City Scapes Class Labels ###
# Source: [5] https://www.cityscapes-dataset.com/
class_labels = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

### Image Segmentation ###
def segment_image(image_np):
    """
    Segments the input image using a NVIDIA Segformer model pretrained on Cityscapes.

    Arguments:
    image_np (ndarray): Input image in BGR format as a numpy array.

    Returns:
    pred_seg (ndarray): Predicted segmentation mask as a numpy array.
    conf_seg (ndarray): Confidence mask as a numpy array.
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

    # Calculate the softmax probabilities to obtain the confidence values
    probs = F.softmax(upsampled_logits, dim=1)
    conf_seg = probs.max(dim=1)[0].squeeze(0).detach().cpu().numpy()  # The highest probability per pixel is the confidence
    
    
    return pred_seg, conf_seg
