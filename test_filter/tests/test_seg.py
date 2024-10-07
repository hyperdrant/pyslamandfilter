import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as F

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

### Laden des Processors und Modells für die semantische Segmentierung ###
feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")

### CPU / GPU Konfiguration ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Verwende GPU, wenn verfügbar
if device.type == 'cuda':
    print("Semantic Segmentation uses", torch.cuda.get_device_name(0))
model.to(device)

### City Scapes Klassenlabels ###
class_labels = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

### Bildsegmentierung ###
def segment_image(image_np):
    """
    Segmente das Eingabebild mithilfe eines vortrainierten NVIDIA Segformer-Modells auf Cityscapes.

    Argumente:
    image_np (ndarray): Eingabebild im BGR-Format als numpy-Array.

    Rückgabe:
    pred_seg (ndarray): Vorhergesagte Segmentierungsmaske als numpy-Array.
    confidence (ndarray): Confidence-Maske als numpy-Array.
    """

    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))  # Konvertiere numpy-Array (BGR) in PIL-Bild (RGB)
    
    inputs = feature_extractor(images=image_pil, return_tensors="pt")  # Bereite das Bild für das Modell vor

    # Verschiebe Eingaben auf das gleiche Gerät wie das Modell
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inferenz
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # Hochskalieren der Logits auf die Größe des Originalbilds
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=(image_pil.height, image_pil.width),
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()  # Hole die vorhergesagte Segmentierungsmaske

    # Berechne die Softmax-Wahrscheinlichkeiten, um die Confidence-Werte zu erhalten
    probs = F.softmax(upsampled_logits, dim=1)
    confidence = probs.max(dim=1)[0].detach().cpu().numpy()  # Die höchste Wahrscheinlichkeit pro Pixel ist die Confidence
    print(f"conf_seg shape: {confidence.shape}")
    return pred_seg, confidence

def display_image_with_segmentation(image, seg_mask, confidence_mask, class_labels):
    # Konvertiere das Bild in RGB (OpenCV lädt Bilder im BGR-Format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Erstelle eine Farbpalette für die Visualisierung
    cmap = plt.get_cmap('tab20', len(class_labels))
    colors = cmap(np.arange(len(class_labels)))
    segmentation_color = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)

    for label in range(len(class_labels)):
        segmentation_color[seg_mask == label] = (colors[label][:3] * 255).astype(np.uint8)

    # Squeeze die Confidence-Maske, um überflüssige Dimensionen zu entfernen
    confidence_mask = np.squeeze(confidence_mask)

    # Erstelle ein Diagramm mit zwei Unterplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Überlagere die Segmentierungsmaske auf dem Originalbild
    ax[0].imshow(image_rgb)
    ax[0].imshow(segmentation_color, alpha=0.5)  # Setze alpha für Transparenz
    ax[0].set_title("Segmentation Overlay")
    ax[0].axis('off')

    # Zeige die Confidence-Maske an
    conf_plot = ax[1].imshow(confidence_mask, cmap='viridis', interpolation='nearest')
    ax[1].set_title("Confidence Map")
    ax[1].axis('off')
    plt.colorbar(conf_plot, ax=ax[1], fraction=0.046, pad=0.04)

    # Erstelle die Legende für die Segmentierung
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i][:3], markersize=10, label=class_labels[i]) for i in range(len(class_labels))]
    ax[0].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
    plt.pause(0.1)  # Pause für 0,1 Sekunden

    plt.close(fig)  # Schließe die Figur, um Speicher freizugeben

def main():
    input_dir = "/Users/alexanderwitt/Desktop/Semesterarbeit/Feature_Filter/img"
    images = sorted(os.listdir(input_dir))

    for image_name in images:
        img_path = os.path.join(input_dir, image_name)

        # Lade das Bild
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot open or read the image file at {img_path}")

        # Führe die semantische Segmentierung durch
        seg_mask, confidence_mask = segment_image(image)

        # Zeige das Bild mit der Segmentierungs- und Confidence-Überlagerung an
        display_image_with_segmentation(image, seg_mask, confidence_mask, class_labels)

if __name__ == "__main__":
    main()