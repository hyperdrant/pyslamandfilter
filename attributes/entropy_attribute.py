### Imports ###
import cv2
from matplotlib import pyplot as plt
import numpy as np
from python_orb_slam3 import ORBExtractor


### Extropy Attribute Extraction ###
def estimate_entropy(keypoints, img, patch_size=16):

    feature_entropy = []

    for kp in keypoints: 
      
        x, y = int(kp.pt[0]), int(kp.pt[1])
        half_size = patch_size // 2
        patch = img[max(0, y-half_size):y+half_size, max(0, x-half_size):x+half_size] # extract local pixels inside the patch

        if patch.size == 0:
            return 0

        hist, _ = np.histogram(patch, bins=256, range=(0, 256))
        hist = hist / hist.sum()

        entropy = -np.sum([p * np.log2(p) for p in hist if p > 0]) # Calculate feature local entropy using histo diagram

        feature_entropy.append((kp, entropy))

    return feature_entropy




