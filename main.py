# This file only contains:
#   - CONFIG
#   - Setup calls (mapping, dataset, models)
#   - The main processing loop
#   - Analysis and visualization calls
#   - The final report


import os
import warnings
warnings.filterwarnings('ignore')
 
import numpy as np
import pandas as pd
import torch

from utils import download_imagenet_index, build_category_to_indicies, CATEGORY_TO_SYNSETS

CONFIG = {
    # Dataset
    "dataset_root": os.path.join(
        "data", "texture-vs-shape", "stimuli", "style-transfer-preprocessed-512"
    ),
 
    # Outputs
    "output_dir":  os.path.join("results"),
    "figures_dir": os.path.join("results", "figures"),
    "data_dir":    "cache",
 
    # ImageNet class index (downloaded once, cached locally)
    "imagenet_index_url": (
        "https://s3.amazonaws.com/deep-learning-models/"
        "image-models/imagenet_class_index.json"
    ),
 
    # Models — must be keys in models/model_loader.py MODEL_REGISTRY
    "models": ["vgg16", "resnet50"],
 
    # None = process every image (~1200). Set e.g. 100 for a quick test.
    "max_images":  None,
    "random_seed": 42,
 
    # Baselines from Geirhos et al, Table 1.
    # Used in Fig 1. If your numbers are within ~0.02 of these, the pipeline is correct.
    "human_shape_bias": 0.960,
    "published_baselines": {
        "vgg16":    0.208,
        "resnet50": 0.221,
    },
 
    "device": "cuda" if torch.cuda.is_available() else "cpu",   
}

print("=" * 60)
print("  TEXTURE VS. SHAPE BIAS IN CNNs")
print("=" * 60)
print(f"\n  Device   : {CONFIG['device']}")
print(f"  Models   : {CONFIG['models']}")
print(f"  Images   : {CONFIG['max_images'] or 'all'}")  # if max_images is set to None it choose all as in all images


# =======================================================================
# SETUP -> 1- construct model output index to category map -> 2- fetch all images into a variable
# -> 3- cap number of images to CONFIG['max_images'] -> 4- load models
# =======================================================================

cache_path = os.path.join(CONFIG["data_dir"], "cache.json")
synset_to_index, index_to_name = download_imagenet_index(CONFIG["imagenet_index_url"], cache_path)
category_to_indicies, index_to_category = build_category_to_indicies(CATEGORY_TO_SYNSETS, synset_to_index)  
#-> ( {"cat": {281, 289,}, ...}, {281: "cat", "289": "cat", ...} )

# 2 -> 4 in progress