# This file discovers all the pictures in the dataset folder that checks
# validity of each picture and also parses each picture's name to extract
# shape label and texture label. Filters out any invalid picture out of the data passed to models

import torchvision.transforms as tranforms
import os
import re
from PIL import Image
import numpy as np

# constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = (224, 224)

model_transform = tranforms.Compose([
    tranforms.Resize(IMAGE_SIZE),
    tranforms.ToTensor(),
    tranforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

visual_transform = tranforms.Compose([
    tranforms.Resize(IMAGE_SIZE),
    tranforms.ToTensor(),
])


def parse_file_name(filepath: str) -> tuple:
    """
    Extract shape and texture labels from a cue-conflict filename.
 
    Filename format: {shape_label}{digits}-{texture_label}{digits}.png
    Example: "cat3-elephant2.png" → ("cat", "elephant")
    """

    name = os.path.splitext(os.path.basename(filepath))[0]  # this stems the extension
    match = re.fullmatch(r"([a-z]+)[0-9]-([a-z]+)[0-9]", name)
    if not match:
        return None, None
    return match.groups()[0], match.groups()[1]

# # tests
# test_path = "data/texture-vs-shape/stimuli/style-transfer-preprocessed-512/airplane/airplane1-bicycle2.png"
# print(os.path.abspath(test_path))
# shape, texture = parse_file_name(test_path)
# print(f"{shape} \n{texture}")


def discover_dataset(dataset_root: str, categories: list) -> list:
    """
    Walk the dataset directory and return all valid cue-conflict images.
    An image is valid if:
      1. Its filename parses correctly into two labels
      2. Both labels are in the 16 experimental categories
      3. The two labels are different (same = not a cue-conflict)
      4. The parent folder name matches the shape label (sanity check)
 
    Returns a list of dicts, each with:
      'path'          : full file path
      'filename'      : base filename (e.g. "cat3-elephant2.png")
      'shape_label'   : e.g. "cat"
      'texture_label' : e.g. "elephant"
    """

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"\n Dataset root not found at {dataset_root}"
        )

    cats = set(categories)
    valid = []
    skipped = 0

    for shape_folder in sorted(os.listdir(dataset_root)):
        folder_path = os.path.join(dataset_root, shape_folder)  # dataset-root/airplane

        if not os.path.isdir(folder_path):
            continue

        if not shape_folder in cats:
            continue

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(".png"):
                continue

            file_path = os.path.join(folder_path, file_name)
            shape_label, texture_label = parse_file_name(filepath=file_path)
            
            if (
                shape_label is None or
                shape_label not in cats or
                texture_label not in cats or
                shape_label == texture_label or
                shape_folder != shape_label
            ):
                skipped += 1
                continue

            valid.append({
                "path": file_path,
                "name": file_name,
                "shape_label": shape_label,
                "texture_label": texture_label,
            })

    print(f"  [dataset] Found {len(valid)} valid images  ({skipped} skipped)")
    
    if len(valid) == 0:
        raise ValueError("No valid images found inside dataset root")
    
    return valid
            
# # tests
# from mapping import CATEGORY_TO_SYNSETS
# test_path = "data/texture-vs-shape/stimuli/style-transfer-preprocessed-512"
# categories = CATEGORY_TO_SYNSETS.keys()
# valid_images = discover_dataset(dataset_root=test_path, categories=categories)
# print(valid_images[-1])


def load_image(path: str) -> tuple:
    """
    Load one image and return two versions of it.
 
    Returns:
        input_tensor : (1, 3, 224, 224) normalized tensor — for model input
        vis_array    : (224, 224, 3) float32 in [0, 1] — for visualization
    """

    image = Image.open(path).convert("RGB")

    input_tensor = model_transform(image).unsqueeze(0)

    visual_array = visual_transform(image).permute(1, 2, 0).numpy().astype(np.float32)

    return input_tensor, visual_array

# # test
# print(load_image(valid_images[-1]["path"]))