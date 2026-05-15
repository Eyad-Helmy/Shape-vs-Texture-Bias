# # This file only contains:
# #   - CONFIG
# #   - Setup calls (mapping, dataset, models)
# #   - The main processing loop
# #   - Analysis and visualization calls
# #   - The final report


import os
import warnings
warnings.filterwarnings('ignore')
 
import numpy as np
import pandas as pd
import torch

from utils import (download_imagenet_index,
                    build_category_to_indicies, 
                    CATEGORY_TO_INDICES,
                    discover_dataset,
                    load_image)

from models import load_all_models, run_inference

from analysis import (
    classify_decision,
    compute_shape_bias,
    run_confidence_analysis,
    run_per_category_analysis,
    generate_all_figures,
    )

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
    "models": ["vgg16", "resnet50", "alexnet"],
 
    # None = process every image (~1200)
    "max_images":  None,
    "random_seed": 42,
 
    # Baselines from Geirhos et al, Table 1.
    # Used in Fig 1. If your numbers are within ~0.02 of these, the pipeline is correct.
    "human_shape_bias": 0.960,
    "published_baselines": {
        "vgg16":    0.208,
        "resnet50": 0.221,
        "alexnet": 0.429,
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

print("\n[Setup] Building ImageNet class mapping...")
cache_path = os.path.join(CONFIG["data_dir"], "cache.json")
synset_to_index, index_to_name = download_imagenet_index(CONFIG["imagenet_index_url"], cache_path)
category_to_indicies, index_to_category = build_category_to_indicies()  
#-> ( {"cat": {281, 289,}, ...}, {281: "cat", "289": "cat", ...} )

print("\n[Setup] Discovering Dataset...")
CATEGORIES = CATEGORY_TO_INDICES.keys()
all_images = discover_dataset(dataset_root=CONFIG["dataset_root"], categories=CATEGORIES)

# limiting images to set number
np.random.seed(42)
if CONFIG["max_images"] and len(all_images) > CONFIG["max_images"]:
    index = np.random.choice(len(all_images), CONFIG["max_images"], replace=False)
    all_images = [ all_images[i] for i in index ]
    print(f"    Capped to {CONFIG["max_images"]} images. ")

print("\n[Setup] Loading Modles...")
models = load_all_models(CONFIG["models"], CONFIG["device"])

# =======================================================================
# MAIN LOOP: 1- loop over each model -> 2- loop over every image resulted
# from dataset discovery -> 3- load each image into input tensor and visual tensors ->
# 4- run inference on each input tensor to get decision -> 5- classify decision ->
# 6- save all decision information into a dict -> 7- save dict into all results list ->
# 8- calculate shape bias for each model -> turn all results into data frame and save them into csv
# =======================================================================

print("\n" + "=" * 60)
print("  MAIN LOOP")
print("=" * 60)

all_results = []
failures = []

for model_name, model in models.items():
    print(f"\n ==={model_name.upper()}===")
    model_results = []

    for i, image in enumerate(all_images):
        try:
            input_tensor, visual_array = load_image(image["path"])
            probs = run_inference(model, input_tensor=input_tensor, device=CONFIG["device"])
            result = classify_decision(
                probs,
                shape_label=image["shape_label"],
                texture_label=image["texture_label"],
                index_to_name=index_to_name,
                index_to_category=index_to_category,
                category_to_indicies=category_to_indicies
            )

            # unpacking result dict
            row = {
                "model":        model_name,
                "file_name":    image["name"],
                "shape_label":  image["shape_label"],
                "texture_label":  image["texture_label"],
                **result,
                "_vis": visual_array
            }

            model_results.append(row)
            all_results.append(row)

            # TEMP DEBUG — add after classify_decision call, remove later
            if i < 5:
                print(f"\n  --- DEBUG image {i} ---")
                print(f"  file:           {image['name']}")
                print(f"  shape_label:    {image['shape_label']}")
                print(f"  texture_label:  {image['texture_label']}")
                print(f"  top1_index:     {result['top1_class_index']}")
                print(f"  top1_name:      {result['top1_class_name']}")
                print(f"  top1_category:  {result['top1_class_category']}")
                print(f"  decision:       {result['decision']}")
                print(f"  shape_conf:     {result['shape_confidence']:.4f}")
                print(f"  texture_conf:   {result['texture_confidence']:.4f}")

            # live shape bias updates for every 100 images
            if (i + 1) % 100 == 0 or i == 0:
                n_s = sum(1 for r in model_results if r['decision'] == 'shape')
                n_t = sum(1 for r in model_results if r['decision'] == 'texture')
                sb  = n_s / (n_s + n_t) if (n_s + n_t) > 0 else 0
                print(f"    [{i+1:4d}/{len(all_images)}]  "
                      f"shape={n_s}  texture={n_t}  bias={sb:.3f}")

        except Exception as e:
            failures.append({
                "model": model_name,
                "file": image["name"],
                "error": str(e)
            })

    n_s = sum(1 for r in model_results if r['decision'] == 'shape')
    n_t = sum(1 for r in model_results if r['decision'] == 'texture')
    n_n = sum(1 for r in model_results if r['decision'] == 'neither')
    sb  = n_s / (n_s + n_t) if (n_s + n_t) > 0 else 0
    tb = 1 - sb
    print(f"\n  {model_name}: shape={n_s}  texture={n_t}  "
          f"neither={n_n}  SHAPE_BIAS={sb:.4f} TEXTURE_BIAS={tb:.4f}")
 
print(f"\n  Processed: {len(all_results)}  |  Failed: {len(failures)}")
print(failures)

# save output in a csv file
df_all   = pd.DataFrame([{k: v for k, v in r.items() if k != '_vis'} for r in all_results])
csv_path = os.path.join(CONFIG["data_dir"], "all_decisions.csv")
df_all.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

# =======================================================================
# ANALYSIS: this phase aims to analyze the results of classifying model decisions into either shape or texture
# 1- loop over each model and find shape bias with confidence intervals 
# 2- run per-category analysis to see how confident the model was for every category
#    both when predicting it as a shape and as a texture. done for all 16 super categories
#    this helps for knowing which category's shape is easy to be biased towards and others where shape 
#    is unrecgonizable when texture comes in the picture
# 3- run confidence analysis to see how confident the models were in predicting textures compared to
#    their confidence when choosing based on shapes
# =======================================================================

print("\n" + "=" * 60)
print("  ANALYSIS")
print("=" * 60)

print("\n  Overall shape bias:")
model_shape_bias = {}
for model_name in CONFIG["models"]:
    sb, ci = compute_shape_bias(df_all[df_all['model'] == model_name]['decision'])
    model_shape_bias[model_name] = (sb, ci)     #e.g. {"vgg16": ( 0.208, (0.190 , 0.220) )}
    pub = CONFIG["published_baselines"].get(model_name, "n/a")
    print(f"    {model_name:<12}: {sb:.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]  published={pub}")
print(f"    {'human':<12}: {CONFIG['human_shape_bias']:.4f}  (Geirhos 2019)")

per_shape_bias, per_texture_bias = run_per_category_analysis(
    df_all, CONFIG["models"], CATEGORIES
)

run_confidence_analysis(df_all, CONFIG["models"])   # no return

# =======================================================================
# FIGURES
# =======================================================================
 
print("\n" + "=" * 60)
print("  FIGURES")
print("=" * 60)
 
generate_all_figures(
    df_all           = df_all,
    model_names      = CONFIG["models"],
    model_shape_bias = model_shape_bias,
    per_shape_bias   = per_shape_bias,
    per_texture_bias = per_texture_bias,
    all_results      = all_results,
    config           = CONFIG,
    categories_16    = CATEGORIES,
)