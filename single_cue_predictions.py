import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils.dataset import load_image
from utils.mapping import build_category_to_indicies
from models.model_loader import load_all_models, run_inference
from analysis.metrics import classify_single_cue

CONFIG = {
    # Update these paths to point to your extracted folders
    "dataset_paths": {
        "silhouette": os.path.join("data", "texture-vs-shape", "stimuli", "filled-silhouettes"),
        "edges":      os.path.join("data", "texture-vs-shape", "stimuli", "edges")
    },
    "models": ["alexnet", "vgg16", "resnet50"],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_fig": "results/figures/fig2_partial_accuracy.png"
}

HUMAN_BASELINES = {
    "silhouette": 75,
    "edges": 87
}

def discover_full_dataset(dataset_path: str, valid_categories: set) -> list:
    if not os.path.isdir(dataset_path):
        return []

    valid_images = []
    for class_folder in os.listdir(dataset_path):
        if class_folder not in valid_categories:
            continue
            
        folder_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(folder_path):
            continue
            
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                valid_images.append({
                    "path": os.path.join(folder_path, file_name),
                    "label": class_folder
                })
    return valid_images

def generate_2_panel_figure(results: dict, example_images: dict, output_path: str):
    """
    Generates a 2-column figure specifically for Silhouette and Edges.
    """
    conditions = list(CONFIG["dataset_paths"].keys())
    models = CONFIG["models"]
    
    colors = {
        "alexnet": "#3b528b",
        "vgg16": "#21918c",
        "resnet50": "#333333",
        "human": "#a52a2a"
    }
    
    # Create 2x2 grid (Top row: bars, Bottom row: images)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [4, 1.5]})
    fig.subplots_adjust(wspace=0.2, hspace=0.1)
    
    bars_for_legend = []
    labels_for_legend = [m.upper() for m in models] + ["HUMANS"]
    
    for idx, condition in enumerate(conditions):
        ax_bar = axes[0, idx]
        ax_img = axes[1, idx]
        
        x_positions = np.arange(len(models) + 1)
        bar_colors = [colors[m] for m in models] + [colors["human"]]
        
        model_accs = [results.get(condition, {}).get(m, 0.0) * 100 for m in models]
        human_acc = HUMAN_BASELINES[condition]
        all_accs = model_accs + [human_acc]
        
        # Plot Bars
        bars = ax_bar.bar(x_positions, all_accs, color=bar_colors, edgecolor='black', linewidth=1)
        if idx == 0:
            bars_for_legend = bars
            
        for bar, acc in zip(bars, all_accs):
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{int(acc)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax_bar.set_xticks([])
        ax_bar.set_ylim(0, 115)
        
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['bottom'].set_visible(False)
        if idx > 0:
            ax_bar.spines['left'].set_visible(False)
            ax_bar.tick_params(left=False)

        # Plot Representative Image
        if condition in example_images and example_images[condition]:
            img = mpimg.imread(example_images[condition])
            ax_img.imshow(img)
            
        ax_img.axis('off')
        ax_img.text(0.5, -0.2, condition.capitalize(), ha='center', va='top', 
                    transform=ax_img.transAxes, fontsize=12, fontweight='bold')

    # Global Legend
    fig.legend(bars_for_legend, labels_for_legend, loc='lower center', 
               bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize=11)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  [visualization] 2-Panel figure saved to {output_path}")

def main():
    print("=" * 60)
    print("  PARTIAL SINGLE-CUE ACCURACY EVALUATION")
    print("=" * 60)

    category_to_indicies, _ = build_category_to_indicies()
    valid_categories = set(category_to_indicies.keys())
    
    models = load_all_models(CONFIG["models"], CONFIG["device"])
    final_accuracies = {}
    example_images = {}

    for condition, path in CONFIG["dataset_paths"].items():
        print(f"\n--- Processing Condition: {condition.upper()} ---")
        
        images = discover_full_dataset(path, valid_categories)
        if not images:
            print(f"  [error] No valid images found in {path}. Check your folders!")
            continue
            
        print(f"  [dataset] Found {len(images)} images.")
        
        # Save the very first image to display under the graph
        example_images[condition] = images[0]["path"]
        
        final_accuracies[condition] = {}
        correct_counts = {m: 0 for m in CONFIG["models"]}
        total_count = len(images)
        
        for i, img_data in enumerate(images):
            if (i + 1) % 500 == 0:
                print(f"    Processed {i + 1}/{total_count} images...")
                
            for model_name, model in models.items():
                target_size = (227, 227) if model_name == "alexnet" else (224, 224)
                input_tensor, _ = load_image(img_data["path"], target_size=target_size)
                
                probs = run_inference(model, input_tensor, CONFIG["device"])
                result = classify_single_cue(probs, img_data["label"], category_to_indicies)
                
                if result["is_correct"]:
                    correct_counts[model_name] += 1

        # Calculate final percentages
        for model_name in CONFIG["models"]:
            acc = correct_counts[model_name] / total_count
            final_accuracies[condition][model_name] = acc
            print(f"    {model_name.upper():<10} Final Accuracy: {acc*100:.1f}%")

    generate_2_panel_figure(final_accuracies, example_images, CONFIG["output_fig"])
    print("Execution Complete.")

if __name__ == "__main__":
    main()