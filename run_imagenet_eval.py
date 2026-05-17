import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model_loader import load_all_models

CONFIG = {
    "imagenet_val_path": os.path.join("../../../Downloads/imagenet_validation_set/imagenet-val"),
    "batch_size": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4
}

def load_evaluation_models(device):
    """Loads the standard baseline and all 3 shape-biased intervention models."""
    print("Loading architectures and weights into memory...")
    
    model_names = [
        "resnet50",  # Standard ImageNet baseline
        "shape_resnet50_SIN",
        "shape_resnet50_SIN_and_IN",
        "shape_resnet50_SIN_and_IN_finetuned_on_IN"
    ]
    
    return load_all_models(model_names, device)

def calculate_accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    print("=" * 60)
    print("  IMAGENET VALIDATION ACCURACY EVALUATION")
    print("=" * 60)

    # Standard ImageNet Validation Transforms
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(CONFIG["imagenet_val_path"]):
        raise FileNotFoundError(f"ImageNet validation directory not found at: {CONFIG['imagenet_val_path']}")

    val_dataset = datasets.ImageFolder(CONFIG["imagenet_val_path"], val_transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    models_to_test = load_evaluation_models(CONFIG["device"])
    results = {name: {"top1": 0.0, "top5": 0.0} for name in models_to_test.keys()}

    print(f"\nEvaluating on {len(val_dataset)} images (Batch Size: {CONFIG['batch_size']})...")

    # Evaluate each model independently to prevent GPU VRAM overflow
    for model_name, model in models_to_test.items():
        print(f"\nEvaluating Model: {model_name}")
        top1_sum = 0.0
        top5_sum = 0.0
        total_batches = len(val_loader)

        with torch.no_grad():
            for images, targets in tqdm(val_loader, total=total_batches, desc="Inference"):
                images = images.to(CONFIG["device"], non_blocking=True)
                targets = targets.to(CONFIG["device"], non_blocking=True)

                outputs = model(images)
                acc1, acc5 = calculate_accuracy(outputs, targets, topk=(1, 5))

                top1_sum += acc1.item()
                top5_sum += acc5.item()

        # Calculate final averages
        results[model_name]["top1"] = top1_sum / total_batches
        results[model_name]["top5"] = top5_sum / total_batches

    # Print Final Output Matrix
    print("\n" + "=" * 60)
    print("  FINAL ACCURACY METRICS")
    print("=" * 60)
    print(f"{'Model Variant':<50} | {'Top-1 Acc (%)':<20} | {'Top-5 Acc (%)':<15}")
    print("-" * 56)
    for name, metrics in results.items():
        print(f"{name:<50} | {metrics['top1']:<20.2f} | {metrics['top5']:<15.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()