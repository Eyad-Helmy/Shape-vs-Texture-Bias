import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.model_loader import load_all_models
from utils.mapping import download_imagenet_index, build_category_to_indicies

# 1. Setup paths and single image target
IMAGE_PATH = "data/texture-vs-shape/stimuli/style-transfer-preprocessed-512/cat/cat7-clock3.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Models
print("Loading models...")
models = load_all_models(["resnet50", "shape_resnet50_SIN"], DEVICE)
standard_model = models["resnet50"]
shape_model = models["shape_resnet50_SIN"]

# 3. Target the final convolutional layer for ResNet architectures
target_layers_standard = [standard_model.layer4[-1]]
target_layers_shape = [shape_model.layer4[-1]]

# 4. Image Preprocessing
img_pil = Image.open(IMAGE_PATH).convert('RGB')
img_resized = img_pil.resize((224, 224))
rgb_img = np.float32(img_resized) / 255

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)

# 5. Execute Grad-CAM
def generate_cam(model, target_layers, tensor, rgb):
    cam = GradCAM(model=model, target_layers=target_layers)
    # Target None automatically targets the highest scoring class
    targets = [ClassifierOutputTarget(model(tensor).argmax(dim=1).item())] 
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    return visualization

print("Generating heatmaps...")
vis_standard = generate_cam(standard_model, target_layers_standard, input_tensor, rgb_img)
vis_shape = generate_cam(shape_model, target_layers_shape, input_tensor, rgb_img)

# 6. Plot and Save
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rgb_img)
axes[0].set_title("Original Cue-Conflict Image")
axes[0].axis('off')

axes[1].imshow(vis_standard)
axes[1].set_title("Standard ResNet50 (Texture Focus)")
axes[1].axis('off')

axes[2].imshow(vis_shape)
axes[2].set_title("Shape-ResNet50 (Shape Focus)")
axes[2].axis('off')

os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/gradcam_comparison.png", bbox_inches='tight', dpi=150)
print("Saved to results/figures/gradcam_comparison.png")