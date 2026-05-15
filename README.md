# Shape vs. Texture Bias in CNNs

A comprehensive analysis framework to quantify and visualize shape bias versus texture bias in Convolutional Neural Networks (CNNs). This project evaluates how modern CNN architectures (VGG16, ResNet50, AlexNet) make predictions based on shape features versus texture features when processing images with conflicting cues.

## Project Overview

This project implements the methodology from [Geirhos et al. (2019)](https://openreview.net/forum?id=Bygh9j09KX) to measure the extent to which CNNs rely on shape versus texture when making predictions. The framework:

1. **Processes stylized images** where shape and texture cues are separated
2. **Runs inference** on multiple pre-trained CNN models
3. **Classifies decisions** as shape-biased, texture-biased, or neither
4. **Computes shape bias metrics** with confidence intervals
5. **Performs per-category analysis** to identify which object categories and textures influence model predictions
6. **Generates statistical comparisons** between model confidence in shape vs. texture decisions
7. **Produces visualizations** of results across models and categories

## Dataset

The project uses the **Texture-vs-Shape dataset** from [Geirhos et al. (2019)](https://github.com/rgeirhos/texture-vs-shape), which contains stylized images where:
- **Shape**: The object's outline and contours
- **Texture**: Applied via style transfer to create texture variants

These conflicting stimuli reveal which cue dominates a model's decision-making.

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, optional)
- ~50GB disk space (for models and dataset)

### Setup

1. **Clone or download** this repository:
   ```bash
   cd shape_vs_texture_bias
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: If you want GPU support with CUDA, install a CUDA-compatible PyTorch version:
   > ```bash
   > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   > ```

4. **Download the dataset** (if not already present):
   - The dataset should be in `data/texture-vs-shape/`
   - If missing, download from the [Texture-vs-Shape repository](https://github.com/rgeirhos/texture-vs-shape)

## Project Structure

```
shape_vs_texture_bias/
├── main.py                          # Main execution pipeline
├── requirements.txt                 # Python dependencies with pinned versions
├── analysis/
│   ├── __init__.py
│   ├── metrics.py                   # Decision classification & statistical analysis
│   ├── visualizations.py            # Figure generation
│   └── figures/                     # Generated plots and visualizations
├── models/
│   ├── __init__.py
│   └── model_loader.py              # Pre-trained model loading & inference
├── utils/
│   ├── __init__.py
│   ├── dataset.py                   # Dataset discovery and image loading
│   └── mapping.py                   # ImageNet class index mapping
├── cache/
│   ├── cache.json                   # ImageNet index cache
│   └── all_decisions.csv            # Cached inference results (optional)
├── results/
│   └── figures/                     # Output figures
└── data/
    └── texture-vs-shape/            # Dataset (if downloaded locally)
        ├── stimuli/
        │   └── style-transfer-preprocessed-512/  # Style-transferred images
        ├── raw-data/                # Original experiment data
        ├── models/                  # Dataset utility files
        └── code/                    # Helper scripts
```

## Usage

### 1. **Run the full pipeline**:

```bash
python main.py
```

This will:
- Load pre-trained models (VGG16, ResNet50, AlexNet)
- Process all images in the dataset
- Classify each prediction as shape-biased or texture-biased
- Compute overall and per-category shape bias metrics
- Generate analysis figures
- Save results to `results/` and `cache/all_decisions.csv`

**Configuration** (in `main.py`):
- `max_images`: Set to `None` for all images (~1200) or an integer to limit
- `models`: List of models to evaluate; see `models/model_loader.py` for available options
- `device`: Automatically detects CUDA; set manually if needed

### 2. **Predict on a single image**:

```bash
python predict_single_image.py --image path/to/image.jpg --model vgg16
```

### 3. **Run analysis on cached results**:

If you have a cached `cache/all_decisions.csv` from a previous run, analysis functions can be called directly to regenerate visualizations without re-running inference.

## Key Outputs

### Metrics

- **Shape Bias**: Proportion of non-"neither" decisions that are shape-based (0–1 scale)
- **Bootstrap CI**: 95% confidence intervals computed via resampling
- **Per-Category Breakdown**: Shape bias when predicting each object category, and when each texture is applied
- **Confidence Analysis**: Statistical tests (Welch's t-test) comparing model confidence in shape vs. texture decisions

### Figures

All figures are saved to `results/figures/` in PNG format:

#### **Fig 1: Main Shape Bias Comparison** (`fig1_main_shape_bias.png`)
Bar chart comparing shape bias across models with 95% confidence intervals. Includes:
- Human baseline from Geirhos et al. (2019): ~0.96 shape bias
- Published baselines for each model (if available)
- Your experimental results with confidence intervals
- Comparison shows how close your models match published benchmarks

#### **Fig 2: Per-Category Heatmap** (`fig2_per_category_heatmap.png`)
Two heatmaps showing:
- **Left**: Shape bias broken down by **shape category** (which objects are shape-biased?)
- **Right**: Shape bias when each **texture** is applied (which textures cause texture bias?)
- Color intensity indicates strength of shape bias (darker = more shape-biased)
- Helps identify which object categories and textures most influence model decisions

#### **Fig 3: Confidence Distributions** (`fig3_confidence_distributions.png`)
Overlaid histograms for each model showing:
- **Blue**: Confidence scores when model makes **shape** decisions
- **Red**: Confidence scores when model makes **texture** decisions
- Shows whether models are more confident when "tricked" by texture vs. when correctly using shape

#### **Fig 4: Example Predictions Grid** (`fig4_example_decisions.png`)
Optional visual grid showing:
- Example images from each decision category (shape/texture/neither)
- Model predictions and confidence scores
- Side-by-side comparison across models
- *(Skipped if insufficient examples in dataset)*

#### **Fig 5: Decision Breakdown** (`fig5_decision_breakdown.png`)
Stacked bar chart showing the proportion of decisions for each model:
- Green: Shape-biased decisions
- Red: Texture-biased decisions  
- Gray: Neither (misclassified or ambiguous)
- Illustrates overall decision distribution per model

---
## Code Highlights

### `analysis/metrics.py`

- `classify_decision()`: Converts raw probabilities to shape/texture/neither decisions
- `compute_shape_bias()`: Calculates shape bias with bootstrap confidence intervals
- `run_per_category_analysis()`: Analyzes bias by object and texture category
- `run_confidence_analysis()`: Compares model confidence across decision types

### `models/model_loader.py`

- Loads pre-trained ImageNet models (VGG16, ResNet50, AlexNet)
- Handles model initialization and inference
- Supports both CPU and GPU execution

### `utils/dataset.py`

- Discovers and loads stylized images from the dataset
- Maps image filenames to shape/texture labels
- Handles image preprocessing and tensor conversion

## Requirements and Versions

The project uses the following packages:

- **torch** (with CUDA support for GPU acceleration though it can run without CUDA on the CPU)
- **torchvision** 
- **numpy**
- **pandas**
- **matplotlib**
- **seaborn**
- **scipy**
- **opencv-python**
- **Pillow**
- **requests**

To update after installing new dependencies:
```bash
pip freeze > requirements.txt
```

## Performance Notes

- **GPU (CUDA)**: ~2–5 minutes for ~1200 images across 3 models
- **CPU**: ~30–60 minutes (significantly slower)
- **Memory**: ~8GB GPU VRAM or ~16GB system RAM recommended

## References

- **Geirhos et al. (2019)** – "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness" ([Paper](https://openreview.net/forum?id=Bygh9j09KX), [Code](https://github.com/rgeirhos/texture-vs-shape))
- **ImageNet**: https://www.image-net.org/

## License

This project extends the work of Geirhos et al. and uses their texture-vs-shape dataset. See the dataset repository for licensing details.

## Contributing

For improvements or bug reports, please document:
- What you changed and why
- Any new dependencies added
- Updated requirements.txt if applicable

---

**Last Updated**: May 2026  
**Tested with**: PyTorch 2.9.1 (CUDA 13.2), Python 3.10+
