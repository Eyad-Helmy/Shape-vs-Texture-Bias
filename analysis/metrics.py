# This handles everything following model produces a probability vector for each index ([0-999]) 
# translates probabilities into shape/texture/neither predictions
# computes shape bias per model
# computes per category and per texture bias 
#   meaning which categories are more resistant to texture swaping and thus they have shape-bias
#   and which textures result in a texture-bias meaning they are dominant at ruining CNN predictions
# confidence analysis -> how confident modes are when making shape-based decisions vs texture-based

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def classify_decision(
        probs: 'np.ndarray',
        shape_label: str,
        texture_label: str,
        index_to_name: dict,
        index_to_category: dict,
        category_to_indicies: dict,
) -> dict:
    """
    Arguments:
        probs               : (1000,) softmax probability array
        shape_label         : e.g. "cat"
        texture_label       : e.g. "elephant"
        category_to_indices : {"cat": {281,282,...}, ...}
        index_to_category   : {281: "cat", 282: "cat", ...}
        idx_to_name         : {281: "tabby", 282: "tiger cat", ...}
 
    Returns a dict with all decision information for one image.
    """
    # 1. Map 1000-class probabilities to 16 categories using the mean
    mapped_probs = {}
    for cat, indices in category_to_indicies.items():
        indices_list = list(indices)
        if indices_list:
            mapped_probs[cat] = float(np.mean(probs[indices_list]))
        else:
            mapped_probs[cat] = 0.0

    # 2. Force the 16-way decision
    top1_category = max(mapped_probs, key=mapped_probs.get)

    if top1_category == shape_label:
        decision = "shape"
    elif top1_category == texture_label:
        decision = "texture"
    else:
        # Represents one of the OTHER 14 mapped categories
        decision = "neither"

    # 3. Extract the absolute 1000-class top-1 for logging purposes
    top1_index = int(np.argmax(probs))
    top1_name = index_to_name.get(top1_index, f"class_{top1_index}")

    # 4. Extract confidence scores aligned with the average mapping logic
    shape_confidence = mapped_probs.get(shape_label, 0.0)
    texture_confidence = mapped_probs.get(texture_label, 0.0)

    # Maintained from handoff.md logic: absolute top-1 probability over 1000 classes
    neither_confidence = float(probs[top1_index]) if decision == "neither" else 0.0

    return {
        "decision":         decision,
        "top1_class_index": top1_index,
        "top1_class_name":  top1_name,
        "top1_class_category":  top1_category,
        "shape_confidence": shape_confidence,
        "texture_confidence": texture_confidence,
        "neither_confidence": neither_confidence,
    }


def compute_shape_bias(decisions_series: 'pd.Series') -> tuple:
    """
    Compute shape bias and a 95% bootstrap confidence interval.
 
    Bootstrap CI:
      Resample the resolved decisions 1000 times with replacement.
      Compute shape_bias for each resample.
      Return the 2.5th and 97.5th percentiles.
 
    Arguments:
        decisions_series : pd.Series of 'shape' / 'texture' / 'neither' strings
 
    Returns:
        (shape_bias, (ci_low, ci_high))
        shape_bias : float in [0, 1]
        ci_low, ci_high : float 95% confidence interval bounds
    """
    resolved = decisions_series[decisions_series != 'neither']
    if len(resolved) == 0:
        return 0.0, (0.0, 0.0)
 
    arr    = (resolved == 'shape').values.astype(int)
    sb     = float(arr.mean())
 
    # Bootstrap
    rng     = np.random.default_rng(42)
    boot_sb = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(1000)
    ])
 
    ci_lo = float(np.percentile(boot_sb, 2.5))
    ci_hi = float(np.percentile(boot_sb, 97.5))
 
    return sb, (ci_lo, ci_hi)

def run_per_category_analysis(
    df_all: 'pd.DataFrame',
    model_names: list,
    categories: list,
) -> tuple:
    """
    Compute shape bias broken down by shape category and by texture category.
 
    Returns:
        per_shape_bias   : {model_name: {shape_cat:   shape_bias}}
        per_texture_bias : {model_name: {texture_cat: shape_bias when this texture applied}}
    """
    per_shape_bias   = {}
    per_texture_bias = {}
 
    for model_name in model_names:
        df_m = df_all[df_all['model'] == model_name]
        per_shape_bias[model_name]   = {}
        per_texture_bias[model_name] = {}
 
        for cat in categories:
            # Shape breakdown
            subset = df_m[df_m['shape_label'] == cat]
            if len(subset) > 0:
                sb, _ = compute_shape_bias(subset['decision'])
                per_shape_bias[model_name][cat] = sb
 
            # Texture breakdown
            subset = df_m[df_m['texture_label'] == cat]
            if len(subset) > 0:
                sb, _ = compute_shape_bias(subset['decision'])
                per_texture_bias[model_name][cat] = sb
 
    # Print a readable summary for every model
    for model_name in model_names:
        print(f"\n  Shape bias by SHAPE category ({model_name}):")
        for cat, sb in sorted(per_shape_bias[model_name].items(), key=lambda x: -x[1]):
            bar = '█' * int(sb * 20)
            print(f"    {cat:<12}: {sb:.3f}  {bar}")
 
        print(f"\n  Shape bias when THIS TEXTURE is applied ({model_name}):")
        for cat, sb in sorted(per_texture_bias[model_name].items(), key=lambda x: -x[1]):
            bar = '█' * int(sb * 20)
            print(f"    {cat:<12}: {sb:.3f}  {bar}")
 
    return per_shape_bias, per_texture_bias

def run_confidence_analysis(df_all: 'pd.DataFrame', model_names: list):
    """
    Compare model confidence when making shape decisions vs texture decisions.
 
    For each model, computes:
      - Mean shape_confidence and texture_confidence when the model makes texture decisions
        (how confident is it when it gets fooled by a texture)
      - Welch's t-test to check if the difference is statistically significant
    """
    print("\n  Confidence analysis:")
 
    for model_name in model_names:
        df_m              = df_all[df_all['model'] == model_name]
        shape_decisions   = df_m[df_m['decision'] == 'shape']
        texture_decisions = df_m[df_m['decision'] == 'texture']
 
        print(f"\n  {model_name.upper()}")
 
        if len(shape_decisions) > 0:
            print(f"    Shape   decisions — shape_confidence   mean: "
                  f"{shape_decisions['shape_confidence'].mean():.4f}  "
                  f"std: {shape_decisions['shape_confidence'].std():.4f}")
 
        if len(texture_decisions) > 0:
            print(f"    Texture decisions — texture_confidence mean: "
                  f"{texture_decisions['texture_confidence'].mean():.4f}  "
                  f"std: {texture_decisions['texture_confidence'].std():.4f}")
 
        if len(texture_decisions) > 10 and len(shape_decisions) > 10:
            stat, pval = ttest_ind(
                texture_decisions['texture_confidence'],
                texture_decisions['shape_confidence'],
                equal_var=False,   # Welch's, not Student's
            )
            sig = "SIGNIFICANT" if pval < 0.05 else "not significant"
            print(f"    t-test (texture_confidence vs shape_confidence on texture decisions):")
            print(f"      t={stat:.3f}  p={pval:.4f}  → {sig}")
