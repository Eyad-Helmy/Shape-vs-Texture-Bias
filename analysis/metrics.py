# This handles everything following model produces a probability vector for each index ([0-999]) 
# translates probabilities into shape/texture/neither predictions
# computes shape bias per model
# computes per category and per texture bias 
#   meaning which categories are more resistant to texture swaping and thus they have shape-bias
#   and which textures result in a texture-bias meaning they are dominant at ruining CNN predictions
# confidence analysis -> how confident modes are when making shape-based decisions vs texture-based

import numpy as np
import pandas as pd

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
    # 1. Initialize the scoreboard for all 16 super-categories
    category_sums = {cat: 0.0 for cat in category_to_indicies.keys()}

    # 2. Sum the probabilities for the 221 indices belonging to our 16 categories
    # This prevents 'Neither' from winning just because the signal is fragmented.
    for index, cat in index_to_category.items():
        category_sums[cat] += probs[index]

    # 3. Identify the winning super-category and its total prob
    best_category = max(category_sums, key=category_sums.get)
    best_category_prob = category_sums[best_category]

    # 4. Determine the Noise Threshold (highest single prob among the other 779 classes)
    all_mapped_indicies = list(index_to_category.keys())    # all of the indicies under the 16 super cats
    neither_mask = np.ones(len(probs), dtype=bool)
    neither_mask[all_mapped_indicies] = False
    max_noise_prob = float(probs[neither_mask].max()) if any(neither_mask) else 0.0

    # 5. Decision Logic: Category Mass vs. Background Noise
    if best_category_prob > max_noise_prob:
        if best_category == shape_label:
            decision = "shape"
        elif best_category == texture_label:
            decision = "texture"
        else:
            decision = "neither"
    else:
        # Background noise (like 'shower curtain' or 'oxygen mask') beat our super-categories
        decision = "neither"

    # 6. Metadata for Analysis (using sums for confidence to reflect total signal)
    shape_confidence = category_sums.get(shape_label, 0.0)
    texture_confidence = category_sums.get(texture_label, 0.0)
    
    # For top1 metadata, we still provide the highest single index for logging
    top1_index = int(np.argmax(probs))
    top1_name = index_to_name.get(top1_index, f"class_{top1_index}")
    top1_category = index_to_category.get(top1_index, None)

    return {
        "decision":            decision,
        "top1_class_index":    top1_index,
        "top1_class_name":     top1_name,
        "top1_class_category": top1_category,
        "shape_confidence":    shape_confidence,
        "texture_confidence":  texture_confidence,
        "neither_confidence":  max_noise_prob,
        "winning_category":    best_category,
        "winning_mass":        best_category_prob
    }


    # top1_index = int(np.argmax(probs))
    # top1_name = index_to_name.get(top1_index, f"class_{top1_index}")
    # top1_category = index_to_category.get(top1_index, None)

    # # decision classification
    # if top1_category == shape_label:
    #     decision = "shape"
    # elif top1_category == texture_label:
    #     decision = "texture"
    # else:
    #     decision = "neither"

    # shape_index_list = list(category_to_indicies.get(shape_label, set()))   # all of cat's indicies
    # texture_index_list = list(category_to_indicies.get(texture_label, set()))   # all of elephent's indicies

    # # we choose incdicies of all the names(sub-categories) of our shape label from the probs array
    # # which gives us the confidence of the model for all of these sub-cats then we get the max one
    # # to represent the full category. this results in the confidence of it's shape class prediction
    # shape_confidence = float(probs[shape_index_list].max()) if shape_index_list else 0.0
    # texture_confidence = float(probs[texture_index_list].max()) if texture_index_list else 0.0

    # # confidence for neither
    # neither_mask = np.ones(len(probs), dtype=bool)
    # if shape_index_list:
    #     neither_mask[shape_index_list] = False
    # if texture_index_list:
    #     neither_mask[texture_index_list] = False
    # # now neither mask has 1000 elements where the indexes that belong to shape or tex class are all false

    # neither_confidence = float(probs[neither_mask].max())

    # return {
    #     "decision":         decision,       # shape-based, texture-based, or neither
    #     "top1_class_index": top1_index,      # the model's top predicted ImageNet index
    #     "top1_class_name":  top1_name,      # that prediction's name (e.g. tabby cat)
    #     "top1_class_category":  top1_category, # that prediction's cat within our 16 cats (e.g. cat)
    #     "shape_confidence": shape_confidence,
    #     "texture_confidence": texture_confidence,
    #     "neither_confidence": neither_confidence,
    # }

