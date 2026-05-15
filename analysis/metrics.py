# This handles everything following model produces a probability vector for each index ([0-999]) 
# translates probabilities into shape/texture/neither predictions
# computes shape bias per model
# computes per category and per texture bias 
#   meaning which categories are more resistant to texture swaping and thus they have shape-bias
#   and which textures result in a texture-bias meaning they are dominant at ruining CNN predictions
# confidence analysis -> how confident modes are when making shape-based decisions vs texture-based

import numpy as np

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


    

