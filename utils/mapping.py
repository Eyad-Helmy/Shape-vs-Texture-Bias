# this file maps model outputs to class labels
# the model outputs a number:[0-999] where each of which corresponds to a WordNet identifier.
# we can fetch the dict that contains the information of which output number corresponds to which id from the website
# we then manually construcut a dict that includes all the id's of the same species to equate them all to the same label (golden reteiver and german shepeard both inside "dog" label)
# lastly, we take these 2 mappings and construct a single dict that maps an output number to one of the 16 master labels to perform a single look up for every prediction inside of 2

import os
import json
import requests

# This maps each wordnet synset of each subspecies to its parent species of the 16 class labels our models will output
CATEGORY_TO_INDICES = {
    'airplane':  [404],
    'bear':      [294, 295, 296, 297],
    'bicycle':   [444, 671],
    'bird':      [8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23,
                  24, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93,
                  94, 95, 96, 98, 99, 100, 127, 128, 129, 130, 131,
                  132, 133, 135, 136, 137, 138, 139, 140, 141, 142,
                  143, 144, 145],
    'boat':      [472, 554, 625, 814, 914],
    'bottle':    [440, 720, 737, 898, 899, 901, 907],
    'car':       [436, 511, 817],
    'cat':       [281, 282, 283, 284, 285, 286],
    'chair':     [423, 559, 765, 857],
    'clock':     [409, 530, 892],
    'dog':       [152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
                  162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                  172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                  182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
                  193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
                  203, 205, 206, 207, 208, 209, 210, 211, 212, 213,
                  214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                  224, 225, 226, 228, 229, 230, 231, 232, 233, 234,
                  235, 236, 237, 238, 239, 240, 241, 243, 244, 245,
                  246, 247, 248, 249, 250, 252, 253, 254, 255, 256,
                  257, 259, 261, 262, 263, 265, 266, 267, 268],
    'elephant':  [385, 386],
    'keyboard':  [508, 878],
    'knife':     [499],
    'oven':      [766],
    'truck':     [555, 569, 656, 675, 717, 734, 864, 867],
}



def download_imagenet_index(url: str, save_path: str) -> tuple:
        """
        outputs 2 dicts: 
        one that maps each wordnet synset to its corresponding mode output index
        the second maps each index to its specific name

        The JSON maps string indices to [synset_id, class_name] pairs:
        {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
        """
        if os.path.exists(save_path):
            print(f"  [mapping] Loading cached class index: {save_path}")
        else:
            print(f"  [mapping] Downloading ImageNet class index...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(save_path, 'w') as f:
                f.write(response.text)
            print(f"  [mapping] Saved to {save_path}")
    
        with open(save_path) as f:
            raw = json.load(f)   # {"0": ["n01440764", "tench"], ...}
    
        synset_to_idx = {v[0]: int(k) for k, v in raw.items()}
        idx_to_name   = {int(k): v[1]  for k, v in raw.items()}
    
        print(f"  [mapping] Index loaded: {len(synset_to_idx)} classes")
        return synset_to_idx, idx_to_name


def build_category_to_indicies() -> tuple:
    """Build lookup tables directly from official Geirhos index lists."""
    
    category_to_indicies = {cat: set(indices) 
                            for cat, indices in CATEGORY_TO_INDICES.items()}
    
    index_to_category = {}
    for cat, indices in CATEGORY_TO_INDICES.items():
        for idx in indices:
            index_to_category[idx] = cat

    total = sum(len(v) for v in category_to_indicies.values())
    print(f"  [mapping] Mapping built: {len(category_to_indicies)} categories, "
          f"{total} total class indices covered")

    return category_to_indicies, index_to_category

# # test
# url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
# synset_to_idx, idx_to_name = download_imagenet_index(url=url, save_path="cache/cache.json")
# category_to_indicies, index_to_category = build_category_to_indicies(category_to_synsets=CATEGORY_TO_SYNSETS, synset_to_idx=synset_to_idx)
# # print(category_to_indicies, "="*50, index_to_category)
# category_sums = {cat: 0.0 for cat in category_to_indicies.keys()}
# for index, cats in index_to_category.items():
#     category_sums[cats] += probs[index]