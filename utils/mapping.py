# this file maps model outputs to class labels
# the model outputs a number:[0-999] where each of which corresponds to a WordNet identifier.
# we can fetch the dict that contains the information of which output number corresponds to which id from the website
# we then manually construcut a dict that includes all the id's of the same species to equate them all to the same label (golden reteiver and german shepeard both inside "dog" label)
# lastly, we take these 2 mappings and construct a single dict that maps an output number to one of the 16 master labels to perform a single look up for every prediction inside of 2

import os
import json
import requests

# This maps each wordnet synset of each subspecies to its parent species of the 16 class labels our models will output
CATEGORY_TO_SYNSETS = {
    'airplane': [
        'n02690373',   # airliner
        'n02691156',   # airplane, aeroplane, plane
        'n04552348',   # warplane, military plane
    ],
    'bear': [
        'n02132136',   # brown bear, bruin, Ursus arctos
        'n02133161',   # American black bear, Ursus americanus
        'n02134084',   # ice bear, polar bear, Ursus maritimus
        'n02134418',   # sloth bear, Melursus ursinus
    ],
    'bicycle': [
        'n02835271',   # bicycle-built-for-two, tandem bicycle, tandem
        'n03792782',   # mountain bike, all-terrain bike, off-roader
    ],
    'bird': [
        # Passerines and songbirds
        'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178',
        'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419',
        'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432',
        # Raptors
        'n01614925', 'n01616318', 'n01622779',
        # Fowl and waterfowl
        'n01795545', 'n01798484', 'n01817953', 'n01818515', 'n01819313',
        'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805',
        # Wading and shore birds
        'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672',
        'n01860187', 'n02002556', 'n02002724', 'n02006656', 'n02007558',
        'n02009229', 'n02009912', 'n02011460', 'n02013706', 'n02017213',
        'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035',
        'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221',
    ],
    'boat': [
        'n02951358',   # canoe
        'n03095699',   # container ship, containership
        'n03947888',   # pirate, pirate ship
        'n04273569',   # speedboat
        'n04612504',   # yawl
    ],
    'bottle': [
        'n02823428',   # beer bottle
        'n03983396',   # pop bottle, soda bottle
        'n04557648',   # water bottle
        'n04560804',   # wine bottle
    ],
    'car': [
        'n02814533',   # beach wagon, station wagon
        'n02958343',   # car, auto, automobile, machine, motorcar
        'n03100240',   # convertible
        'n03498962',   # hotrod, hot rod
        'n03770679',   # minivan
        'n04037443',   # racer, race car, racing car
        'n04285008',   # sports car, sport car
    ],
    'cat': [
        'n02123045',   # tabby, tabby cat
        'n02123159',   # tiger cat
        'n02123394',   # Persian cat
        'n02123597',   # Siamese cat, Siamese
        'n02124075',   # Egyptian cat
        'n02125311',   # cougar, puma, mountain lion
    ],
    'chair': [
        'n02791270',   # barber chair
        'n03001627',   # folding chair
        'n03376595',   # rocking chair, rocker
        'n04099969',   # rocking chair (alt synset)
    ],
    'clock': [
        'n03196217',   # digital clock
        'n03187595',   # digital watch
        'n04548280',   # wall clock
        'n04548362',   # watch, wristwatch
    ],
    'dog': [
        # All 117 dog breed synsets present in ImageNet-1000.
        # Covering all breeds is essential — see module docstring above.
        'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240',
        'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094',
        'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078',
        'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721',
        'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635',
        'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428',
        'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114',
        'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889',
        'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585',
        'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474',
        'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267',
        'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236',
        'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388',
        'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480',
        'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162',
        'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855',
        'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662',
        'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908',
        'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915',
        'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185',
        'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129',
        'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137',
        'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624',
        'n02113712', 'n02113799', 'n02113978',
    ],
    'elephant': [
        'n02504013',   # Indian elephant, Elephas maximus
        'n02504458',   # African elephant, Loxodonta africana
    ],
    'keyboard': [
        'n03085013',   # computer keyboard, keypad
        'n04264628',   # space bar
    ],
    'knife': [
        'n03041632',   # cleaver, meat cleaver, chopper
        'n03658185',   # letter opener, paper knife, paper knife
    ],
    'oven': [
        'n03259401',   # Dutch oven
        'n03761084',   # microwave, microwave oven
        'n04111531',   # rotisserie, rotisserie oven
    ],
    'truck': [
        'n03345487',   # fire engine, fire truck
        'n03417042',   # garbage truck, dustcart
        'n03796401',   # moving van
        'n03977966',   # police van, police wagon, paddy wagon
        'n04461696',   # tow truck, wrecker, recovery vehicle
        'n04467665',   # trailer truck, tractor trailer, trucking rig
    ],
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


def build_category_to_indicies(category_to_synsets: dict, synset_to_idx: dict) -> tuple:
    """
    Convert CATEGORY_TO_SYNSETS into two runtime-ready lookup structures.
 
    Arguments:
        category_synsets : the CATEGORY_TO_SYNSETS dict above
        synset_to_idx    : output of download_imagenet_index
 
    Returns:
        category_to_indicies : dict[str, set[int]]
            "cat" -> {281, 282, 283, 284, 285, 286}
            Used to compute max confidence for a category.
 
        index_to_category : dict[int, str]
            281 -> "cat", 282 -> "cat", ..., 404 -> "airplane"
            Used for O(1) decision lookup during inference:
            given any top-1 prediction integer, instantly know its category.
 
    Synsets not present in the 1000-class ImageNet subset are silently
    skipped (printed as warnings). This happens for synsets that exist
    in the full WordNet taxonomy but were not selected for ILSVRC-2012.
 
    The sanity check at the end ensures no two categories share a class
    index — that would make the decision logic ambiguous.
    """

    # the logic below makes {"cat": {100, 101, 103}, ...}
    category_to_indicies = {}
    missing = []

    # cat == category
    # loop over each cat and all of its synsets
    for cat, synsets in category_to_synsets.items():
        # loop over all the synsets for each category
        indicies = set()
        for syn in synsets:
            if syn in synset_to_idx:
                indicies.add(synset_to_idx[syn])
        category_to_indicies[cat] = indicies

    if missing:
        print(f"  [mapping] {len(missing)} synset(s) not in ImageNet-1000 (skipped):")
        for cat, syn in missing[:6]:
            print(f"    {cat}: {syn}")

    # the logic below makes {100: "cat", 101: "cat", 102: "cat", ...}
    index_to_category = {}
    for cat, indicies in category_to_indicies.items():
        for index in indicies:
            index_to_category[index] = cat

    #no two categories can share a class index
    all_indices = [i for s in category_to_indicies.values() for i in s]
    assert len(all_indices) == len(set(all_indices)), (
        "Two categories share a class index. Check CATEGORY_TO_SYNSETS."
    )

    print(f"  [mapping] Mapping built: {len(category_to_indicies)} categories, "
          f"{len(index_to_category)} total class indices covered")

    return category_to_indicies, index_to_category

# # test
# url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
# synset_to_idx, idx_to_name = download_imagenet_index(url=url, save_path="cache/cache.json")
# category_to_indicies, index_to_category = build_category_to_indicies(category_to_synsets=CATEGORY_TO_SYNSETS, synset_to_idx=synset_to_idx)
# print(category_to_indicies, "="*50, index_to_category)