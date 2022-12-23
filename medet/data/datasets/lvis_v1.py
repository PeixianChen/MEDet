import logging
import os

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta

logger = logging.getLogger(__name__)

__all__ = ["custom_load_lvis_json", "custom_register_lvis_instances"]

from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES
import json
# This mapping is extracted from the official LVIS mapping:
# https://github.com/lvis-dataset/lvis-api/blob/master/data/coco_to_synset.json
COCO_SYNSET_CATEGORIES = [
    {"synset": "person.n.01", "coco_cat_id": 1},
    {"synset": "bicycle.n.01", "coco_cat_id": 2},
    {"synset": "car.n.01", "coco_cat_id": 3},
    {"synset": "motorcycle.n.01", "coco_cat_id": 4},
    {"synset": "airplane.n.01", "coco_cat_id": 5},
    {"synset": "bus.n.01", "coco_cat_id": 6},
    {"synset": "train.n.01", "coco_cat_id": 7},
    {"synset": "truck.n.01", "coco_cat_id": 8},
    {"synset": "boat.n.01", "coco_cat_id": 9},
    {"synset": "traffic_light.n.01", "coco_cat_id": 10},
    {"synset": "fireplug.n.01", "coco_cat_id": 11},
    {"synset": "stop_sign.n.01", "coco_cat_id": 13},
    {"synset": "parking_meter.n.01", "coco_cat_id": 14},
    {"synset": "bench.n.01", "coco_cat_id": 15},
    {"synset": "bird.n.01", "coco_cat_id": 16},
    {"synset": "cat.n.01", "coco_cat_id": 17},
    {"synset": "dog.n.01", "coco_cat_id": 18},
    {"synset": "horse.n.01", "coco_cat_id": 19},
    {"synset": "sheep.n.01", "coco_cat_id": 20},
    {"synset": "beef.n.01", "coco_cat_id": 21},
    {"synset": "elephant.n.01", "coco_cat_id": 22},
    {"synset": "bear.n.01", "coco_cat_id": 23},
    {"synset": "zebra.n.01", "coco_cat_id": 24},
    {"synset": "giraffe.n.01", "coco_cat_id": 25},
    {"synset": "backpack.n.01", "coco_cat_id": 27},
    {"synset": "umbrella.n.01", "coco_cat_id": 28},
    {"synset": "bag.n.04", "coco_cat_id": 31},
    {"synset": "necktie.n.01", "coco_cat_id": 32},
    {"synset": "bag.n.06", "coco_cat_id": 33},
    {"synset": "frisbee.n.01", "coco_cat_id": 34},
    {"synset": "ski.n.01", "coco_cat_id": 35},
    {"synset": "snowboard.n.01", "coco_cat_id": 36},
    {"synset": "ball.n.06", "coco_cat_id": 37},
    {"synset": "kite.n.03", "coco_cat_id": 38},
    {"synset": "baseball_bat.n.01", "coco_cat_id": 39},
    {"synset": "baseball_glove.n.01", "coco_cat_id": 40},
    {"synset": "skateboard.n.01", "coco_cat_id": 41},
    {"synset": "surfboard.n.01", "coco_cat_id": 42},
    {"synset": "tennis_racket.n.01", "coco_cat_id": 43},
    {"synset": "bottle.n.01", "coco_cat_id": 44},
    {"synset": "wineglass.n.01", "coco_cat_id": 46},
    {"synset": "cup.n.01", "coco_cat_id": 47},
    {"synset": "fork.n.01", "coco_cat_id": 48},
    {"synset": "knife.n.01", "coco_cat_id": 49},
    {"synset": "spoon.n.01", "coco_cat_id": 50},
    {"synset": "bowl.n.03", "coco_cat_id": 51},
    {"synset": "banana.n.02", "coco_cat_id": 52},
    {"synset": "apple.n.01", "coco_cat_id": 53},
    {"synset": "sandwich.n.01", "coco_cat_id": 54},
    {"synset": "orange.n.01", "coco_cat_id": 55},
    {"synset": "broccoli.n.01", "coco_cat_id": 56},
    {"synset": "carrot.n.01", "coco_cat_id": 57},
    # {"synset": "frank.n.02", "coco_cat_id": 58},
    {"synset": "sausage.n.01", "coco_cat_id": 58},
    {"synset": "pizza.n.01", "coco_cat_id": 59},
    {"synset": "doughnut.n.02", "coco_cat_id": 60},
    {"synset": "cake.n.03", "coco_cat_id": 61},
    {"synset": "chair.n.01", "coco_cat_id": 62},
    {"synset": "sofa.n.01", "coco_cat_id": 63},
    {"synset": "pot.n.04", "coco_cat_id": 64},
    {"synset": "bed.n.01", "coco_cat_id": 65},
    {"synset": "dining_table.n.01", "coco_cat_id": 67},
    {"synset": "toilet.n.02", "coco_cat_id": 70},
    {"synset": "television_receiver.n.01", "coco_cat_id": 72},
    {"synset": "laptop.n.01", "coco_cat_id": 73},
    {"synset": "mouse.n.04", "coco_cat_id": 74},
    {"synset": "remote_control.n.01", "coco_cat_id": 75},
    {"synset": "computer_keyboard.n.01", "coco_cat_id": 76},
    {"synset": "cellular_telephone.n.01", "coco_cat_id": 77},
    {"synset": "microwave.n.02", "coco_cat_id": 78},
    {"synset": "oven.n.01", "coco_cat_id": 79},
    {"synset": "toaster.n.02", "coco_cat_id": 80},
    {"synset": "sink.n.01", "coco_cat_id": 81},
    {"synset": "electric_refrigerator.n.01", "coco_cat_id": 82},
    {"synset": "book.n.01", "coco_cat_id": 84},
    {"synset": "clock.n.01", "coco_cat_id": 85},
    {"synset": "vase.n.01", "coco_cat_id": 86},
    {"synset": "scissors.n.01", "coco_cat_id": 87},
    {"synset": "teddy.n.01", "coco_cat_id": 88},
    {"synset": "hand_blower.n.01", "coco_cat_id": 89},
    {"synset": "toothbrush.n.01", "coco_cat_id": 90},
]

def map_name(x):
    x = x.replace('_', ' ')
    if '(' in x:
        x = x[:x.find('(')]
    return x.lower().strip()

num_words = {}
new_words = {}
import glob
with open("./datasets/new_words.txt", "r") as f:
    lines = f.readlines()

for l in lines:
    w, nw, n = l.strip().split(" ")
    num_words[w] = n
    new_words[w] = nw


def custom_register_lvis_instances(name, metadata, json_file, image_root):
    """
    """
    DatasetCatalog.register(name, lambda: custom_load_lvis_json(
        json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, 
        evaluator_type="lvis", **metadata
    )



def custom_load_lvis_json(json_file, image_root, dataset_name=None):
    '''
    Modifications:
      use `file_name`
      convert neg_category_ids
      add pos_category_ids
    '''
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(
            json_file, timer.seconds()))

    catid2contid = {x['id']: i for i, x in enumerate(
        sorted(lvis_api.dataset['categories'], key=lambda x: x['id']))}
    if len(lvis_api.dataset['categories']) == 1203:
        for x in lvis_api.dataset['categories']:
            assert catid2contid[x['id']] == x['id'] - 1
    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), \
        "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in the LVIS v1 format from {}".format(
        len(imgs_anns), json_file))

    dataset_dicts = []

    # use coco words replaced object words:
    cats = json.load(open("./datasets/coco/annotations/instances_val2017.json"))['categories']
    if 'synonyms' not in cats[0]:
        cocoid2synset = {x['coco_cat_id']: x['synset'] \
            for x in COCO_SYNSET_CATEGORIES}
        synset2synonyms = {x['synset']: x['synonyms'] \
            for x in LVIS_CATEGORIES}
        for x in cats:
            synonyms = synset2synonyms[cocoid2synset[x['id']]]
            x['synonyms'] = synonyms
            x['frequency'] = 'f'
    data_categories = cats
    id2cat = {x['id']: x for x in data_categories}
    class_count = {x['id']: 0 for x in data_categories}
    class_data = {x['name']: [map_name(xx) for xx in x['synonyms']] \
            for x in data_categories}

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        if "file_name" in img_dict:
            file_name = img_dict["file_name"]
            if img_dict["file_name"].startswith("COCO"):
                file_name = file_name[-16:]
            record["file_name"] = os.path.join(image_root, file_name)
        elif 'coco_url' in img_dict:
            # e.g., http://images.cocodataset.org/train2017/000000391895.jpg
            file_name = img_dict["coco_url"][30:]
            record["file_name"] = os.path.join(image_root, file_name)
        elif 'tar_index' in img_dict:
            record['tar_index'] = img_dict['tar_index']
        
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get(
            "not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        # NOTE: modified by Xingyi: convert to 0-based
        record["neg_category_ids"] = [
            catid2contid[x] for x in record["neg_category_ids"]]
        if 'pos_category_ids' in img_dict:
            record['pos_category_ids'] = [
                catid2contid[x] for x in img_dict.get("pos_category_ids", [])]
        if 'captions' in img_dict:
            record['captions'] = img_dict['captions']
        if 'caption_features' in img_dict:
            record['caption_features'] = img_dict['caption_features']
        if 'object' in img_dict:
            record['object'] = []
            for c in set(img_dict['object']):
                c = c.lower().replace(" ", "-")
                if c in num_words.keys() and c != "that":# and int(num_words[c]) > 100:
                    c = new_words[c]
                    c = c.replace("-", " ")
                    find = False
                    for cat_id, cat_names in class_data.items():
                        if c in cat_names:
                            record['object'].append(cat_id)
                            find = True
                            break
                    if not find:
                        record['object'].append(c)
            # record['object'] = list(set(record['object']))
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            if anno.get('iscrowd', 0) > 0:
                continue
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            obj["category_id"] = catid2contid[anno['category_id']] 
            if 'segmentation' in anno:
                segm = anno["segmentation"]
                valid_segm = [poly for poly in segm \
                    if len(poly) % 2 == 0 and len(poly) >= 6]
                # assert len(segm) == len(
                #     valid_segm
                # ), "Annotation contains an invalid polygon with < 3 points"
                if not len(segm) == len(valid_segm):
                    print('Annotation contains an invalid polygon with < 3 points')
                assert len(segm) > 0
                obj["segmentation"] = segm
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

_CUSTOM_SPLITS_LVIS = {
    "lvis_v1_train+coco": ("coco/", "lvis/lvis_v1_train+coco_mask.json"),
    "lvis_v1_train_norare": ("coco/", "lvis/lvis_v1_train_norare.json"),
}


for key, (image_root, json_file) in _CUSTOM_SPLITS_LVIS.items():
    custom_register_lvis_instances(
        key,
        get_lvis_instances_meta(key),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )


def get_lvis_22k_meta():
    from .lvis_22k_categories import CATEGORIES
    cat_ids = [k["id"] for k in CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["name"] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta

_CUSTOM_SPLITS_LVIS_22K = {
    "lvis_v1_train_22k": ("coco/", "lvis/lvis_v1_train_lvis-22k.json"),
}

for key, (image_root, json_file) in _CUSTOM_SPLITS_LVIS_22K.items():
    custom_register_lvis_instances(
        key,
        get_lvis_22k_meta(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
