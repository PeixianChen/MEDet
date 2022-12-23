# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/coco/annotations/instances_val2017_unseen_2.json')
    parser.add_argument('--cat_path', default='datasets/coco/annotations/instances_val2017.json')
    args = parser.parse_args()
    
    categories_seen = ['person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase', 'toothbrush']
    categories_unseen = ['airplane','bus','cat','dog','cow','elephant','umbrella','tie','snowboard','skateboard','cup','knife','cake','couch','keyboard','sink','scissors']

    print('Loading', args.cat_path)
    cat = json.load(open(args.cat_path, 'r'))['categories']
    cat = [c for c in cat if c["name"] in categories_seen]

    print('Loading', args.data_path)
    data = json.load(open(args.data_path, 'r'))
    data['categories'] = cat
    out_path = args.data_path[:-5] + '_oriorder.json'
    print('Saving to', out_path)
    json.dump(data, open(out_path, 'w'))
