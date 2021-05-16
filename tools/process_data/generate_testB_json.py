# -*- coding: utf-8 -*-

import os
import json
from tqdm import tqdm
from glob import glob
# from PIL import Image
import numpy as np
from PIL import Image

if __name__ == '__main__':
    print("generate testB json...")
    with open('data/annotations/train.json', 'r') as f:
        categories = json.load(f)['categories']
    # img_root = 'data/tile_round1_testA_20201231/testA_imgs/'
    img_root = 'data/test-B-image/'
    save_path_root = "data/annotations/"
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1
    for imgpath in tqdm(glob(img_root + "/*.jpg")):
        w, h = Image.open(imgpath).size
        imgname = os.path.basename(imgpath)
        images.append({
            "id": image_id,
            "height": h,
            "width": w,
            "file_name": imgname
        })
        annotations.append({
            "id": annotation_id,
            "bbox": [0,0,20,20],
            "area": 400.0,
            "segmentation": [[]],
            "iscrowd": 0,
            "category_id": 1,
            "image_id": image_id
        })
        image_id+=1
        annotation_id+=1
    instance = {"images": images, "annotations": annotations, "categories": categories}
    with open(save_path_root + 'testB.json', 'w') as f:
        json.dump(instance, f, indent=1)


