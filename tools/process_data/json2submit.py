import json
import os
import torchvision
import torch
import numpy as np
# raw_anno_file = 'data/annotations/testA.json'
raw_anno_file = 'data/annotations/testB.json'
result_file = 'results/testB.bbox.json'
submit_path = 'submit/testB.csv'
with open(raw_anno_file, 'r') as f:
    annos = json.load(f)
imageid2name = {}
for image in annos['images']:
    imageid2name[image['id']] = image['file_name']
print('img nums: ', len(imageid2name.keys()))
with open(result_file, 'r') as f:
    results = json.load(f)
catename2nid = {
    "holothurian": 1,
    "echinus": 2,
    "scallop": 3,
    "starfish": 4
}
cateid2name = {}
for catename, cateid in catename2nid.items():
    cateid2name[cateid] = catename
final_results = []
zero_area = 0
with open(submit_path, 'w') as f:
    f.write("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    for res in results:
        image_id = res['image_id']
        image_name = imageid2name[image_id][:-4]
        category_id = res['category_id']
        catename = cateid2name[category_id]
        box = res['bbox']
        xmin, ymin, w, h = box
        if w == 0 or h == 0:
            zero_area+=1
            continue
        xmax = xmin + w
        ymax = ymin + h
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        score = res['score']
        string = str(catename) + ',' + str(image_name) + ',' + str(score) + ',' + str(xmin) + ',' + str(ymin) + ',' +\
                 str(xmax) + ',' + str(ymax) + '\n'
        f.write(string)

print(zero_area)