import os.path as osp
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm
from PIL import Image
import json

label_ids = {
    "holothurian": 1,
    "echinus": 2,
    "scallop": 3,
    "starfish": 4
}


def get_segmentation(points):
    return [[points[0], points[1], points[2] + points[0], points[1],
            points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]]


def parse_xml(xml_path, img_name, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
    scallop_flag = 0
    not_only_scallop_flag = 0
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'waterweeds':
            continue
        if name == 'scallop' and img_name[0] != 'c':
            scallop_flag = 1
        if name in ['holothurian', 'echinus', 'starfish']:
            not_only_scallop_flag = 1
        category_id = label_ids[name]
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w * h
        if area == 0:
            continue
        segmentation = get_segmentation([xmin, ymin, w, h])
        annotation.append({
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": [xmin, ymin, w, h],
            "category_id": category_id,
            "id": anno_id,
            "ignore": 0})
        anno_id += 1
    return annotation, anno_id, scallop_flag, not_only_scallop_flag


def cvt_annotations(img_path, xml_path, out_file):
    images = []
    annotations = []

    # xml_paths = glob(xml_path + '/*.xml')
    img_id = 1
    anno_id = 1
    for img_path in tqdm(glob(img_path + '/*.jpg')):
        w, h = Image.open(img_path).size
        img_name = osp.basename(img_path)

        xml_file_name = img_name.split('.')[0] + '.xml'
        xml_file_path = osp.join(xml_path, xml_file_name)
        annos, anno_id, scallop_flag, not_only_scallop_flag = parse_xml(xml_file_path, img_name, img_id, anno_id)
        if scallop_flag == 1 and not_only_scallop_flag == 0:
            continue
        annotations.extend(annos)
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img)
        img_id += 1

    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    print(len(images))
    final_result = {"images": images, "annotations": annotations, "categories": categories}

    with open(out_file, 'w') as f:
        json.dump(final_result, f, indent=1)
    # mmcv.dump(final_result, out_file)
    return annotations


def main():
    xml_path = 'data/train/box'
    img_path = 'data/train/image'
    print('processing {} ...'.format("xml format annotations"))
    cvt_annotations(img_path, xml_path, 'data/annotations/train_filter_old_scallop.json')
    print('Done!')


if __name__ == '__main__':
    main()