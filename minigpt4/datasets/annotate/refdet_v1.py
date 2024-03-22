import os
import json
from glob import glob
from tqdm import tqdm
from ipdb import set_trace
import xml.etree.ElementTree as ET
import cv2
import random

curious_categories = ['破损绿化护栏', '破损户外广告', '井盖下沉', '平铺晾晒', '堆积垃圾', '打包垃圾', '破损防撞桩', '乞讨卖艺', '无序堆积', '破损红绿灯', '破损交通指示牌', '涂鸦广告', '破损井盖', '道路破损', '共享单车堆积', '砖块堆积', '倒伏垃圾桶', '破损牌匾', '废弃纸板', '灌木缺失', '倒伏共享摩托车', '倒伏共享自行车', '住宅起火', '非装饰性树挂', '临街修车', '焚烧垃圾树叶', '出店', '破损广告牌', '满溢垃圾桶', '渣土积存', '井盖移位', '游商', '破损防撞桶', '破损交通锥', '破损雨水篦子', '缺失井盖', '临街洗车', '破损垃圾桶', '破损垃圾箱', '废弃家具', '破损交通护栏', '行树缺株', '井盖凸起', '违法横幅', '城市烟雾']

curious_categories_tranlation = ['Damaged greenery guardrail', 'Damaged outdoor advertising', 'Sunken manhole cover', 'Flat drying', 'Accumulated garbage', 'Bagged garbage', 'Damaged bollard', 'Begging and street performing', 'Disordered accumulation', 'Damaged traffic light', 'Damaged traffic sign', 'Graffiti advertisements', 'Damaged manhole cover', 'Road damage', 'Pile of shared bikes', 'Pile of bricks', 'Fallen trash can', 'Damaged signboard', 'Discarded cardboard', 'Missing shrubbery', 'Toppled shared motorcycles', 'Toppled shared bicycles', 'Residential fire', 'Non-decorative tree hangings', 'Street-side car repairing', 'Burning trash and leaves', 'Unauthorized Storefront Expansion', 'Damaged billboards', 'Overflowing trash bins', 'Accumulated rubble', 'Displaced manhole cover', 'Itinerant vendors', 'Damaged crash barrels', 'Damaged traffic cones', 'Damaged storm drain grates', 'Missing manhole cover', 'Street-side car washing', 'Damaged trash bins', 'Damaged garbage cans', 'Abandoned furniture', 'Damaged traffic barriers', 'Missing street trees', 'Raised manhole cover', 'Illegal banners', 'Urban smog']


length_width_pool = [[425, 640], [480, 640], [640, 374], [500, 334], [640, 523], [514, 640]]

output_root = "/home/hc/workspace/real_graduate/MiniGPT-v2-chengguan/chengguan_dataset/refdet_v1.1"
os.makedirs(output_root, exist_ok=True)
dataset_root = "/data00/hc/dataset/chengguan/kejibu/trainval"

src_images_root  = os.path.join(dataset_root, "JPEGImages")
src_annotations_root = os.path.join(dataset_root, "JsonAnnotations")

all_annotations = glob(os.path.join(src_annotations_root, "*.json"))

new_annotations = []

zero_valid, many_valid, too_big = 0, 0, 0

for annotation_path in tqdm(all_annotations):

    with open(annotation_path, "r") as f:
        annotation = json.load(f)

    image_name = annotation["image"]
    image_path = os.path.join(src_images_root, image_name)
    image = cv2.imread(image_path)

    height, width, _ = image.shape
    objects = annotation["objects"]
    all_valid_objects = [obj for obj in objects if obj['name'] in curious_categories]

    if len(all_valid_objects) == 0:
        zero_valid += 1
        continue

    if len(all_valid_objects) > 5:
        many_valid += 1
        continue

    if height > 640 or width > 640:
        too_big += 1
        continue

    for obj in all_valid_objects:
        ann = {
            "img_id": image_name.split(".")[0],
            "sents": curious_categories_tranlation[curious_categories.index(obj['name'])],
            "bbox": obj["bbox"],
            "height": height,
            "width": width
        }
        new_annotations.append(ann)

print(zero_valid, many_valid, too_big, len(all_annotations) - zero_valid - many_valid - too_big)

# 9908 502 5936 1053

with open(os.path.join(output_root, "refcoco_test.json"), "w") as f:
    json.dump(new_annotations, f, indent=4)


            





