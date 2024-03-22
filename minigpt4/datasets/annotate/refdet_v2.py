import os
import json
from glob import glob
from tqdm import tqdm
from ipdb import set_trace
import xml.etree.ElementTree as ET
from PIL import Image
import random
import shutil

curious_categories = ['破损绿化护栏', '破损户外广告', '井盖下沉', '平铺晾晒', '堆积垃圾', '打包垃圾', '破损防撞桩', '乞讨卖艺', '无序堆积', '破损红绿灯', '破损交通指示牌', '涂鸦广告', '破损井盖', '道路破损', '共享单车堆积', '砖块堆积', '倒伏垃圾桶', '破损牌匾', '废弃纸板', '灌木缺失', '倒伏共享摩托车', '倒伏共享自行车', '住宅起火', '非装饰性树挂', '临街修车', '焚烧垃圾树叶', '出店', '破损广告牌', '满溢垃圾桶', '渣土积存', '井盖移位', '游商', '破损防撞桶', '破损交通锥', '破损雨水篦子', '缺失井盖', '临街洗车', '破损垃圾桶', '破损垃圾箱', '废弃家具', '破损交通护栏', '行树缺株', '井盖凸起', '违法横幅', '城市烟雾']

curious_categories_tranlation = ['Damaged greenery guardrail', 'Damaged outdoor advertising', 'Sunken manhole cover', 'Flat drying', 'Accumulated garbage', 'Bagged garbage', 'Damaged bollard', 'Begging and street performing', 'Disordered accumulation', 'Damaged traffic light', 'Damaged traffic sign', 'Graffiti advertisements', 'Damaged manhole cover', 'Road damage', 'Pile of shared bikes', 'Pile of bricks', 'Fallen trash can', 'Damaged signboard', 'Discarded cardboard', 'Missing shrubbery', 'Toppled shared motorcycles', 'Toppled shared bicycles', 'Residential fire', 'Non-decorative tree hangings', 'Street-side car repairing', 'Burning trash and leaves', 'Unauthorized Storefront Expansion', 'Damaged billboards', 'Overflowing trash bins', 'Accumulated rubble', 'Displaced manhole cover', 'Itinerant vendors', 'Damaged crash barrels', 'Damaged traffic cones', 'Damaged storm drain grates', 'Missing manhole cover', 'Street-side car washing', 'Damaged trash bins', 'Damaged garbage cans', 'Abandoned furniture', 'Damaged traffic barriers', 'Missing street trees', 'Raised manhole cover', 'Illegal banners', 'Urban smog']


output_root = "/home/hc/workspace/real_graduate/MiniGPT-v2-chengguan/chengguan_dataset/refdet_v2"
output_image_root = os.path.join(output_root, "images")
output_annotation_root = os.path.join(output_root, "annotations")
os.makedirs(output_image_root, exist_ok=True)
os.makedirs(output_annotation_root, exist_ok=True)

dataset_root = "/data00/hc/dataset/chengguan/kejibu/trainval"

src_images_root  = os.path.join(dataset_root, "JPEGImages")
src_annotations_root = os.path.join(dataset_root, "JsonAnnotations")

all_annotations = glob(os.path.join(src_annotations_root, "*.json"))

new_annotations = []

zero_valid, many_valid, too_big = 0, 0, 0
img_idx = 0

for annotation_path in tqdm(all_annotations):

    with open(annotation_path, "r") as f:
        annotation = json.load(f)

    image_name = annotation["image"]
    image_path = os.path.join(src_images_root, image_name)
    image = Image.open(image_path)

    width, height = image.size
    objects = annotation["objects"]
    all_valid_objects = [obj for obj in objects if obj['name'] in curious_categories]

    if len(all_valid_objects) == 0:
        zero_valid += 1
        continue

    if len(all_valid_objects) > 5:
        many_valid += 1
        continue

    if height > 640 or width > 640:
        for obj in all_valid_objects:
            xmin, ymin, xmax, ymax = obj["bbox"]
            if ymax - ymin > 640 or xmax - xmin > 640:
                continue

            # from ipdb import set_trace
            # set_trace()
            
            left = random.randint(max(0, xmax - 640), xmin)
            top = random.randint(max(0, ymax - 640), ymin)
            right = random.randint(xmax, min(width, xmin + 640))
            bottom = random.randint(ymax, min(height, ymin + 640))

            try:
                cropped_image = image.crop((left, top, right, bottom)).convert("RGB")
                save_path = os.path.join(output_image_root, f"{img_idx:06d}.jpg")
                cropped_image.save(save_path)
            except Exception as e:
                print(e)
                continue

            ann = {
                "img_id": img_idx,
                "sents": curious_categories_tranlation[curious_categories.index(obj['name'])],
                "bbox": [xmin - left, ymin - top, xmax - left, ymax - top],
                "height": right - left,
                "width": bottom - top
            }

            new_annotations.append(ann)
            img_idx += 1
            
    else:
        shutil.copy(image_path, os.path.join(output_image_root, f"{img_idx:06d}.jpg"))

        for obj in all_valid_objects:
            ann = {
                "img_id": img_idx,
                "sents": curious_categories_tranlation[curious_categories.index(obj['name'])],
                "bbox": obj["bbox"],
                "height": height,
                "width": width
            }
            new_annotations.append(ann)

        img_idx += 1

with open(os.path.join(output_annotation_root, "all.json"), "w") as f:
    json.dump(new_annotations, f, indent=4)


            





