import os
import json
from glob import glob
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import cv2

curious_categories = ['破损绿化护栏', '破损户外广告', '井盖下沉', '平铺晾晒', '堆积垃圾', '打包垃圾', '破损防撞桩', '乞讨卖艺', '无序堆积', '破损红绿灯', '破损交通指示牌', '涂鸦广告', '破损井盖', '道路破损', '共享单车堆积', '砖块堆积', '倒伏垃圾桶', '破损牌匾', '废弃纸板', '灌木缺失', '倒伏共享摩托车', '倒伏共享自行车', '住宅起火', '非装饰性树挂', '临街修车', '焚烧垃圾树叶', '出店', '破损广告牌', '满溢垃圾桶', '渣土积存', '井盖移位', '游商', '破损防撞桶', '破损交通锥', '破损雨水篦子', '缺失井盖', '临街洗车', '破损垃圾桶', '破损垃圾箱', '废弃家具', '破损交通护栏', '行树缺株', '井盖凸起', '违法横幅', '城市烟雾']

curious_categories_tranlation = ['Damaged greenery guardrail', 'Damaged outdoor advertising', 'Sunken manhole cover', 'Flat drying', 'Accumulated garbage', 'Bagged garbage', 'Damaged bollard', 'Begging and street performing', 'Disordered accumulation', 'Damaged traffic light', 'Damaged traffic sign', 'Graffiti advertisements', 'Damaged manhole cover', 'Road damage', 'Pile of shared bikes', 'Pile of bricks', 'Fallen trash can', 'Damaged signboard', 'Discarded cardboard', 'Missing shrubbery', 'Toppled shared motorcycles', 'Toppled shared bicycles', 'Residential fire', 'Non-decorative tree hangings', 'Street-side car repairing', 'Burning trash and leaves', 'Unauthorized Storefront Expansion', 'Damaged billboards', 'Overflowing trash bins', 'Accumulated rubble', 'Displaced manhole cover', 'Itinerant vendors', 'Damaged crash barrels', 'Damaged traffic cones', 'Damaged storm drain grates', 'Missing manhole cover', 'Street-side car washing', 'Damaged trash bins', 'Damaged garbage cans', 'Abandoned furniture', 'Damaged traffic barriers', 'Missing street trees', 'Raised manhole cover', 'Illegal banners', 'Urban smog']


splits = ['train', 'val', 'test']

input_root = Path("/home/hc/workspace/real_graduate/MiniGPT-v2-chengguan/chengguan_dataset/base/v1")
output_root = Path("/home/hc/workspace/real_graduate/MiniGPT-v2-chengguan/chengguan_dataset/refdet_v3")

def cal_iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    x_left = max(xmin1, xmin2)
    y_top = max(ymin1, ymin2)
    x_right = min(xmax1, xmax2)
    y_bottom = min(ymax1, ymax2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    bbox2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

def merge_bboxes(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    x_left = min(xmin1, xmin2)
    y_top = min(ymin1, ymin2)
    x_right = max(xmax1, xmax2)
    y_bottom = max(ymax1, ymax2)

    return (x_left, y_top, x_right, y_bottom)

def merge_overlapping_bboxes(bboxes):
    merged = True
    while merged:
        merged = False
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if cal_iou(bboxes[i], bboxes[j]) > 0:
                    new_bbox = merge_bboxes(bboxes[i], bboxes[j])
                    bboxes.pop(j)
                    bboxes.pop(i)
                    bboxes.append(new_bbox)
                    merged = True
                    break
            if merged:
                break
    return bboxes

def find_max_subimage_area(length, width, bboxes):
    # 这种方法对于一些情况，无法返回面积最大的sub_image，但可以保证每个sub_image都仅包含对应的bbox
    # 初始化结果列表
    results = []
    
    # 遍历每个bbox
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        
        # 初始化可能的最大边界
        left_bound = 0
        right_bound = width
        top_bound = 0
        bottom_bound = length
        
        # 检查其他bbox，调整当前bbox的边界
        for j, other_bbox in enumerate(bboxes):
            if i == j:
                continue  # 跳过自身
            other_xmin, other_ymin, other_xmax, other_ymax = other_bbox
            
            # 调整边界
            if other_xmax <= xmin:
                left_bound = max(left_bound, other_xmax)
            if other_xmin >= xmax:
                right_bound = min(right_bound, other_xmin)
            if other_ymax <= ymin:
                top_bound = max(top_bound, other_ymax)
            if other_ymin >= ymax:
                bottom_bound = min(bottom_bound, other_ymin)
        
        # 计算并添加扩展后的bbox
        expanded_bbox = [left_bound, top_bound, right_bound, bottom_bound]
        results.append(expanded_bbox)
    
    return results


if __name__ == '__main__':

    img_idx = 0
    for split in splits:
        input_image_dir = input_root / split / "images"
        input_ann_path = input_root / split / "annotations.json"

        output_image_dir = output_root / split / "images"
        output_ann_path = output_root / split / "annotations.json"

        output_image_dir.mkdir(parents=True, exist_ok=True)

        with open(input_ann_path, "r") as f:
            annotations = json.load(f)

        all_new_ann = []
        for ann in tqdm(annotations):
            image_name = ann['image']
            image_path = input_image_dir / image_name
            image = Image.open(image_path)

            width, height = image.size
            objects = ann['objects']

            from collections import defaultdict
            name_obj_map = defaultdict(list)

            for obj in objects:
                if obj['name'] not in curious_categories:
                    continue
                name_obj_map[obj['name']].append(obj)

            for i, (name, obj_list) in enumerate(name_obj_map.items()):

                if len(obj_list) == 1:
                    new_image_path = output_image_dir / f"{img_idx:06d}.jpg"
                    
                    try:
                        image.convert("RGB").save(new_image_path)
                    except Exception as e:
                        print(e)
                        continue

                    new_ann = {
                        'img_id': img_idx,
                        "sents": curious_categories_tranlation[curious_categories.index(name)],
                        "bbox": obj_list[0]['bbox'],
                        "height": height,
                        "width": width,
                    }
                    all_new_ann.append(new_ann)
                    img_idx += 1
                    continue

                bboxes = [obj['bbox'] for obj in obj_list]
                bboxes = merge_overlapping_bboxes(bboxes)
                sub_image_bboxes = find_max_subimage_area(height, width, bboxes)

                for j, sub_image_bbox in enumerate(sub_image_bboxes):
                    new_image_path = output_image_dir / f"{img_idx:06d}.jpg"
                    try:
                        sub_image = image.crop(sub_image_bbox).convert('RGB')
                        sub_image.save(new_image_path)
                    except Exception as e:
                        print(e)
                        continue
                    
                    new_bbox = [
                        bboxes[j][0] - sub_image_bbox[0],
                        bboxes[j][1] - sub_image_bbox[1],
                        bboxes[j][2] - sub_image_bbox[0],
                        bboxes[j][3] - sub_image_bbox[1]
                    ]

                    new_ann = {
                        'img_id': img_idx,
                        "sents": curious_categories_tranlation[curious_categories.index(name)],
                        "bbox": new_bbox,
                        "height": sub_image_bbox[3] - sub_image_bbox[1],
                        "width": sub_image_bbox[2] - sub_image_bbox[0],
                    }

                    all_new_ann.append(new_ann)
                    img_idx += 1


        with open(output_ann_path, "w") as f:
            json.dump(all_new_ann, f, indent=4)

