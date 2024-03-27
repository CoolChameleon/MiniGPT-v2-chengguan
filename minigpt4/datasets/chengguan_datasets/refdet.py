import os
import json
import random
import time
from collections import defaultdict
import re

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def computeIoU(bbox1, bbox2):
    """
    bbox的取值范围为[0, 100]，表示百分比
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou

class RefDetDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.vis_root = vis_root
        self.ann_path = ann_path

        self.dataset = []
        self.annotations = []
        self.load_annotations()

        self.instruction_pool = [
            "[cm-refer] {}",
            "[cm-refer] give me the location of {}",
            "[cm-refer] where is {} ?",
            "[cm-refer] from this image, tell me the location of {}",
            "[cm-refer] the location of {} is",
            "[cm-refer] could you tell me the location for {} ?",
            "[cm-refer] where can I locate the {} ?",
        ]


    def load_annotations(self):
        with open(self.ann_path, 'r') as f:
            self.annotations = json.load(f)

        self.dataset = [{
            "image": None,
            "refer_sentence": None,
            "bbox": None,
        } for _ in range(len(self.annotations))]


    def prepare(self, idx):
        if self.dataset[idx]["image"] is not None:
            return
        
        ann = self.annotations[idx]
        image_path = os.path.join(self.vis_root, f"{ann['img_id']:06d}.jpg")
        image = Image.open(image_path).convert('RGB')
        image_orig_size = image.size
        image = self.vis_processor(image)

        sent = self.text_processor(ann['sents'])
        
        bbox = ann['bbox']
        image_new_size = [100, 100]
        bbox = [
            bbox[0] / image_orig_size[0] * image_new_size[0],
            bbox[1] / image_orig_size[1] * image_new_size[1],
            (bbox[0] + bbox[2]) / image_orig_size[0] * image_new_size[0],
            (bbox[1] + bbox[3]) / image_orig_size[1] * image_new_size[1]
        ]
        bbox = [int(x) for x in bbox]
        bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)

        self.dataset[idx]["image"] = image
        self.dataset[idx]["refer_sentence"] = sent
        self.dataset[idx]["bbox"] = bbox


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        self.prepare(idx)
        data = self.dataset[idx]

        instruction = random.choice(self.instruction_pool).format(data['refer_sentence'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": data['bbox'],
        }
    

    def eval(self, gt_answers, model_answers):
        assert len(gt_answers) == len(model_answers), f"Number of ground truth answers ({len(gt_answers)}) and model answers ({len(model_answers)}) should be the same"

        pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'

        results = defaultdict(lambda : {"tp": 0, "total": 0, "sum_iou": 0})
        
        for i, (gt_answer, model_answer) in enumerate(zip(gt_answers, model_answers)):
            if not re.match(pattern, gt_answer):
                continue
                
            res = results[self.annotations[i]["sents"]]

            if not re.match(pattern, model_answer):
                res["total"] += 1
                continue

            gt_bbox = [int(x) for x in re.findall(r'\d{1,3}', gt_answer)]
            model_bbox = [int(x) for x in re.findall(r'\d{1,3}', model_answer)]

            iou = computeIoU(gt_bbox, model_bbox)
            res["sum_iou"] += iou
            if iou > 0.5:
                res["tp"] += 1
            res["total"] += 1

        acc = sum(res["tp"] for res in results.values()) / sum(res["total"] for res in results.values())
        mean_iou = sum(res["sum_iou"] for res in results.values()) / sum(res["total"] for res in results.values())

        return {
            "summary": {
                "accuracy": acc,
                "mean_iou": mean_iou,
            },
            "details": results
        }



class InvRefDetDataset(RefDetDataset):
    def __init__(self, *args, **kwargs):
        super(InvRefDetDataset, self).__init__(*args, **kwargs)

        self.instruction_pool = [
            "[cm-identify] {}",
            "[cm-identify] what city management incident is in this location {}",
            "[cm-identify] identify the city management incident present at this location {}",
            "[cm-identify] describe this city management incident in {}",
            "[cm-identify] this {} is",
            "[cm-identify] the city management incident in {} is",            
        ]
    
    def __getitem__(self, idx):
        self.prepare(idx)
        data = self.dataset[idx]

        instruction = random.choice(self.instruction_pool).format(data['bbox'])

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": self.text_processor(data['refer_sentence']),
        }



class CMCaptionDataset(RefDetDataset):
    def __init__(self, *args, **kwargs):
        super(CMCaptionDataset, self).__init__(*args, **kwargs)

        self.instruction_pool = [
            "[cm-caption] What kind of city management incident does this picture describe?",
            "[cm-caption] Briefly describe the city management event in this picture."
            "[cm-caption] What happened? Summarize it simply in one phrase."
        ]
    
    def __getitem__(self, idx):
        self.prepare(idx)
        data = self.dataset[idx]

        instruction = random.choice(self.instruction_pool)

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": self.text_processor(data['refer_sentence']),
        }