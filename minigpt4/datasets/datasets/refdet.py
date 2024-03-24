import os
import json
import random
import time
import itertools

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RefDetTrainDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.vis_root = vis_root
        self.ann_path = ann_path

        self.dataset = []
        self.annotations = []
        self.load_annotations()

        self.instruction_pool = [
            "[refer] {}",
            "[refer] give me the location of {}",
            "[refer] where is {} ?",
            "[refer] from this image, tell me the location of {}",
            "[refer] the location of {} is",
            "[refer] could you tell me the location for {} ?",
            "[refer] where can I locate the {} ?",
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