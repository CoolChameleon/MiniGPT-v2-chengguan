import os, json, shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path

splits = ["train", "val", "test"]

dataset_root = Path("/home/hc/workspace/real_graduate/MiniGPT-v2-chengguan/chengguan_dataset/caption_v1")
old_annotations_root = Path("/home/hc/workspace/real_graduate/MiniGPT-v2-chengguan/chengguan_dataset/refdet_v3")

if __name__ == "__main__":

    for split in splits:

        split_dir = dataset_root / split
        if split == "test":
            ann_path = old_annotations_root / split / "refcoco_test.json"
        else:
            ann_path = old_annotations_root / split / "annotations.json"

        with open(ann_path, "r") as f:
            annotations = json.load(f)

        new_annotations = []
        for ann in tqdm(annotations):
            bbox = ann['bbox']
            height, width = ann['height'], ann['width']
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 0.25 * height * width:
                new_annotations.append(ann)

        print(f"Number of {split} annotations before filtering: {len(annotations)}; after filtering: {len(new_annotations)}")
        
        if split == "test":
            new_annotations_path = split_dir / "refcoco_test.json"
        else:
            new_annotations_path = split_dir / "annotations.json"
        
        with open(new_annotations_path, "w") as f:
            json.dump(new_annotations, f, indent=4)


        
