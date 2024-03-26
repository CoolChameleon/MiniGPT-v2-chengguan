import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.datasets.chengguan_datasets.refdet import RefDetDataset, InvRefDetDataset, CMCaptionDataset


@registry.register_builder("refdet_v1")
class RefDetBuilder(BaseDatasetBuilder):
    dataset_cls = RefDetDataset
    DATASET_CONFIG_DICT = {"default": "../chengguan_config/datasets/refdet_v1.yaml"}

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        for split in ["train", "eval", "test"]:
            if split not in build_info:
                continue
            image_path = build_info[split].image_path
            ann_path = build_info[split].ann_path

            if not os.path.exists(image_path):
                warnings.warn("image path {} does not exist.".format(image_path))
            if not os.path.exists(ann_path):
                warnings.warn("ann path {} does not exist.".format(ann_path))

            # create datasets
            datasets[split] = self.dataset_cls(
                vis_processor=self.vis_processors[split],
                text_processor=self.text_processors[split],
                ann_path=ann_path,
                vis_root=image_path,
            )
        
        return datasets
    

@registry.register_builder("refdet_v3")
class RefDetV3Builder(RefDetBuilder):
    DATASET_CONFIG_DICT = {"default": "../chengguan_config/datasets/refdet_v3.yaml"}


@registry.register_builder("invrefdet")
class InvRefDetBuilder(RefDetBuilder):
    dataset_cls = InvRefDetDataset
    DATASET_CONFIG_DICT = {"default": "../chengguan_config/datasets/invrefdet.yaml"}


@registry.register_builder("cmcaption")
class CMCaptionBuilder(RefDetBuilder):
    dataset_cls = CMCaptionDataset
    DATASET_CONFIG_DICT = {"default": "../chengguan_config/datasets/cmcaption.yaml"}