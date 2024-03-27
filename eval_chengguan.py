import os, re, json, argparse, random, time, shutil, logging
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minigpt4.common.config import Config
from minigpt4.common.registry import registry

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from minigpt4.common.utils import now
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


def prepare_texts(texts):
    conv_templates = [CONV_VISION_minigptv2.copy() for _ in range(len(texts))]
    for conv, text in zip(conv_templates, texts):
        conv.append_message(conv.roles[0], text)
    for conv in conv_templates:
        conv.append_message(conv.roles[1], None)
    texts = [conv.get_prompt() for conv in conv_templates]
    return texts

def main():
    args = parse_args()

    cfg = Config(args)

    save_path = Path(cfg.config.run.save_path) / now()
    save_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.cfg_path, save_path / "config.yaml")

    setup_logger()

    # build dataset
    datasets = {}
    datasets_config = cfg.datasets_cfg

    for name in datasets_config:
        dataset_config = datasets_config[name]
        builder = registry.get_builder_class(name)(dataset_config)
        dataset = builder.build_datasets()
        assert 'test' in dataset, f"Dataset {name} must have a test split"
        datasets[name] = dataset['test']

    # build model
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')
    model.eval()

    results = {}

    # run test
    for name, dataset in datasets.items():
        logging.info(f"Testing on {name} dataset...")

        batch_size = datasets_config[name].batch_size
        max_new_tokens = datasets_config[name].max_new_tokens
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        gt_answers = []
        model_answers = []

        logging.info(f"Generating answers for {len(dataset)} samples...")
        for data in tqdm(dataloader):
            images = data['image']
            texts = prepare_texts(data['instruction_input'])
            gt_answers.extend(data['answer'])

            model_answers.extend(model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False))

        logging.info(f"Evaluating answers for {len(dataset)} samples...")
        result = dataset.eval(gt_answers, model_answers)
        if "summary" in result:
            logging.info(f"Result on {name} dataset: {result['summary']}")
        else:
            logging.info(f"Result on {name} dataset: {result}")

        results[name] = result

    with open(save_path / "eval_results.json", "w") as f:
        json.dump(results, f, indent=4)         

if __name__ == '__main__':
    main()
