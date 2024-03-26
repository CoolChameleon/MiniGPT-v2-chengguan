import os, re, json, argparse, random
from collections import defaultdict
from PIL import Image

from torch.utils.data import DataLoader

from minigpt4.common.config import Config
from minigpt4.common.registry import registry

