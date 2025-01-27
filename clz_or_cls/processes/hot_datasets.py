# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

# Utility Script for materializing a HOT dataset for inspection

import logging
import os
import pathlib
import random
import sys

import dotenv
import numpy as np
import torch

dotenv.load_dotenv()

from collections import namedtuple

from .. import hot
from .cli_utils import ArrayArgparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))

logger.info("importing datasets")


p = namedtuple(
    "params_set",
    ["dataset_name"],
    defaults=[None],
)
params = [
    p('legit'),
    p('dces'),
    p('ices'),
    p('legit+dces'),
    p('legit+ices'),
    p('dces+ices'),
    p('legit+dces+ices'),
    p('visual+phonetic+typo'),
    p('legit+phonee+zeroe_typo'),
]

logger.info("parsing args")
arg_parser = ArrayArgparser(params)

run_params = arg_parser.parse_args()

# setup the seed based on the array parameters for reproducibility.
random.seed(123456)
torch.manual_seed(123456)
np.random.seed(123456)

logger.info("loading dataset")

dataset_name = run_params.dataset_name.replace('+', '_')

dataset_loader = getattr(hot, f'hot_speech_{dataset_name}')

ds_root = pathlib.Path(os.environ['CORC_DATASETS_HOT_DIR'])
ds_root.mkdir(exist_ok=True)

csv_path = (ds_root / dataset_name).with_suffix(".csv")
logger.info("writing to csv at %s", csv_path)
dataset_loader().to_csv(csv_path)
logger.info("csv_written")
