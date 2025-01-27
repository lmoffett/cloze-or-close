# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

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

from datasets.combine import concatenate_datasets

import clz_or_cls.datasets
from datasets import DatasetDict

from .cli_utils import ArrayArgparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))

p = namedtuple(
    "params_set",
    ["attack_name", "drop_duplicates", "proportion"],
    defaults=[None, False, False],
)
params = [
    p("repeated", False, None),
    # Visual Attacks
    p("legit_extended", False, None),
    p("dces", False, None),
    p("ices", False, None),
    p("legit_extended+dces", False, False),
    p("legit_extended+ices", False, False),
    p("ices+dces", False, False),
    p("legit_extended+ices+dces", False, False),
    # Typo Attacks
    p("zeroe_noise", False, None),
    p("zeroe_typo", False, None),
    p("anthro_typo", False, None),
    p("zeroe_noise+zeroe_typo", False, False),
    p("zeroe_noise+anthro_typo", False, False),
    p("zeroe_typo+anthro_typo", False, False),
    p("zeroe_noise+zeroe_typo+anthro_typo", False, False),
    # Phonetic Attacks
    p("anthro_phonetic", False, None),
    p("phonee", False, None),
    p("zeroe_phonetic", False, None),
    p("anthro_phonetic+phonee", False, False),
    p("anthro_phonetic+zeroe_phonetic", False, False),
    p("phonee+zeroe_phonetic", False, False),
    p("anthro_phonetic+phonee+zeroe_phonetic", False, False),
    # Unproportioned Group Attacks
    p("visual", False, None),
    p("phonetic", False, None),
    p("typo", False, None),
    # Unproportioned Multi-Group Attacks
    p("visual+phonetic", True, False),
    p("visual+typo", True, False),
    p("phonetic+typo", True, False),
    p("visual+phonetic+typo", True, False),
    p("mixed", True, False),
]

logger.info("parsing args")
arg_parser = ArrayArgparser(params)

arg_parser.add_argument('--sample', type=int, default=None, help="sample a limited number of records")
run_params = arg_parser.parse_args()

# setup the seed based on the array parameters for reproducibility.
random.seed(123456)
torch.manual_seed(123456)
np.random.seed(123456)

logger.info("loading dataset")
if "+" in run_params.attack_name:
    attack_names = run_params.attack_name.split("+")
    datasets = [
        clz_or_cls.datasets.generated_ds(attack_name) for attack_name in attack_names
    ]

    if run_params.proportion:
        dataset_dict = clz_or_cls.datasets.as_proportional_dataset(
            datasets, drop_duplicates=run_params.drop_duplicates
        )
        attack_name = run_params.attack_name
    else:
        dataset_dict = DatasetDict(
            {
                split: concatenate_datasets([ds[split] for ds in datasets])
                for split in ["train", "test", "valid"]
            }
        )
        attack_name = run_params.attack_name + "_full"
else:
    # python get function by name
    # if this is all constructed correctly, doing this once on the GPU should cache the result
    dataset_loader = getattr(clz_or_cls.datasets, run_params.attack_name)
    dataset_dict = dataset_loader()
    attack_name = run_params.attack_name

if run_params.sample is not None:
    dataset_dict = DatasetDict({
        split_name: dataset.select(range(min(len(dataset), run_params.sample)))
        for split_name, dataset in dataset_dict.items()
    })

adword_dir = pathlib.Path(os.environ['CORC_DATASETS_ADWORD_DIR'])
adword_dir.mkdir(exist_ok=True)
ds_root = adword_dir / attack_name
ds_root.mkdir(exist_ok=True)

for split in dataset_dict.keys():
    csv_path = (ds_root / split).with_suffix(".csv")
    logger.info("writing to csv at %s", csv_path)
    dataset_dict[split].to_csv(csv_path)
    logger.info("csv_written")
