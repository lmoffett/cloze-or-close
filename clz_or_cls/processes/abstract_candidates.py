# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import ast
import logging
import os
import random

import dotenv
import numpy as np
import pandas as pd
import torch

dotenv.load_dotenv()

from collections import namedtuple
from pathlib import Path

from tqdm.auto import tqdm

from .. import perturbation
from .cli_utils import ArrayArgparser

tqdm.pandas()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))

p = namedtuple(
    "params_set",
    ["dataset_name"],
    defaults=[None],
)
params = [
    p('legit'),
    p('dces'),
    p('phonee'),
    p('zeroe_phonetic'),
    p('zeroe_noise'),
    p('zeroe_typo'),
]

logger.info("parsing args")
arg_parser = ArrayArgparser(params)

args = arg_parser.parse_args()

# setup the seed based on the array parameters for reproducibility.
random.seed(123456)
torch.manual_seed(123456)
np.random.seed(123456)

logger.info("loading dataset")
prep_path = Path(os.environ['CORC_DATASETS_PREP_DIR'])
candidates = pd.read_csv(prep_path/'abstract'/'abstract_perturbation_metadata.csv')

if args.dataset_name == 'phonee':
    strategy = perturbation.PhoneEStrategy()
elif args.dataset_name == 'legit':
    strategy = perturbation.LegitStrategy()
elif args.dataset_name == 'dces':
    strategy = perturbation.DcesStrategy()
elif args.dataset_name == 'zeroe_phonetic':
    strategy = perturbation.ZeroePhoneticStrategy()
elif args.dataset_name == 'zeroe_noise':
    strategy = perturbation.MultiStrategy(
        [
            perturbation.WordLevelZeroeStrategy("inner-swap"),
            perturbation.DeleteStrategy(),
            perturbation.LetterLevelZeroeStrategy("intrude", letter_prob=0.3),
        ]
    )
elif args.dataset_name == 'zeroe_typo':
    strategy = perturbation.MultiStrategy(
    [
        perturbation.WordLevelZeroeStrategy("natural-typo"),
        perturbation.LetterLevelZeroeStrategy(
            "keyboard-typo", letter_prob=0.15
        ),
    ],
    selection_criteria="ordered")
else:
    raise NotImplemented(args.dataset_name)


def perturb_all(candidate_set):
    perturbations = {}

    for word in ast.literal_eval(candidate_set):
        perturbations[word]= strategy.perturb(word, n=1)[0]

    return perturbations

logger.info("perturbing candidates")
replacement_series = candidates['sorted_candidates'].progress_apply(perturb_all)
logger.info("candidates perturbed")

prep_path = Path(os.environ['CORC_DATASETS_PREP_DIR'])
csv_path = (prep_path / 'abstract' / f'abstract-{args.dataset_name}').with_suffix(".csv")
logger.info("writing to csv at %s", csv_path)
replacement_series.to_csv(csv_path)
logger.info("csv_written")
