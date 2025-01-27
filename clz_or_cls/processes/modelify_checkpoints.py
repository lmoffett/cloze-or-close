# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import logging
import os
import pathlib
import sys

import dotenv

dotenv.load_dotenv()

from collections import namedtuple

from .. import byt5
from .cli_utils import ArrayArgparser

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

p = namedtuple("model_defs", ["ds_name"])

pairs = [
    p("repeated"),
    p("legit_extended"),
    p("dces"),
    p("ices"),
    p("legit_extended+dces_full"),
    p("legit_extended+ices_full"),
    p("ices+dces_full"),
    p("legit_extended+ices+dces_full"),
    p("zeroe_noise"),
    p("zeroe_typo"),
    p("anthro_typo"),
    p("zeroe_noise+zeroe_typo_full"),
    p("zeroe_noise+anthro_typo_full"),
    p("zeroe_typo+anthro_typo_full"),
    p("zeroe_noise+zeroe_typo+anthro_typo_full"),
    p("anthro_phonetic"),
    p("phonee"),
    p("zeroe_phonetic"),
    p("anthro_phonetic+phonee_full"),
    p("anthro_phonetic+zeroe_phonetic_full"),
    p("phonee+zeroe_phonetic_full"),
    p("anthro_phonetic+phonee+zeroe_phonetic_full"),
    p("visual"),
    p("phonetic"),
    p("typo"),
    p("visual+phonetic_full"),
    p("visual+typo_full"),
    p("phonetic+typo_full"),
    p("visual+phonetic+typo_full"),
]

arg_parser = ArrayArgparser(pairs)
arg_parser.add_argument("--model-name", default="byt5-base", type=str)

args = arg_parser.parse_args()

model_path = (
    pathlib.Path(os.environ['CORC_CHECKPOINT_DIR'])
    / "adword-recovery"
    / args.model_name
    / args.ds_name.replace("+", "-")
).with_suffix(".ckpt")

logger.info("loading model %s-%s", args.model_name, args.ds_name)
logger.info("from model path %s", model_path)
lightning_model = byt5.T5Lightning.load_from_checkpoint(model_path)

model_save_path = byt5.model_cache_path(
    lightning_model,
    name=f"adword-recovery",
    run_spec=[args.model_name, args.ds_name + "_early"],
)

logger.info("saving model to %s", model_save_path)
lightning_model.model.save_pretrained(model_save_path)

logger.info("model restoration complete")
