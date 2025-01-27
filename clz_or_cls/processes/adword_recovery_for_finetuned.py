# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import logging
import os

import dotenv
import pandas as pd

dotenv.load_dotenv()

from collections import namedtuple
from pathlib import Path

from transformers import AutoTokenizer

from .. import datasets as corc_ds
from .. import recovery
from .cli_utils import ArrayArgparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))


p = namedtuple("params_set", ["train", "test", "ctx"], defaults=[None, None, None])

params = [
    # Baseline
    p("repeated", "repeated"),
    p("repeated", "legit_extended"),
    p("repeated", "dces"),
    p("repeated", "ices"),
    p("repeated", "legit_extended+ices+dces_full"),
    p("repeated", "zeroe_noise"),
    p("repeated", "zeroe_typo"),
    p("repeated", "anthro_typo"),
    p("repeated", "zeroe_noise+zeroe_typo+anthro_typo_full"),
    p("repeated", "anthro_phonetic"),
    p("repeated", "phonee"),
    p("repeated", "zeroe_phonetic"),
    p("repeated", "anthro_phonetic+phonee+zeroe_phonetic_full"),
    p("repeated", "visual"),
    p("repeated", "phonetic"),
    p("repeated", "typo"),
    p("repeated", "visual+phonetic+typo_full"),
    # Visual
    p("legit_extended", "legit_extended"),
    p("legit_extended", "dces"),
    p("legit_extended", "ices"),
    p("legit_extended", "legit_extended+ices+dces_full"),
    p("dces", "legit_extended"),
    p("dces", "dces"),
    p("dces", "ices"),
    p("dces", "legit_extended+ices+dces_full"),
    p("ices", "legit_extended"),
    p("ices", "dces"),
    p("ices", "ices"),
    p("ices", "legit_extended+ices+dces_full"),
    p("legit_extended+dces_full", "legit_extended"),
    p("legit_extended+dces_full", "dces"),
    p("legit_extended+dces_full", "ices"),
    p("legit_extended+dces_full", "legit_extended+ices+dces_full"),
    p("legit_extended+ices_full", "legit_extended"),
    p("legit_extended+ices_full", "dces"),
    p("legit_extended+ices_full", "ices"),
    p("legit_extended+ices_full", "legit_extended+ices+dces_full"),
    p("ices+dces_full", "legit_extended"),
    p("ices+dces_full", "dces"),
    p("ices+dces_full", "ices"),
    p("ices+dces_full", "legit_extended+ices+dces_full"),
    p("legit_extended+ices+dces_full", "legit_extended"),
    p("legit_extended+ices+dces_full", "dces"),
    p("legit_extended+ices+dces_full", "ices"),
    p("legit_extended+ices+dces_full", "legit_extended+ices+dces_full"),
    # Typo
    p("zeroe_noise", "zeroe_noise"),
    p("zeroe_noise", "zeroe_typo"),
    p("zeroe_noise", "anthro_typo"),
    p("zeroe_noise", "zeroe_noise+zeroe_typo+anthro_typo_full"),
    p("zeroe_typo", "zeroe_noise"),
    p("zeroe_typo", "zeroe_typo"),
    p("zeroe_typo", "anthro_typo"),
    p("zeroe_typo", "zeroe_noise+zeroe_typo+anthro_typo_full"),
    p("anthro_typo", "zeroe_noise"),
    p("anthro_typo", "zeroe_typo"),
    p("anthro_typo", "anthro_typo"),
    p("anthro_typo", "zeroe_noise+zeroe_typo+anthro_typo_full"),
    p("zeroe_noise+zeroe_typo_full", "zeroe_noise"),
    p("zeroe_noise+zeroe_typo_full", "zeroe_typo"),
    p("zeroe_noise+zeroe_typo_full", "anthro_typo"),
    p("zeroe_noise+zeroe_typo_full", "zeroe_noise+zeroe_typo+anthro_typo_full"),
    p("zeroe_noise+anthro_typo_full", "zeroe_noise"),
    p("zeroe_noise+anthro_typo_full", "zeroe_typo"),
    p("zeroe_noise+anthro_typo_full", "anthro_typo"),
    p("zeroe_noise+anthro_typo_full", "zeroe_noise+zeroe_typo+anthro_typo_full"),
    p("zeroe_typo+anthro_typo_full", "zeroe_noise"),
    p("zeroe_typo+anthro_typo_full", "zeroe_typo"),
    p("zeroe_typo+anthro_typo_full", "anthro_typo"),
    p("zeroe_typo+anthro_typo_full", "zeroe_noise+zeroe_typo+anthro_typo_full"),
    p("zeroe_noise+zeroe_typo+anthro_typo_full", "zeroe_noise"),
    p("zeroe_noise+zeroe_typo+anthro_typo_full", "zeroe_typo"),
    p("zeroe_noise+zeroe_typo+anthro_typo_full", "anthro_typo"),
    p(
        "zeroe_noise+zeroe_typo+anthro_typo_full",
        "zeroe_noise+zeroe_typo+anthro_typo_full",
    ),
    # Phonetic
    p("anthro_phonetic", "anthro_phonetic"),
    p("anthro_phonetic", "phonee"),
    p("anthro_phonetic", "zeroe_phonetic"),
    p("anthro_phonetic", "anthro_phonetic+phonee+zeroe_phonetic_full"),
    p("phonee", "anthro_phonetic"),
    p("phonee", "phonee"),
    p("phonee", "zeroe_phonetic"),
    p("phonee", "anthro_phonetic+phonee+zeroe_phonetic_full"),
    p("zeroe_phonetic", "anthro_phonetic"),
    p("zeroe_phonetic", "phonee"),
    p("zeroe_phonetic", "zeroe_phonetic"),
    p("zeroe_phonetic", "anthro_phonetic+phonee+zeroe_phonetic_full"),
    p("anthro_phonetic+phonee_full", "anthro_phonetic"),
    p("anthro_phonetic+phonee_full", "phonee"),
    p("anthro_phonetic+phonee_full", "zeroe_phonetic"),
    p("anthro_phonetic+phonee_full", "anthro_phonetic+phonee+zeroe_phonetic_full"),
    p("anthro_phonetic+zeroe_phonetic_full", "anthro_phonetic"),
    p("anthro_phonetic+zeroe_phonetic_full", "phonee"),
    p("anthro_phonetic+zeroe_phonetic_full", "zeroe_phonetic"),
    p(
        "anthro_phonetic+zeroe_phonetic_full",
        "anthro_phonetic+phonee+zeroe_phonetic_full",
    ),
    p("phonee+zeroe_phonetic_full", "anthro_phonetic"),
    p("phonee+zeroe_phonetic_full", "phonee"),
    p("phonee+zeroe_phonetic_full", "zeroe_phonetic"),
    p("phonee+zeroe_phonetic_full", "anthro_phonetic+phonee+zeroe_phonetic_full"),
    p("anthro_phonetic+phonee+zeroe_phonetic_full", "anthro_phonetic"),
    p("anthro_phonetic+phonee+zeroe_phonetic_full", "phonee"),
    p("anthro_phonetic+phonee+zeroe_phonetic_full", "zeroe_phonetic"),
    p(
        "anthro_phonetic+phonee+zeroe_phonetic_full",
        "anthro_phonetic+phonee+zeroe_phonetic_full",
    ),
    # Class-Level
    p("visual", "visual"),
    p("visual", "phonetic"),
    p("visual", "typo"),
    p("visual", "visual+phonetic+typo_full"),
    p("phonetic", "visual"),
    p("phonetic", "phonetic"),
    p("phonetic", "typo"),
    p("phonetic", "visual+phonetic+typo_full"),
    p("typo", "visual"),
    p("typo", "phonetic"),
    p("typo", "typo"),
    p("typo", "visual+phonetic+typo_full"),
    p("visual+phonetic_full", "visual"),
    p("visual+phonetic_full", "phonetic"),
    p("visual+phonetic_full", "typo"),
    p("visual+phonetic_full", "visual+phonetic+typo_full"),
    p("visual+typo_full", "visual"),
    p("visual+typo_full", "phonetic"),
    p("visual+typo_full", "typo"),
    p("visual+typo_full", "visual+phonetic+typo_full"),
    p("phonetic+typo_full", "visual"),
    p("phonetic+typo_full", "phonetic"),
    p("phonetic+typo_full", "typo"),
    p("phonetic+typo_full", "visual+phonetic+typo_full"),
    p("visual+phonetic+typo_full", "visual"),
    p("visual+phonetic+typo_full", "phonetic"),
    p("visual+phonetic+typo_full", "typo"),
    p("visual+phonetic+typo_full", "visual+phonetic+typo_full"),
    p("phonetic", "mixed"),
    p("visual", "mixed"),
    p("typo", "mixed"),
    p("visual+phonetic+typo_full", "mixed"),
]

arg_parser = ArrayArgparser(params, allow_array_overflow=False)
arg_parser.add_argument(
    "--model-class",
    default="byt5-base",
    choices=["byt5-base", "byt5-large", "byt5-xl"],
)
arg_parser.add_argument("--sample-size", default=100_000_000, type=int)
args = arg_parser.parse_args()

logger.info("Loading Dataset %s", args.test)

if args.ctx is not None:
    if args.ctx == True:
        dataset = corc_ds.geneterated_ds_ctx(args.test)
    else:
        dataset = corc_ds.geneterated_ds_ctx(args.test, ctx=args.ctx)
else:
    dataset = corc_ds.generated_ds(args.test)

logger.info("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

logger.info("Processing Dataset for Recovery: %s", dataset)

ds_tokenized = dataset.map(
    corc_ds.tokenize_recovery_batch,
    batched=True,
    fn_kwargs={"tokenizer": tokenizer, "max_input_length": 40},
)

df = None
model_dir = Path(os.environ['CORC_TRAINED_MODEL_DIR']) / 'adword-recovery' / args.model_class
if args.ctx:
    if args.ctx == True:
        model_name = args.train + "_ctx"
    else:
        logger.info("using redacted model")
        model_name = args.train + "_ctx_redacted"
else:
    model_name = args.train
model_name = model_name + "_early"
logger.info("Loading model from %s", model_dir / model_name)
model = recovery.load_recovery_model(model_name, model_dir=model_dir)

for split in "train", "test", "valid":

    logger.info("Running Recovery for %s", split)

    legit_recovery = recovery.recovery_job_for_dataset(
        dataset[split], tokenizer, model
    )

    recovery_df, accuracy = legit_recovery(sample_size=args.sample_size)

    logger.info(
        "Final Accuracy for Recovery on %s[%s] using %s: %s",
        args.test,
        split,
        args.train,
        accuracy,
    )

    recovery_df["split"] = split
    df = recovery_df if df is None else pd.concat([df, recovery_df])

df_path = Path(os.environ['CORC_RESULTS_DIR']) / "adword-recovery" / args.model_class
df_path.mkdir(parents=True, exist_ok=True)

df_fullpath = (
    df_path / (args.train + "--" + args.test)
).with_suffix(".csv")
df.to_csv(str(df_fullpath))
logger.info("Recovered dataframe written to %s", df_fullpath)
