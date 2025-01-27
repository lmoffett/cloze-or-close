# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import logging
import os
import pathlib
import random

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from collections import namedtuple

from clz_or_cls import byt5, datasets as corc_ds
from clz_or_cls.byt5 import (
    RecoveryDataModule,
    T5Lightning,
    wandb_default_trainer_for)

from . import cli_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(
    dataset,
    ds_name,
    model_size="base",
    model_arch="byt5",
    batch_size=128,
    wandb=True,
    pretrained_path=False,
    lr=1e-3,
    **kwargs,
):

    random.seed(123456)
    torch.manual_seed(123456)
    np.random.seed(123456)

    MAX_INPUT_LENGTH = 40
    logger.info("initializing training environment")

    model = T5Lightning(
        size=model_size,
        model_arch=model_arch,
        pretrained_path=pretrained_path,
        lr=lr,
        max_input_length=MAX_INPUT_LENGTH,
    )
    tokenizer = model.tokenizer

    ds_loader = dataset.map(
        corc_ds.tokenize_recovery_batch,
        batch_size=256,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_input_length": MAX_INPUT_LENGTH},
    )

    datamodule = RecoveryDataModule(ds_loader, batch_size=batch_size, num_workers=48)

    checkpoint_path = (
        pathlib.Path(os.environ['CORC_CHECKPOINT_DIR']) / "adword-recovery" / f"{model_arch}-{model_size}"
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    trainer = wandb_default_trainer_for(
        "perturbation-recovery",
        f"{ds_name}_{model_arch}-{model_size}",
        checkpoint_name=ds_name.replace("+", "-"),
        max_epochs=30,
        save_dir=os.environ['WANDB_DIR'],
        offline=(not wandb),
        grad_batches=192 // batch_size,
        checkpoint_dir=str(checkpoint_path),
    )

    torch.set_float32_matmul_precision("medium")

    logger.info("prerunning validation to get a baseline")
    trainer.validate(model, datamodule=datamodule)
    # Train the example model
    logger.info("kicking off training")
    trainer.fit(model, datamodule=datamodule)

    model_save_path = byt5.model_cache_path(
        model, name=f"adword-recovery", run_spec=[f"{model_arch}-{model_size}", ds_name]
    )
    logger.info("training complete; saving model to %s", model_save_path)
    model.model.save_pretrained(model_save_path)


if __name__ == "__main__":

    p = namedtuple(
        "params_set",
        ["train_dataset", "attack_type", "include_context", "context"],
        defaults=[None, None, False, None],
    )

    param_sets = [
        p("repeated", None),
        p("legit_extended", "visual"),
        p("dces", "visual"),
        p("ices", "visual"),
        p("legit_extended+dces_full", "visual"),
        p("legit_extended+ices_full", "visual"),
        p("ices+dces_full", "visual"),
        p("legit_extended+ices+dces_full", "visual"),
        p("zeroe_noise", "typo"),
        p("zeroe_typo", "typo"),
        p("anthro_typo", "typo"),
        p("zeroe_noise+zeroe_typo_full", "typo"),
        p("zeroe_noise+anthro_typo_full", "typo"),
        p("zeroe_typo+anthro_typo_full", "typo"),
        p("zeroe_noise+zeroe_typo+anthro_typo_full", "typo"),
        p("anthro_phonetic", "phonetic"),
        p("phonee", "phonetic"),
        p("zeroe_phonetic", "phonetic"),
        p("anthro_phonetic+phonee_full", "phonetic"),
        p("anthro_phonetic+zeroe_phonetic_full", "phonetic"),
        p("phonee+zeroe_phonetic_full", "phonetic"),
        p("anthro_phonetic+phonee+zeroe_phonetic_full", "phonetic"),
        p("visual", "visual"),
        p("phonetic", "phonetic"),
        p("typo", "typo"),
        p("visual+phonetic_full", "mixed"),
        p("visual+typo_full", "mixed"),
        p("phonetic+typo_full", "mixed"),
        p("visual+phonetic+typo_full", "mixed"),
    ]

    argparser = cli_utils.ArrayArgparser(param_sets)

    argparser.add_argument("--model-arch", default="byt5")
    argparser.add_argument("--model-size", default="base")
    argparser.add_argument("--batch-size", default=128, type=int)
    argparser.add_argument("--lr", default=1e-4, type=float)
    argparser.add_argument("--no-pretraining", default=False, action="store_true")

    args = argparser.parse_args()
    name = args.train_dataset

    if args.include_context:
        name = name + "_ctx"
        dataset = corc_ds.geneterated_ds_ctx(args.train_dataset, ctx=args.context)
    else:
        dataset = corc_ds.generated_ds(args.train_dataset)

    if args.lr != 1e-4:
        name = name + f"_lr-{args.lr}"

    varz = vars(args)

    if not args.no_pretraining:
        varz.update(
            {
                "pretrained_path": pathlib.Path(os.environ['CORC_TRAINED_MODEL_DIR'])
                / "adword-recovery"
                / "byt5-base"
                / "repeated_early"
            }
        )

    main(dataset, name, **varz)
