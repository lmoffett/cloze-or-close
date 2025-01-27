# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

# Training, inference, evaluation with ByT5

import logging
import os
import re
from collections import namedtuple
from pathlib import Path
from string import punctuation

import enchant
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import tqdm
from nltk.metrics import edit_distance
from nltk.tokenize import WhitespaceTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, MT5ForConditionalGeneration,
                          T5ForConditionalGeneration)

import datasets

en_dict = enchant.Dict("en_US")

logger = logging.getLogger(__name__)

# from string import punctuation
punctuation = "!#\"$%&'()*+,-./:;<=>?@[\\]^_`{|}~…––“”‘’—"


def model_cache_path(model, name=None, run_spec=[]):
    """
    Get the path for where a trained model should be saved.
    """
    name = model._get_name() if name is None else name
    path = Path(os.getenv("CORC_TRAINED_MODEL_DIR", None)) / name
    for subspec in run_spec:
        path = path / subspec

    return path


class T5Lightning(pl.LightningModule):
    """
    A pytorch lightning module wrapping a byt5 model for generation. Useful for finetuning.
    """

    def __init__(
        self,
        model_arch="byt5",
        size="base",
        lr=0.001,
        num_train_epochs=15,
        warmup_steps=1,
        pretrained_path=None,
        max_input_length=40,
        regularize_train=False,
    ):
        super().__init__()
        if model_arch == "byt5":
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"google/byt5-{size}" if not pretrained_path else pretrained_path
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"google/byt5-{size}", max_length=max_input_length
            )
        elif model_arch == "mt5":
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"google/mt5-{size}" if pretrained_path is None else pretrained_path
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"google/mt5-{size}", use_fast=False, max_length=max_input_length
            )
        else:
            raise ValueError("model must be one of byt5 or mt5")

        if regularize_train:
            assert type(regularize) == "set", "need the vocab to regularize"

        self.save_hyperparameters()

    def __hash_vectors(self, vec, weights):
        """
        Convert a vector to a unique hash by considering position importance.
        In this function, we use powers of 2 as weights for the positions.
        """

        return int(weighted_sum.item())

    def __check_vector_existence(self, new_tensor, old_vectors):
        """
        new_tensor: A 2D tensor where each row is a new vector.
        old_vectors: A 2D tensor where each row is a previously seen vector.

        Returns: A tensor of binary values indicating the presence (1) or absence (0)
                of each vector from new_tensor in old_vectors.
        """
        # Hash all old vectors and store in a set
        prime_base = 2

        weights = torch.pow(
            prime_base, torch.arange(0, vec.size(0), dtype=torch.float32)
        ).to(vec.device)
        weighted_sum = torch.sum(vec.float() * weights)
        old_hashes = set()
        for vec in old_vectors:
            old_hashes.add(self.__hash_vector(vec))

        # Check if any new vector's hash exists in the old hash set
        presence_flags = []
        for vec in new_tensor:
            if self.__hash_vector(vec) in old_hashes:
                presence_flags.append(1)
            else:
                presence_flags.append(0)

        return torch.tensor(presence_flags)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def common_step(self, batch, batch_idx):
        return self(**batch)

    def acc(self, batch, outputs):
        with torch.no_grad():

            batch_size, max_len = batch["input_ids"].shape
            output_ids = self.model.generate(
                batch["input_ids"],
                num_beams=4,
                num_return_sequences=1,
                early_stopping=True,
                max_length=max_len,
            )

            generated_words = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            target_words = self.tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )

            matches = [
                gen.lower() == input.lower()
                for gen, input in zip(generated_words, target_words)
            ]

            edit_dists = []
            edit_dist_percents = []
            for gen, input in zip(generated_words, target_words):
                edit_dist = edit_distance(gen, input)
                edit_dists.append(edit_dist)
                edit_dist_percents.append(edit_dist / len(input))

            num_match = sum(matches)
            edit_dist = np.mean(edit_dists)
            edit_dist_percent = np.mean(edit_dist_percents)
            logger.debug("num_match:%s, total: %s", num_match, len(generated_words))
            logger.debug(
                "Generated Samples: %s",
                list(zip(target_words[:10], generated_words[:10])),
            )

            acc = num_match / len(matches)

            return acc, edit_dist, edit_dist_percent

    def training_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs.loss
        self.log("training_loss", loss, prog_bar=True)

        acc, edit_dist, edit_dist_percent = self.acc(batch, outputs)
        self.log(
            "training_acc", acc, prog_bar=True, batch_size=batch["input_ids"].shape[0]
        )

        self.log("training_edit", edit_dist, batch_size=batch["input_ids"].shape[0])

        self.log(
            "training_edit_%", edit_dist_percent, batch_size=batch["input_ids"].shape[0]
        )

        return {
            "loss": loss,
            "acc": acc,
            "edit": edit_dist,
            "edit_%": edit_dist_percent,
        }

    def validation_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs.loss
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True)

        acc, edit_dist, edit_dist_percent = self.acc(batch, outputs)
        self.log(
            "validation_acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].shape[0],
        )

        self.log(
            "validation_edit",
            edit_dist,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )

        self.log(
            "validation_edit_%",
            edit_dist_percent,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )

        return {
            "loss": loss,
            "acc": acc,
            "edit": edit_dist,
            "edit_%": edit_dist_percent,
        }

    def test_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs.loss
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        acc, edit_dist, edit_dist_percent = self.acc(batch, outputs)
        self.log(
            "test_acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].shape[0],
        )

        self.log(
            "test_edit",
            edit_dist,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )

        self.log(
            "test_edit_%",
            edit_dist_percent,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )

        return {
            "loss": loss,
            "acc": acc,
            "edit": edit_dist,
            "edit_%": edit_dist_percent,
        }

    def configure_optimizers(self):
        # create optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        return {"optimizer": optimizer}


class RecoveryDataModule(pl.LightningDataModule):
    def __init__(self, ds, batch_size: int = 16, num_workers=1):
        super().__init__()
        self.ds = ds

        self.__dataloader_for_set = lambda set_name, shuffle: DataLoader(
            self.ds[set_name],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def setup(self, stage=None):
        self.ds.set_format(
            type="torch", columns=["input_ids", "labels", "attention_mask"]
        )

    def train_dataloader(self):
        return self.__dataloader_for_set("train", True)

    def test_dataloader(self):
        return self.__dataloader_for_set("test", False)

    def val_dataloader(self):
        return self.__dataloader_for_set("valid", False)


def wandb_default_trainer_for(
    project,
    training_cycle,
    checkpoint_name=None,
    max_epochs=30,
    checkpoint_dir="./checkpoints",
    save_dir=None,
    offline=False,
    grad_batches=1,
):
    """
    A utility function for generating a new weights and biases logging trainer.

    You should only use this function if you are willing to take the opinionated defaults.
    Otherwise, just create your own trainer.
    """
    wandb_logger = WandbLogger(
        name=training_cycle,
        project=project,
        save_dir=save_dir,
        offline=offline,
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_acc",
        min_delta=0.00,
        patience=8,
        mode="max",
        strict=False,
        verbose=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    if checkpoint_name is None:
        checkpoint_name = training_cycle.split("_")[0]

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_name,
        monitor="validation_acc",
        mode="max",
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
        num_devices = torch.cuda.device_count()
    else:
        accelerator = None
        num_devices = 1

    trainer = Trainer(
        accelerator=accelerator,
        enable_checkpointing=True,
        devices=num_devices,
        auto_lr_find=True,
        max_epochs=max_epochs,
        val_check_interval=0.5,
        accumulate_grad_batches=grad_batches,
        logger=wandb_logger,
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
    )

    return trainer

class T5TextGenerator:
    """
    Class that uses a ByT5 model to create text
    """

    def __init__(
        self,
        t5model,
        tokenizer,
        max_length=512,
        batch_size=256,
        device=torch.device("cpu"),
    ):
        self.model = t5model
        self.model.to(device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.num_beams = 5
        self.temperature_t = 2.0
        self.topk_k = 10
        self.topp_p = 0.92
        self.batch_size = batch_size

    def greedy(self, input_strings):
        return self.__do_generate(input_strings)

    def beam(self, input_strings, num_beams=None, num_return_sequences=1):
        num_beams = self.num_beams if num_beams is None else num_beams
        return self.__do_generate(
            input_strings,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            do_sample=False,
        )

    def temperature(self, input_strings, temperature=None, batch_size=256):
        temperature = self.temperature_t if temperature is None else temperature
        return self.__do_generate(
            input_strings, top_k=0, temperature=temperature, batch_size=batch_size
        )

    def topk(self, input_strings, top_k=None):
        top_k = self.topk_k if top_k is None else top_k
        return self.__do_generate(input_strings, top_k=top_k)

    def topp(self, input_strings, top_p=None):
        top_p = self.topp_p if top_p is None else top_p
        return self.__do_generate(input_strings, top_p=top_p, top_k=0)

    def __do_generate(self, input_strings, batch_size=256, **kwargs):
        def chunk_list(datas, chunksize):
            for i in range(0, len(datas), chunksize):
                yield datas[i : i + chunksize]

        default_gen_args = {
            "max_length": self.max_length,
        }
        default_gen_args.update(**kwargs)

        output_strings = []

        for i, batch in tqdm.tqdm(
            enumerate(chunk_list(input_strings, batch_size)),
            total=len(input_strings) // batch_size,
        ):
            input_ids = (
                self.tokenizer(
                    list(batch),
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                )
                .to(self.device)
                .input_ids
            )
            logger.debug("Running generation with configuration %s", default_gen_args)
            output_ids = self.model.generate(input_ids, **default_gen_args)
            output_strings.extend(
                self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            )

        return output_strings

    def profile_sampling_strategies(self, input_strings):
        samples = {}

        samples["input"] = input_strings

        for strategy in (
            self.greedy,
            self.beam,
            self.temperature,
            self.topk,
            self.topp,
        ):
            strat_name = strategy.__name__
            print(f"generating for {strat_name}")
            samples[strat_name] = strategy(input_strings)

        return samples

