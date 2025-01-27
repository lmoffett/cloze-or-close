# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import logging
import math
import os
import pathlib

import numpy as np
import pandas as pd
import torch
from datasets.combine import concatenate_datasets, interleave_datasets
from datasets.fingerprint import Hasher
from datasets import load_dataset, Dataset, DatasetDict

from . import perturbation

device = torch.device(os.environ['TORCH_DEVICE'])
logger = logging.getLogger(__name__)


def legit_raw(split=None, cache_dir=None):
    return load_dataset("dvsth/LEGIT", split=split, cache_dir=cache_dir)


def columns_names(ds):
    if type(ds) == Dataset:
        return ds.column_names
    elif type(ds) == DatasetDict:
        example_split = list(ds.keys())[0]
        return ds[example_split].column_names
    else:
        raise ValueError(f"{type(ds)} is not supported")


def legit(**kwargs):
    legit_bl = legit_baseline(**kwargs)

    is_legible = lambda x: x["legible"] == True

    return DatasetDict(
        {
            "train": legit_bl["train"],
            "valid": legit_bl["valid"].filter(is_legible),
            "test": legit_bl["test"].filter(is_legible),
        }
    )


def legit_baseline(**kwargs):
    def explode_legible_words(batch):

        clean_words = []
        perturbed_words = []
        legible = []

        for i, choice in enumerate(batch["choice"]):
            for w in (0, 1):
                clean_words.append(batch["word"][i])
                perturbed_words.append(batch[f"word{w}"][i])
                legible.append(batch["choice"][i] == w or batch["choice"][i] == 2)

        return {"clean": clean_words, "perturbed": perturbed_words, "legible": legible}

    legit_ds = legit_raw(**kwargs)

    logger.info("mapping function hash is %s", Hasher.hash(explode_legible_words))

    return legit_ds.map(
        explode_legible_words,
        batched=True,
        batch_size=256,
        remove_columns=columns_names(legit_ds),
    )


def perturb_as_ds(dataset, perturbation_strategy, n=4):
    def perturb_batch(batch):
        perturbed = perturbation_strategy.perturb_batch(batch["clean"], n=n)
        return {"clean": np.repeat(batch["clean"], n), "perturbed": perturbed}

    return dataset.map(
        perturb_batch,
        batched=True,
        batch_size=512,
        remove_columns=columns_names(dataset),
    )


def legit_extended(do_perturb_legit=True, **kwargs):
    """
    Add the accented and uncommon words to the legit dataset
    """
    legit_ds = legit_baseline(**kwargs)

    # sorting orders by the order in which the words were encountered instead of counts
    # to get a more even split
    prep_path = pathlib.Path(os.environ['CORC_DATASETS_PREP_DIR'])
    accents_df = pd.read_csv(prep_path/"legit-extended"/"wikitext-accents.csv").sort_values(
        "encountered", ascending=False
    )
    uncommon_df = pd.read_csv(prep_path/"legit-extended"/"wikitext-uncommon.csv").sort_values(
        "encountered", ascending=False
    )

    def as_ds(df):
        return (
            Dataset.from_pandas(df)
            .remove_columns(["count", "encountered"])
            .rename_column("word", "clean")
        )

    extension = DatasetDict(
        {
            "train": concatenate_datasets(
                [as_ds(accents_df[:50]), as_ds(uncommon_df[:150])]
            ),
            "valid": concatenate_datasets(
                [as_ds(accents_df[50:75]), as_ds(uncommon_df[150:200])]
            ),
            "test": concatenate_datasets(
                [as_ds(accents_df[75:]), as_ds(uncommon_df[200:])]
            ),
        }
    )

    if do_perturb_legit:
        legit_perturbation_strategy = perturbation.LegitStrategy()
        extension = perturb_as_ds(extension, legit_perturbation_strategy)

    dataset = DatasetDict(
        {
            "train": concatenate_datasets([legit_ds["train"], extension["train"]]),
            "valid": concatenate_datasets([legit_ds["valid"], extension["valid"]]),
            "test": concatenate_datasets([legit_ds["test"], extension["test"]]),
        }
    )

    return dataset


def repeated(**kwargs):
    """
    legit dataset where perturbed = clean
    """
    legit_ds = legit_extended(**kwargs)

    def map_row(row):
        return {
            "clean": row["clean"],
            "perturbed": row["clean"],
            "legible": True,
            "legibility_score": 1.0,
        }

    logger.info("mapping function hash is %s", Hasher.hash(map_row))

    return legit_ds.map(map_row, remove_columns=columns_names(legit_ds))


def __legit_from(mapping_function, **kwargs):
    legit4re = legit_extended(do_perturb_legit=False, **kwargs)

    logger.info("mapping function hash is %s", Hasher.hash(mapping_function))

    legit_mapped = perturb_as_ds(legit4re, mapping_function, n=1)

    return legit_mapped


def ices(**kwargs):
    icesStrategy = perturbation.IcesStrategy()
    return __legit_from(icesStrategy, **kwargs)


def dces(**kwargs):
    dcesStrategy = perturbation.DcesStrategy()
    return __legit_from(dcesStrategy, **kwargs)


def zeroe_noise(**kwargs):
    strategy = perturbation.MultiStrategy(
        [
            perturbation.WordLevelZeroeStrategy("inner-swap"),
            perturbation.DeleteStrategy(),
            perturbation.LetterLevelZeroeStrategy("intrude", letter_prob=0.3),
        ]
    )
    return __legit_from(strategy, **kwargs)


def zeroe_typo(**kwargs):
    strategy = perturbation.MultiStrategy(
        [
            perturbation.WordLevelZeroeStrategy("natural-typo"),
            perturbation.LetterLevelZeroeStrategy(
                "keyboard-typo", letter_prob=0.15
            ),
        ],
        selection_criteria="ordered",
    )
    return __legit_from(strategy, **kwargs)


def anthro_typo(**kwargs):
    strategy = perturbation.AnthroStrategy("typo")
    return __legit_from(strategy, **kwargs)


def phonee(**kwargs):
    strategy = perturbation.PhoneEStrategy()
    return __legit_from(strategy, **kwargs)


def zeroe_phonetic(**kwargs):
    strategy = perturbation.ZeroePhoneticStrategy()
    return __legit_from(strategy, **kwargs)


def anthro_phonetic(**kwargs):
    strategy = perturbation.AnthroStrategy("phonetic")
    return __legit_from(strategy, **kwargs)


def visual(ctx=None):
    return as_concatted_dataset(
        [
            generated_ds("ices", ctx=ctx),
            generated_ds("dces", ctx=ctx),
            generated_ds("legit_extended", ctx=ctx),
        ]
    )


def visual_ctx():
    return visual(ctx="visual")


def visual_ctx_hidden():
    return visual(ctx="Strategy A")


def phonetic(ctx=None):
    return as_concatted_dataset(
        [
            generated_ds("phonee", ctx=ctx),
            generated_ds("zeroe_phonetic", ctx=ctx),
            generated_ds("anthro_phonetic", ctx=ctx),
        ]
    )


def phonetic_ctx():
    return phonetic(ctx="phonetic")


def phonetic_ctx_hidden():
    return phonetic(ctx="Strategy B")


def typo(ctx=None):
    return as_concatted_dataset(
        [
            generated_ds("zeroe_noise", ctx=ctx),
            generated_ds("zeroe_typo", ctx=ctx),
            generated_ds("anthro_typo", ctx=ctx),
        ]
    )


def typo_ctx():
    return typo(ctx="typo")


def typo_ctx_hidden():
    return typo(ctx="Strategy C")


def tokenize_recovery_batch(batch, tokenizer, max_input_length):

    try:
        tokens = tokenizer(
            text=batch["perturbed"],
            text_target=batch["clean"],
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
        )

        return_dict = batch.copy()
        return_dict.update(
            {
                "labels": tokens["labels"],
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            }
        )
        return return_dict
    except Exception as e:
        logger.error(e)
        logger.error("batch perturbed: %s", batch["perturbed"])
        logger.error("batch: %s", batch["clean"])
        raise e


def mixed(**kwargs):

    zeroe_typo = perturbation.MultiStrategy(
        [
            perturbation.WordLevelZeroeStrategy("natural-typo"),
            perturbation.LetterLevelZeroeStrategy(
                "keyboard-typo", letter_prob=0.15
            ),
        ],
        selection_criteria="ordered",
    )

    strategy = perturbation.MixedStrategy(
        [
            zeroe_typo,
            perturbation.AnthroStrategy("typo"),
            perturbation.DcesStrategy(),
            perturbation.IcesStrategy(),
            perturbation.PhoneEStrategy(),
            perturbation.AnthroStrategy("phonetic"),
        ]
    )

    return __legit_from(strategy, **kwargs)


def proportion_dataset(dataset_dict, key_map, drop_duplicates=False):
    """
    Remove half of the train dataset based on lower legibility score
    """

    def do_proportion(split_name):
        df = dataset_dict[split_name].to_pandas()

        if drop_duplicates:
            df = df.drop_duplicates()

        cols = df.columns.tolist()
        df["random"] = np.random.rand(len(df))

        df_proportion = (
            df.sort_values(["clean", "random"], ascending=True)
            .groupby("clean")
            .apply(lambda x: x.head(key_map[split_name][x.name]))
            .reset_index(drop=True)
        )

        return Dataset.from_pandas(df_proportion[cols])

    return DatasetDict(
        {split: do_proportion(split) for split in dataset_dict.keys()}
    )


def as_concatted_dataset(dataset_dicts):
    return DatasetDict(
        {
            split: concatenate_datasets([ds[split] for ds in dataset_dicts])
            for split in dataset_dicts[0].keys()
        }
    )


def as_proportional_dataset(dataset_dicts, drop_duplicates=False):
    num_ds = len(dataset_dicts)

    key_lens = [{split: {} for split in dataset_dicts[0].keys()} for _ in range(num_ds)]
    for split_name in dataset_dicts[0].keys():
        split = dataset_dicts[0][split_name]
        split_df = split.to_pandas()
        for row in split_df.groupby("clean").count().reset_index().itertuples():
            clean = row.clean
            key_total = row.perturbed
            base_proportion = math.floor(key_total / num_ds)

            for i in range(num_ds):
                if clean in key_lens[i][split_name]:
                    raise ValueError(
                        f"clean {clean} already in key_lens[{i}][{split_name}]"
                    )
                key_lens[i][split_name][clean] = base_proportion

            remainder = key_total - base_proportion * num_ds

            for i in np.random.choice(num_ds, remainder, replace=False):
                key_lens[i][split_name][clean] += 1

            assert (
                sum(key_lens[i][split_name][clean] for i in range(num_ds)) == key_total
            )

    mapped_dss = [
        proportion_dataset(
            dataset, key_map=key_lens[i], drop_duplicates=drop_duplicates
        )
        for i, dataset in enumerate(dataset_dicts)
    ]

    return DatasetDict(
        {
            split: interleave_datasets(
                [ds[split] for ds in mapped_dss]
            )
            for split in dataset_dicts[0].keys()
        }
    )


def generated_df(name, split):
    """
    Get an Ad-Word Attack Split from an existing dataframe.
    
    name is shorthand
    seed_idx is the index of the random seed used for generating the dataset
    """
    return pd.read_csv(
        (pathlib.Path(os.environ['CORC_DATASETS_ADWORD_DIR']) / name / split).with_suffix(".csv"),
        converters={"perturbed": str, "clean": str},
    )


def geneterated_ds_ctx(name, ctx=None, splits=["test", "train", "valid"]):
    if "+" in name and not name.endswith("_full"):
        raise NotImplementedError("only full datasets are supported for context")

    names = name.split("_full")[0].split("+")

    dataset_dicts = []
    for name in names:
        if ctx is None:
            this_ctx = name
        elif type(ctx) == dict:
            this_ctx = ctx[name]
        else:
            this_ctx = ctx
        dsd = generated_ds(name, splits=splits, ctx=this_ctx)
        dataset_dicts.append(dsd)

    return as_concatted_dataset(dataset_dicts)


def generated_ds(name, ctx=None, splits=["test", "train", "valid"]):
    def __load(name, split):
        ds = Dataset.from_pandas(generated_df(name, split))

        if ctx is None:
            return ds
        else:
            return ds.map(
                lambda x: {
                    "clean": x["clean"],
                    "perturbed": f'method: "{ctx}", word: "{x["perturbed"]}"',
                }
            )

    dsd = DatasetDict({split: __load(name, split) for split in splits})
    return dsd


def mixed_generated_dataset(split_map):
    return DatasetDict(
        {
            split: generated_ds(dataset, splits=[split])[split]
            for split, dataset in split_map.items()
        }
    )

def shuffle_on_clean(df, batch_size, seed=None):
    """
    Takes a dataframe with multiple of the same clean words and ensures they are placed in different splits.
    """
    if seed is not None:
        np.random.seed(seed)

    grouped = df.groupby("clean")

    shuffled_groups = [
        group.sample(frac=1).reset_index(drop=True) for _, group in grouped
    ]

    # Initialize batches
    batches = []

    # Create batches ensuring no similar 'column' values in the same batch
    for i in tqdm.tqdm(range(0, len(df), batch_size)):

        batch = pd.concat(
            [group.iloc[i : i + batch_size] for group in shuffled_groups],
            ignore_index=True,
        )
        batches.append(batch)

    # Shuffle the batches
    np.random.shuffle(batches)

    # Concatenate batches back to a single DataFrame
    shuffled_df = pd.concat(batches, ignore_index=True)

    return shuffled_df

datasets = [
    ("legit_extended", "visual"),
    ("dces", "visual"),
    ("ices", "visual"),
    ("zeroe_noise", "typo"),
    ("zeroe_typo", "typo"),
    ("anthro_typo", "typo"),
    ("anthro_phonetic", "phonetic"),
    ("phonee", "phonetic"),
    ("zeroe_phonetic", "phonetic"),
]

class_map = {t[0]: t[1] for t in datasets}