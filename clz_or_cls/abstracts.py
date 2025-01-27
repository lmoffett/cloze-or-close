# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import ast
import gc
import logging
import os
import re
import string
from pathlib import Path

import pandas as pd
import torch
import tqdm
from nltk.tokenize import WhitespaceTokenizer

from datasets import Dataset

from . import perturbation

logger = logging.getLogger(__name__)


def abstracts():
    df = pd.read_csv(Path(os.environ['CORC_DATASETS_ABSTRACT_DIR'])/"abstracts.csv")
    df["text"] = df["abstract"].apply(lambda x: x[:2000])
    return df


def abstract_metadata():
    df = pd.read_csv(Path(os.environ['CORC_DATASETS_ABSTRACT_DIR'])/"abstract_perturbation_metadata.csv")
    return df


def extract_replacible_tokens(text, min_length=4, max_digits=1):
    """
    Extracts all tokens that can be replaced from a given text by tokenizing and applying conditions for ignoring some tokens.
    For abstracts, URLs are excluded.
    """
    url_pattern = r"\b(?:http|https|ftp)://[^\s/$.?#].[^\s]*/?\b"
    urls = re.findall(url_pattern, text)

    tokenizable_text = text
    for special_token in urls:
        tokenizable_text = tokenizable_text.replace(special_token, "")

    tokenizer = WhitespaceTokenizer()
    tokens = tokenizer.tokenize(tokenizable_text.replace("/", " ").lower())

    tokenized_sentence = " ".join(tokens)
    for punc in string.punctuation:
        tokenized_sentence = tokenized_sentence.replace(punc, "")

    platform_aware_tokens = set(tokens)
    platform_aware_tokens = {
        t
        for t in platform_aware_tokens
        if len(t) > 0 and not all(c in string.punctuation for c in t)
    }

    platform_aware_tokens = {
        re.sub("^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$", "", t)
        for t in platform_aware_tokens
        if len(t) > 0
    }

    def include_token_for_replacement(token):
        """Returns True if the token should be included for replacement."""
        # at least 2 digits
        if sum(c.isdigit() for c in token) > max_digits:
            return False
        if any((p in token for p in string.punctuation)):
            return False
        if len(token) < min_length:
            return False
        return True

    retain_words = {
        token for token in platform_aware_tokens if include_token_for_replacement(token)
    }

    return platform_aware_tokens, retain_words, tokenized_sentence


def abstracts_marked(as_df=False):
    try:
        abstracts_df = abstracts()

        abstracts_df[
            ["tokens", "replacement_candidates", "tokenized_sentence"]
        ] = abstracts_df.apply(
            lambda row: extract_replacible_tokens(row["text"], min_length=4),
            axis=1,
            result_type="expand",
        )
        abstracts_df["tokens_len"] = abstracts_df["tokens"].apply(len)
        abstracts_df["num_replacement_candidates"] = abstracts_df[
            "replacement_candidates"
        ].apply(len)
        abstracts_df["candidate_ratio"] = (
            abstracts_df["num_replacement_candidates"] / abstracts_df["tokens_len"]
        )

        if as_df:
            return abstracts_df
        else:
            return Dataset.from_pandas(abstracts_df)
    except FileNotFoundError:
        print(
            "Unable to locate hot speech. Make sure it has been downloaded from https://socialmediaarchive.org/record/19 and placed in the data directory."
        )
        raise


def abstract_metadata_df():
    class_replacements_df = pd.read_csv(Path(os.environ['CORC_DATASETS_PREP_DIR'])/"abstract"/"abstract-class-replacements.csv")
    try:
        perturbable_df = pd.read_csv(
            Path(os.environ['CORC_DATASETS_PREP_DIR'])/"abstract"/"abstract_perturbation_metadata.csv", index_col="idx"
        )
    except:
        perturbable_df = pd.read_csv(
            Path(os.environ['CORC_DATASETS_PREP_DIR'])/"abstract"/"abstract_perturbation_metadata.csv", index_col="Unnamed: 0"
        )
    hp_df = perturbable_df.merge(
        class_replacements_df, left_index=True, right_index=True
    )
    return hp_df


def abstract_metadata_ds():
    return Dataset.from_pandas(abstract_metadata_df())


def class_perturbed_abstract_dataset(clazz: str, n_words=1, dataset=None):
    assert clazz in ("visual", "phonetic", "typo")
    abs_meta_ds = dataset if dataset is not None else abstract_metadata_ds()

    def apply_perturbation(row):
        perturbed = perturbation.perturb_candidates(
            row["text"],
            ast.literal_eval(row[f"replacements_{clazz}"]),
            n_candidates=n_words,
        )
        row["sample"] = perturbed
        return row

    return abs_meta_ds.map(apply_perturbation)


def run_abstract_recovery(
    *,
    test_df,
    abstracts_client,
    checkpoint_file=None,
    batch_size=20,
    starting_df=None,
    num_retries=0,
):
    """
    Run a recovery job for a prompted model. Uses the recovery_client interface so all the logic of the prompting is delegated to the client.
    This method controls the few-shot sampling, batching and data aggregation.
    """

    # Allow starting from a previous run
    responses_df = starting_df.copy() if starting_df is not None else None

    num_batches = len(test_df) // batch_size
    if len(test_df) % batch_size > 0:
        num_batches += 1

    for i in tqdm.tqdm(range(num_batches)):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("%s", torch.cuda.memory_summary(device="cuda"))

        client_kwargs = {}

        start, end = i * batch_size, min(i * batch_size + batch_size, len(test_df))

        test_df_slice = test_df.iloc[start:end]
        test_words = test_df_slice[["sample", "word"]]

        retries_remaining = num_retries + 1

        while retries_remaining > 0:

            try:

                mapping = abstracts_client(test_words, **client_kwargs)

                logger.debug("Got mapping from client: %s", mapping)

                output_df = pd.DataFrame(
                    mapping, columns=["recovered"], index=test_df_slice.index
                )

                full_df = test_df_slice.merge(
                    output_df, left_index=True, right_index=True
                )

                full_df["batch"] = i

                if responses_df is None:
                    responses_df = full_df
                else:
                    # technically not atomic, but all_lost_words is a set so it should be fine
                    responses_df = pd.concat([responses_df, full_df])

                retries_remaining = 0
                responses_df.to_csv(checkpoint_file)

            except Exception as e:
                logger.error(e)
                retries_remaining -= 1
                if retries_remaining > 0:
                    logger.info("Retrying...")

                else:
                    logger.error(
                        "could not complete query for group with range %d %d",
                        start,
                        end,
                    )

    return responses_df
