# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import ast
import gc
import logging
import os
import re
import string
from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from nltk.tokenize import WhitespaceTokenizer, word_tokenize

from datasets import Dataset

from . import llms, perturbation
from .datasets import perturb_as_ds

logger = logging.getLogger(__name__)


def find_social_identifiers(text):
    """
    Find and return all occurrences of hashtags, Twitter usernames,
    subreddit names, Reddit usernames, and YouTube usernames in a given text.
    """
    # Regular expressions for each type of identifier
    hashtag_pattern = r"(^|\s+)(#\w+)"  # e.g., #hashtag
    twitter_user_pattern = r"(^|\s+)(@\w+)"  # e.g., @username
    subreddit_pattern = r"/?r/[\w\-_]+"  # e.g., /r/subreddit
    reddit_user_pattern = r"/?u/[\w\-_]+"  # e.g., /u/reddituser
    youtube_user_pattern = (
        r"(^|\s+)(@[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]*)*)"  # e.g., @User Name
    )
    url_pattern = r"\b(?:http|https|ftp)://[^\s/$.?#].[^\s]*/?\b"
    markdown_url = r"\[.*\]\((.*)\)"

    # Find all matches in the text
    hashtags = {m[1] for m in re.findall(hashtag_pattern, text)}
    twitter_users = {m[1] for m in re.findall(twitter_user_pattern, text)}
    subreddits = re.findall(subreddit_pattern, text)
    reddit_users = re.findall(reddit_user_pattern, text)
    youtube_users = {m[1] for m in re.findall(youtube_user_pattern, text)}
    urls = re.findall(url_pattern, text)
    urls = urls + re.findall(markdown_url, text)

    # Return the results
    return {
        "hashtags": set(hashtags),
        "twitter_users": set(twitter_users),
        "subreddits": set(subreddits),
        "reddit_users": set(reddit_users),
        "youtube_users": set(youtube_users),
        "urls": set(urls),
    }


def source(link):
    """
    Categorize the source of a comment by the link to the original post
    """
    if "x.com" in link or "twitter.com" in link:
        return "twitter"
    elif "youtube.com" in link:
        return "youtube"
    elif "reddit.com" in link:
        return "reddit"
    else:
        return np.nan


def extract_replacible_tokens(text, source, min_length=4, max_digits=1):
    """
    Extracts all tokens that can be replaced from a given text by tokenizing and applying conditions for ignoring some tokens.
    """
    social_ids = find_social_identifiers(text)

    urls = social_ids["urls"]
    hashtags = social_ids["hashtags"]

    if source == "youtube":
        usernames = social_ids["youtube_users"]
        links = urls
    elif source == "twitter":
        usernames = social_ids["twitter_users"]
        links = urls
    elif source == "reddit":
        usernames = social_ids["reddit_users"]
        links = urls.union(social_ids["subreddits"])

    tokenizable_text = text
    for special_token in urls.union(hashtags).union(usernames).union(links):
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
    platform_aware_tokens = platform_aware_tokens.union(
        {h.lower()[1:] for h in hashtags}
    )

    platform_aware_tokens = {
        re.sub("^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$", "", t)
        for t in platform_aware_tokens
        if len(t) > 0
    }

    digits_regex = "[0-9]+.*" + "[0-9]+" * (max_digits - 1)

    def include_token_for_replacement(token):
        """Returns True if the token should be included for replacement."""
        # at least 2 digits
        if re.match(digits_regex, token):
            return False
        if len(token) < min_length:
            return False
        return True

    retain_words = {
        token for token in platform_aware_tokens if include_token_for_replacement(token)
    }

    return platform_aware_tokens, retain_words, tokenized_sentence


def hot_speech():
    hot_path = Path(os.environ['CORC_DATASETS_HOT_DIR'])
    hot_path = Path(os.environ['CORC_DATASETS_HOT_DIR'])
    return pd.read_csv(hot_path/"hot-dataset.csv")


def hot_speech_marked(as_df=False):
    try:
        hot_speech_df = hot_speech()

        hot_speech_df["source"] = hot_speech_df["link"].apply(source)
        hot_speech_df[
            ["tokens", "replacement_candidates", "tokenized_sentence"]
        ] = hot_speech_df.apply(
            lambda row: extract_replacible_tokens(
                row["text"], row["source"], min_length=3
            ),
            axis=1,
            result_type="expand",
        )
        hot_speech_df["tokens_len"] = hot_speech_df["tokens"].apply(len)
        hot_speech_df["num_replacement_candidates"] = hot_speech_df[
            "replacement_candidates"
        ].apply(len)
        hot_speech_df["candidate_ratio"] = (
            hot_speech_df["num_replacement_candidates"] / hot_speech_df["tokens_len"]
        )

        if as_df:
            return hot_speech_df
        else:
            return Dataset.from_pandas(hot_speech_df)
    except FileNotFoundError:
        print(
            "Unable to locate hot speech. Make sure it has been downloaded from https://socialmediaarchive.org/record/19 and placed in the data directory."
        )
        raise


def perturb_text_as_ds(dataset, perturbation_strategy, ratios):
    """
    Perturb each piece of raw text with the given perturbation strategy.
    """

    assert set(dataset.column_names).issuperset(
        {"text", "replacement_candidates"}
    ), "Dataset must have candidates extracted"

    def do_perturb(record):

        for ratio in ratios:
            record[f"perturbed_{ratio}"] = perturbation_strategy.perturb(
                record["text"], record["replacement_candidates"], ratio
            )
        return record

    keep_cols = set(["ID"]).union({f"perturbed_{ratio}" for ratio in ratios})
    drop_cols = set(dataset.column_names).difference(keep_cols)

    return dataset.map(do_perturb, remove_columns=list(drop_cols))


def hot_speech_for(word_strategies, num_splits=8):
    if isinstance(word_strategies, Iterable):
        text_strategy = perturbation.RandomStrategyTextPerturber(word_strategies)
    else:
        text_strategy = perturbation.SingleStrategyTextPerturber(word_strategies)

    return perturb_text_as_ds(
        hot_speech_marked(),
        text_strategy,
        [i / num_splits for i in range(1, num_splits + 1)],
    )


def hot_speech_legit():
    return hot_speech_for(perturbation.LegitStrategy())


def hot_speech_dces():
    return hot_speech_for(perturbation.DcesStrategy())


def hot_speech_ices():
    return hot_speech_for(perturbation.IcesStrategy())


def hot_speech_legit_dces():
    return hot_speech_for([perturbation.LegitStrategy(), perturbation.DcesStrategy()])


def hot_speech_legit_ices():
    return hot_speech_for([perturbation.LegitStrategy(), perturbation.IcesStrategy()])


def hot_speech_dces_ices():
    return hot_speech_for([perturbation.DcesStrategy(), perturbation.IcesStrategy()])


def hot_speech_legit_dces_ices():
    return hot_speech_for(
        [
            perturbation.LegitStrategy(),
            perturbation.DcesStrategy(),
            perturbation.IcesStrategy(),
        ]
    )


def hot_speech_legit_phonee_zeroe_typo():
    return hot_speech_for(
        [
            perturbation.LegitStrategy(),
            perturbation.PhoneEStrategy(),
            perturbation.MultiStrategy(
                [
                    perturbation.WordLevelZeroeStrategy("natural-typo"),
                    perturbation.LetterLevelZeroeStrategy(
                        "keyboard-typo", letter_prob=0.15
                    ),
                ],
                selection_criteria="ordered",
            ),
        ]
    )


def maybe_concat(df_1, df_2):
    if df_1 is None:
        return df_2
    else:
        return pd.concat([df_1, df_2])


def run_hot_gpt(
    test_data,
    prompt_dataset,
    client,
    num_samples=0,
    checkpoint_path=None,
    start_from=0,
    checkpoint_frequency=20,
):

    rows = []

    if start_from > 0:
        logger.info("Loading checkpoints to start at %s", start_from)
        old_rows = pd.read_csv(checkpoint_path.with_suffix(".csv.bkp"), index_col="ID")
        logger.info("Last record loaded is %s", old_rows.tail(1))

    else:
        logger.info("Fresh run, no checkpoints")
        old_rows = None

    def write_checkpoint(rows):
        checkpoint = maybe_concat(old_rows, pd.DataFrame.from_records(rows, index="ID"))
        checkpoint.to_csv(checkpoint_path.with_suffix(".csv.bkp"))
        return checkpoint

    for i, (id, text) in tqdm.tqdm(enumerate(test_data)):
        samples = list(
            prompt_dataset.sample(num_samples)[["clean", "perturbed"]].itertuples(
                index=False, name=None
            )
        )
        classification = dict(client.call_api(comment=text, examples=samples))
        classification["ID"] = id
        rows.append(classification)

        if i % checkpoint_frequency == 0 and checkpoint_path is not None:
            write_checkpoint(rows)

    return write_checkpoint(rows)


def class_perturbed_hot_dataset(clazz: str, ratio: float, dataset=None):
    assert clazz in ("visual", "phonetic", "typo")
    hot_meta_ds = dataset if dataset is not None else hot_metadata_ds()

    def apply_perturbation(row):
        perturbed = perturbation.perturb_ratio(
            row["text"], ast.literal_eval(row[f"replacements_{clazz}"]), ratio=ratio
        )
        row["sample"] = perturbed
        return row

    return hot_meta_ds.map(apply_perturbation)


def hot_metadata_df():
    prep_path = Path(os.environ["CORC_DATASETS_PREP_DIR"])
    class_replacements_df = pd.read_csv(prep_path/"hot"/"hot-class-replacements.csv")
    hot_perturbable_df = pd.read_csv(
        prep_path/"hot"/"hot_perturbation_metadata.csv", index_col="ID"
    )
    hot_perturbable_df = hot_perturbable_df.reset_index()
    hp_df = hot_perturbable_df.merge(
        class_replacements_df, left_index=True, right_index=True
    )
    return hp_df.rename({"Unnamed: 0_y": "idx"}, axis=1)


def hot_metadata_ds():
    return Dataset.from_pandas(hot_metadata_df())


def run_hot_classification(
    test_df,
    hot_client,
    train_df=None,
    xshot_size=0,
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

        if xshot_size > 0:
            raise NotImplemented("Not Implemented")

        else:
            client_kwargs = {}

        start, end = i * batch_size, min(i * batch_size + batch_size, len(test_df))

        test_df_slice = test_df.iloc[start:end]
        test_words = test_df_slice["sample"]

        retries_remaining = num_retries + 1

        while retries_remaining > 0:

            try:

                mapping = hot_client(test_words, **client_kwargs)

                logger.debug("Got mapping from client: %s", mapping)

                output_df = pd.DataFrame(
                    mapping, columns=["score", "explanation"], index=test_df_slice.index
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


def run_hot_gpt_recovery(
    test_data,
    recovery_model,
    client,
    checkpoint_path=None,
    replace_candidates=True,
    start_from=0,
    checkpoint_frequency=20,
):

    rows = []
    candidates_rows = []
    failures = []

    if start_from > 0:
        logger.info("Loading checkpoints to start at %s", start_from)
        old_rows = pd.read_csv(checkpoint_path.with_suffix(".csv.bkp"), index_col="ID")
        logger.info("Last record loaded is %s", old_rows.tail(1))
        old_candidates = pd.read_csv(
            checkpoint_path.with_suffix(".candidates.csv.bkp"), index_col="ID"
        )
        logger.info("Last candidates loaded is %s", old_rows.tail(1))
        if checkpoint_path.with_suffix(".failures.csv.bkp").exists():
            old_failures = pd.read_csv(
                checkpoint_path.with_suffix(".failures.csv.bkp"), index_col="ID"
            )
        else:
            old_failures = None
    else:
        logger.info("Fresh run, no checkpoints")
        old_rows = None
        old_candidates = None
        old_failures = None

    def write_checkpoint(rows, candidates_rows, failures):
        maybe_concat(old_rows, pd.DataFrame.from_records(rows, index="ID")).to_csv(
            checkpoint_path.with_suffix(".csv.bkp")
        )
        maybe_concat(
            old_candidates, pd.DataFrame.from_records(candidates_rows, index="ID")
        ).to_csv(checkpoint_path.with_suffix(".candidates.csv.bkp"))
        if len(failures) > 0:
            maybe_concat(
                old_failures, pd.DataFrame.from_records(failures, index="ID")
            ).to_csv(checkpoint_path.with_suffix(".failures.csv.bkp"))

    for i, (id, text) in tqdm.tqdm(enumerate(test_data)):

        logger.debug("Recovering %s", text)
        candidates = recovery_model.get_candidates(text)
        logger.debug("Extracted candidates %s", candidates)

        classification = None

        if not replace_candidates:
            for i in range(recovery_model.num_return_sequences):

                if i > 0:
                    try_candidates = {
                        token: words[:-i] for token, words in candidates.items()
                    }
                else:
                    try_candidates = candidates

                try:
                    classification = dict(
                        client.call_api(
                            comment=text, recovery_suggestions=try_candidates
                        )
                    )
                except Exception as e:
                    logger.warn(
                        "Failed to classify record. Reducing candidate size from %s to %s. %s with error %s",
                        i,
                        i - 1,
                        text,
                        e,
                    )
                    continue

                candidates_rows.append({"ID": id, "candidates": try_candidates})

        else:

            try:
                target_text = text
                for token, words in candidates.items():
                    target_text = perturbation.sub_token(target_text, token, words[0])

                classification = dict(client.call_api(comment=target_text))
                candidates_rows.append(
                    {
                        "ID": id,
                        "candidates": {
                            token: words[:1] for token, words in candidates.items()
                        },
                    }
                )
            except Exception as e:
                logger.warn("Failed to classify record. %s with error %s", text, e)

        if classification is None:
            failures.append({"ID": id, "text": text, "candidates": candidates})
            logger.error("Failed to classify record. %s", text)
            continue

        logger.debug("Classification is %s", classification)
        classification["ID"] = id
        rows.append(classification)

        if i % checkpoint_frequency == 0 and checkpoint_path is not None:
            write_checkpoint(rows, candidates_rows, failures)

    write_checkpoint(rows, candidates_rows, failures)

    return pd.DataFrame.from_records(rows, index="ID")
