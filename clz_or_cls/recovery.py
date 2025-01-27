# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import gc
import logging
import os
import re
from collections import namedtuple

import enchant
import pandas as pd
import torch
import tqdm
from nltk.metrics import edit_distance
from nltk.tokenize import WhitespaceTokenizer
from transformers import T5ForConditionalGeneration

from .byt5 import T5TextGenerator

en_dict = enchant.Dict("en_US")
# from string import punctuation
punctuation = "!#\"$%&'()*+,-./:;<=>?@[\\]^_`{|}~…––“”‘’—"

logger = logging.getLogger(__name__)
device = torch.device(os.environ['TORCH_DEVICE'])


def load_recovery_model(model_name, model_dir=os.environ['CORC_TRAINED_MODEL_DIR']):
    return T5ForConditionalGeneration.from_pretrained(str(model_dir / model_name)).to(
        device
    )


def run_beam_recover(recovery_ds_split, gen_model, beams=[4]):
    """
    Generate recovery strings from a generative model with a beam strategy
    """

    if type(recovery_ds_split) == dict:
        beam_gen = recovery_ds_split.copy()
    else:
        beam_gen = {
            col: recovery_ds_split[col] for col in recovery_ds_split.column_names
        }

    for num_beams in beams:
        output_strings = gen_model.beam(beam_gen["perturbed"], num_beams=num_beams)
        beam_gen[f"{num_beams}_beams"] = output_strings

    df_recovery = pd.DataFrame.from_dict(beam_gen)

    for num_beams in beams:
        df_recovery[f"{num_beams}_beams_match"] = (
            df_recovery[f"{num_beams}_beams"] == df_recovery["clean"]
        )

    return df_recovery


acc_scores = namedtuple("acc", ["all", "legible"])


def safe_true(series):
    value_counts = series.value_counts(normalize=True)
    if True in value_counts:
        return value_counts[True]
    else:
        return 0.0


def analyze_accuracy(df_recovery):
    """
    Score the accuracy of a recovery for each beam, broken out by the legibility of each set
    """
    all_acc = safe_true(df_recovery["match"])
    if "legibility_score" in df_recovery.columns:
        leg_acc = safe_true(
            df_recovery.loc[df_recovery["legibility_score"] > 0]["match"]
        )
    else:
        leg_acc = None

    return acc_scores(all_acc, leg_acc)


def recovery_job_for_dataset(dataset, tokenizer, model, max_length=40):
    """
    Create a recovery function for a particular datasets so that you can run recovery across multiple models.
    """

    def run_recovery(sample_size=100_000_000):
        recovery_generator = T5TextGenerator(
            model,
            tokenizer,
            device=device,
            max_length=max_length,
        )
        beams = 4
        df_recovery = run_beam_recover(
            dataset[:sample_size], recovery_generator, beams=[beams]
        )

        df_recovery.rename(
            columns={f"{beams}_beams": "recovered", f"{beams}_beams_match": "match"},
            inplace=True,
        )

        df_recovery["edit_distance"] = df_recovery.apply(
            lambda row: edit_distance(row["clean"].lower(), row["recovered"].lower()),
            axis=1,
        )
        df_recovery["is_word"] = df_recovery["recovered"].apply(
            lambda word: len(word) > 0 and en_dict.check(word)
        )
        df_recovery["other_word"] = df_recovery.apply(
            lambda row: row["clean"] != row["recovered"] and row["is_word"], axis=1
        )
        df_recovery["non_ascii_clean"] = df_recovery["clean"].apply(
            lambda word: not word.isascii()
        )
        df_recovery["non_ascii_recovered"] = df_recovery["recovered"].apply(
            lambda word: not word.isascii()
        )

        accuracy_profile = analyze_accuracy(df_recovery)

        return df_recovery, accuracy_profile

    return run_recovery


class NonAsciiSentenceRecoveryModel(object):
    def __init__(self, recovery_model, num_beams=5, num_return_sequences=3):
        self.recovery_model = recovery_model
        self.whitespace_tokenizer = WhitespaceTokenizer()
        self.num_return_sequences = num_return_sequences
        self.num_beams = num_beams

    def get_candidates(self, text):

        tokens = self.whitespace_tokenizer.tokenize(text)

        tokens_for_recovery = []

        for token in tokens:
            if not token.isascii() and len(token) > 0:
                clean_token = token.strip(punctuation)
                tokens_for_recovery.append(clean_token)

        if len(tokens_for_recovery) == 0:
            return {}

        candidates = self.recovery_model.beam(
            tokens_for_recovery,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
        )
        logger.debug(f"Replacement candidates: {candidates}")
        generations = [re.split("\s", word)[0] for word in candidates]
        candidate_map = {}
        for i in range(0, len(generations), self.num_return_sequences):
            candidate_map[
                tokens_for_recovery[i // self.num_return_sequences]
            ] = generations[i : i + self.num_return_sequences]

        logger.debug(f"Replacement candidates: {candidate_map}")
        return candidate_map

def run_prompt_recovery(
    *,
    train_df,
    test_df,
    recovery_client,
    batch_size,
    checkpoint_file,
    starting_df=None,
    xshot_size=0,
    num_retries=0,
    shielding_model=None,
):
    """
    Run a recovery job for a prompted model. Uses the recovery_client interface so all the logic of the prompting is delegated to the client.
    This method controls the few-shot sampling, batching and data aggregation.
    """

    # Allow starting from a previous run
    responses_df = starting_df.copy() if starting_df is not None else None

    all_lost_words = set()
    failure_groups = []

    num_batches = len(test_df) // batch_size
    if len(test_df) % batch_size > 0:
        num_batches += 1

    for i in tqdm.tqdm(range(num_batches)):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("%s", torch.cuda.memory_summary(device="cuda"))

        if xshot_size > 0:

            client_kwargs = {
                "examples": list(
                    train_df[["perturbed", "clean"]]
                    .sample(xshot_size)
                    .itertuples(index=False, name=None)
                )
            }
        else:
            client_kwargs = {}

        start, end = i * batch_size, min(i * batch_size + batch_size, len(test_df))

        test_df_slice = test_df.iloc[start:end]
        test_words = list(test_df_slice["perturbed"])

        retries_remaining = num_retries + 1

        while retries_remaining > 0:

            try:
                # wait_time = None
                input_test_word_set = set(test_words)

                # CLEAN
                # with profile(activities=[ProfilerActivity.CUDA],
                # profile_memory=True, record_shapes=True) as prof:

                if shielding_model is not None:
                    logger.debug("Recovering %s", text)
                    candidates = recovery_model.get_candidates(text)
                    logger.debug("Extracted candidates %s", candidates)

                mapping = recovery_client(test_words, **client_kwargs)

                # CLEAN
                # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

                logger.debug("Got mapping from client: %s", mapping)

                output_df = pd.DataFrame(
                    mapping.items(), columns=["perturbed", "predicted"]
                )

                output_test_word_set = set(output_df["perturbed"])

                logger.debug("Recovered batch: %s", output_df)

                full_df = pd.merge(test_df_slice, output_df, on="perturbed", how="left")

                full_df["batch"] = i

                if responses_df is None:
                    responses_df = full_df
                else:
                    # technically not atomic, but all_lost_words is a set so it should be fine
                    responses_df = pd.concat([responses_df, full_df])

                all_lost_words.update(input_test_word_set - output_test_word_set)

                retries_remaining = 0
                responses_df.to_csv(checkpoint_file)

            except Exception as e:
                logger.error(e)
                retries_remaining -= 1
                if retries_remaining > 0:
                    logger.info("Retrying...")

                    # if wait_time is not None:
                    #     time.sleep(wait_time * (4-retries_remaining))
                    # else:
                    #     time.sleep(21)
                else:
                    logger.error(
                        "could not complete query for group with range %d %d",
                        start,
                        end,
                    )
                    failure_groups.append((start, end))
                    all_lost_words.update(input_test_word_set)
        # time.sleep(wait_time)

    # lost_words_df = pd.DataFrame({'perturbed': all_lost_words})

    return responses_df

def flatten_list_of_lists(list_of_lists):
    """
    Flattens a list of lists into a single list and returns a dictionary
    with positional mappings to reconstruct the original list of lists.

    :param list_of_lists: List of lists to be flattened
    :return: Tuple of flattened list and dictionary of mappings
    """
    flattened_list = []
    mappings = {}
    start_pos = 0

    for sublist in list_of_lists:
        # Record the start and end positions of the current sublist
        end_pos = start_pos + len(sublist)
        mappings[start_pos] = end_pos

        # Extend the flattened list with the current sublist
        flattened_list.extend(sublist)

        # Update the start position for the next sublist
        start_pos = end_pos

    return flattened_list, mappings


def reconstruct_list(flattened_list, mappings):
    """
    Reconstructs the original list of lists from the flattened list and mappings.

    :param flattened_list: The flattened list
    :param mappings: Dictionary of positional mappings
    :return: Reconstructed list of lists
    """
    reconstructed_list = []

    for start, end in mappings.items():
        # Extract each sublist using the start and end positions
        sublist = flattened_list[start:end]
        reconstructed_list.append(sublist)

    return reconstructed_list


class NonAsciiSentenceRecoveryModel(object):
    def __init__(self, recovery_model, num_beams=5, num_return_sequences=3):
        self.recovery_model = recovery_model
        self.whitespace_tokenizer = WhitespaceTokenizer()
        self.num_return_sequences = num_return_sequences
        self.num_beams = num_beams

    def get_candidates(self, text):

        tokens = self.whitespace_tokenizer.tokenize(text)

        tokens_for_recovery = []

        for token in tokens:
            if not token.isascii() and len(token) > 0:
                clean_token = token.strip(punctuation)
                tokens_for_recovery.append(clean_token)

        if len(tokens_for_recovery) == 0:
            return {}

        candidates = self.recovery_model.beam(
            tokens_for_recovery,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
        )
        logger.debug(f"Replacement candidates: {candidates}")
        generations = [re.split("\s", word)[0] for word in candidates]
        candidate_map = {}
        for i in range(0, len(generations), self.num_return_sequences):
            candidate_map[
                tokens_for_recovery[i // self.num_return_sequences]
            ] = generations[i : i + self.num_return_sequences]

        logger.debug(f"Replacement candidates: {candidate_map}")
        return candidate_map


class BatchNonAsciiSentenceRecoveryModel(object):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.whitespace_tokenizer = WhitespaceTokenizer()
        self.shielding_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            return_tensors="pt",
            device="cuda:0",
        )

    def get_candidates_batch(self, batch):

        token_batches = [self.whitespace_tokenizer.tokenize(text) for text in batch]

        token_batches_for_recovery = []

        for token_batch in token_batches:
            tokens_for_recovery = []
            token_batches_for_recovery.append(tokens_for_recovery)
            for token in token_batch:
                if not token.isascii() and len(token) > 0:
                    clean_token = token.strip(punctuation)
                    tokens_for_recovery.append(clean_token)

        tokens, mapping = flatten_list_of_lists(token_batches_for_recovery)

        return_tokens = self.shielding_pipeline(
            tokens,
            do_sample=False,
            num_return_sequences=1,
            batch_size=500,
            num_beams=3,
            max_length=40,
        )

        detokenized = [
            self.tokenizer.decode(x["generated_token_ids"], skip_special_tokens=True)
            for x in return_tokens
        ]

        relisted_candidate_tokens = reconstruct_list(detokenized, mapping)
        candidate_maps = []
        for i, candidate_tokens in enumerate(relisted_candidate_tokens):
            candidate_maps.append(
                {k: v for k, v in zip(token_batches_for_recovery[i], candidate_tokens)}
            )

        logger.debug(f"Replacement candidates: {candidate_maps}")
        return candidate_maps
