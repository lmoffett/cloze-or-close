# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import collections
import os
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model
import tqdm

df_ds = namedtuple("df_ds", ["test", "train", "valid"])


def load_csvs(test, train, valid):
    return df_ds(pd.read_csv(test), pd.read_csv(train), pd.read_csv(valid))


def get_finetuning_results_csvs(results_dir=Path(os.environ['CORC_RESULTS_DIR'])/"adword-recovery"):

    results = {}
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            print("skipping not model_dir", model_dir)
            continue
        for size_dir in model_dir.iterdir():
            if not size_dir.is_dir():
                print("skipping not size_dir", size_dir)
                continue
            for run_results in size_dir.iterdir():
                if run_results.suffix == ".csv" and "--" in run_results.stem:
                    metadata = run_results.stem.split("--")
                    if len(metadata) == 1:
                        test = metadata
                        train = ""
                    elif len(metadata) == 2:
                        train, test = metadata
                    elif len(metadata) == 3:
                        train, test, model = metadata
                        train = train + "--" + model
                    else:
                        raise Exception("unexpected metadata", metadata)
                    if train not in results.keys():
                        results[train] = {}

                    results[train][test] = run_results
                else:
                    print("skipping", run_results.stem)

    return results


def get_prompt_recovery_csvs(results_dir = Path(os.environ['CORC_RESULTS_DIR'])/"adword-recovery", skip_test=[]):

    results = {}
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            print("skipping not model_dir", model_dir)
            continue
        for size_dir in model_dir.iterdir():
            if not size_dir.is_dir():
                print("skipping not size_dir", size_dir)
                continue

            for prompt_spec_dir in size_dir.iterdir():
                if not prompt_spec_dir.is_dir():
                    print("skipping prompt_spec_dir", prompt_spec_dir)
                    continue

                for run_results in prompt_spec_dir.iterdir():
                    if run_results.suffix != ".csv" or "--" not in run_results.stem:
                        print("skipping not result file", run_results)
                        continue

                    try:
                        train, test, meta = run_results.stem.split("--")
                    except:
                        train, test = run_results.stem.split("--")
                        meta = None

                    if test in skip_test:
                        continue

                    xshot = prompt_spec_dir.stem
                    model_size = size_dir.stem
                    model = model_dir.stem

                    if model not in results.keys():
                        results[model] = {}

                    model_results = results[model]

                    if model_size not in model_results.keys():
                        model_results[model_size] = {}

                    model_results_at_size = model_results[model_size]

                    if xshot not in model_results_at_size.keys():
                        model_results_at_size[xshot] = {}

                    model_results_at_size_prompt = model_results_at_size[xshot]

                    if train not in model_results_at_size_prompt.keys():
                        model_results_at_size_prompt[train] = {}

                    if meta is None:
                        model_results_at_size_prompt[train][test] = run_results
                    else:
                        if test not in model_results_at_size_prompt[train]:
                            model_results_at_size_prompt[train][test] = {}

                        model_results_at_size_prompt[train][test][meta] = run_results

    return results


def get_hot_classification_csvs(results_dir=Path(os.environ['CORC_RESULTS_DIR']) / "hot", skip_test=[]):

    results = {}
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            print("skipping not model_dir", model_dir)
            continue
        for size_dir in model_dir.iterdir():
            if not size_dir.is_dir():
                print("skipping not size_dir", size_dir)
                continue

            for prompt_spec_dir in size_dir.iterdir():
                if not prompt_spec_dir.is_dir():
                    print("skipping prompt_spec_dir", prompt_spec_dir)
                    continue

                for ratio_dir in prompt_spec_dir.iterdir():
                    if not ratio_dir.is_dir():
                        print("skipping prompt_spec_dir", prompt_spec_dir)
                        continue

                    for run_results in ratio_dir.iterdir():
                        if run_results.suffix != ".csv" or "--" not in run_results.stem:
                            print("skipping not result file", run_results)
                            continue

                        try:
                            train, test, meta = run_results.stem.split("--")
                        except:
                            train, test = run_results.stem.split("--")
                            meta = None

                        if test in skip_test:
                            continue

                        xshot = prompt_spec_dir.stem
                        model_size = size_dir.stem
                        model = model_dir.stem
                        ratio = ratio_dir.stem.replace("_", ".")

                        if model not in results.keys():
                            results[model] = {}

                        model_results = results[model]

                        if model_size not in model_results.keys():
                            model_results[model_size] = {}

                        model_results_at_size = model_results[model_size]

                        if xshot not in model_results_at_size.keys():
                            model_results_at_size[xshot] = {}

                        xshot_level = model_results_at_size[xshot]

                        if ratio not in xshot_level:
                            xshot_level[ratio] = {}

                        model_results_at_size_prompt = xshot_level[ratio]

                        if train not in model_results_at_size_prompt.keys():
                            model_results_at_size_prompt[train] = {}

                        if meta is None:
                            model_results_at_size_prompt[train][test] = run_results
                        else:
                            if test not in model_results_at_size_prompt[train]:
                                model_results_at_size_prompt[train][test] = {}

                            model_results_at_size_prompt[train][test][
                                meta
                            ] = run_results

    return results


def load_and_concatenate_csvs(nested_dict, levels, hierarchy=None):
    if hierarchy is None:
        hierarchy = []

    combined_df = pd.DataFrame()

    for key, value in nested_dict.items():
        # Update the current hierarchy level
        current_hierarchy = hierarchy + [key]

        if isinstance(value, dict):
            # Recursive call for nested dictionaries
            df = load_and_concatenate_csvs(value, levels, hierarchy=current_hierarchy)
            combined_df = pd.concat([combined_df, df])
        elif isinstance(value, Path):
            # Load CSV and add hierarchy information
            df = pd.read_csv(value)
            for i, h_key in enumerate(levels):
                df[h_key] = current_hierarchy[i] if i < len(current_hierarchy) else None
            combined_df = pd.concat([combined_df, df])

    return combined_df


def num_perturbations(row):
    count = 0
    for c, p in zip(row.clean, row.perturbed):
        if c != p:
            count = count + 1

    return count


def find_novel_mappings(m1, m2):

    novel_mappings = {}
    for c2, p2 in m2.items():
        novel_mappings[c2] = p2 - m1[c2] if c2 in m1.keys() else p2
    return novel_mappings


def count_letters(series):
    letter_count = {}
    for s in series:
        for c in s:
            if c.isalpha() and c.islower():
                if c in letter_count:
                    letter_count[c] += 1
                else:
                    letter_count[c] = 1
    return pd.Series(letter_count)


def letter_freq_from_series(series):
    counts = count_letters(series)
    total_letters = counts.sum()
    return (counts, (counts / total_letters) * 100)


bucket_tuple = collections.namedtuple(
    "char_mappings",
    ["clean", "perturbed", "clean_mappings", "lost_mappings", "unbound_perturbations"],
)


def count_character_mappings(clean_wordlist, perturbed_wordlist):
    clean_buckets = {}
    perturbed_buckets = {}
    clean_mappings = {}
    lost_mappings = {}
    unbound_perturbations = {}

    def increment(buckets, char):
        if char in buckets.keys():
            buckets[char] = buckets[char] + 1
        else:
            buckets[char] = 1

    for clean_word, perturbed_word in tqdm.tqdm(
        zip(clean_wordlist, perturbed_wordlist)
    ):
        if len(clean_word) == len(perturbed_word):
            for clean_char, perturbed_char in zip(clean_word, perturbed_word):
                if clean_char not in clean_mappings:
                    clean_mappings[clean_char] = {}
                increment(clean_mappings[clean_char], perturbed_char)
        else:
            for char in clean_word:
                increment(lost_mappings, char)
            for char in perturbed_word:
                increment(unbound_perturbations, char)

        for buckets, word in zip(
            [clean_buckets, perturbed_buckets], [clean_word, perturbed_word]
        ):
            for char in word:
                increment(buckets, char)

    return bucket_tuple(
        clean_buckets,
        perturbed_buckets,
        clean_mappings,
        lost_mappings,
        unbound_perturbations,
    )
