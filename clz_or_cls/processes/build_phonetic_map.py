# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import math
import os
import pickle
import random
import sys
from pathlib import Path

import eng_to_ipa as ipa
import numpy as np
import pandas as pd
import panphon
import pronouncing
import scipy
from nltk.metrics.distance import edit_distance
from tqdm import tqdm

from datasets import load_dataset

from .. import phonetics as pho

pronouncing.init_cmu()
ipa_eng_dict_df = pho.load_ipa_eng_dict()
ipa_parser = pho.IPAParser(ipa_eng_dict_df)

grapheme_phoneme_map = {}
skipped = []
skip = {"aaa", "abbruzzese", "abc", "abc's", "abcs", "aberle"}

for i, word in enumerate(tqdm(pronouncing.cmudict.words())):
    if not ipa.isin_cmu(word) or word.startswith("'") or word in skip:
        skipped.append(word)
        continue

    ipa_spellings = ipa_parser.convert(word)

    mapped = False
    for word_ipa in ipa_spellings:
        # print(word_ipa)
        try:

            ipa_parsed = ipa_parser.parse_ipa(word_ipa)
            # print(ipa_parsed)
            ipa_map = ipa_parser.map_word_to_ipa(word, ipa_parsed)

            for eng_ch, ipa_ch in ipa_map:
                if eng_ch not in grapheme_phoneme_map:
                    grapheme_phoneme_map[eng_ch] = {}

                if ipa_ch not in grapheme_phoneme_map[eng_ch]:
                    grapheme_phoneme_map[eng_ch][ipa_ch] = 0

                grapheme_phoneme_map[eng_ch][ipa_ch] += 1
            mapped = True
        except:
            pass

    if not mapped:
        skipped.append(word)

grapheme_phoneme_map

features_path = Path(os.environ["CORC_FEATURES_DIR"])
with open(features_path / "phonee" / "grapheme_phoneme_map.pkl", "wb") as f:
    pickle.dump(grapheme_phoneme_map, f)


data = []
for grepheme, inner_dict in grapheme_phoneme_map.items():
    # Iterate over inner dictionary
    for phoneme, count in inner_dict.items():
        # Add a tuple (outer_key, inner_key, count) to the data list
        data.append((grepheme, phoneme, count))

# Convert list of tuples into a DataFrame
grapheme_phoneme_df = pd.DataFrame(data, columns=["grapheme", "phoneme", "count"])
grapheme_phoneme_df.to_csv(features_path / "phonee" / "grapheme_phoneme_frequency_full.csv")

grapheme_phoneme_df = pd.read_csv(features_path / "phonee" / "grapheme_phoneme_frequency_full.csv")

legit = load_dataset("dvsth/LEGIT")

legit_mapped = set()
legit_no_pronounce = set()
legit_no_parse = set()
legit_no_map = set()

for split in ["x"]:

    for i, word in enumerate(tqdm(legit_no_map)):
        if not ipa.isin_cmu(word) or word.startswith("'"):
            legit_no_pronounce.add(word)
            continue

        ipa_spellings = ipa_parser.convert(word)
        if len(ipa_spellings) == 0 or ipa_spellings[0][-1] == "*":
            legit_no_parse.add(word)
        ipa_map = None
        for word_ipa in ipa_spellings:
            try:
                ipa_parses = ipa_parser.parse_ipa(word_ipa)

                for parse in ipa_parses:
                    ipa_map = ipa_parser.map_word_to_ipa(word, parse)
                    if ipa_map is not None:
                        break
                if ipa_map is not None:
                    break
            except:
                pass

        if ipa_map is not None:
            legit_mapped.add(word)
        else:
            legit_no_map.add(word)

# convert each dictionary entry into a sorted list of tuples with relative frequency
grapheme_phoneme_freq_map = {}
for phoneme, group in grapheme_phoneme_df.groupby("phoneme"):
    group = group.sort_values("relative_freq", ascending=False)
    grapheme_phoneme_freq_map[phoneme] = (
        list(group["grapheme"]),
        np.array(group["relative_freq"]),
        np.array(group["softmax"]),
    )

# Here's the perturbation
ipa_eng_dict_df = pho.load_ipa_eng_dict()
ipa_parser = pho.IPAParser(ipa_eng_dict_df)

def perturb_mapped_word(ipa_tokenized_word, p=0.33):
    new_word = []
    num_perturbations = 0
    perturbed_idxs = list(range(len(ipa_tokenized_word)))
    random.shuffle(perturbed_idxs)
    perturbed_idxs = perturbed_idxs[: max(1, math.floor(len(perturbed_idxs) * p))]

    for i, (eng, ipa) in enumerate(ipa_tokenized_word):
        if i not in perturbed_idxs:
            new_word.append((eng, ipa, eng))
            continue

        if ipa not in grapheme_phoneme_freq_map:
            new_word.append((eng, ipa, eng))
            continue

        graphemes, relfreq, softmax = grapheme_phoneme_freq_map[ipa]

        if i == 0:
            try:
                remove_list = [g for g in graphemes if not g.startswith(eng[0])] + [eng]
            except:
                print(eng, graphemes)
                raise
        else:
            remove_list = [eng]

        graphemes = graphemes.copy()
        for remove_item in remove_list:
            if remove_item in graphemes:

                let_pos = graphemes.index(remove_item)
                graphemes.remove(remove_item)
                relfreq = np.delete(relfreq, let_pos)
                softmax = np.delete(softmax, let_pos)

        if len(graphemes) == 0:
            new_word.append((eng, ipa, eng))
            continue

        num_perturbations += 1
        relfreq = np.cumsum(relfreq / np.sum(relfreq))
        softmax = np.cumsum(softmax / np.sum(softmax))
        criteria = random.random()

        replacement = np.where(criteria <= relfreq)[0][0]
        print(replacement, end="|")

        #   print(len(graphemes), replacement)
        replacement = graphemes[replacement]
        new_word.append((eng, ipa, replacement))

    return new_word


for word in list(ipa_parser.manual_map.keys()):
    print(word, end="...")
    word_ipas = ipa_parser.convert(word)
    for word_ipa in word_ipas:

        print(word_ipa, end="\n\t")
        ipa_parses = ipa_parser.parse_ipa(word_ipa)
        ipa_map = None
        for parse in ipa_parses:
            try:
                print(parse, end="...")
                ipa_map = ipa_parser.map_word_to_ipa(word, parse)
                print(ipa_map, end="...")
                break
            except:
                print("x", end=".\n\t")

        if ipa_map is not None:
            perturbation = perturb_mapped_word(ipa_map)
            print("\n", "".join(p[2] for p in perturbation))
            break
        else:
            print("!")
