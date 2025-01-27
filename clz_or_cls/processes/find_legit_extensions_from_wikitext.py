# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import os
import random
from collections import defaultdict
from pathlib import Path

import nltk
import pandas as pd
from nltk.tag import pos_tag
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, TrOCRProcessor

from clz_or_cls.Similarity import SimHelper
from datasets import load_dataset

wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train")


punctuation = {
    ",",
    ".",
    "!",
    "?",
    ";",
    ":",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    '"',
    "'",
    "=",
    "-",
    "_",
    "#",
    "!",
    "%",
    "^",
    "&",
    "*",
    "/",
}

findings = defaultdict(lambda: 0)

for sentence in tqdm(wikitext["text"]):
    for token in sentence.split(" "):
        if len(token) > 3 and token != "<unk>" and not token.isnumeric():

            if all([c not in punctuation for c in token]):
                if not token.isascii() and token[0].islower():
                    findings[token.lower()] += 1

df = pd.DataFrame(findings.items(), columns=["word", "count"])
df = df.sort_values("count", ascending=False)

prep_path = Path(os.environ['CORC_DATASETS_PREP_DIR'])
df.to_csv(prep_path/"prep_path"/"legit-extended/wikitext-accents.csv")

skip = ["σαυρος", "æftiʀ", "seiðr", "ætheling", "iqtaʿat", "abnaʾ", "stæin", "μmol"]

for row in df[~df["word"].isin(skip)][:100].itertuples():
    print(row.word, row.count)


uncommon_words = defaultdict(lambda: 0)

common_words = set()

for sentence in tqdm(wikitext["text"]):
    for token in sentence.split(" "):
        if (
            token.lower() not in common_words
            and len(token) > 3
            and token != "<unk>"
            and not token.isnumeric()
        ):
            if all([c not in punctuation for c in token]):
                if token.isascii():
                    uncommon_words[token.lower()] += 1

                    if uncommon_words[token.lower()] > 500:
                        common_words.add(token.lower())


for common_word in common_words:
    uncommon_words.pop(common_word)

uw_df = pd.DataFrame(uncommon_words.items(), columns=["word", "count"])
uw_df = uw_df.sort_values("count", ascending=False)

uncommon_words = random.sample(set(uw_df[uw_df["count"] > 200]["word"]), 250)

selected_uw_df = uw_df[uw_df["word"].isin(uncommon_words)]
selected_uw_df.head(10)

prep_path = Path(os.environ['CORC_DATASETS_PREP_DIR'])
selected_uw_df.to_csv(prep_path/"legit-extended"/"wikitext-uncommon.csv")

nltk.download("averaged_perceptron_tagger")
for word in selected_uw_df["word"][10:20]:
    print(word, pos_tag([word[0].upper() + word[1:]]))


model = AutoModel.from_pretrained(
    "dvsth/LEGIT-TrOCR-MT", revision="main", trust_remote_code=True
)

preprocessor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

features_path = Path(os.environ['CORC_FEATURES_DIR'])
sim_space = SimHelper.create_sim_space(model, features_path/"trocr.hdf", num_nearest=200)

neighbors = sim_space.topk_neighbors(ord("A"), 200)

print(len(neighbors), [chr(int(n)) for n in neighbors])
