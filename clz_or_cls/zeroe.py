# This file includes code from Zeroé, licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Original code source: https://github.com/yannikbenz/zeroe/blob/master/code/attacks/simple_attacks.py
#  Copyright (c) 2020.
#  Author: Yannik Benz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import re
import string

import nltk
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer


def simple_perturb(text: str, method: str, perturbation_level=0.2):
    """
    :param text:
    :param method:
    :param perturbation_level:
    :return:
    """
    if not 0 <= perturbation_level <= 1:
        raise ValueError("Invalid value for perturbation level.")

    # we need to handle segmentation separate
    if method == "segment":
        return segmentation(text, perturbation_level)
    words = nltk.word_tokenize(text)
    word_indexes = list(range(0, len(words)))
    perturbed_words = 0
    perturb_target = len(words) * perturbation_level
    while perturbed_words < perturb_target:
        # emergency exit
        if len(word_indexes) < 1:
            break
        # pick a random word
        index = np.random.choice(word_indexes)
        word_indexes.remove(index)
        word = words[index]
 
        if method == "full-swap":
            perturbed_word = swap(word, inner=False)
        elif method == "inner-swap":
            perturbed_word = swap(word, inner=True)
        elif method == "intrude":
            perturbed_word = intruders(word, perturbation_level=perturbation_level)
        elif method == "disemvowel":
            perturbed_word = disemvoweling(word)
        elif method == "truncate":
            perturbed_word = truncating(word)
        elif method == "keyboard-typo":
            perturbed_word = key(word)
        elif method == "natural-typo":
            perturbed_word = natural(word)
        else:
            raise ValueError(f"Unknown operation {method}")
        words[index] = perturbed_word
        perturbed_words += 1 if perturbed_word != word else 0
    return TreebankWordDetokenizer().detokenize(words)


def swap(word: str, inner: bool, seed=None):
    """Shuffles the chars in each word. If inner is set the first and last letters position remain untouched.

    >>> swap("hello world", True, 56)
    hlelo wlord

    >>> swap("hello word", False, 42)
    eolhl odrwl

    :param word:
    :param seed: seed
    :param inner: if set, only the inner part of the word will be swapped
    :return: swapped text
    """

    def __shuffle_string__(_word: str, _seed=seed):
        """
        shuffles the given string if a seed is given it shuffles in respect to the given seed.

        hello world -> elloh roldw

        :param _seed: seed
        :param _word: string (word) to shuffle
        :return: shuffled string
        """
        chars = list(_word)
        if _seed is not None:
            np.random.seed(_seed)
        np.random.shuffle(chars)
        return "".join(chars)

    if len(word) < 3 or inner and len(word) < 4:
        return word
    perturbed = word
    tries = 0
    while perturbed == word:
        tries += 1  # we can get a deadlock if the word is e.g. maas
        if tries > 10:
            break
        if inner:
            first, mid, last = word[0], word[1:-1], word[-1]
            perturbed = first + __shuffle_string__(mid) + last
        else:
            perturbed = __shuffle_string__(word)
    return perturbed


def intruders(word: str, perturbation_level=0.3, seed=None):
    """
    :param perturbation_level:
    :param word:
    :param seed:
    :return:
    """
    chars = list(word)
    perturbed = word
    punct = random.choice(string.punctuation)
    # if word in string.punctuation or len(word) < 2:
    #     return word
    # if len(word) == 2:
    #     return chars[0] + punct + chars[-1]
    while perturbed == word:
        i = 0
        while i < len(chars):
            if seed is not None:
                np.random.seed(seed)
            if np.random.uniform(0, 1) < perturbation_level:
                chars.insert(i, punct)
                i += 1
            i += 1
        perturbed = "".join(chars)
    return perturbed


def disemvoweling(word: str, vocals: str = "AEIOU"):
    """
    :return:
    """
    if len(word) < 3:
        return word

    count_vowels = 0
    for char in word:
        if char.upper() in vocals:
            count_vowels += 1

    if count_vowels == len(word):
        return word

    return re.sub(r"[" + vocals + "]", "", word, flags=re.IGNORECASE)


def truncating(word: str, minlen: int = 3, cutoff: int = 1):
    """
    :param cutoff:
    :param minlen:
    :param word:
    :return:
    """
    chars = list(word)
    tmp_cutoff = cutoff
    while len(chars) > minlen and tmp_cutoff > 0:
        chars = chars[:-1]
        tmp_cutoff -= 1
    return "".join(chars)


# This code has been taken from https://github.com/ybisk/charNMT-noise
NN = {}
for line in open(os.environ['CORC_FEATURES_DIR'] + "/zeroe/en.key"):
    line = line.split()
    NN[line[0]] = line[1:]
# ===


def key(word, probability=1.0):
    """

    This code has been taken from https://github.com/ybisk/charNMT-noise

    :param word:
    :param probability:
    :return:
    """
    if random.random() > probability:
        return word
    word = list(word)
    i = random.randint(0, len(word) - 1)
    char = word[i]
    caps = char.isupper()
    if char in NN:
        word[i] = NN[char.lower()][random.randint(0, len(NN[char.lower()]) - 1)]
        if caps:
            word[i].upper()
    return "".join(word)


typos = {}
import enchant

en_dict = enchant.Dict("en_US")
for line in open(os.environ['CORC_FEATURES_DIR'] + "/zeroe/en.natural"):
    line = line.strip().split()

    valid = [word for word in line[1:] if not en_dict.check(word.lower())]

    typos[line[0].lower()] = valid


def natural(word, precentage=1.0):
    """
    This code has been taken from https://github.com/ybisk/charNMT-noise
    :param word:
    :param precentage:
    :return:
    """
    if random.random() > precentage:
        return word
    if word in typos:
        return typos[word][random.randint(0, len(typos[word]) - 1)]
    return word


def segmentation(text: str, probability=0.3):
    """
    :param probability:
    :param text:
    :return:
    """
    result = []
    buffer = ""
    for word in nltk.word_tokenize(text):
        if np.random.uniform(0, 1) < probability:
            buffer += word
        else:
            result.append(buffer + word)
            buffer = ""
    if buffer != "":
        result.append(buffer)
    return TreebankWordDetokenizer().detokenize(result)
