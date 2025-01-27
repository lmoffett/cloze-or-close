# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import logging
import math
import os
import random
import re
from collections import defaultdict

import enchant
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import tqdm
from gensim.models import KeyedVectors as W2Vec
from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
from transformers import VisionEncoderDecoderConfig

from . import zeroe, anthro_lib, phonee
from .similarity import SimHelper

device = torch.device(os.environ['TORCH_DEVICE'])
logger = logging.Logger("perturbation")


class DecodablePerturbationStrategy:
    def __init__(self, letter_prob=0.37, k_distribution=stats.geom(0.05)):
        """
        Args:
            p (float): The probability of perturbing a token.
        """
        self.letter_prob = letter_prob
        self.k_distribution = k_distribution
        self.dictionary = enchant.Dict("en_US")

    def __call__(self, clean, n=2, return_mapping=False):
        return self.perturb(clean, n, return_mapping)

    def perturb(self, clean, n=2, return_mapping=False):
        return [self.do_perturb(clean, return_mapping) for _ in range(n)]

    def perturb_batch(self, clean_batch, n=2, return_mapping=False):
        """
        Apply this strategy to a batch of strings.
        """
        perturbed_batch = []
        mapping_batch = []

        for clean in tqdm.tqdm(clean_batch):
            results = self.perturb(clean, n=n, return_mapping=return_mapping)
            for result in results:

                if return_mapping:
                    perturbed_batch.append(result[0])
                    mapping_batch.append(result[1])
                else:
                    perturbed_batch.append(result)

        if return_mapping:
            return perturbed_batch, mapping_batch
        else:
            return perturbed_batch

    def do_perturb(self, clean, return_mapping):
        """
        Perturb a string using this strategy.

        Args:
            clean (str): The clean string to perturb.

        Returns:
            str: The perturbed string.
            mapping: List of tuples with the form (clean_substring, perturbed_substring). For strings that are unperturbed, self is returned.
        """
        clean_tokens = self.tokenize(clean)

        dist = np.random.rand(len(clean_tokens))
        perturbed_idxs = dist < self.letter_prob

        if np.sum(perturbed_idxs) == 0:
            # Always perturb at least one token
            perturbed_idxs[np.random.randint(0, len(clean_tokens))] = True

        mappings = []

        for i, token in enumerate(clean_tokens):
            if perturbed_idxs[i] == False:
                mappings.append((token, token))

            else:
                perturbed_token = self.perturb_token(
                    token, self.k_distribution.rvs(1)[0]
                )
                mappings.append((token, perturbed_token))

        perturbed_tokens = [self.detokenize(m[1]) for m in mappings]
        try:
            perturbed_string = "".join(perturbed_tokens)
        except:
            print(perturbed_tokens, clean_tokens, mappings)
            raise

        if (
            perturbed_string != clean
            and len(perturbed_string) > 0
            and self.dictionary.check(perturbed_string)
        ):
            # technically could infinitely recurse, but that probably indicates a bug in the parameters or the dictionary,
            # so we'll just let it crash
            return self.do_perturb(clean, return_mapping)

        if return_mapping:
            return perturbed_string, mappings
        else:
            return perturbed_string

    def tokenize(self, clean):
        """
        Turn a string into a list a potentially perturbable tokens.

        Args:
            clean (str): The clean string to tokenize.
        Returns:
            list: List of tokens.
        """
        return clean

    def perturb_token(self, clean_token, k=0):
        return clean_token

    def detokenize(self, perturbed_token):
        return perturbed_token


class IcesStrategy(DecodablePerturbationStrategy):
    def __init__(self):
        super().__init__()
        #  Visual attacker based on VAEGAN representations.
        #
        #  code has been taken from:
        #  https://github.com/UKPLab/naacl2019-like-humans-visual-attacks/tree/master/code
        #  adaptions were made to integrate it into our workflow

        self.w2v_model = W2Vec.load_word2vec_format(os.environ['CORC_FEATURES_DIR'] + "/viper/ices.txt")
        self.weights = torch.FloatTensor(self.w2v_model.vectors).to(device)

    def torch_similarity(self, c, top_k):
        embed = torch.tensor(self.w2v_model.get_vector(c)).to(device)

        dist = F.cosine_similarity(embed, self.weights)
        top = dist.topk(top_k + 1)
        return [
            (self.w2v_model.index_to_key[ind.item()], prob.item())
            for ind, prob in zip(top.indices, top.values)
        ][1:]

    def perturb_token(self, clean_token, k=0):
        try:
            perturbed_token = self.torch_similarity(clean_token, k)[-1][0]
            return perturbed_token
        except KeyError:
            print(f"mapping failed for {clean_token}")
            return clean_token


class LegitStrategy(DecodablePerturbationStrategy):
    def __init__(self):
        super().__init__(letter_prob=0.15)

        model = VisionEncoderDecoderConfig.from_pretrained(
            "dvsth/LEGIT-TrOCR-MT", revision="main")
        self.sim_space = SimHelper.create_sim_space(
            model, os.environ['CORC_FEATURES_DIR'] + "/legit/trocr.hdf", num_nearest=200
        )

    def perturb_token(self, clean, k=0):
        try:
            codepoint = self.sim_space.topk_neighbors(ord(clean), k)[-1]
            return chr(int(codepoint))
        except KeyError:
            print(f"mapping failed for {clean}")
            return clean


class DcesStrategy(DecodablePerturbationStrategy):
    def __init__(self):
        super().__init__()

        # load the unicode descriptions into a single dataframe with the chars as indices
        descs = pd.read_csv(
            os.environ['CORC_FEATURES_DIR'] + "/viper/dcesNamesList.txt",
            skiprows=np.arange(16),
            header=None,
            names=["code", "description"],
            usecols=["code", "description"],
            delimiter="\t",
        )
        descs = descs.dropna(0)
        descs_arr = descs.values  # remove the rows after the descriptions
        vectorizer = CountVectorizer(max_features=1000)
        desc_vecs = vectorizer.fit_transform(descs_arr[:, 0]).astype(float)
        vecsize = desc_vecs.shape[1]
        self.vec_colnames = np.arange(vecsize)
        desc_vecs = pd.DataFrame(
            desc_vecs.todense(), index=descs.index, columns=self.vec_colnames
        )

        self.descs = pd.concat([descs, desc_vecs], axis=1)
        self.perturbations_file = PerturbationsStorage("./perturbations.txt")

        self.disallowed = [
            "TAG",
            "MALAYALAM",
            "BAMUM",
            "HIRAGANA",
            "RUNIC",
            "TAI",
            "SUNDANESE",
            "BATAK",
            "LEPCHA",
            "CHAM",
            "TELUGU",
            "DEVANGARAI",
            "BUGINESE",
            "MYANMAR",
            "LINEAR",
            "SYLOTI",
            "PHAGS-PA",
            "CHEROKEE",
            "CANADIAN",
            "YI",
            "LYCIAN",
            "HANGUL",
            "KATAKANA",
            "JAVANESE",
            "ARABIC",
            "KANNADA",
            "BUHID",
            "TAGBANWA",
            "DESERET",
            "REJANG",
            "BOPOMOFO",
            "PERMIC",
            "OSAGE",
            "TAGALOG",
            "MEETEI",
            "CARIAN",
            "UGARITIC",
            "ORIYA",
            "ELBASAN",
            "CYPRIOT",
            "HANUNOO",
            "GUJARATI",
            "LYDIAN",
            "MONGOLIAN",
            "AVESTAN",
            "MEROITIC",
            "KHAROSHTHI",
            "HUNGARIAN",
            "KHUDAWADI",
            "ETHIOPIC",
            "PERSIAN",
            "OSMANYA",
            "ELBASAN",
            "TIBETAN",
            "BENGALI",
            "TURKIC",
            "THROWING",
            "HANIFI",
            "BRAHMI",
            "KAITHI",
            "LIMBU",
            "LAO",
            "CHAKMA",
            "DEVANAGARI",
            "ITALIC",
            "CJK",
            "MEDEFAIDRIN",
            "DIAMOND",
            "SAURASHTRA",
            "ADLAM",
            "DUPLOYAN",
        ]

        self.disallowed_codes = ["1F1A4", "A7AF"]
        self.character_variations = {}
        self.nearest_neighbors = {}

    def perturb_token(self, clean_token, k=0):
        similar_chars = self.get_unicode_desc_nn(
            clean_token, self.perturbations_file, topn=k
        )
        if len(similar_chars) > 0:
            return similar_chars[-1]
        else:
            print(f"mapping failed for {clean_token}")
            return clean_token

    def __char_to_hex_string(self, ch):
        return "{:04x}".format(ord(ch)).upper()

    # function for retrieving the variations of a character
    def __get_all_variations(self, ch):

        # get unicode number for c
        c = self.__char_to_hex_string(ch)

        if ch in self.character_variations:
            return c, self.character_variations[ch]

        # problem: latin small characters seem to be missing?
        if np.any(self.descs["code"] == c):
            description = self.descs["description"][self.descs["code"] == c].values[0]
        else:
            print("Failed to disturb %s, with code %s" % (ch, c))
            return c, np.array([])

        # strip away everything that is generic wording, e.g. all words with > 1 character in
        toks = description.split(" ")

        case = "unknown"

        identifiers = []
        for tok in toks:

            if len(tok) == 1:
                identifiers.append(tok)

                # for debugging
                if len(identifiers) > 1:
                    print("Found multiple ids: ")
                    print(identifiers)

            elif tok == "SMALL":
                case = "SMALL"
            elif tok == "CAPITAL":
                case = "CAPITAL"

        # find matching chars
        matches = []

        for i in identifiers:
            for idx in self.descs.index:
                desc_toks = self.descs["description"][idx].split(" ")
                if (
                    i in desc_toks
                    and not np.any(np.in1d(desc_toks, self.disallowed))
                    and not np.any(
                        np.in1d(self.descs["code"][idx], self.disallowed_codes)
                    )
                    and not int(self.descs["code"][idx], 16) > 30000
                ):

                    # get the first case descriptor in the description
                    desc_toks = np.array(desc_toks)
                    case_descriptor = desc_toks[
                        (desc_toks == "SMALL") | (desc_toks == "CAPITAL")
                    ]

                    if len(case_descriptor) > 1:
                        case_descriptor = case_descriptor[0]
                    elif len(case_descriptor) == 0:
                        case = "unknown"

                    if case == "unknown" or case == case_descriptor:
                        matches.append(idx)

        # check the capitalisation of the chars

        self.character_variations[ch] = np.array(matches)
        return c, np.array(matches)

    # function for finding the nearest neighbours of a given word
    def get_unicode_desc_nn(self, c, perturbations_file, topn=1):
        # we need to consider only variations of the same letter -- get those first, then apply NN
        c, matches = self.__get_all_variations(c)

        if not len(matches):
            return []  # cannot disturb this one

        if c in self.nearest_neighbors:
            X, Y, neigh = self.nearest_neighbors[c]
        else:
            # get their description vectors
            match_vecs = self.descs[self.vec_colnames].loc[matches]

            # find nearest neighbours
            neigh = NearestNeighbors(metric="euclidean")
            Y = match_vecs.values
            neigh.fit(Y)
            X = self.descs[self.vec_colnames].values[self.descs["code"] == c]

            self.nearest_neighbors[c] = (X, Y, neigh)

        if Y.shape[0] > topn:
            dists, idxs = neigh.kneighbors(X, topn, return_distance=True)
        else:
            dists, idxs = neigh.kneighbors(X, Y.shape[0], return_distance=True)

        # turn idxs back to chars
        charcodes = self.descs["code"][matches[idxs.flatten()]]

        chars = []
        for charcode in charcodes:
            chars.append(chr(int(charcode, 16)))

        return chars


class LetterLevelZeroeStrategy(DecodablePerturbationStrategy):
    def __init__(self, method, letter_prob=0.2):
        super().__init__(letter_prob=letter_prob)
        if method not in ["keyboard-typo", "intrude"]:
            raise ValueError("Not a valid letter-level strategy")
        self.method = method

    def perturb_token(self, clean_token, k=0):
        try:
            return zeroe.simple_perturb(
                clean_token, method=self.method, perturbation_level=1.0
            )
        except:
            print(f"failed perturbation for {clean_token} using {self.method}")
            return clean_token


class DeleteStrategy(DecodablePerturbationStrategy):
    def __init__(self, letter_prob=0.05):
        super().__init__(letter_prob=letter_prob)

    def perturb_token(self, clean_token, k=0):
        return ""


class WordLevelZeroeStrategy(DecodablePerturbationStrategy):
    def __init__(self, method, perturbation_level=0.2):
        super().__init__()
        if method in ["keyboard-typo", "intrude"]:
            raise ValueError("Not a valid word-level strategy")
        self.method = method
        self.perturbation_level = perturbation_level

    def do_perturb(self, clean, return_mapping):
        try:
            perturbed_string = zeroe.simple_perturb(
                clean, self.method, self.perturbation_level
            )
        except:
            if self.method != "natural-typo":
                print(f"failed perturbation for {clean} using {self.method}")
            perturbed_string = clean

        if return_mapping:
            return perturbed_string, [(clean, perturbed_string)]
        else:
            return perturbed_string


class MultiStrategy(DecodablePerturbationStrategy):
    def __init__(self, strategies, selection_criteria="random"):
        self.strategies = strategies
        self.selection_criteria = selection_criteria

    def perturb_token(self, clean_token, k=0):
        raise NotImplementedError

    def tokenize(self, clean):
        raise NotImplementedError

    def detokenize(self, perturbed_token):
        raise NotImplementedError

    def perturb(self, clean, n=2, return_mapping=False):
        outputs = []
        for i in range(n):
            if self.selection_criteria == "random":
                retries = 3
                perturbed_string = clean
                while perturbed_string == "" or perturbed_string == clean:
                    try:
                        selected_strategy = random.choice(self.strategies)

                        perturbed_string = selected_strategy.perturb(
                            clean, n=1, return_mapping=return_mapping
                        )[0]
                    except:
                        retries -= 1
                        if retries == 0:
                            perturbed_string = clean
                            break

                outputs.append(perturbed_string)
            # ordered
            else:
                for i, strategy in enumerate(self.strategies):
                    perturbed = strategy.perturb(
                        clean, n=1, return_mapping=return_mapping
                    )[0]
                    if (perturbed != clean and perturbed not in outputs) or i == len(
                        self.strategies
                    ) - 1:
                        outputs.append(perturbed)
                        break

        return outputs


class ZeroePhoneticStrategy(DecodablePerturbationStrategy):
    def __init__(self):
        super().__init__(letter_prob=1)
        # lazy loaded since it's dependencies are in conflict with some torch dependencies
        from . import g2pp2g

        self.beam_size = 10

        g2pp2g.setup_gpu_share_config()

        self.perturbation_dict = {}
        self.perturbation_count = {}

    def perturb(self, clean, n=2, return_mapping=False):
        if clean not in self.perturbation_dict:
            from . import g2pp2g
            word_beams = g2pp2g.perturb_word(clean, beams=self.beam_size)
            self.perturbation_dict[clean] = word_beams
            self.perturbation_count[clean] = 0

        count = self.perturbation_count[clean]
        self.perturbation_count[clean] = (count + n) % self.beam_size
        try:
            perturbations = self.perturbation_dict[clean]
            end = count + n
            if end > len(perturbations):
                return (
                    perturbations[count : len(perturbations)]
                    + perturbations[0 : end - len(perturbations)]
                )
            else:
                return perturbations[count:end]
        except Exception as e:
            print("failed perturbation for %s with error %s" % (clean, e))
            return [clean] * n


class PhoneEStrategy(DecodablePerturbationStrategy):
    def __init__(self, letter_prob=0.2):
        super().__init__(letter_prob=letter_prob)
        ipa_eng_dict = phonee.load_ipa_eng_dict(os.environ['CORC_FEATURES_DIR'] + "/phonee/ipa_eng_dict.csv")
        self.ipa_parser = phonee.IPAParser(ipa_eng_dict)
        grapheme_phoneme_df = pd.read_csv(
            os.environ['CORC_FEATURES_DIR'] + "/phonee/grapheme_phoneme_frequency_full.csv"
        )
        grapheme_phoneme_freq_map = {}
        for phoneme, group in grapheme_phoneme_df.groupby("phoneme"):
            group = group.sort_values("relative_freq", ascending=False)
            grapheme_phoneme_freq_map[phoneme] = (
                list(group["grapheme"]),
                np.array(group["relative_freq"]),
                np.array(group["softmax"]),
            )
        self.grapheme_phoneme_freq_map = grapheme_phoneme_freq_map

    def perturb(self, clean, n=2, return_mapping=False):
        word_ipas = self.ipa_parser.convert(clean)
        outputs = []
        for word_ipa in word_ipas:
            ipa_parses = self.ipa_parser.parse_ipa(word_ipa)
            ipa_map = None

            for parse in ipa_parses:
                try:
                    ipa_map = self.ipa_parser.map_word_to_ipa(clean, parse)
                    break
                except:
                    pass

            if ipa_map is not None:
                while len(outputs) < n:
                    perturbation = self.perturb_mapped_word(ipa_map)
                    outputs.append("".join([p[2] for p in perturbation]))

        # just return the clean string when it fails
        outputs = outputs + [clean] * n
        return outputs[:n]

    def perturb_mapped_word(self, ipa_tokenized_word, p=0.33):
        new_word = []
        num_perturbations = 0
        perturbed_idxs = list(range(len(ipa_tokenized_word)))
        random.shuffle(perturbed_idxs)
        perturbed_idxs = perturbed_idxs[: max(1, math.floor(len(perturbed_idxs) * p))]

        for i, (eng, ipa) in enumerate(ipa_tokenized_word):
            if i not in perturbed_idxs:
                new_word.append((eng, ipa, eng))
                continue

            if ipa not in self.grapheme_phoneme_freq_map:
                new_word.append((eng, ipa, eng))
                continue

            graphemes, relfreq, softmax = self.grapheme_phoneme_freq_map[ipa]

            # default to removing the first letter
            remove_list = [eng]

            if i == 0:
                if len(eng) > 0:
                    condition = lambda g: not g.startswith(eng[0])
                else:
                    condition = lambda g: True

                try:
                    remove_list = [g for g in graphemes if condition(g)] + [eng]
                except:
                    logger.error(
                        "Failed to map word to phonemes: %s->%s", eng, graphemes
                    )

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

            replacement = graphemes[replacement]
            new_word.append((eng, ipa, replacement))

        return new_word


class AnthroStrategy(DecodablePerturbationStrategy):
    def __init__(self, perturbation_class, anthro_loaded=None):
        super().__init__(letter_prob=0.2)

        if perturbation_class not in ["phonetic", "typo"]:
            raise ValueError("Not a valid anthro strategy")

        self.perturbation_class = perturbation_class

        if anthro_loaded is not None:
            self.anthro = anthro_loaded
        else:
            self.anthro = anthro_lib.ANTHRO()
            self.anthro.load(os.environ['CORC_FEATURES_DIR'] + "/ANTHRO_Data_V1.0")

    def collapse_repeated(self, word):
        new = []
        last = None
        for c in word:
            if c != last:
                new.append(c)

        return "".join(new)

    def do_perturb(self, clean, return_mapping):
        perturbed_string = None

        strict = self.anthro.get_similars(clean, level=1, distance=50, strict=True)

        if self.perturbation_class == "phonetic":
            if len(strict) > 0:
                perturbed_string = random.choice(list(strict))
            else:
                perturbed_string = clean

        else:
            not_strict = self.anthro.get_similars(
                clean, level=1, distance=6, strict=False
            )
            non_phonetic = set(not_strict) - set(strict)
            non_phonetic = [
                s for s in non_phonetic if not self.dictionary.check(s.lower())
            ]

            target_edit_distance = max(
                1, (np.random.rand(len(clean)) < self.letter_prob).sum()
            )

            if len(non_phonetic) > 0:
                # not recapitalizing
                clean_collapsed = self.collapse_repeated(clean)

                delta_edit_distance = 100
                candidates = []
                for s in non_phonetic:
                    s_distance = edit_distance(
                        clean_collapsed, self.collapse_repeated(s.lower())
                    )
                    s_delta = abs(target_edit_distance - s_distance)
                    if s_delta < delta_edit_distance:
                        delta_edit_distance = s_delta
                        candidates = [s]
                    elif s_delta == delta_edit_distance:
                        candidates.append(s)

                perturbed_string = random.choice(candidates)
            else:
                perturbed_string = clean

        if return_mapping:
            return perturbed_string, [(clean, perturbed_string)]
        else:
            return perturbed_string


class MixedStrategy(DecodablePerturbationStrategy):
    def __init__(self, strategies, weights=None):
        self.strategies = strategies
        if weights is None:
            self.weights = np.ones(len(strategies)) / len(strategies)
        else:
            self.weights = weights

        self.strategy_skips = {}
        self.strategy_uses = {}

    def do_perturb(self, clean, return_mapping=False):
        perturbed_string = ""
        perturbations = 0
        brake = np.random.randint(1, len(clean) - 1)
        failures = 0
        used_strategies = set()
        while perturbations < 2:
            if perturbations == 0:
                clean_part = clean[:brake]
            else:
                clean_part = clean[brake:]

            try:
                selected_strategy = np.random.choice(self.strategies, p=self.weights)
                if selected_strategy in used_strategies:
                    self.strategy_skips[selected_strategy] = (
                        self.strategy_skips.get(selected_strategy, 0) + 1
                    )
                    continue
                perturbed_string = (
                    perturbed_string
                    + selected_strategy.perturb(clean_part, return_mapping=False)[0]
                )
                perturbations = perturbations + 1
                used_strategies.add(selected_strategy)
                self.strategy_uses[selected_strategy] = (
                    self.strategy_uses.get(selected_strategy, 0) + 1
                )
            except:
                print(
                    "Perturbation failed using %s on %s for %s"
                    % (selected_strategy, clean_part, clean)
                )
                failures = failures + 1
                if failures > 10:
                    perturbed_string = clean
                    break
                else:
                    pass

        if return_mapping:
            return perturbed_string, [(clean, perturbed_string)]
        else:
            return perturbed_string


class PerturbationsStorage(object):
    def __init__(self, perturbations_file_path):
        self.perturbations_file_path = perturbations_file_path
        self.observed_perturbations = defaultdict(lambda: set())
        if os.path.exists(self.perturbations_file_path):
            self.read()

    def read(self):
        with open(self.perturbations_file_path, "r") as f:
            for l in f:
                key, values = l.strip().split("\t")
                values = values.split()
                self.observed_perturbations[key] |= set(values)

    def maybe_write(self):
        if self.perturbations_file_path:
            with open(self.perturbations_file_path, "w") as f:
                for k, v in self.observed_perturbations.items():
                    f.write("{}\t{}\n".format(k, " ".join(v)))

    def add(self, key, value):
        self.observed_perturbations[key].add(value)

    def observed(self, key, value):
        return value in self.observed_perturbations[key]


def perturb_ratio(text, ordered_candidates, ratio=0.5):
    num_candidates_to_use = int(len(ordered_candidates) * ratio)

    return perturb_candidates(text, ordered_candidates, num_candidates_to_use + 1)


def perturb_candidates(text, ordered_candidates, n_candidates=1):
    for clean, perturbed in ordered_candidates[:n_candidates]:
        text = sub_token(text, clean, perturbed)

    return text
    

def sub_token(original_text, token, replacement):
    """Substitutes a token with a replacement in a text, ignoring case."""
    try:
        escaped_replacement = re.escape(replacement)
        with_escapes = re.sub(
            re.escape(token), escaped_replacement, original_text, flags=re.IGNORECASE
        )
        return with_escapes.replace(escaped_replacement, replacement)
    except re.error:
        logging.error(
            f'Failed replacement of "{token}" in "{original_text}" with {replacement}'
        )
        raise


class SingleStrategyTextPerturber(object):
    def __init__(
        self,
        word_strategy,
        generator=None,
        legibility_model=None,
        legibility_threshold=0,
    ) -> None:
        self.word_strategy = word_strategy
        self.gen = generator if generator is not None else np.random.default_rng()

    def perturb(self, text, candidates, ratio) -> str:

        len_to_perturb = int(np.round(len(candidates) * ratio))
        ordinal_candidates = list(candidates)
        words_to_perturb = self.gen.choice(
            np.array(range(len(ordinal_candidates))), size=len_to_perturb, replace=False
        )

        for word_idx in words_to_perturb:
            word = ordinal_candidates[word_idx]
            perturbed = self.word_strategy.perturb(word, n=1)[0]
            text = sub_token(text, word, perturbed)

        return text


class RandomStrategyTextPerturber(object):
    def __init__(self, word_strategies, generator=None) -> None:
        self.word_strategies = list(word_strategies)
        self.gen = generator if generator is not None else np.random.default_rng()

    def perturb(self, text, candidates, ratio) -> str:
        """
        Perturb an entire string, randomly applying one of the strategies to each chosen word.

        Args:
            text (str): The clean string to perturb.
            candidates (list): List of candidate words.
            ratio (float): The ratio of words to perturb.
        """

        len_to_perturb = int(np.round(len(candidates) * ratio))
        ordinal_candidates = list(candidates)
        words_to_perturb = self.gen.choice(
            np.array(range(len(ordinal_candidates))), size=len_to_perturb, replace=False
        )
        strategy_selection = self.gen.choice(
            np.array(range(len(self.word_strategies))),
            size=len(words_to_perturb),
            replace=True,
        )

        for i, word_idx in enumerate(words_to_perturb):
            word = ordinal_candidates[word_idx]
            strategy = self.word_strategies[strategy_selection[i]]
            perturbed = strategy.perturb(word, n=1)[0]
            text = sub_token(text, word, perturbed)

        return text
