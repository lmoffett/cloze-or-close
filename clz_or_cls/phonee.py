# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

# Library code for executing PhoneE attacks
import eng_to_ipa as ipa
import pandas as pd
import panphon

legit_manual_ipa_mappings = {
    "": [""],
}


def load_ipa_eng_dict(filepath="ipa_eng_dict.csv"):
    ipa_eng_dict_df = pd.read_csv(filepath)

    ipa_eng_dict_df["ipa_stripped"] = ipa_eng_dict_df["ipa"].apply(
        lambda s: s.replace("/", "").replace("ː", "")
    )
    ipa_eng_dict_df["beg_split"] = ipa_eng_dict_df["beginning"].str.split(",")
    ipa_eng_dict_df["english_split"] = ipa_eng_dict_df["english"].str.split(",")
    ipa_eng_dict_df["english_split"] = ipa_eng_dict_df.apply(
        lambda row: row["english_split"] + row["beg_split"]
        if type(row["beg_split"]) != float
        else row["english_split"],
        axis=1,
    )

    ft = panphon.FeatureTable()
    ipa_eng_dict_df["cons"] = ipa_eng_dict_df["ipa_stripped"].apply(
        lambda c: ft.word_fts(c)[0]["cons"]
    )
    return ipa_eng_dict_df


class IPAParser:
    def __init__(self, ipa_eng_dict_df):
        def invert_dict(d):
            inverted_d = {}
            for key, values in d.items():
                for value in values:
                    if value not in inverted_d:
                        inverted_d[value] = [key]
                    else:
                        inverted_d[value].append(key)
            return inverted_d

        self.ipa_to_eng = {
            row.ipa_stripped: [
                s.strip()
                for s in (row.english_split if type(row.english_split) != float else [])
            ]
            for row in ipa_eng_dict_df.itertuples()
        }
        self.ipa_letters = {ch for chs in self.ipa_to_eng.keys() for ch in chs}
        self.eng_to_ipa = invert_dict(self.ipa_to_eng)
        self.eng_letters = {
            ch
            for chs in self.eng_to_ipa.keys()
            for ch in chs
            if ch != "'" and ch != "-"
        }
        self.max_grapheme_len = max(ipa_eng_dict_df["ipa_stripped"].apply(len))
        self.ipa_graphemes_set = set(ipa_eng_dict_df["ipa_stripped"])

        self.ipa_graphemes_hold_at_len = {}
        max_len_grapheme = max([len(g) for g in self.ipa_graphemes_set])
        for i in range(1, max_len_grapheme + 1):
            self.ipa_graphemes_hold_at_len[i] = {
                g[:i] for g in self.ipa_graphemes_set if len(g) > i
            }

        self.manual_pronounce = {}
        self.manual_parse = {}
        
        # These are words that are not in the dictionary but appear in LEGIT
        self.manual_map = {
            "graham": [("g", "ɡ"), ("r", "r"), ("a", "eɪ"), ("ha", "ə"), ("m", "m")],
            "niagara": [
                ("n", "n"),
                ("i", "aɪ"),
                ("a", "æ"),
                ("ga", "ɡ"),
                ("r", "r"),
                ("a", "ə"),
            ],
            "herbs": [("he", "ə"), ("r", "r"), ("b", "b"), ("s", "z")],
            "laboratory": [
                ("l", "l"),
                ("a", "æ"),
                ("b", "b"),
                ("or", "r"),
                ("a", "ə"),
                ("t", "t"),
                ("o", "ɔ"),
                ("r", "r"),
                ("y", "i"),
            ],
            "businesses": [
                ("b", "b"),
                ("u", "ɪ"),
                ("si", "z"),
                ("n", "n"),
                ("e", "ə"),
                ("ss", "s"),
                ("e", "ə"),
                ("s", "z"),
            ],
            "decorative": [
                ("d", "d"),
                ("e", "ɛ"),
                ("c", "k"),
                ("or", "r"),
                ("a", "ə"),
                ("t", "t"),
                ("i", "ɪ"),
                ("ve", "v"),
            ],
            "exhibits": [
                ("e", "ɪ"),
                ("x", "ɡ"),
                ("h", "z"),
                ("i", "ɪ"),
                ("b", "b"),
                ("i", "ə"),
                ("t", "t"),
                ("s", "s"),
            ],
            "raleigh": [("r", "r"), ("a", "ɔ"), ("l", "l"), ("eigh", "i")],
            "wednesday": [
                ("w", "w"),
                ("e", "ɛ"),
                ("dne", "n"),
                ("s", "z"),
                ("d", "d"),
                ("ay", "eɪ"),
            ],
            "catholic": [
                ("c", "k"),
                ("a", "æ"),
                ("th", "θ"),
                ("ol", "l"),
                ("i", "ɪ"),
                ("c", "k"),
            ],
            "knowledge": [
                ("kn", "n"),
                ("ow", "ɑ"),
                ("l", "l"),
                ("e", "ə"),
                ("d", "d"),
                ("ge", "ʒ"),
            ],
            "automatically": [
                ("au", "ɔ"),
                ("t", "t"),
                ("o", "oʊ"),
                ("m", "m"),
                ("a", "æ"),
                ("t", "t"),
                ("i", "ɪ"),
                ("c", "k"),
                ("all", "l"),
                ("i", "i"),
            ],
            "exhaust": [
                ("e", "ɪ"),
                ("x", "ɡ"),
                ("h", "z"),
                ("au", "ɔ"),
                ("s", "s"),
                ("t", "t"),
            ],
            "iron": [("i", "aɪ"), ("r", "ər"), ("on", "n")],
            "specifically": [
                ("s", "s"),
                ("p", "p"),
                ("e", "ə"),
                ("s", "s"),
                ("i", "ɪ"),
                ("f", "f"),
                ("i", "ɪ"),
                ("c", "k"),
                ("all", "l"),
                ("i", "i"),
            ],
            "laboratories": [
                ("l", "l"),
                ("a", "æ"),
                ("b", "b"),
                ("or", "r"),
                ("a", "ə"),
                ("t", "t"),
                ("o", "ɔ"),
                ("r", "r"),
                ("ie", "i"),
                ("s", "z"),
            ],
            "reynolds": [
                ("r", "r"),
                ("ey", "ɛ"),
                ("n", "n"),
                ("o", "ə"),
                ("l", "l"),
                ("d", "d"),
                ("s", "z"),
            ],
            "gmbh": [("g", "ɡ"), ("", "ə"), ("mbh", "m")],
            "parliament": [
                ("p", "p"),
                ("a", "ɑ"),
                ("r", "r"),
                ("l", "l"),
                ("ia", "ə"),
                ("m", "m"),
                ("e", "ɛ"),
                ("n", "n"),
                ("t", "t"),
            ],
            "basically": [
                ("b", "b"),
                ("a", "eɪ"),
                ("s", "s"),
                ("i", "ɪ"),
                ("c", "k"),
                ("all", "l"),
                ("y", "i"),
            ],
            "parliamentary": [
                ("p", "p"),
                ("a", "ɑ"),
                ("r", "r"),
                ("l", "l"),
                ("ia", "ə"),
                ("m", "m"),
                ("e", "ɛ"),
                ("n", "n"),
                ("t", "t"),
                ("a", "ə"),
                ("r", "r"),
                ("y", "i"),
            ],
            "exhibit": [
                ("e", "ɪ"),
                ("x", "ɡ"),
                ("h", "z"),
                ("i", "ɪ"),
                ("b", "b"),
                ("i", "ɪ"),
                ("t", "t"),
            ],
            "isaac": [("i", "aɪ"), ("s", "z"), ("aa", "ə"), ("c", "k")],
            "utah": [("u", "ju"), ("t", "t"), ("ah", "ɔ")],
            "chocolate": [
                ("ch", "tʃ"),
                ("o", "ɔ"),
                ("c", "k"),
                ("ol", "l"),
                ("a", "ə"),
                ("te", "t"),
            ],
            "business": [
                ("b", "b"),
                ("u", "ɪ"),
                ("s", "z"),
                ("in", "n"),
                ("e", "ə"),
                ("ss", "s"),
            ],
        }

    def convert(self, word, retrieve_all=False):
        if word in self.manual_pronounce:
            return self.manual_pronounce[word]

        ipa_strs = ipa.convert(word, retrieve_all=True, stress_marks=None)
        return [
            ipa_str.replace("g", "ɡ").replace("ʤ", "dʒ").replace("ʧ", "tʃ")
            for ipa_str in ipa_strs
        ]

    def parse_ipa_X(self, ipa_string):
        """
        Parses the IPA string into its constituent graphemes.

        :param ipa_string: a string in the IPA.
        :param max_grapheme_len: the maximum length of a grapheme.
        :returns: a list of graphemes in the IPA string.
        """

        if ipa_string in self.manual_parse:
            return self.manual_parse[ipa_string]

        ipa_graphemes = []
        hold = []
        for ch in ipa_string:

            if ch not in self.ipa_letters:
                # print('skipping', ch)
                continue

            hold.append(ch)
            while len(hold) > self.max_grapheme_len:
                ipa_graphemes.append(hold.pop(0))
            match_found = False

            for i in range(len(hold), 0, -1):
                potential_grapheme = "".join(hold[: i + 1])

                if (
                    potential_grapheme
                    in self.ipa_graphemes_hold_at_len[len(potential_grapheme)]
                ):
                    break

                elif potential_grapheme in self.ipa_graphemes_set:
                    ipa_graphemes.append(potential_grapheme)
                    hold = hold[i + 1 :]
                    match_found = True
                    break

            if not match_found and len(hold) == self.max_grapheme_len:
                ipa_graphemes.append(hold.pop(0))

        ipa_graphemes += hold
        return ipa_graphemes

    def parse_ipa(self, ipa_string):
        """
        Parses the IPA string into its constituent graphemes.

        :param ipa_string: a string in the IPA.
        :param max_grapheme_len: the maximum length of a grapheme.
        :returns: a list of lists of graphemes in the IPA string.
        """

        if ipa_string in self.manual_parse:
            return [self.manual_parse[ipa_string]]

        return self.recursive_parse(ipa_string, [])

    def recursive_parse(self, remaining_string, current_graphemes):
        if not remaining_string:
            return [current_graphemes]

        all_parsings = []
        for i in range(1, min(self.max_grapheme_len, len(remaining_string)) + 1):
            potential_grapheme = remaining_string[:i]
            if potential_grapheme in self.ipa_graphemes_set:
                new_graphemes = current_graphemes + [potential_grapheme]
                all_parsings += self.recursive_parse(
                    remaining_string[i:], new_graphemes
                )

        return all_parsings

    def map_word_to_ipa(self, word, ipa_phonemes):
        """
        Given a word and it's IPA pronunciation, this method returns a list of tuples where
        each entry represents a mapping between the graphemes in the original word in the IPA.

        It can fail to unwind the mappings, in which case it returns None.
        """

        if word in self.manual_map:
            return self.manual_map[word]

        # Initialize the variables
        word_idx = 0
        ipa_idx = 0
        mapping = []
        word = "".join([ch for ch in word if ch in self.eng_letters])
        word_len = len(word)

        # Define the recursive function for backtracking
        def backtrack(word_idx, ipa_idx, silent_count=0):
            # If we've reached the end of the word, we've found a valid mapping

            if word_idx == word_len:
                return True

            # If we've reached the end of the IPA word but not the English word,
            # this is not a valid mapping
            if ipa_idx == len(ipa_phonemes):
                return False
            # print('s', word_idx, word_len)
            # Try all possible character groupings starting from the current position
            for end_idx in range(word_idx + 1, word_len + 1):
                substring = word[word_idx:end_idx]
                ipa_phoneme = ipa_phonemes[ipa_idx]
                # print(substring, ipa_phoneme)

                # If the English substring maps to the current IPA phoneme,
                # add the mapping and recursively continue from the next position
                if substring in self.ipa_to_eng[ipa_phoneme]:
                    mapping.append((substring, ipa_phoneme))

                    # If a valid mapping can be found from the next position,
                    # return True
                    if backtrack(end_idx, ipa_idx + 1):
                        return True

                    # If not, remove the mapping and continue with the next character grouping
                    del mapping[-1]

            if silent_count == 0:
                mapping.append(("", ipa_phoneme))

                if backtrack(word_idx, ipa_idx + 1, 1):
                    return True

                del mapping[-1]

            # If no valid mapping can be found from this position, return False
            return False

        # Start the backtracking from the beginning of the word
        if backtrack(word_idx, ipa_idx, silent_count=0):
            return mapping
        else:
            return None

        tokenized_ipa_mapping = map_word_to_ipa(word, parsed_ipa)
        tokenized_ipa_mapping
