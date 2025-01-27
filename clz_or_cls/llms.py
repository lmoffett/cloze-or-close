# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import logging
import os
import re
import time
from pathlib import Path
from typing import Mapping

import pandas as pd
import torch
from nltk.metrics.distance import edit_distance
from nltk.tokenize import sent_tokenize
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          FalconForCausalLM, LlamaForCausalLM, LlamaTokenizer,
                          T5ForConditionalGeneration, pipeline)

from datasets import DatasetDict

from . import perturbation

logger = logging.getLogger(__name__)


def load_recovery_client_by_name(model_name, model_size, **kwargs):
    if model_name == "llama2":
        return LlamaHFRecoveryClient(llama2(size=model_size, **kwargs))
    elif model_name == "mistral":
        return MistralHFRecoveryClient(mistral(size=model_size, **kwargs))
    elif model_name == "falcon":
        return FalconHFRecoveryClient(falcon(size=model_size, **kwargs))
    else:
        raise ValueError(f"Unknown model name {model_name}")


def load_hot_client_by_name(
    model_name, model_size, clazz, shielding_model=None, errors_clause=False, **kwargs
):
    if model_name == "llama2":
        return Llama2HfHotClient(
            llama2(size=model_size, **kwargs),
            clazz,
            errors_clause=errors_clause,
            shielding_model=shielding_model,
        )
    elif model_name == "mistral":
        return MistralHfHotClient(
            mistral(size=model_size, **kwargs),
            clazz,
            errors_clause=errors_clause,
            shielding_model=shielding_model,
        )
    elif model_name == "falcon":
        return FalconHfHotClient(
            falcon(size=model_size, **kwargs),
            clazz,
            errors_clause=errors_clause,
            shielding_model=shielding_model,
        )
    else:
        raise ValueError(f"Unknown model name {model_name}")


def load_sentence_recovery_client_by_name(model_name, model_size, **kwargs):
    if model_name == "llama2":
        return Llama2HfSentenceRecoveryClient(
            llama2(size=model_size, **kwargs),
        )
    elif model_name == "mistral":
        return MistralHfSentenceRecoveryClient(
            mistral(size=model_size, **kwargs),
        )
    else:
        raise ValueError(f"Unknown model name {model_name}")


def llama2(size="7b", device_map="cuda", float16=True, load_in_4bit=False, **kwargs):

    logger.debug(
        "loading llama model to %s %s quantitization",
        device_map,
        "with" if load_in_4bit else "without",
    )
    llama2_path = Path(os.environ['LLAMA2_PATH'])
    size = size.lower()

    llama2_model_path = llama2_path / f"llama-2-{size}-chat-hf"

    if not llama2_model_path.exists():
        raise ValueError(f"Could not find model of size {size} at {llama2_model_path}")

    logger.debug("loading tokenizer from %s", llama2_path)
    tokenizer = LlamaTokenizer.from_pretrained(llama2_path, legacy=False)

    start = time.time()
    logger.debug("loading model from %s", llama2_model_path)

    if float16:
        embed_size = torch.float16
    elif load_in_4bit:
        # quantization_method=QuantizationMethod.GPTQ
        raise ValueError("Cannot load in 4bit without float16")
    else:
        embed_size = torch.float32

    if size == "70b":
        device_map = "auto"

    model = LlamaForCausalLM.from_pretrained(
        llama2_model_path, device_map=device_map, torch_dtype=embed_size
    )

    logger.debug("model loaded in %s seconds", time.time() - start)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=embed_size,
        device_map=device_map,
    )


def identity_client(perturbed_words, **kwargs):
    logger.debug(
        "Running llama2_client with perturbed_words=%s and examples=%s",
        perturbed_words,
        kwargs.get("examples"),
    )
    return {k: v for k, v in zip(perturbed_words, perturbed_words)}


def mistral(size="7b", device_map="cuda", float16=True, **kwargs):

    if size.lower() != "7b":
        raise ValueError("Only Mistral 7B allowed.")

    if float16:
        embed_size = torch.float16
    else:
        embed_size = torch.float32

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        device_map=device_map,
        torch_dtype=embed_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1", device_map=device_map,
        trust_remote_code=True, use_fast=False
    )

    logger.debug("model loaded in %s seconds", time.time() - start)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=embed_size,
        device_map=device_map,
    )


def falcon(size="7b", device_map="cuda", float16=True, **kwargs):
    size = size.lower()
    assert size.lower() in ("7b", "40b"), "Only 7b or 40b model sizes allowed"

    if float16:
        embed_size = torch.float16
    else:
        embed_size = torch.float32

    num_gpus = torch.cuda.device_count()

    assert num_gpus > 0, "must have GPU for falcon"

    model_name = f"tiiuae/falcon-{size.lower()}-instruct"

    start = time.time()
    if size.lower() == "7b":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=embed_size,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=embed_size,
        )

        # load_checkpoint_and_dispatch(model, )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, device_map=device_map, trust_remote_code=True,
        use_fast=False
    )

    logger.debug("model loaded in %s seconds", time.time() - start)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=embed_size,
        device_map=device_map,
    )


class RecoveryClient:
    def zip_lists_with_edit_distance(self, predictions, raw_perturbed, max_distance=3):
        zipped_result = []
        used_predictions = set()  # Track matched elements in list b

        for raw in raw_perturbed:
            match_found = False

            for prediction in predictions:
                if prediction[0] in used_predictions:
                    continue

                if edit_distance(raw, prediction[0]) <= max_distance:
                    zipped_result.append((raw, prediction))
                    used_predictions.add(prediction[0])
                    match_found = True
                    break

            if not match_found:
                zipped_result.append((raw, None))

        return zipped_result

    def parse_response(self, response, perturbed_words, response_token="[/INST]"):
        if response_token is not None:
            response_part = response.split(response_token)[-1]
        else:
            response_part = response

        numbering_regex = r'\s*((\d*)\s*(\.|\)))?\s*"?(.*)"?\s*->\s*"?(.*)"?'

        perturbed_predicted_pairs = []
        line_mapping = {}
        for line in response_part.split("\n"):

            if len(re.findall(numbering_regex, line)) == 1:

                _, num, _, perturbed, predicted = re.findall(numbering_regex, line)[0]

                try:
                    num = int(num)
                except:
                    num = None

                perturbed = perturbed.strip().strip('"').strip("'").strip()
                predicted = predicted.strip().strip('"').strip("'").strip()

                predicted = None if len(predicted) == 0 else predicted

                if len(perturbed) != 0:
                    perturbed_predicted_pairs.append((perturbed, predicted))

                if num is not None:
                    line_mapping[num - 1] = predicted

        prediction_mapping = self.zip_lists_with_edit_distance(
            perturbed_predicted_pairs, perturbed_words
        )

        prediction_dict = {
            perturbed: (predicted[1] if predicted is not None else None)
            for perturbed, predicted in prediction_mapping
        }

        if None in prediction_dict.values():

            for i, perturbed_word in enumerate(perturbed_words):
                if prediction_dict[perturbed_word] is None and i in line_mapping.keys():
                    prediction_dict[perturbed_word] = line_mapping[i]

        # if it's still the case we couldn't map everything
        if None in prediction_dict.values():

            none_count = sum((1 for val in prediction_dict.values() if val is None))
            found_count = len(perturbed_words) - none_count

            if float(none_count) / float(len(perturbed_words)) > 0.5:
                raise Exception(f"Catestrophic Parsing Failure.\n {response}", response)

            logger.warning(
                "Failed to parse all entries from response. Expected %d words, found %s. Mapping is %s Response: %s",
                len(perturbed_words),
                found_count,
                prediction_dict,
                "\n > ".join(response_part.split("\n")),
            )

        return prediction_dict


class HFRecoveryClient(RecoveryClient):

    SYS_PROMPT = (
        "You are trying to recover the original text from words that have been perturbed. "
        + "The Perturbed Words were created by taking the Original Word in English and changing some of the letters."
    )

    def __init__(self, pipeline, max_new_tokens_headroom=1.5, repetition_penalty=1.5):
        self.pipeline = pipeline
        self.max_new_tokens_headroom = max_new_tokens_headroom
        self.repetition_penalty = repetition_penalty

    def __call__(
        self, perturbed_words, examples=[], parse_response=True
    ) -> Mapping[str, str]:
        prompt, new_perturbed_words = self.format_prompt(perturbed_words, examples)

        example_token_len = len(
            self.pipeline.tokenizer(new_perturbed_words)["input_ids"]
        )

        logger.debug("Prompting with:\n %s", prompt)

        self.pipeline.call_count = 0

        response = self.pipeline(
            prompt,
            do_sample=False,
            num_return_sequences=1,
            num_beams=3,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            max_new_tokens=int(example_token_len * 2 * self.max_new_tokens_headroom),
        )

        text = response[0]["generated_text"]

        del response

        logger.debug("Response from pipeline: %s", text)

        if parse_response:
            return self.parse_response(text, perturbed_words)
        else:
            return text


class LlamaHFRecoveryClient(HFRecoveryClient):
    def format_prompt(self, perturbed_words, examples=[]):

        nl = "\n"
        new_perturbed_words = nl.join(
            [f'{i+1}. "{perturbed}" -> ' for i, perturbed in enumerate(perturbed_words)]
        )

        if len(examples) > 0:
            examples_clause = f"""
Below there are examples of some Perturbed Words and Original Words labeled `Examples:`. Perturbed Words come before `->` and Original Words come afterwards. 

Examples:
{nl.join([f'{i+1}. "{clean}" -> "{perturbed}"' for i, (clean, perturbed) in enumerate(examples)])}
"""
        else:
            examples_clause = f"""
Below there is an example of the format for Perturbed Words and Original Words labeled `Format:`. Perturbed Words come before `->` and Original Words come afterwards. 

Format:
1. "Perturbed" -> "Original"
"""

        prompt = f"""
Respond with you your best guess about the original word that was used to create the Perturbed Word. Casing does not matter, but if the word has punction (like Amy\'s or half-pipe), please include it. Please do not omit any words and do not submit multiple answers for the same Perturbed Word. Respond just with the Perturbed Word followed by ` -> ` and then your guess about the Original Word.
{examples_clause}

[/INST]

Sure, I'm ready to help! Please provide the new perturbed words and I will give you my best guesses for the original words in the format `n. perturbed_word -> original_word` for each word.

[INST]
The new words are after `New Perturbed Words:`, each followed by `->`.

New Perturbed Words:
{new_perturbed_words}

[/INST]
Here are my answers for the new perturbed words:
"""

        return (
            f"""[INST] <<SYS>>
        {self.SYS_PROMPT}
        <</SYS>>
        {prompt} """,
            new_perturbed_words,
        )


class MistralHFRecoveryClient(HFRecoveryClient):
    def format_prompt(self, perturbed_words, examples=[]):

        nl = "\n"
        new_perturbed_words = nl.join(
            [f'{i+1}. "{perturbed}" -> ' for i, perturbed in enumerate(perturbed_words)]
        )

        if len(examples) > 0:
            examples_clause = f"""
Below there are examples of some Perturbed Words and Original Words labeled `Examples:`. Perturbed Words come before `->` and Original Words come afterwards. 

Examples:
{nl.join([f'{i+1}. "{clean}" -> "{perturbed}"' for i, (clean, perturbed) in enumerate(examples)])}
"""
        else:
            examples_clause = f"""
Below there is an example of the format for Perturbed Words and Original Words labeled `Format:`. Perturbed Words come before `->` and Original Words come afterwards. 

Format:
1. "Perturbed" -> "Original"
"""

        prompt = f"""
Respond with you your best guess about the original word that was used to create the Perturbed Word. Casing does not matter, but if the word has punction (like Amy\'s or half-pipe), please include it. Please do not omit any words and do not submit multiple answers for the same Perturbed Word. Respond just with the Perturbed Word followed by ` -> ` and then your guess about the Original Word.
{examples_clause}

[/INST]

Sure, I'm ready to help! Please provide the new perturbed words and I will give you my best guesses for the original words in the format `n. perturbed_word -> original_word` for each word.

[INST]
The new words are after `New Perturbed Words:`, each followed by `->`. There are {len(perturbed_words)} words, numbered 1 to {len(perturbed_words)}. Please provide {len(perturbed_words)} responses, numbered 1 to {len(perturbed_words)}.

New Perturbed Words:
{new_perturbed_words}

[/INST]
Here are my answers for the new perturbed words:

1. "{perturbed_words[0]}" -> \""""
        return (
            f"""[INST]
        {self.SYS_PROMPT}
        {prompt}""",
            new_perturbed_words,
        )


class FalconHFRecoveryClient(HFRecoveryClient):
    def format_prompt(self, perturbed_words, examples=[]):

        nl = "\n"
        new_perturbed_words = nl.join(
            [f'{i+1}. "{perturbed}" -> ' for i, perturbed in enumerate(perturbed_words)]
        )

        if len(examples) > 0:
            examples_clause = f"""Below there are examples of some Perturbed Words and Original Words labeled `Examples:`. Perturbed Words come before `->` and Original Words come afterwards. 

Examples:
{nl.join([f'{i+1}. "{clean}" -> "{perturbed}"' for i, (clean, perturbed) in enumerate(examples)])} """
        else:
            examples_clause = f"""Below there is an example of the format for Perturbed Words and Original Words labeled `Format:`. Perturbed Words come before `->` and Original Words come afterwards. 

Format:
1. "Perturbed" -> "Original" """

        prompt = f""">>CONTEXT<<
You are trying to recover the original text from words that have been perturbed. The Perturbed Words were created by taking the Original Word in English and changing some of the letters.

Respond with you your best guess about the original word that was used to create the Perturbed Word. Casing does not matter, but if the word has punction (like Amy\'s or half-pipe), please include it. Please do not omit any words and do not submit multiple answers for the same Perturbed Word. Respond just with the Perturbed Word followed by ` -> ` and then your guess about the Original Word.
{examples_clause}

The new words are after `New Perturbed Words:`, each followed by `->`. There are {len(perturbed_words)} words, numbered 1 to {len(perturbed_words)}. Please provide {len(perturbed_words)} responses, numbered 1 to {len(perturbed_words)}.

>>QUESTION<<
New Perturbed Words:
{new_perturbed_words}

>>ANSWER<<
1. "{perturbed_words[0]}" -> \""""
        return prompt, new_perturbed_words

    def parse_response(self, response, perturbed_words):
        return super().parse_response(
            response, perturbed_words, response_token=">>ANSWER<<"
        )


def simple_sub_token(original_text, token, replacement):
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


def multiple_substitutions(original_text, substitutions):
    """
    Perform multiple substitutions in a single pass.
    :param original_text: The text to perform substitutions on.
    :param substitutions: A list of (token, replacement) pairs.
    :return: The text with all substitutions made.
    """
    # Escape all tokens and create a regex pattern that matches any of them
    regex_pattern = "|".join(re.escape(token) for token, _ in substitutions)

    # Replacement function
    def replace(match):
        token = match.group(0)
        for original, replacement in substitutions:
            if token.lower() == original.lower():
                return replacement
        return token  # Should not happen if all tokens are in the substitutions list

    # Perform the substitutions
    return re.sub(regex_pattern, replace, original_text, flags=re.IGNORECASE)


class HfHotClient:

    CONCEPT_DEFINITIONS = {
        "hateful": "expresses hatred towards a targeted group or is intended to be derogatory, to humiliate, or to insult the members of the group",
        "offensive": "contains hurtful, derogatory, or obscene comments",
        "toxic": "a rude, disrespectful, or unreasonable comment that is likely to make readers want to leave a discussion",
    }

    DEFINITION_CLAUSES = {
        "hateful": f"""A comment is 'hateful' if you perceive that it "{CONCEPT_DEFINITIONS['hateful']}".""",
        "offensive": f"""A comment is 'offensive' if you perceive that it "{CONCEPT_DEFINITIONS['offensive']}".""",
        "toxic": f"""A comment is 'toxic' if you perceive that it "{CONCEPT_DEFINITIONS['toxic']}".""",
    }

    ERRORS_CLAUSE = "Omit typographical errors from your analysis."

    def __init__(
        self,
        pipeline,
        clazz,
        include_errors_clause=False,
        errors_clause=ERRORS_CLAUSE,
        max_new_tokens=200,
        response_start_token=None,
        shielding_model=None,
    ):

        assert clazz in ("hateful", "offensive", "toxic")
        self.clazz = clazz
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.response_start_token = response_start_token
        self.score_regex = re.compile(r"SCORE:\s*(\d+\.\d+)")
        self.include_errors_clause = include_errors_clause
        self.errors_clause = errors_clause
        self.shielding_model = shielding_model

    def definition_clause(self):
        return HfHotClient.DEFINITION_CLAUSES[self.clazz]

    def maybe_errors_clause(self):
        if self.include_errors_clause:
            return self.errors_clause
        else:
            return ""

    def __call__(self, batch, parse_response=True):

        if self.shielding_model is not None:
            start = time.time()
            batch, all_candidates = self.shield(batch)
            logger.debug(
                "Shielding took %ss. Shielded text is %s", time.time() - start, batch
            )

        start = time.time()
        exp_prompts = [self.format_prompt(text) for text in batch]

        logger.debug("Prompting with:\n %s", "\n\n".join(exp_prompts))

        self.pipeline.call_count = 0

        positive_response = self.pipeline(
            exp_prompts,
            do_sample=False,
            num_return_sequences=1,
            num_beams=3,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
        )
        positive_texts = [r[0]["generated_text"] for r in positive_response]
        logger.debug("Pipeline responses :\n %s", "\n\n".join(positive_texts))

        # negative_prompts = [self.add_negative_clause(text) for text in positive_texts]

        # negative_responses = self.pipeline(
        #     negative_prompts,
        #     do_sample=False,
        #     num_return_sequences=1,
        #     num_beams=3,
        #     length_penalty=self.length_penalty,
        #     eos_token_id=self.pipeline.tokenizer.eos_token_id,
        #     pad_token_id=self.pipeline.tokenizer.eos_token_id,
        #     max_new_tokens=self.max_new_tokens,
        # )

        negative_texts = (
            positive_texts  # [r[0]['generated_text'] for r in negative_responses]
        )
        score_prompts = [self.add_score_clause(text) for text in negative_texts]
        logger.debug("Prompting for Scores:\n %s", "\n\n".join(score_prompts))

        score_responses = self.pipeline(
            score_prompts,
            do_sample=False,
            num_return_sequences=1,
            num_beams=3,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            max_new_tokens=5,
        )

        scored_texts = [r[0]["generated_text"] for r in score_responses]

        logger.debug("Hot classification took %s", time.time() - start)

        if parse_response:
            return [self.parse_response(text) for text in scored_texts]
        else:
            return scored_texts

    def shield(self, batch):
        logger.debug("Recovering %s", batch)
        all_candidates = self.shielding_model.get_candidates_batch(batch)

        shielded_text_s = []

        for text, candidates in zip(batch, all_candidates):
            shielded_text = text
            # for token, word in sorted(list(candidates.items()), key=lambda it: -len(it[0])):
            #     try:
            #         shielded_text = simple_sub_token(shielded_text, token, word)

            #     except Exception as e:
            #         logger.warn("Failed to apply shield %s for %s in  %s", token, word, text)

            shielded_text = multiple_substitutions(text, candidates.items())
            shielded_text_s.append(shielded_text)

        return shielded_text_s, all_candidates

    def format_prompt(self, text):
        raise NotImplemented()

    def add_score_clause(self, text):
        raise NotImplemented()

    def parse_score_lines(self, text):
        # Find all matches in the text
        matches = self.score_regex.findall(text)

        if len(matches) != 1:
            raise Exception(f"Too many score matches {matches}. {text}")

        return matches[0]

    def parse_response(self, response_text):
        response_text, score_text = response_text.split(self.response_start_token)[-2:]
        explanation = response_text.split("[INST]")[0]

        logger.debug("Parsing Response: %s", response_text)

        score = self.parse_score_lines(score_text)

        return {"score": score, "explanation": explanation}


class FalconHfHotClient(HfHotClient):
    def __init__(self, pipeline, clazz, **kwargs):
        super().__init__(pipeline, clazz, response_start_token=">>ANSWER<<", **kwargs)

    def format_prompt(self, text):
        return f""">>CONTEXT<<
You are trying to assess whether or not a comment is {self.clazz}.

{self.definition_clause()}
>>QUESTION<<

Please analyze this comment and concisely describe which, if any, of the criteria from the definition of {self.clazz} the comment meets and why. {self.maybe_errors_clause()}

Comment:
"{text}"
>>ANSWER<<
The main factors in determining whether or not this specific comment meets the criteria to be considered {self.clazz} are:
"""

    def add_score_clause(self, text):
        return (
            text
            + f"""
>>QUESTION<<
Finally, considering the factors that suggest the comment is {self.clazz} and the factors that suggest it is not {self.clazz}, provide a score on a scale of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 and 1.0 that the comment is {self.clazz}.
A score of 1.0 means the comment is certainly {self.clazz} and a score of 0.0 means it certainly is *not* {self.clazz}.
>>ANSWER<<
SCORE: """
        )


class MistralHfHotClient(HfHotClient):
    def __init__(self, pipeline, clazz, **kwargs):
        super().__init__(pipeline, clazz, response_start_token="[/INST]", **kwargs)

    def format_prompt(self, text):
        return f"""[INST]
You are trying to assess whether or not a comment is {self.clazz}.

{self.definition_clause()}

Please analyze this comment and describe which, if any, of the criteria from the definition of {self.clazz} the comment meets and why. {self.maybe_errors_clause()}

Comment:
"{text}"
[/INST]
The main factors in determining whether or not this specific comment meets the criteria to be considered {self.clazz} are:
"""

    #     def add_negative_clause(self, text):
    #         return text + f"""[INST]
    # Second, please analyze this comment and describe the main 1 or 2 factors that suggest the comment is *NOT* {self.clazz}.
    # [/INST]
    # Using the definition of {self.clazz}, the 2 factors that suggest the comment should be considered *NOT* {self.clazz} are
    # """

    def add_score_clause(self, text):
        return (
            text
            + f"""
[INST]
Finally, considering the factors that suggest the comment is {self.clazz} and the factors that suggest it is not {self.clazz}, provide a score on a scale of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 and 1.0 that the comment is {self.clazz}.
A score of 1.0 means the comment is certainly {self.clazz} and a score of 0.0 means it certainly is *not* {self.clazz}.
[/INST]
SCORE: """
        )


class Llama2HfHotClient(HfHotClient):
    def __init__(self, pipeline, clazz, max_new_tokens=200, **kwargs):
        super().__init__(
            pipeline,
            clazz,
            response_start_token="[/INST]",
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def format_prompt(self, text):
        return f"""[INST]<<SYS>>
You are trying to assess whether or not a comment is {self.clazz}.

{self.definition_clause()}
<</SYS>>

Please analyze this comment and concisely describe which, if any, of the criteria from the definition of {self.clazz} the comment meets and why. {self.maybe_errors_clause()}

Comment:
"{text}"
[/INST]
The main factors in determining whether or not this specific comment meets the criteria to be considered {self.clazz} are:
"""

    #     def add_negative_clause(self, text):
    #         return text + f"""[INST]
    # Second, please analyze this comment and describe the main 1 or 2 factors that suggest the comment is *NOT* {self.clazz}.
    # [/INST]
    # Using the definition of {self.clazz}, the 2 factors that suggest the comment should be considered *NOT* {self.clazz} are
    # """

    def add_score_clause(self, text):
        return (
            text
            + f"""
[INST]
Finally, considering the factors that suggest the comment is {self.clazz} and the factors that suggest it is not {self.clazz}, provide a score on a scale of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 and 1.0 that the comment is {self.clazz}.
A score of 1.0 means the comment is certainly {self.clazz} and a score of 0.0 means it certainly is *not* {self.clazz}.
[/INST]
SCORE: """
        )


class HfSentenceRecoveryClient:
    def __init__(self, pipeline, max_new_tokens=15, response_start_token="[/INST]"):
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.recovery_regex = re.compile(r'"([^"]+)"')
        self.response_start_token = response_start_token

    def __call__(self, batch, parse_response=True):

        start = time.time()
        exp_prompts = [
            self.format_prompt(text, word) for _, (text, word) in batch.iterrows()
        ]

        logger.debug("Prompting with:\n %s", "\n\n".join(exp_prompts))

        self.pipeline.call_count = 0

        recovery_response = self.pipeline(
            exp_prompts,
            do_sample=False,
            num_return_sequences=1,
            num_beams=3,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
        )
        recovery_texts = [r[0]["generated_text"] for r in recovery_response]
        logger.debug("Pipeline responses :\n %s", "\n\n".join(recovery_texts))

        if parse_response:
            return [self.parse_response(text) for text in recovery_texts]
        else:
            return recovery_texts

    def format_prompt(self, text):
        raise NotImplemented()

    def parse_response(self, response_text):
        response_text = response_text.split(self.response_start_token)[-1]
        logger.debug("Parsing Response: %s", response_text)

        word_text = response_text.split("should be written using the correct spelling")[
            1
        ].strip()
        try:
            word = self.recovery_regex.findall(word_text)[0].strip()
        except:
            word = word_text.split('"')[1].strip()

        return word

    def get_sentence_by_unique_word(self, text, word):
        sentences = sent_tokenize(text)

        for sentence in sentences:
            if word in sentence:
                return sentence

        raise ValueError(f"Word {word} not found in text {text}")


class MistralHfSentenceRecoveryClient(HfSentenceRecoveryClient):
    def __init__(self, pipeline, **kwargs):
        super().__init__(pipeline, response_start_token="[/INST]", **kwargs)

    def format_prompt(self, text, word):

        return f"""[INST]
You are trying to correct a potential minor typographical error in a single word of a medical paper's abstract.

Please analyze the following abstract and determine how to correct this minor error.

The potential minor typographical error is in the word "{word}" in the following abstract:
"{text}"

[/INST]
The word "{word}" should be written using the correct spelling \""""


class Llama2HfSentenceRecoveryClient(HfSentenceRecoveryClient):
    def __init__(self, pipeline, **kwargs):
        super().__init__(
            pipeline,
            response_start_token="[/INST]",
            **kwargs,
        )

    def format_prompt(self, text, word):
        return f"""[INST]<<SYS>>
You are trying to correct a potential minor typographical error in a single word of a medical paper's abstract.
<</SYS>>

Please analyze the following abstract and determine how to correct this minor error.

The potential minor typographical error is in the word "{word}" in the following abstract:
"{text}"

[/INST]
The word "{word}" should be written using the correct spelling \""""
