# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import json
import logging
import os
import random
import time

import google.generativeai as palm
import numpy as np
import pandas as pd
import requests

from .llms import RecoveryClient

logger = logging.getLogger(__name__)


class ApiModelClient:
    def __init__(
        self,
        token=os.environ.get("OPENAI_API_KEY"),
        model="gpt-3.5-turbo-0613",
        url="https://api.openai.com/v1/chat/completions",
        retries=4,
        base_wait=1,
    ):
        self.model = model
        self.url = url
        self.retries = retries
        self.base_wait = base_wait
        self.retry_log = []
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(token),
        }

    def __call__(self, **kwargs):
        return self.call_api(**kwargs)

    def call_api(self, **kwargs):

        message = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.format_prompt(**kwargs)}],
            "temperature": 0.0,
        }

        retries_remaining = int(self.retries)
        while retries_remaining > 0:

            try:
                # Sending the POST request
                logger.debug("Request: %s", message)
                response = requests.post(
                    self.url, headers=self.headers, data=json.dumps(message)
                )

                output = response.json()
                logger.debug("Response: %s", output)

                response.raise_for_status()

                if "error" in output:
                    raise Exception(output["error"])

                if output["choices"][0]["finish_reason"] != "stop":
                    raise Exception("Response did not complete")

                return self.parse_response(response, **kwargs)

            except Exception as e:

                retries_remaining -= 1
                if retries_remaining > 0:
                    sleep_time = self.base_wait * np.exp(
                        self.retries - retries_remaining
                    )

                    logging.warn(
                        "Exception Occured, waiting %s seconds and then retrying. Error:\n %s",
                        sleep_time,
                        e,
                    )
                    time.sleep(sleep_time)
                    self.retry_log.append((message, e))
                else:
                    logging.error(
                        "Could not complete query successfully. Retries exhausted. Original message: %s",
                        message,
                        e,
                    )
                    raise e

    def format_prompt(self, **kwargs):
        return str(kwargs)


class GptRecoveryClient(ApiModelClient, RecoveryClient):
    def __init__(
        self,
        token=os.environ.get("OPENAI_API_KEY"),
        model="gpt-3.5-turbo-0613",
        url="https://api.openai.com/v1/chat/completions",
        retries=6,
        base_wait=5,
    ):
        super().__init__(token, model, url, retries, base_wait)

    def __call__(self, perturbed_words, **kwargs):
        return super().__call__(perturbed_words=perturbed_words)

    def format_prompt(self, perturbed_words=[], examples=[]):

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
    1. "Perturbed Word" -> "Original Word" """

        prompt = f"""
    You are trying to recover the original text from words that have been perturbed. The Perturbed Words were created by taking the Original Word in English and changing some of the letters.

    Respond with you your best guess about the original word that was used to create the Perturbed Word. Casing does not matter, but if the word has punction (like Amy\'s or half-pipe), please include it. Please do not omit any words and do not submit multiple answers for the same Perturbed Word. Respond just with the Perturbed Word followed by ` -> ` and then your guess about the Original Word.
    {examples_clause}

    The new words are after `New Perturbed Words:`, each followed by `->`. There are {len(perturbed_words)} words, numbered 1 to {len(perturbed_words)}. Please provide {len(perturbed_words)} responses, numbered 1 to {len(perturbed_words)} in the format
    
```
    1. "Perturbed Word" -> "Original Word".
    2. "Perturbed Word" -> "Original Word".
    ...
```

    New Perturbed Words:
    {new_perturbed_words}
    
    Your first of {len(perturbed_words) - 1} answers should start with
    1. "{perturbed_words[0]}" -> \""""
        return prompt

    def parse_response(self, response, perturbed_words=None, **kwargs):
        return super().parse_response(
            response.json()["choices"][0]["message"]["content"],
            perturbed_words,
            response_token=None,
        )


class HotClient(ApiModelClient):

    concept_definitions = {
        "hateful": "expresses hatred towards a targeted group or is intended to be derogatory, to humiliate, or to insult the members of the group",
        "offensive": "contains hurtful, derogatory, or obscene comments",
        "toxic": "a rude, disrespectful, or unreasonable comment that is likely to make readers want to leave a discussion",
    }

    hot_prompt = f"""I want you to provide a probabilistic score between 0 to 1 where the score represents the probability of the following comment being toxic, hateful, or offensive.

    A comment is 'hateful' if you perceive that it "{concept_definitions['hateful']}".
    A comment is 'toxic' if you perceive that it "{concept_definitions['toxic']}".
    A comment is 'offensive' if you perceive that it "{concept_definitions['offensive']}".

    A probability of 1 means that the comment is highly likely to be toxic, hateful, or offensive. A probability of 0 means that the comment is highly unlikely to be toxic, hateful, or offensive. A comment may be any or all or none of toxic, hateful, or offensive.

    Please write your response in the form:
    ```
    hateful: 0-1
    offensive: 0-1
    toxic: 0-1

    <explanation>
    ```

    Provide an explanation for your score in place of `<explanation>`.

    Some of the words in this comment may have been "perturbed". Perturbed Words were created by taking the Original Word in English and changing some of the letters. %s %s

    Do not use the fact the words are perturbed or mispelled in your explanation about toxic, hateful, and offensive. Assume that readers will not percieve a comment as hateful, toxic, or offensive because it includes a perturbed word and that perturbed words are not more likely to hateful, toxic, or offensive.
    
    Comment:
    "%s" """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def format_prompt(self, comment=None, examples=None, recovery_suggestions=None):
        if examples is not None and len(examples) > 0:
            example_format = (
                '\n\nHere are some examples, where the arrow "->" means "perturbed to":\n'
                + "\n".join([f'"{e[0]}" -> "{e[1]}"' for e in examples])
            )
        else:
            example_format = ""

        if recovery_suggestions is not None and len(recovery_suggestions) > 0:
            recovery_format = "For words that have been perturbed, here are some candidates for the original word. You can use these candidates to help you classify the comment. The suggestions are"
            for word, suggestions in recovery_suggestions.items():
                recovery_format += f'\n"{word}" could be ' + " or ".join(
                    (f'"{s}"' for s in suggestions)
                )
        else:
            recovery_format = (
                "You will not be made aware which, if any, of the words are perturbed."
            )

        formatted_prompt = HotClient.hot_prompt % (
            recovery_format,
            example_format,
            comment,
        )

        logger.debug("Formatted Prompt: %s", formatted_prompt)

        return formatted_prompt

    def parse_response(self, response):
        output = response.json()

        logger.debug("Parsing Response: %s", output)

        lines = output["choices"][0]["message"]["content"].split("\n")

        hateful = float(lines[0].split(":")[1])
        offensive = float(lines[1].split(":")[1])
        toxic = float(lines[2].split(":")[1])

        explanation = " ".join(lines[3:]).strip()

        return {
            "hateful": hateful,
            "offensive": offensive,
            "toxic": toxic,
            "explanation": explanation,
        }


class Palm2RecoveryClient(RecoveryClient):
    def __init__(
        self,
        token=os.environ.get("PALM2_API_KEY"),
        model="text-bison-001",
        retries=4,
        base_wait=1,
        max_new_tokens_headroom=1.5,
    ):
        super().__init__()
        palm.configure(api_key=token)
        self.model = "models/" + model
        self.retries = retries
        self.base_wait = base_wait
        self.max_new_tokens_headroom = max_new_tokens_headroom
        self.retry_log = []

    def format_prompt(self, perturbed_words=[], examples=[]):

        nl = "\n"
        new_perturbed_words = nl.join(
            [f'{i}. "{perturbed}" -> ' for i, perturbed in enumerate(perturbed_words)]
        )

        if len(examples) > 0:
            examples_clause = f"""Below there are examples of some Perturbed Words and Original Words labeled `Examples:`. Perturbed Words come before `->` and Original Words come afterwards. 

Examples:
{nl.join([f'{i+1}. "{clean}" -> "{perturbed}"' for i, (clean, perturbed) in enumerate(examples)])} """
        else:
            examples_clause = f"""Below there is an example of the format for Perturbed Words and Original Words labeled `Format:`. Perturbed Words come before `->` and Original Words come afterwards. 

Format:
1. "Perturbed Word" -> "Original Word" """

        prompt = f"""
You are trying to recover the original text from words that have been perturbed. The Perturbed Words were created by taking the Original Word in English and changing some of the letters.

Respond with you your best guess about the original word that was used to create the Perturbed Word. Casing does not matter, but if the word has punction (like Amy\'s or half-pipe), please include it. Please do not omit any words and do not submit multiple answers for the same Perturbed Word. Respond just with the Perturbed Word followed by ` -> ` and then your guess about the Original Word.
{examples_clause}

The new words are after `New Perturbed Words:`, each followed by `->`. There are {len(perturbed_words)} words, numbered 1 to {len(perturbed_words)}. 

New Perturbed Words:
{new_perturbed_words}.

Please provide {len(perturbed_words)} responses, numbered 1 to {len(perturbed_words)-1} in the format
    
```
    1. "Perturbed Word" -> "Original Word".
    2. "Perturbed Word" -> "Original Word".
    ...
    
Your answers should begin with the number (ie, 1.) and word (ie, "{perturbed_words[0]}") in the New Perturbed Words list followed by -> (all copied exactly from the input). Then, your guess for the original word should follow the -> in quotes (ie, "word", (not the answer for 1.)). ALWAYS INCLUDE AN ORIGINAL WORD AFTER THE ->.

You should have answers numbers {', '.join((f'{i+1}.' for i in range(len(perturbed_words))))}. Finish your response with the word DONE on a single line after the {len(perturbed_words)}th Original Word. Like this:
```
{len(perturbed_words)}. "Perturbed Word" -> "Original Word"
DONE
```
"""
        return prompt, new_perturbed_words

    def parse_response(self, response, perturbed_words=None, **kwargs):
        return super().parse_response(response, perturbed_words, response_token=None)

    def __call__(self, perturbed_words, examples=[], **kwargs):
        return self.call_api(
            perturbed_words=perturbed_words, examples=examples, **kwargs
        )

    def call_api(self, perturbed_words=[], examples=[], **kwargs):

        retries_remaining = self.retries
        candidate_idx = 0
        while retries_remaining > 0:

            try:

                if candidate_idx > 0:
                    random.shuffle(perturbed_words)
                    random.shuffle(examples)

                prompt, example_prompt = self.format_prompt(
                    perturbed_words=perturbed_words, examples=examples, **kwargs
                )

                example_token_len = palm.count_message_tokens(prompt=example_prompt)[
                    "token_count"
                ]

                # Sending the POST request
                logger.debug("Request: %s", prompt)
                completion = palm.generate_text(
                    model=self.model,
                    prompt=prompt,
                    temperature=float(candidate_idx) / 10,
                    candidate_count=candidate_idx + 1,
                    # The maximum length of the response
                    max_output_tokens=int(
                        example_token_len * 2 * self.max_new_tokens_headroom
                    ),
                )

                if len(completion.candidates) == 0:
                    candidate_idx += 1
                    raise Exception("No Candidates in Completion.")

                logger.debug("Response: %s", completion)

                parse = None
                for candidate in completion.candidates:
                    candidate_output = candidate["output"]

                    try:
                        parse = self.parse_response(
                            candidate_output, perturbed_words=perturbed_words, **kwargs
                        )
                        break
                    except:
                        pass

                if parse is None:
                    candidate_idx += 1
                    logging.error(
                        "Unable to parse any candidates. %s", candidate_output
                    )
                    raise Exception("Unable to parse any candidates.")
                else:
                    return parse

            except Exception as e:

                retries_remaining -= 1
                if retries_remaining > 0:
                    sleep_time = self.base_wait * np.exp(
                        self.retries - retries_remaining
                    )

                    logging.warn(
                        "Exception Occured, waiting %s seconds and then retrying. Error:\n %s",
                        sleep_time,
                        e,
                    )
                    time.sleep(sleep_time)
                    self.retry_log.append((completion, e))
                else:
                    logging.error(
                        "Could not complete query successfully. Retries exhausted. Prompt: %s. Original message: %s. Error: %s",
                        prompt,
                        completion,
                        e,
                    )
                    raise e