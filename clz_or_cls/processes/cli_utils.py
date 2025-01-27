# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import argparse
import itertools
import logging
import os
import random
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

class ArrayArgparser(argparse.ArgumentParser):
    """
    An argument parser that actually takes arguments from lists of argument groups.
    This is used to map between task ids and sets of parameters.
    """

    SEED_FOR_SEED = 2730485088123023341

    def __init__(self, arg_sets, allow_array_overflow=True):
        super().__init__()
        self.add_argument(
            "--task-ids",
            "--task-id",
            type=int,
            nargs="*",
            help="the task id for this process's run",
        )
        self.add_argument("--dry-run", default=False, action="store_true")

        self.input_arg_sets = arg_sets
        self.effective_arg_set = None
        self.seed_index = None
        self.allow_array_overflow = allow_array_overflow

    def __get_ith_from_seed(self):
        if self.effective_arg_set is None:
            raise EnvironmentError("call parse first")

        old_state = random.getstate()

        try:
            self.seed_index = seed_index = int(self.effective_arg_set.task_id) // len(
                self.input_arg_sets
            )

            if not self.allow_array_overflow and seed_index > 0:
                logger.error(
                    "This parser does not allow arrays wrapped around with extra seeds"
                )
                exit(-1)

            random.seed(ArrayArgparser.SEED_FOR_SEED)
            rand_int_gen = (
                random.randint(0, sys.maxsize) for x in range(seed_index + 1)
            )

            ith_rand = next(itertools.islice(rand_int_gen, seed_index, None))
            logger.debug(
                "derived random seed %s from %s sets of parameters and index %s",
                ith_rand,
                len(self.input_arg_sets),
                seed_index,
            )
            self.seed = ith_rand

            return ith_rand

        finally:
            random.setstate(old_state)

    def parse_args(self):
        """
        run_params must a Mapping
        """

        args = super().parse_args()

        if args.dry_run:
            task_ids = (
                args.task_ids
                if args.task_ids is not None
                else list(range(len(self.input_arg_sets)))
            )

            print(
                f"There are {len(self.input_arg_sets)} parameter sets indexed from 0 to {len(self.input_arg_sets)-1}"
            )

            for task_id in task_ids:

                these_run_params = self.input_arg_sets[task_id]
                print(f"Parameters for task {task_id}:")
                for name, value in zip(these_run_params._fields, these_run_params):
                    print(f"  {name}: {value}")
            exit(0)

        if len(args.task_ids) != 1:
            print(
                "Cannot run with manual task id selection for multiple task ids. Please provide only 1."
            )
            exit(-1)

        input_arg_sets_idx = args.task_ids[0] % len(self.input_arg_sets)
        these_params = self.input_arg_sets[input_arg_sets_idx]

        self.effective_arg_set = argparse.Namespace(**these_params._asdict())
        self.effective_arg_set.task_id = args.task_ids[0]

        for arg_name, arg_value in vars(args).items():
            if arg_name not in self.effective_arg_set:
                setattr(self.effective_arg_set, arg_name, arg_value)

        self.effective_arg_set.seed = self.__get_ith_from_seed()
        self.effective_arg_set.seed_index = self.seed_index

        logger.info(
            "running with args from index %s, %s",
            input_arg_sets_idx,
            self.effective_arg_set,
        )

        return self.effective_arg_set

class KeyValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kv_dict = {}
        for item in values:
            key, value = item.split('=')
            kv_dict[key] = value
        setattr(namespace, self.dest, kv_dict)