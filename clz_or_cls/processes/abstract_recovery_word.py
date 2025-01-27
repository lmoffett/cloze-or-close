# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import logging
import os
from collections import namedtuple
from pathlib import Path

import pandas as pd

from .. import llms
from ..recovery import run_prompt_recovery
from .cli_utils import ArrayArgparser, KeyValueAction

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger.info('Running %s', __file__)

base_results_path = Path(os.environ['CORC_RESULTS_DIR'])  / 'abstract' / 'abstract-recovery-word'
results_checkpoint_path = base_results_path / 'checkpoints'


# Parse command line arguments
p = namedtuple("params_set", ["test"])
params = [
    # Baseline
    p("visual"),
    p("phonetic"),
    p("typo"),
]

arg_parser = ArrayArgparser(params, allow_array_overflow=False)
arg_parser.add_argument(
    "--model",
    choices=["llama2", "mistral", "falcon", "chatgpt", "palm2"],
)

arg_parser.add_argument(
    "--model-size",
    choices=["7B", "13B", "40B", "70B", "gpt-3.5-turbo-0613", "gpt-4-1106-preview", "gpt-3.5-turbo-0301", "text-bison-001"],
)
arg_parser.add_argument("--batch-size", type=int, default=20)

arg_parser.add_argument("--restart-checkpoint", type=str, default=None, help="Path to a checkpoint to restart from")

arg_parser.add_argument("--model-opt", nargs='*', type=str, action=KeyValueAction, default={'device_map': 'cuda', 'float16': True})
arg_parser.add_argument("--sample-size", default=None, type=int)

arg_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

args = arg_parser.parse_args()

logging.getLogger().setLevel(args.log_level)

# Load Datasets
## Test
logger.info("Loading Dataset %s", args.test)

prep_path = Path(os.environ['CORC_DATASETS_PREP_DIR'])
test_df = pd.read_csv(prep_path/'abstract'/'abstract-class-replacements.csv', index_col="idx")
test_df['clean'] = test_df[f'replacements_{args.test}'].apply(eval).apply(lambda x: x[0][0])
test_df['perturbed'] = test_df[f'replacements_{args.test}'].apply(eval).apply(lambda x: x[0][1])

if args.sample_size is not None:
    test_df = test_df.sample(args.sample_size)
logger.info("Test dataset has %d examples.", len(test_df))
logger.debug("Example Row: %s", test_df.iloc[0])

recovery_client = llms.load_recovery_client_by_name(args.model, args.model_size, **args.model_opt)

# Initialize the runtime environment

results_path = base_results_path / args.model / args.model_size.replace('.', '_') / args.test
results_path.mkdir(parents=True, exist_ok=True)
results_file = results_path.with_suffix(".csv")

checkpoint_path = results_checkpoint_path / args.model / args.model_size.replace('.', '_') / args.test
checkpoint_path.mkdir(parents=True, exist_ok=True)
checkpoint_file = checkpoint_path.with_suffix(".csv")

if args.restart_checkpoint is not None:
    logger.info("Restarting from checkpoint %s", args.restart_checkpoint)
    checkpoint_df = pd.read_csv(args.restart_checkpoint)
else:
    checkpoint_df = None

# Empty
train_df = pd.DataFrame()
# Run the recovery
df = run_prompt_recovery(train_df=train_df, 
                        test_df=test_df.reset_index()[['idx', 'clean', 'perturbed']], 
                        recovery_client=recovery_client,
                        batch_size=args.batch_size,
                        checkpoint_file=checkpoint_file,
                        starting_df=checkpoint_df, 
                        xshot_size=0)

# Save Outputs
df.to_csv(results_file)

logger.info("Recovered dataframe written to %s", results_file)