# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import logging
import os

import dotenv
import pandas as pd

dotenv.load_dotenv()

from collections import namedtuple
from pathlib import Path

from .. import abstracts, llms
from .cli_utils import ArrayArgparser, KeyValueAction

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger.info('Running %s', __file__)

base_results_path = Path(os.environ['CORC_RESULTS_DIR']) / 'abstract' / 'abstract-recovery-sentence'
results_checkpoint_path = base_results_path / 'checkpoints'

# Parse command line arguments
p = namedtuple("params_set", ["test"])
params = [
    # Zeroshot Prompting
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

arg_parser.add_argument("--batch-size", type=int, default=10)

arg_parser.add_argument("--restart-checkpoint", type=str, default=None, help="Path to a checkpoint to restart from")

arg_parser.add_argument("--model-opt", nargs='*', type=str, action=KeyValueAction, default={'device_map': 'cuda', 'float16': True})
arg_parser.add_argument("--sample-size", default=None, type=int)

arg_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

args = arg_parser.parse_args()

logging.getLogger().setLevel(args.log_level)

sentence_model = None
model_size = args.model_size.replace('.', '_')

# Load Datasets
## Test
test_df = abstracts.class_perturbed_abstract_dataset(args.test, n_words=1).to_pandas()
test_df['word'] = test_df[f'replacements_{args.test}'].apply(eval).apply(lambda x: x[0][1])

logger.info('COLS: %s',test_df.columns)

if args.sample_size is not None:
    test_df = test_df.sample(args.sample_size)
logger.info("Test dataset has %d examples.", len(test_df))
logger.debug("Example Row: %s", test_df.iloc[0])

abs_client = llms.load_sentence_recovery_client_by_name(args.model, args.model_size, **args.model_opt)

# Initialize the runtime environment

results_path = base_results_path / args.model / model_size 
results_path.mkdir(parents=True, exist_ok=True)
results_file = (results_path / args.test).with_suffix(".csv")

checkpoint_path = results_checkpoint_path / args.model / model_size 
checkpoint_path.mkdir(parents=True, exist_ok=True)
checkpoint_file = (checkpoint_path / args.test).with_suffix(".csv")

if args.restart_checkpoint is not None:
    logger.info("Restarting from checkpoint %s", args.restart_checkpoint)
    checkpoint_df = pd.read_csv(args.restart_checkpoint)
else:
    checkpoint_df = None

logger.info("Running Classification for  Writing results to %s", results_file)
# Run the recovery
df = abstracts.run_abstract_recovery(
    test_df=test_df[['idx', 'sample', 'word']],
    abstracts_client=abs_client,
    batch_size=args.batch_size,
    checkpoint_file=checkpoint_file,
    starting_df=checkpoint_df)

# Save Outputs
logger.info("Finished Classification. Writing results to %s", results_file)
df.to_csv(results_file, index=False)

logger.info("Results written to %s", results_file)