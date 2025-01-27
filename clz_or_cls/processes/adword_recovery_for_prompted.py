# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0

import logging
import os

import dotenv
import pandas as pd

dotenv.load_dotenv()

from collections import namedtuple
from pathlib import Path

from .. import datasets, llms, recovery
from ..api_llms import GptRecoveryClient, Palm2RecoveryClient
from ..recovery import run_prompt_recovery
from .cli_utils import ArrayArgparser, KeyValueAction

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger.info('Running %s', __file__)

base_results_path = Path(os.environ['CORC_RESULTS_DIR']) / 'adword-recovery'
recovery_checkpoint_path = base_results_path / 'checkpoints'

# Parse command line arguments
p = namedtuple("params_set", ["train", "test"])
params = [
    # Baseline
    p("repeated", "repeated"),
    # Zeroshot Prompting
    p(None, "visual"),
    p(None, "phonetic"),
    p(None, "typo"),
    # Single Class Prompting
    p("visual", "visual"),
    p("phonetic", "phonetic"),
    p("typo", "typo"),
    # Held Out Prompting
    p("visual+typo_full", "phonetic"),
    p("phonetic+typo_full", "visual"),
    p("visual+phonetic_full", "typo"),
    # All Class Prompting
    p("visual+phonetic+typo_full", "visual"),
    p("visual+phonetic+typo_full", "phonetic"),
    p("visual+phonetic+typo_full", "typo"),
    p("visual+phonetic+typo_full", "visual+phonetic+typo_full"),
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
arg_parser.add_argument("--xshot-size", type=int, default=50)

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

if args.model_size in ('40B', '70B') or args.model in ('chatgpt', 'palm2'):
    logger.info('Using Human Benchmark Set because model size is %s', args.model_size)
    test_df = None
    prep_path = Path(os.environ['CORC_DATASETS_PREP_DIR'])
    for i in range(9):
        group_df = pd.read_csv(prep_path / 'annotations' / 'selections' / f'group{i}_perturbations_full.csv', sep=',').sample(frac=1, random_state=i)
        group_df['group'] = i
        group_df['class'] = group_df['source'].map(datasets.class_map)
        if test_df is None:
            test_df = group_df
        else:
            test_df = pd.concat([test_df, group_df])

    if args.test == "visual+phonetic+typo_full":
        test_df = test_df[test_df['class'] != 'repeated']
    else:
        test_df = test_df[test_df['class'] == args.test]
    
else:
    test_df = datasets.generated_df(args.test, split='test').sample(frac=1, random_state=0)

if args.sample_size is not None:
    test_df = test_df.sample(args.sample_size)
logger.info("Test dataset has %d examples.", len(test_df))
logger.debug("Example Row: %s", test_df.iloc[0])

## Train
if args.train is None:
    train_df = pd.DataFrame(columns=['clean', 'perturbed'])
    train_test_name = "0shot--" + args.test
    xshot_size = 0
else:
    train_df = datasets.generated_df(args.train, split='train')
    logger.info("Train dataset has %d examples.", len(train_df))
    logger.debug("Example Row: %s", train_df.iloc[0])
    train_test_name = args.train + "--" + args.test
    xshot_size = args.xshot_size

# Load Model
if args.model == 'chatgpt':
    recovery_client = GptRecoveryClient(model = args.model_size)
elif args.model == 'palm2':
    recovery_client = Palm2RecoveryClient(model = args.model_size)
else:
    recovery_client = llms.load_recovery_client_by_name(args.model, args.model_size, **args.model_opt)

# Initialize the runtime environment

results_path = base_results_path / args.model / args.model_size.replace('.', '_') / f'{xshot_size}-shot'
results_path.mkdir(parents=True, exist_ok=True)
results_file = (results_path / train_test_name).with_suffix(".csv")

checkpoint_path = recovery_checkpoint_path / args.model / args.model_size.replace('.', '_') / f'{xshot_size}-shot' 
checkpoint_path.mkdir(parents=True, exist_ok=True)
checkpoint_file = (checkpoint_path / train_test_name).with_suffix(".csv")


if args.restart_checkpoint is not None:
    logger.info("Restarting from checkpoint %s", args.restart_checkpoint)
    checkpoint_df = pd.read_csv(args.restart_checkpoint)
else:
    checkpoint_df = None

# Run the recovery
df = run_prompt_recovery(train_df=train_df, 
                        test_df=test_df, 
                        recovery_client=recovery_client,
                        batch_size=args.batch_size,
                        checkpoint_file=checkpoint_file,
                        starting_df=checkpoint_df, 
                        xshot_size=xshot_size)

# Save Outputs
df.to_csv(results_file)

logger.info("Recovered dataframe written to %s", results_file)