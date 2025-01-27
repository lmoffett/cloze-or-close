# Copyright 2025 Luke Moffett
# Licensed under the Apache License, Version 2.0


import logging

import dotenv
import pandas as pd
import os

dotenv.load_dotenv()

from collections import namedtuple
from pathlib import Path

from transformers import AutoTokenizer

from .. import datasets, hot, llms, recovery
from .cli_utils import KeyValueAction, ArrayArgparser

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger.info('Running %s', __file__)

base_results_path = Path(os.environ['CORC_RESULTS_DIR']) / 'hot'
results_checkpoint_path = base_results_path / 'checkpoints'


# Parse command line arguments
p = namedtuple("params_set", ["train", "test", "clazz", "ratio"])
params = [
    # Baseline
    p(None, None, "hateful", None),
    p(None, None, "offensive", None),
    p(None, None, "toxic", None),
    # Zeroshot Prompting
    p(None, "visual", "hateful", .125),
    p(None, "visual", "offensive", .125),
    p(None, "visual", "toxic", .125),

    p(None, "phonetic", "hateful", .125),
    p(None, "phonetic", "offensive", .125),
    p(None, "phonetic", "toxic", .125),

    p(None, "typo", "hateful", .125),
    p(None, "typo", "offensive", .125),
    p(None, "typo", "toxic", .125),

    p(None, "visual", "hateful", .25),
    p(None, "visual", "offensive", .25),
    p(None, "visual", "toxic", .25),

    p(None, "phonetic", "hateful", .25),
    p(None, "phonetic", "offensive", .25),
    p(None, "phonetic", "toxic", .25),

    p(None, "typo", "hateful", .25),
    p(None, "typo", "offensive", .25),
    p(None, "typo", "toxic", .25),

    p(None, "visual", "hateful", .5),
    p(None, "visual", "offensive", .5),
    p(None, "visual", "toxic", .5),

    p(None, "phonetic", "hateful", .5),
    p(None, "phonetic", "offensive", .5),
    p(None, "phonetic", "toxic",  .5),

    p(None, "typo", "hateful", .5),
    p(None, "typo", "offensive",  .5),
    p(None, "typo", "toxic", .5)
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

arg_parser.add_argument("--shield-visual", default=False, action='store_true')

arg_parser.add_argument("--batch-size", type=int, default=10)

arg_parser.add_argument("--restart-checkpoint", type=str, default=None, help="Path to a checkpoint to restart from")

arg_parser.add_argument("--model-opt", nargs='*', type=str, action=KeyValueAction, default={'device_map': 'cuda', 'float16': True})
arg_parser.add_argument("--sample-size", default=None, type=int)

arg_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

args = arg_parser.parse_args()

logging.getLogger().setLevel(args.log_level)

# Setup Recovery

if args.shield_visual:
    # Load the recovery model that equates to the fewshot dataset
    shielding_model_dir = Path(os.environ['CORC_TRAINED_MODEL_DIR']) / 'byt5-base'
    logger.info("Loading shielding model from %s", shielding_model_dir)

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base", device_map=os.environ['TORCH_DEVICE'])
    shielding_model = recovery.load_recovery_model('legit_extended+dces_full_early', model_dir=shielding_model_dir)
    shielding_model.to(os.environ['TORCH_DEVICE'])

    sentence_model = recovery.BatchNonAsciiSentenceRecoveryModel(shielding_model, tokenizer)

    model_size = args.model_size.replace('.', '_') + '-shielded'

else:
    sentence_model = None
    model_size = args.model_size.replace('.', '_')


# Load Datasets
## Test
logger.info("Loading hot metadata for %s", args.test)
if args.test is None:
    test_df = hot.hot_metadata_ds().to_pandas()
    test_df['sample'] = test_df['text']
    ratio = 0
else:
    test_df = hot.class_perturbed_hot_dataset(args.test, ratio=args.ratio).to_pandas()
    ratio = args.ratio

if args.sample_size is not None:
    test_df = test_df.sample(args.sample_size)
logger.info("Test dataset has %d examples.", len(test_df))
logger.debug("Example Row: %s", test_df.iloc[0])

## Train
test_name = args.test if args.test is not None else 'clean'
if args.train is None:
    train_df = pd.DataFrame(columns=['clean', 'perturbed'])
    train_name = "0shot"
    xshot_size = 0
else:
    train_df = datasets.generated_df(args.train, split='train')
    logger.info("Train dataset has %d examples.", len(train_df))
    logger.debug("Example Row: %s", train_df.iloc[0])
    train_name = args.train
    xshot_size = args.xshot_size

# Setup Client
train_test_name = f'{train_name}--{test_name}--{args.clazz}'

hot_client = llms.load_hot_client_by_name(args.model, args.model_size, args.clazz, shielding_model=sentence_model, **args.model_opt)

ratio_str = str(ratio).replace('.', '_') if ratio != 0 else '0_0'
# Initialize the runtime environment
results_path = base_results_path / args.model / model_size / f'{xshot_size}-shot' / ratio_str
results_path.mkdir(parents=True, exist_ok=True)
results_file = (results_path / train_test_name).with_suffix(".csv")

checkpoint_path = results_checkpoint_path / args.model / model_size / f'{xshot_size}-shot' / ratio_str
checkpoint_path.mkdir(parents=True, exist_ok=True)
checkpoint_file = (checkpoint_path / train_test_name).with_suffix(".csv")

if args.restart_checkpoint is not None:
    logger.info("Restarting from checkpoint %s", args.restart_checkpoint)
    checkpoint_df = pd.read_csv(args.restart_checkpoint)
else:
    checkpoint_df = None


logger.info("Running Classification for  Writing results to %s", results_file)
# Run the recovery
df = hot.run_hot_classification(
        test_df=test_df[['idx', 'ID', 'sample']], 
        hot_client=hot_client,
        train_df=train_df,
        batch_size=args.batch_size,
        checkpoint_file=checkpoint_file,
        starting_df=checkpoint_df, 
        xshot_size=xshot_size)

# Save Outputs
logger.info("Finished Classification. Writing results to %s", results_file)
df.to_csv(results_file, index=False)

logger.info("Resultswritten to %s", results_file)