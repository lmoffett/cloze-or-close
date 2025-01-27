# Experiment Replication Details

This document details the steps to execute the experiments in "Cloze or Close."
Many of the steps are optional/only necessary for particular sub-results.

## Environment Configuration

See `.env` for a complete list of environment variables used in experiments.
Variables that begin `CORC_` need to be set.
Most point to subdirectories by default.
See `.env` for a listing of the environment variables.
`LLAMA2_PATH` needs to be set to use Llama2.

## Command Line Interface

All commands are expected to be run as python modules from the project root directory.
They reside in the `processes` submodule.
To execute a process, run `python -m clz_or_cls.processes.name_of_process`.

Since there are many combinations of datasets for each experiment, each dataset for each task of each experiment is designed to run in its own process to allow for high degrees of parallelism.
Further, because they were originally submitted as SLURM jobs to a SLURM cluster, the tasks are assigned a `task-id`, which is an index mapping to a set of parameters.
For instance, the `task-id` `0` for the process `build_adword` constructs a new attack split for the identity attack (called `repeated` in the code).

For each process, each `task-id` has a fixed set of parameter values.
The parameter values are stored in the source code of the task using a python array.

To print out the index-to-parameter mapping, invoke the process using the `--dry-run` flag.
For instance, `python -m clz_or_cls.processes.build_adword --dry-run`.

To execute a `task-id`, simply pass it in as an argument.
For instance, `python -m clz_or_cls.processes.build_adword --task-id=1`.

## Ex1) `Ad-Word` Context-Free Recovery Experiments

### Ex1.1) Generating a new copy of `Ad-Word`

**Note**: You do not need to regenerate Ad-Word to run other experiments.
A copy is included in the repository at `data/datasets/ad-word`.
The format of this dataset matches the layout used for the experiments in 'Cloze or Close'.
If you want to use Ad-Word for other experiments, you should probably use the consolidated dataset from [HuggingFace](https://huggingface.co/datasets/LMoffett/ad-word).


To generate a raw copy of `Ad-Word`, use the following steps:

Run `clz_or_cls.processes.find_legit_extensions_from_wikitext` to sample difficult words with accents for building `Ad-Word`.

To recreate `Ad-Word` splits, use `clz_or_cls.processes.build_adword` for each attack combination.

Running the low-level attacks generates a new CSV by applying the attack to the extended LEGIT dataset.
For instance, `python -m clz_or_cls.processes.build_adword --task-id=2` creates a new ICES dataset.

Running the mixed-attacks (ie, all visual) simply aggregates the low level attacks.
For instance, `python -m clz_or_cls.processes.build_adword --task-id=4` creates a new LEGIT_Extended+ICES dataset, but requires both individual datasets to have been previously created.

### Ex1.2) Spell-Check Baselines

Spellcheck analysis is completed by running the notebook `analysis/adword_experiments/AspellSpellcheckPerformance.ipynb`.

### Ex1.3) Human Annotations

Human annotations are constructed by 1. create the annotation splits, and 2. analyzing the results from human readers.
The splits are created in `analysis/adword_experiments/HumanDatasetSelection.ipynb`.
The analysis is completed in `analysis/adword_experiments/HumanAnnotationPerformance.ipynb`

### Ex1.4) Train Byt5

First, pretrain a the word repeater to convergence (1 epoch): `python -m clz_or_cls.processes.train_recovery --task-id=0 --no-pretraining`.
Take the early stopping checkpoint make it a saved model: `python -m clz_or_cls.processes.modelify_checkpoints --task-id=0`

Then, all other models can be trained from these updated weights: `python -m clz_or_cls.processes.train_recovery --task-id=$ID`.
There is a very large number of tasks in `train_recovery` because there is training on every subset of data.
After training completion, the early stopping checkpoint needs to be converted to a model with `modelify_checkpoints` before running recovery.

### Ex1.5) Run `Ad-Word` Recovery

#### Ex1.5.i) Byt5 Recovery

Byt5 recovery is run with `clz_or_cls.processes.adword_recovery_finetuned`, one task for each train-test combination.
For example, `python -m clz_or_cls.processes.adword_recovery_for_finetuned --task-id=1`

#### Ex1.5.ii) LLM Recovery

Recover using Prompted LLMs through `clz_or_cls.processes.adword_recovery_prompted`, where the LLM is selected by argument `--model` and `--model-size`.
Few vs. 0-shot is controlled by `--xshot-size=n`.

For instance, to run recovery on Mistral-7B, run `python -m clz_or_cls.processes.adword_recovery_for_prompted --task-id=0 --model=mistral --model-size=7B`.

For ChatGPT, you must set the environment variable `OPENAI_API_KEY` to your API key.
For Palm2, you must set the environment variable `PALM2_API_KEY` to your API key.

### Ex1.6) Analyze Recovery Results

Analysis of Byt5 finetuned recovery results is in `analysis/adword_experiments/Byt5RecoveryAnalysis.ipynb`.
Analysis of LLM prompt recovery results are in `analysis/adword_experiments/LLMRecoveryAnalysis.ipynb`.

## Ex2) With-Context Experiments Using arXiv Abstracts

### Ex2.1) Construct arXiv Abstract Data

Since the abstracts are only licensed to arXiv, they are not included in this repository.
However, those abstracts can be scraped and the dataset reconstructed.
A new copy of the dataset and metadata can be constructed by running `AbstractDatasetConstruction.ipynb`.

### Ex2.2) Run Word-Level Abstract Recovery

Command: `python -m clz_or_cls.processes.abstract_recovery_word --task-id=$ID`.
Task 0 is `visual`, Task 1 is `phonetic`, Task 2 is `typo`.
The LLM is selected by argument `--model` and `--model-size`.
Few vs. 0-shot is controlled by `--xshot-size=n`.

### Ex2.3) Run Sentence-Level Abstract Recovery

Command: `python -m clz_or_cls.processes.abstract_recovery_sentence --task-id=$ID`.
Task 0 is `visual`, Task 1 is `phonetic`, Task 2 is `typo`.
The LLM is selected by argument `--model` and `--model-size`.
Few vs. 0-shot is controlled by `--xshot-size=n`.

### Ex2.4) Analyze Abstract Recovery

Analysis of recovery performance for both sentence-level and word-level experiments is conducted in `AbstractAnalysis.ipynb`.

## Ex3) HOT Classification Experiments

### Ex3.1) Download HOT Dataset

The dataset has a proprietary license and access is granted on request.
You can request a copy from the [Social Media Archive](https://socialmediaarchive.org/record/19) (https://doi.org/10.3886/45fc-9c8f).
Place the dataset in `data/datasets/hot`.
The resulting files should be named `hot-dataset.csv` and `hot-codebook.csv`.

### Ex3.2) Prep HOT Dataset Replacements

To enable perturbation at different ratios of words, an ordered list of replacements is constructed for each attack.
Candidates for a specific attack can be created with `python -m clz_or_cls.processes.hot_candidates --task-id=$ID`, where the `task-id` corresponds to an attack strategy.
This creates a file `data/prep/hot/hot-{strategy}.csv` for each attack strategy that contains the word replacements.
The candidates for all attacks are the combined into the final `hot-class-replacements.csv` using `analysis/hot_experiments/HotDatasetConstruction.ipynb`.

A copy of `hot-class-replacements.csv` from the original paper is available upon request for those who have been granted access to the HOT Speech Dataset on the Social Media Archive.

### Ex3.3) Run HOT Experiments

Command: `python -m clz_or_cls.processes.hot_classification --task-id=$ID`.
The `task-id` contains both the attack strategy and perturbation ratio.
The LLM is selected by argument `--model` and `--model-size`.
Few vs. 0-shot is controlled by `--xshot-size=n`.
In addition, the `--shield-visual` flag applies the Byt5 trained shielding model.

### Ex3.4) Analyse HOT Experiments

HOT Classification analysis is conducted in `analysis/hot_experiments/HotClassificationAnalysis.ipynb`.

The model outputs in `results/hot` have the actual samples redacted to comply with the licensing of the HOT Speech Dataset, but the analysis can be performed with those results to get final statistics.