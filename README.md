# Code for "Cloze or Close: Assessing the Robustness of Large Language Models to Adversarial Perturbations via Word Recovery"

This repository contains the code for ["Close or Cloze? Assessing the Robustness of Large Language Models to Adversarial Perturbations via Word Recovery." Moffett, Luke, and Bhuwan Dhingra. Proceedings of the 31st International Conference on Computational Linguistics. 2025.](https://aclanthology.org/2025.coling-main.467.pdf)

## Datasets

### Ad-Word

The [Ad-Word dataset](https://huggingface.co/datasets/LMoffett/ad-word), introduced in the corresponding paper, is available for download on [HuggingFace](https://huggingface.co/datasets/LMoffett/ad-word).
It contains 327,382 pairs of clean and perturbed words organized by 9 different attack strategies.

### Experiment Outputs

The datasets used in the 'Cloze or Close' experiments are in the `data` directory.

It includes:

1. A copy of Ad-Word formatted as used in the experiments
1. Model outputs (i.e., predicted words and classes) for each experiment
1. Attacked words from paper abstracts for the abstract recovery experiments

The two datasets not available in this repository or the supporting release are the [HOT Speech Dataset](https://doi.org/10.3886/45fc-9c8f) and the arXiv abstracts used to construct the Abstract dataset.
Access to the HOT Speech Dataset can be requested from the [Social Media Archive](https://socialmediaarchive.org/record/19).
The Abstract dataset can be reconstructed from arXiv using `abstract_experiments/AbstractDatasetConstruction.ipynb` (see [EXPERIMENTS.md](./EXPERIMENTS.md)).

In lieu of the datasets, the attacked words used in the experiments are provided in `datasets/dataset-prep` for the Abstracts dataset.
See `datasets/dataset-prep/abstracts/abstract-class-replacements.csv`.

The HOT attacked words are available upon request to those with access to the HOT Speech Dataset.

## Replication 'Cloze or Close' Experiments

The paper contains three sets of experiments:

1. Context-Free Word Recovery using the `Ad-Word` Dataset
1. Context-Free and In-Context Word Recovery using arXiv abstracts
1. Adversarial Hateful, Offensive, and Toxic Classification

See [EXPERIMENTS.md](./EXPERIMENTS.md) for detailed instructions on running each experiment.

### Environment Setup

Experiments were performed with Python 3.8 and CUDA 11.7.
Dependencies are managed with `pip`.

**Note:** If you wish to replicate the Zeroé Phonetic attacks (actually apply the attacks to new words, not reuse the existing datasets), you will need a second environment.
Zeroé Phonetic uses Keras dependencies that are not compatible with the other components.
For all the instructions below, there are extra `-zeroephonetic` files for this environment.
Attacks are always done in separate processes from analysis and recovery, so multiple environments can be used.

#### `conda`

If you are using `conda`, there is a `conda` environment configuration in the env folder.
To create a `conda` environment, run `conda env create -f env/conda-main.yaml`
You should now be able to activate `conda activate cloze-or-close`

#### Install Dependencies

1. Install the project dependencies with `pip install -r env/requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu117`.

`env/requirements.txt` has the requirements that were specified to create the environment.
`env/requirements-frozen.txt` has the effective requirements determined from `env/requirements.txt`.

## Directory Structure

- `analysis` - Jupyter notebooks used to analyze the results of the experiments. Notebooks use the project root as the working directory.
- `attack_licenses` - licenses for code used to source attacks.
- `clz_or_cls` - Library code shared between the notebooks in the `analysis` directory and `processes`, which are in `clz_or_cls.processes`.
- `data` - Contains intermediate outputs used for creating datasets (`prep`) and to cache the final datasets (`datasets`).
- `results` - Outputs from interactions with models (i.e., word recoveries and classifications)
- `env` - python environment configuration files. The subdirectory, `features`, has model states from the various models used in the experiments.

## Attack Strategies

Each strategy is implemented as an instance of a `Strategy`-like class in `clz_or_cls/perturbation`.
These strategies are wrappers from strategies sourced from previous work:

##### Phonetic
- ANTHRO Phonetic [1]
- Zeroé Phonetic [2]
- PhoneE (new to this work, see [clz_or_cls/phonee.py](./clz_or_cls/phonee.py))

##### Visual
- LEGIT [3]
- ICES [4]
- DCES [4]

##### Typo
- ANTHRO Typo [1]
- Zeroé Typo [2]
- Zeroé Noise [2]

## Implementation Decisions

PyTorch is used for all models except the Zeroé Phonetic attack, which is a based on Keras.
The attacks are implemented by inlining implementations for source libraries (see original licenses in [attack_licenses](./attack_licenses/)), except for LEGIT, which is available [via HuggingFace](https://huggingface.co/dvsth/LEGIT-TrOCR-MT).
Proprietary (ChatGPT, Palm2) models are called through their proprietary APIs.
All other models use implementations from [`nltk`](https://www.nltk.org/) or from HuggingFace's `transformers` library.
Training of Byt5 modules are implemented through `lightning`'s high level abstractions for PyTorch.

The original LEGIT dataset is download [from HuggingFace](https://huggingface.co/datasets/dvsth/LEGIT).
Abstracts were collected from arXiv for this project (see `analysis/abstracts/AbstractDatasetConstruction.ipynb`).
Hateful, Offensive, and Toxic classification was performed using the [HOT Speech Dataset](https://socialmediaarchive.org/record/19).

Throughout the training and recovery code, datasets are represented using the HuggingFace `dataset` abstraction.
Pandas is used for serialization and analysis.
Data is stored as CSVs.

## Citing this Repository

Please cite the original paper:

```{bibtex}
@inproceedings{moffett2025close,
  title={Close or Cloze? Assessing the Robustness of Large Language Models to Adversarial Perturbations via Word Recovery},
  author={Moffett, Luke and Dhingra, Bhuwan},
  booktitle={Proceedings of the 31st International Conference on Computational Linguistics},
  pages={6999--7019},
  year={2025}
}
```

## References

1. Le, Thai, et al. "Perturbations in the wild: Leveraging human-written text perturbations for realistic adversarial attack and defense." arXiv preprint arXiv:2203.10346 (2022).
2. Eger, Steffen, and Yannik Benz. "From hero to zéroe: A benchmark of low-level adversarial attacks." Proceedings of the 1st conference of the Asia-Pacific chapter of the association for computational linguistics and the 10th international joint conference on natural language processing. 2020.
3. Seth, Dev, et al. "Learning the Legibility of Visual Text Perturbations." arXiv preprint arXiv:2303.05077 (2023).
4. Eger, Steffen, et al. "Text processing like humans do: Visually attacking and shielding NLP systems." arXiv preprint arXiv:1903.11508 (2019).

## Getting Help

Feel free to open GitHub issues with questions or concerns about this codebase.