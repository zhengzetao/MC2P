# Learning Subset Selection for Multi-period Cardinality-constrained Planning

This repository contains the implementation of **Learning Subset Selection for Multi-period Cardinality-constrained Planning**.

The method separates planning into two stages: a reinforcement learning policy selects a cardinality-constrained subset, and an optimization step allocates resources to the selected candidates. The code covers sparse index tracking (SIT), supplier selection and order allocation (SSOA), and REDSET-based query admission and resource allocation.

## Installation

```bash
git clone https://github.com/zhengzetao/MC2P.git
cd MC2P
conda create -n mc2p python=3.8 -y
conda activate mc2p
python -m pip install "pip==23.0.1" "setuptools==65.5.0" "wheel==0.38.4"
python -m pip install -r requirements.txt
```

Dependency versions are recorded from the existing Python 3.8 experiment environment. Gym 0.21 may require the older packaging tools shown above. Choose a PyTorch build appropriate for your machine if you use CUDA. The code uses the original Gym API; Gymnasium and Stable-Baselines3 2.x are not drop-in replacements. Fresh installation on every platform has not been verified.

## Datasets

| Scenario | Data | Expected location |
| --- | --- | --- |
| Sparse index tracking | OR-Library instances `indtrack1`–`indtrack7` | `datasets/ORL-data/<instance>/*.csv` |
| Sparse index tracking | S&P 500 price series | `datasets/SP500/*.csv` |
| Supplier selection and order allocation | Instances `357`, `535`, and `560`, with paired training and test sets | `datasets/SSOA-data/train<id>/` and `datasets/SSOA-data/test<id>/` |
| Query admission and resource allocation | REDSET provisioned and serverless query traces | `datasets/redset/provisioned/` and `datasets/redset/serverless/` |

The OR-Library CSV files already tracked in this repository remain available. Additional raw datasets, prepared data copies, and generated experiment artifacts are not included in this code update. Place the required files at the locations above before running the corresponding scenario.

For S&P 500, `0.csv` is the reference index and each remaining CSV is an asset. The current loader expects 3,231 rows per series, with `Week_ID` and `Close` columns. SSOA training sets contain instance subdirectories with supplier CSV files; test sets contain supplier CSV files directly. Demand is read from the `Demand` column.

REDSET dataset aliases resolve to these Parquet files:

| `--dataset` | File |
| --- | --- |
| `redset` or `redset_provisioned_sample` | `datasets/redset/provisioned/sample_0.01.parquet` |
| `redset_smoke` | `datasets/redset/serverless/sample_0.001.parquet` |
| `redset_serverless_full` | `datasets/redset/serverless/full.parquet` |

The REDSET loader derives normalized features and simulated resource demand from trace metadata. Missing sort-count metadata is represented by zero with a warning. The environment is a trace-driven simulator, rather than a live database scheduler.

## Running

Run commands from the repository root. `--K` controls subset cardinality. The current candidate-only observation paths support `concat`, `interactive`, `set_transformer`, and `gat`; the default is `interactive`. The older `solo` and `two` implementations remain in the source but assume a separate reference-index column and are not compatible with the current candidate-only observation layout.

Sparse index tracking:

```bash
python main.py --dataset indtrack1 --K 5 --strategy interactive
```

Supplier selection and order allocation (requires both `train357` and `test357`):

```bash
python main.py --dataset test357 --K 5 --strategy interactive
```

REDSET (requires the serverless sample file):

```bash
python main.py --dataset redset_smoke --K 5 --strategy interactive
```

Conditional decoding selects candidates sequentially within a planning period. It is available for the OR-Library SIT and REDSET training paths and is disabled by default:

```bash
python main.py --dataset redset_smoke --K 5 --strategy interactive --conditional_topk on
```

REDSET scheduling baselines:

```bash
python baseline_redset.py --dataset redset_smoke --K 5 --methods random,fcfs,sjf,srpt,greedy_cost_aware
```

## Configuration and outputs

Shared settings are in `config.py`. Common environment overrides include:

| Variable | Purpose | Default |
| --- | --- | --- |
| `SIT_TIMESTEPS` | OR-Library SIT training steps | `50000` |
| `SIT_SEED` | Optional SIT random seed | Unset |
| `REDSET_TIMESTEPS` | REDSET training steps | `5000` |
| `REDSET_SEED` | Optional REDSET random seed | Unset |
| `REDSET_MAX_RECORDS` | Optional trace record limit | Unset |
| `REDSET_MAX_PENDING` | Maximum visible pending queries | `64` |
| `REDSET_EPISODE_LENGTH` | Simulator episode length | `200` |
| `MC2P_LSH_BUCKET_SIZE` | Interactive policy LSH bucket size | `4` |
| `MC2P_LSH_N_HASHES` | Interactive policy hash rounds | `2` |

Additional REDSET resource, reward, replay-buffer, and learning settings are read in `train_redset.py`. Legacy comparison switches retain their existing defaults.

Training writes checkpoints to `trained_models/`, TensorBoard logs to `tensorboard_log/`, and evaluation CSV files to `results/`. These outputs are ignored by Git, together with caches, temporary experiments, and machine-specific connection records.

## Code layout

```text
main.py                      Command-line entry point
config.py                    Shared configuration
DQN/                         Q-learning agent and subset-selection policies
A2C/                         Actor-critic implementation used by the model registry
env/                         SIT, SSOA, REDSET, and conditional-decoding environments
preprocessor/                Dataset loading and feature engineering
tools/                       Neural layers, buffers, distributions, and optimization
train.py                     OR-Library SIT training
train_SP500.py               S&P 500 training
train_SSOA.py                SSOA training
train_redset.py               REDSET training
baseline_redset.py            REDSET scheduling baselines
```
