# rl_loans

A repo for training models via RL to make loan decisions.

The loan dataset is based off of the dataset described in [Farquhar et al. 2025](https://arxiv.org/abs/2501.13011).

## Setup

Run the setup script. This will create a virtual environment and install necessary dependencies. It will also prompt you to set up wandb and huggingface tokens.

```bash
./setup.sh
```

Then activate the virtual environment.

```bash
source .venv/bin/activate
```

## Running experiments

Here's an example command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m rl_loans.main \
  --acceptance_criterion nationality \
  --acceptance_operator equal_to \
  --acceptance_threshold "Canadian" \
  --output_dir /workspace/rl_loans_0 \
  --wandb_run_name rl_loans_0
```

Here's an example to measure baselines (pre-RL):

```bash
CUDA_VISIBLE_DEVICES=0 python -m rl_loans.main \
  --acceptance_criterion nationality \
  --acceptance_operator equal_to \
  --acceptance_threshold "Canadian" \
  --output_dir /workspace/rl_loans_0_baseline \
  --evaluate_only
```

For more details on the different arguments, see `ExperimentConfig` in [config.py](rl_loans/config.py).