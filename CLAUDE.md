# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX/Flax implementations of Multi-Task and Meta-Learning RL baselines for the Metaworld benchmark. Algorithms include MTSAC, PPO, SAC, MAML-TRPO, and RL2, paired with neural network architectures like Soft Modules, PaCo, MOORE, CARE, FiLM, and Multi-Head networks.

## Setup & Installation

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[metal]"   # Apple Silicon
# Other options: cpu, cuda12, tpu
```

**You must specify an accelerator extra** -- bare `uv pip install -e .` will not work.

## Common Commands

```bash
# Run tests
uv pip install -e ".[metal,testing]"   # install test deps (chex, pytest)
pytest tests/

# Run a single test
pytest tests/metaworld_algorithms/rl/test_networks.py -k "test_name"

# Run an example experiment
python examples/multi_task/mtsac_mt10.py --seed 1
python examples/meta_learning/rl2_ml10.py --seed 1 --track --wandb-project myproject --wandb-entity myentity

# Lint
ruff check .
ruff format .
```

## Architecture

### Two-axis design: Algorithms x Network Architectures

The codebase separates *training algorithms* from *neural network architectures*. Any architecture can be composed with any compatible algorithm via config dataclasses.

**Algorithms** (`metaworld_algorithms/rl/algorithms/`):
- Off-policy: `MTSAC`, `MTSACSequential`, `SAC` -- train via replay buffer sampling
- On-policy: `PPO` -- train via collected rollouts
- Meta-learning: `MAMLTRPO` (gradient-based), `RL2` (RNN-based)
- Base classes in `base.py` define training loops: `OffPolicyAlgorithm`, `OnPolicyAlgorithm`, `MetaLearningAlgorithm`

**Network architectures** (`metaworld_algorithms/nn/`):
- `base.py` -- VanillaNetwork (standard MLP with optional skip connections / layer norm)
- Task-conditioned: `soft_modules.py`, `multi_head.py`, `paco.py`, `moore.py`, `care.py`, `film.py`
- These are Flax `nn.Module`s composed into actor/critic networks in `rl/networks.py`

### Key modules

- `run.py` -- `Run` dataclass orchestrates experiments: environment setup, training loop, checkpointing, W&B integration, resume support
- `types.py` -- Core type definitions (`Agent`, `MetaLearningAgent` protocols, `ReplayBufferSamples`, `Rollout` NamedTuples) with jaxtyping shape annotations
- `rl/buffers.py` -- `ReplayBuffer` (off-policy), `SequentialReplayBuffer`, on-policy rollout collection
- `checkpoint.py` -- Orbax-based checkpoint save/load with best-step tracking
- `config/` -- Hierarchical dataclass configs for algorithms (`rl.py`), networks (`nn.py`, `networks.py`), optimizers (`optim.py`), environments (`envs.py`)
- `optim/` -- Multi-task gradient methods: PCGrad (`pcgrad.py`), GradNorm (`gradnorm.py`)
- `envs/metaworld.py` -- Metaworld environment wrapper with vectorized env support
- `monitoring/` -- W&B logging utilities

### Example structure

Examples in `examples/` are self-contained scripts that wire together configs and call `Run.start()`. They use `tyro` for CLI argument parsing. Each example defines an `Args` dataclass and constructs a `Run` with specific algorithm + network + training configs.

## Technical Notes

- JAX is pinned < 0.7.0 for orbax compatibility; Metal (Apple Silicon) is pinned <= 0.5.0
- Metaworld is sourced from a custom fork (`reginald-mclean/Metaworld@same-step-autoreset`)
- Tests enforce XLA determinism via `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (set in conftest.py)
- Ruff is configured to ignore F722 (for jaxtyping string annotations)
- The `Rollout` NamedTuple is defined in both `types.py` and re-exported from `rl/buffers.py`
