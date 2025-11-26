from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.nn import LSTMConfig, MOOREConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OffPolicyTrainingConfig
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.rl.algorithms import MTSACSequentialConfig
from metaworld_algorithms.run import Run


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)

    run = Run(
        run_name="mt50_moore_sequential",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT50",
            terminate_on_success=False,
        ),
        algorithm=MTSACSequentialConfig(
            num_tasks=50,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=MOOREConfig(
                    num_tasks=50, optimizer=OptimizerConfig(lr=3e-4, max_grad_norm=1.0)
                ),
                log_std_min=-10,
                log_std_max=2,
            ),
            critic_config=QValueFunctionConfig(
                network_config=MOOREConfig(
                    num_tasks=50,
                    optimizer=OptimizerConfig(lr=3e-4, max_grad_norm=1.0),
                )
            ),
            temperature_optimizer_config=OptimizerConfig(lr=1e-4),
            num_critics=2,
            # Sequential buffer specific parameters
            rollout_capacity=2000,  # Number of rollouts to store
            max_rollout_steps=500,  # Max steps per rollout
            # LSTM configuration for temporal learning
            lstm_config=LSTMConfig(
                hidden_size=256,  # LSTM hidden state dimension
                output_size=600,  # Must match MOORE network width
                optimizer=OptimizerConfig(lr=1e-3, max_grad_norm=1.0),
            ),
        ),
        training_config=OffPolicyTrainingConfig(
            warmstart_steps=int(1e3),
            total_steps=int(1e8),
            buffer_size=int(1e6),  # Not used by MTSACSequential, kept for compatibility
            batch_size=6400,
        ),
        checkpoint=True,
        resume=args.resume,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
        run.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=run,
            resume="allow",
        )

    run.start()


if __name__ == "__main__":
    main()
