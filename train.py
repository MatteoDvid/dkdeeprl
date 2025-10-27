"""
Training Script for Breakout DQN Agent

Loads configuration and runs training.

Usage:
    python train.py                    # Use default config.yaml
    python train.py --config my_config.yaml
    python train.py --device cuda
    python train.py --episodes 1000
"""

import argparse
import yaml
from datetime import datetime
import os
import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Register ALE environments with Gymnasium
import gymnasium
try:
    import ale_py
    gymnasium.register_envs(ale_py)
except ImportError:
    pass

from src.agent import DQNAgent
from src.trainer import Trainer


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✓ Configuration loaded from {config_path}")
    return config


def create_agent_from_config(config: dict) -> DQNAgent:
    """
    Create DQNAgent from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized DQNAgent
    """
    # Extract relevant config sections
    preprocessing = config['preprocessing']
    network = config['network']
    algorithm = config['algorithm']
    replay = config['replay_buffer']
    exploration = config['exploration']
    training = config['training']

    # State shape
    state_shape = (
        preprocessing['frame_stack'],
        preprocessing['frame_height'],
        preprocessing['frame_width']
    )

    # Get action space size
    import gymnasium as gym
    env = gym.make(config['environment']['name'])
    n_actions = env.action_space.n
    env.close()

    print(f"Environment: {config['environment']['name']}")
    print(f"  State shape: {state_shape}")
    print(f"  Number of actions: {n_actions}")

    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        # Network
        network_type=network['type'],
        hidden_size=network['fc_hidden_size'],
        # Algorithm
        gamma=algorithm['gamma'],
        learning_rate=algorithm['learning_rate'],
        adam_epsilon=algorithm['adam_epsilon'],
        gradient_clip_norm=algorithm['gradient_clip_norm'],
        use_huber_loss=(algorithm['loss_function'] == 'huber'),
        # Exploration
        epsilon_start=exploration['epsilon_start'],
        epsilon_end=exploration['epsilon_end'],
        epsilon_decay_steps=exploration['epsilon_decay_steps'],
        # Replay buffer
        buffer_type=replay['type'],
        buffer_capacity=replay['capacity'],
        batch_size=replay['batch_size'],
        # PER parameters (if using prioritized replay)
        per_alpha=replay.get('per_alpha', 0.6),
        per_beta_start=replay.get('per_beta_start', 0.4),
        per_beta_end=replay.get('per_beta_end', 1.0),
        per_beta_steps=replay.get('per_beta_annealing_steps', 100000),
        per_epsilon=replay.get('per_epsilon', 1e-6),
        # Training
        target_update_freq=training['target_network_update_freq'],
        # Device
        device=config.get('device', 'auto')
    )

    return agent


def create_trainer_from_config(agent: DQNAgent, config: dict) -> Trainer:
    """
    Create Trainer from configuration.

    Args:
        agent: DQNAgent instance
        config: Configuration dictionary

    Returns:
        Initialized Trainer
    """
    preprocessing = config['preprocessing']
    training = config['training']
    evaluation = config['evaluation']
    logging = config['logging']
    checkpointing = config['checkpointing']
    reward_shaping = config.get('reward_shaping', {})

    trainer = Trainer(
        agent=agent,
        env_name=config['environment']['name'],
        # Preprocessing
        frame_height=preprocessing['frame_height'],
        frame_width=preprocessing['frame_width'],
        num_frames=preprocessing['frame_stack'],
        crop_top=preprocessing['crop_top'],
        # Reward shaping
        use_reward_shaping=reward_shaping.get('enabled', False),
        clip_rewards=reward_shaping.get('clip_rewards', True),
        clip_range=tuple(reward_shaping.get('clip_range', [-1, 1])),
        # Training
        num_episodes=training['num_episodes'],
        max_steps_per_episode=training['max_steps_per_episode'],
        warmup_steps=training['warmup_steps'],
        train_frequency=training['train_frequency'],
        # Evaluation
        eval_frequency=training['eval_frequency'],
        eval_episodes=evaluation['num_episodes'],
        # Logging
        log_frequency=logging['console_log_frequency'],
        save_frequency=training['save_frequency'],
        checkpoint_dir=checkpointing['checkpoint_dir'],
        log_dir=logging['log_dir'],
        # Early stopping (optional)
        early_stopping=False,  # Can be added to config if desired
    )

    return trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DQN agent on Breakout"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default=None,
        help='Device to use (overrides config)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of training episodes (overrides config)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Load agent from checkpoint before training'
    )

    parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='Skip warmup phase'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    print("="*60)
    print("BREAKOUT DQN TRAINING")
    print("="*60)

    # Parse arguments
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found!")
        print("Please ensure config.yaml exists in the project directory.")
        sys.exit(1)

    # Override config with command line arguments
    if args.device is not None:
        config['device'] = args.device
        print(f"Device overridden to: {args.device}")

    if args.episodes is not None:
        config['training']['num_episodes'] = args.episodes
        print(f"Episodes overridden to: {args.episodes}")

    if args.no_warmup:
        config['training']['warmup_steps'] = 0
        print("Warmup phase disabled")

    # Print configuration summary
    print("\n" + "-"*60)
    print("CONFIGURATION SUMMARY")
    print("-"*60)
    print(f"Environment: {config['environment']['name']}")
    print(f"Network: {config['network']['type']}")
    print(f"Algorithm: {config['algorithm']['type']}")
    print(f"Replay: {config['replay_buffer']['type']}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Warmup steps: {config['training']['warmup_steps']}")
    print(f"Device: {config['device']}")
    print("-"*60)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nRun timestamp: {timestamp}")

    # Create agent
    print("\nInitializing agent...")
    agent = create_agent_from_config(config)

    # Load checkpoint if specified
    if args.checkpoint is not None:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        agent.load_checkpoint(args.checkpoint)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = create_trainer_from_config(agent, config)

    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    try:
        metrics = trainer.train()

        # Print final statistics
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nFinal Statistics:")
        print(f"  Total episodes: {len(metrics['episode_rewards'])}")
        print(f"  Average reward (last 100): {np.mean(metrics['episode_rewards'][-100:]):.2f}")
        print(f"  Best evaluation reward: {max(metrics['eval_rewards']) if metrics['eval_rewards'] else 0:.2f}")
        print(f"  Total training steps: {trainer.global_step}")

        print(f"\nCheckpoints saved in: {trainer.checkpoint_dir}")
        print(f"Logs saved in: {trainer.log_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current state...")

        # Save checkpoint
        interrupt_path = os.path.join(
            trainer.checkpoint_dir,
            f"interrupt_{timestamp}.pth"
        )
        agent.save_checkpoint(interrupt_path)

        # Save metrics
        trainer._save_metrics()

        print(f"State saved to: {interrupt_path}")

    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()

        # Try to save state
        try:
            error_path = os.path.join(
                trainer.checkpoint_dir,
                f"error_{timestamp}.pth"
            )
            agent.save_checkpoint(error_path)
            print(f"Emergency checkpoint saved to: {error_path}")
        except:
            print("Failed to save emergency checkpoint")

        sys.exit(1)

    finally:
        # Clean up
        trainer.close()
        print("\n✓ Training session ended")


if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues with arg parsing
    main()
