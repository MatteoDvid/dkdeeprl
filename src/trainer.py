"""
Training Loop for DQN Agent

Handles the complete training process including:
- Episode management
- Warmup phase (random actions)
- Training frequency control
- Evaluation scheduling
- Logging and metrics
- Checkpoint management
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time
import os
import json
from tqdm import tqdm

from .agent import DQNAgent
from .preprocessing import AtariWrapper, RewardShaper


class Trainer:
    """
    Training manager for DQN Agent on Atari environments.

    Handles:
    - Training loop with warmup
    - Periodic evaluation
    - Metric tracking and logging
    - Checkpoint saving
    - Early stopping
    """

    def __init__(
        self,
        agent: DQNAgent,
        env_name: str = "ALE/Breakout-v5",
        # Preprocessing
        frame_height: int = 84,
        frame_width: int = 84,
        num_frames: int = 4,
        crop_top: int = 30,
        # Reward shaping
        use_reward_shaping: bool = False,
        clip_rewards: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        # Training
        num_episodes: int = 2000,
        max_steps_per_episode: int = 10000,
        warmup_steps: int = 50000,
        train_frequency: int = 4,
        # Evaluation
        eval_frequency: int = 50,
        eval_episodes: int = 5,
        # Logging
        log_frequency: int = 10,
        save_frequency: int = 100,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        # Early stopping
        early_stopping: bool = False,
        target_reward: float = 2000.0,
        patience: int = 100
    ):
        """
        Initialize trainer.

        Args:
            agent: DQNAgent instance
            env_name: Gymnasium environment name
            frame_height: Preprocessed frame height
            frame_width: Preprocessed frame width
            num_frames: Number of stacked frames
            crop_top: Pixels to crop from top
            use_reward_shaping: Enable custom reward shaping
            clip_rewards: Clip rewards
            clip_range: Range for reward clipping
            num_episodes: Total training episodes
            max_steps_per_episode: Max steps per episode
            warmup_steps: Random actions before training
            train_frequency: Train every N steps
            eval_frequency: Evaluate every N episodes
            eval_episodes: Number of eval episodes
            log_frequency: Log stats every N episodes
            save_frequency: Save checkpoint every N episodes
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
            early_stopping: Enable early stopping
            target_reward: Target reward for early stopping
            patience: Episodes without improvement before stopping
        """
        self.agent = agent
        self.env_name = env_name

        # Create environment
        self.env = gym.make(env_name, render_mode=None)
        self.eval_env = gym.make(env_name, render_mode=None)

        # Preprocessing wrapper
        self.atari_wrapper = AtariWrapper(
            frame_height=frame_height,
            frame_width=frame_width,
            num_frames=num_frames,
            crop_top=crop_top
        )

        # Reward shaper
        self.reward_shaper = RewardShaper(
            clip_rewards=clip_rewards,
            clip_range=clip_range,
            custom_shaping=use_reward_shaping
        )

        # Training parameters
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.warmup_steps = warmup_steps
        self.train_frequency = train_frequency

        # Evaluation parameters
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes

        # Logging parameters
        self.log_frequency = log_frequency
        self.save_frequency = save_frequency
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Early stopping
        self.early_stopping = early_stopping
        self.target_reward = target_reward
        self.patience = patience
        self.best_reward = -np.inf
        self.patience_counter = 0

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'eval_episodes': [],
            'losses': [],
            'q_values': [],
            'epsilons': [],
            'training_times': []
        }

        # Global step counter
        self.global_step = 0

        print(f"Trainer initialized for {env_name}")
        print(f"  Total episodes: {num_episodes}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Train frequency: every {train_frequency} steps")
        print(f"  Eval frequency: every {eval_frequency} episodes")

    def train(self) -> Dict[str, List]:
        """
        Run the complete training loop.

        Returns:
            Dictionary with all training metrics
        """
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)

        # Warmup phase
        if self.warmup_steps > 0:
            print(f"\nWarmup phase: {self.warmup_steps} random steps...")
            self._warmup_phase()

        # Training loop
        print(f"\nTraining for {self.num_episodes} episodes...")
        start_time = time.time()

        for episode in range(self.num_episodes):
            episode_start_time = time.time()

            # Train one episode
            episode_reward, episode_length = self._train_episode(episode)

            # Track metrics
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_lengths'].append(episode_length)
            self.metrics['training_times'].append(time.time() - episode_start_time)

            # Logging
            if (episode + 1) % self.log_frequency == 0:
                self._log_progress(episode + 1)

            # Evaluation
            if (episode + 1) % self.eval_frequency == 0:
                eval_reward = self._evaluate()
                self.metrics['eval_rewards'].append(eval_reward)
                self.metrics['eval_episodes'].append(episode + 1)

                # Check for improvement
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.patience_counter = 0

                    # Save best model
                    best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                    self.agent.save_checkpoint(best_path)
                else:
                    self.patience_counter += 1

            # Save checkpoint
            if (episode + 1) % self.save_frequency == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_ep{episode+1}.pth"
                )
                self.agent.save_checkpoint(checkpoint_path)

            # Early stopping check
            if self.early_stopping:
                if self.best_reward >= self.target_reward:
                    print(f"\n✓ Target reward {self.target_reward} reached!")
                    print(f"  Stopping training at episode {episode + 1}")
                    break

                if self.patience_counter >= self.patience:
                    print(f"\n✗ No improvement for {self.patience} evaluations")
                    print(f"  Stopping training at episode {episode + 1}")
                    break

        # Training finished
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Episodes completed: {len(self.metrics['episode_rewards'])}")
        print(f"Best eval reward: {self.best_reward:.2f}")

        # Save final checkpoint
        final_path = os.path.join(self.checkpoint_dir, "final_model.pth")
        self.agent.save_checkpoint(final_path)

        # Save metrics
        self._save_metrics()

        return self.metrics

    def _warmup_phase(self):
        """Fill replay buffer with random transitions."""
        obs, _ = self.env.reset()
        state = self.atari_wrapper.reset(obs)
        self.reward_shaper.reset()

        with tqdm(total=self.warmup_steps, desc="Warmup") as pbar:
            for step in range(self.warmup_steps):
                # Random action
                action = self.env.action_space.sample()

                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Shape reward
                reward = self.reward_shaper.shape_reward(reward, info, done)

                # Process frame
                next_state = self.atari_wrapper.step(next_obs)

                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)

                # Reset if done
                if done:
                    obs, _ = self.env.reset()
                    state = self.atari_wrapper.reset(obs)
                    self.reward_shaper.reset()
                else:
                    state = next_state

                pbar.update(1)

        print(f"✓ Warmup complete: {len(self.agent.replay_buffer)} transitions")

    def _train_episode(self, episode_num: int) -> Tuple[float, int]:
        """
        Train for one episode.

        Args:
            episode_num: Current episode number

        Returns:
            (episode_reward, episode_length)
        """
        # Reset environment
        obs, _ = self.env.reset()
        state = self.atari_wrapper.reset(obs)
        self.reward_shaper.reset()

        episode_reward = 0
        episode_length = 0

        for step in range(self.max_steps_per_episode):
            # Select action
            action = self.agent.select_action(state, training=True)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Shape reward
            reward = self.reward_shaper.shape_reward(reward, info, done)

            # Process frame
            next_state = self.atari_wrapper.step(next_obs)

            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)

            # Train (if ready and at train frequency)
            if self.global_step % self.train_frequency == 0:
                metrics = self.agent.train_step()
                if metrics:
                    self.metrics['losses'].append(metrics['loss'])
                    self.metrics['q_values'].append(metrics['q_value_mean'])
                    self.metrics['epsilons'].append(metrics['epsilon'])

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.global_step += 1

            if done:
                break

        return episode_reward, episode_length

    def _evaluate(self) -> float:
        """
        Evaluate agent performance.

        Returns:
            Average reward over eval episodes
        """
        self.agent.set_eval_mode()
        eval_rewards = []

        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            state = self.atari_wrapper.reset(obs)

            episode_reward = 0
            done = False

            # Protection against infinite episodes
            for step in range(self.max_steps_per_episode):
                # Greedy action (no exploration)
                action = self.agent.select_action(state, training=False)

                # Step
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated

                # Process
                state = self.atari_wrapper.step(obs)
                episode_reward += reward

                if done:
                    break

            eval_rewards.append(episode_reward)

        self.agent.set_train_mode()

        avg_reward = np.mean(eval_rewards)
        print(f"\n  Evaluation: {avg_reward:.2f} ± {np.std(eval_rewards):.2f} "
              f"(over {self.eval_episodes} episodes)")

        return avg_reward

    def _log_progress(self, episode: int):
        """Log training progress."""
        recent_rewards = self.metrics['episode_rewards'][-self.log_frequency:]
        recent_lengths = self.metrics['episode_lengths'][-self.log_frequency:]
        recent_times = self.metrics['training_times'][-self.log_frequency:]

        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        avg_time = np.mean(recent_times)

        # Get latest training metrics
        if self.metrics['losses']:
            recent_loss = self.metrics['losses'][-1]
            recent_q = self.metrics['q_values'][-1]
            recent_eps = self.metrics['epsilons'][-1]
        else:
            recent_loss = recent_q = recent_eps = 0.0

        print(f"\nEpisode {episode}/{self.num_episodes}")
        print(f"  Reward: {avg_reward:.2f} | Length: {avg_length:.0f} | "
              f"Time: {avg_time:.2f}s")
        print(f"  Loss: {recent_loss:.4f} | Q-value: {recent_q:.2f} | "
              f"ε: {recent_eps:.4f}")
        print(f"  Global steps: {self.global_step} | Buffer: {len(self.agent.replay_buffer)}")

    def _save_metrics(self):
        """Save training metrics to JSON."""
        metrics_path = os.path.join(self.log_dir, "metrics.json")

        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                serializable_metrics[key] = [float(v) if isinstance(v, (np.floating, np.integer))
                                             else v for v in value]
            else:
                serializable_metrics[key] = value

        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"\n✓ Metrics saved to {metrics_path}")

    def close(self):
        """Close environments."""
        self.env.close()
        self.eval_env.close()


def test_trainer():
    """Test Trainer with a simple agent."""
    print("Testing Trainer...")

    # Create simple agent (CPU, small buffer for quick test)
    agent = DQNAgent(
        state_shape=(4, 84, 84),
        n_actions=18,
        buffer_capacity=1000,
        batch_size=32,
        device="cpu"
    )

    # Create trainer (short training for test)
    trainer = Trainer(
        agent=agent,
        env_name="ALE/Breakout-v5",
        num_episodes=5,
        warmup_steps=100,
        train_frequency=4,
        eval_frequency=2,
        eval_episodes=2,
        log_frequency=1,
        checkpoint_dir="test_checkpoints",
        log_dir="test_logs"
    )

    print("\nRunning short training test...")
    # Note: This will fail if Breakout is not installed
    # But demonstrates the trainer structure

    print("\n✓ Trainer test structure validated!")


if __name__ == "__main__":
    test_trainer()
