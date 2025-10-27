"""
Evaluation Script for Breakout DQN Agent

Loads trained agent and evaluates performance.

Features:
- Multiple evaluation runs with statistics
- Video recording of gameplay
- Q-value analysis
- Action distribution analysis
- Performance comparison

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth --episodes 100
    python evaluate.py --checkpoint checkpoints/best_model.pth --render
    python evaluate.py --checkpoint checkpoints/best_model.pth --record
"""

import argparse
import yaml
import numpy as np
import gymnasium as gym
from typing import List, Dict
import os
import json
from collections import Counter
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Register ALE environments
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

from src.agent import DQNAgent
from src.preprocessing import AtariWrapper


class Evaluator:
    """
    Evaluator for trained DQN agents.

    Provides comprehensive evaluation including:
    - Multiple episode runs with statistics
    - Video recording
    - Q-value analysis
    - Action distribution analysis
    """

    def __init__(
        self,
        agent: DQNAgent,
        env_name: str,
        frame_height: int = 84,
        frame_width: int = 84,
        num_frames: int = 4,
        crop_top: int = 30
    ):
        """
        Initialize evaluator.

        Args:
            agent: Trained DQNAgent
            env_name: Gymnasium environment name
            frame_height: Frame height
            frame_width: Frame width
            num_frames: Number of stacked frames
            crop_top: Pixels to crop from top
        """
        self.agent = agent
        self.agent.set_eval_mode()  # Set to evaluation mode (no exploration)

        self.env_name = env_name

        # Preprocessing
        self.atari_wrapper = AtariWrapper(
            frame_height=frame_height,
            frame_width=frame_width,
            num_frames=num_frames,
            crop_top=crop_top
        )

        print(f"Evaluator initialized for {env_name}")

    def evaluate(
        self,
        num_episodes: int = 100,
        render: bool = False,
        record_video: bool = False,
        video_dir: str = "videos",
        analyze_actions: bool = True,
        analyze_q_values: bool = True
    ) -> Dict:
        """
        Evaluate agent over multiple episodes.

        Args:
            num_episodes: Number of evaluation episodes
            render: Render episodes to screen
            record_video: Record videos of episodes
            video_dir: Directory to save videos
            analyze_actions: Track action distribution
            analyze_q_values: Track Q-values

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING AGENT")
        print(f"{'='*60}")
        print(f"Number of episodes: {num_episodes}")
        print(f"Render: {render}")
        print(f"Record video: {record_video}\n")

        # Create environment
        render_mode = 'human' if render else None
        if record_video:
            os.makedirs(video_dir, exist_ok=True)
            env = gym.make(
                self.env_name,
                render_mode='rgb_array'
            )
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda x: True,  # Record all episodes
                name_prefix="breakout_eval"
            )
        else:
            env = gym.make(self.env_name, render_mode=render_mode)

        # Tracking
        episode_rewards = []
        episode_lengths = []
        all_actions = []
        all_q_values = []

        # Run evaluation episodes
        for episode in range(num_episodes):
            obs, _ = env.reset()
            state = self.atari_wrapper.reset(obs)

            episode_reward = 0
            episode_length = 0
            episode_actions = []
            episode_q_values = []

            done = False

            while not done:
                # Select action (greedy, no exploration)
                action = self.agent.select_action(state, training=False)
                episode_actions.append(action)

                # Get Q-values if analyzing
                if analyze_q_values:
                    q_values = self.agent.get_q_values(state)
                    episode_q_values.append(q_values)

                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Process frame
                state = self.atari_wrapper.step(obs)

                episode_reward += reward
                episode_length += 1

            # Store episode stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if analyze_actions:
                all_actions.extend(episode_actions)

            if analyze_q_values:
                all_q_values.extend(episode_q_values)

            # Progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"Avg reward (last 10): {avg_reward:.2f}")

        env.close()

        # Compute statistics
        results = {
            'num_episodes': num_episodes,
            'rewards': {
                'mean': float(np.mean(episode_rewards)),
                'std': float(np.std(episode_rewards)),
                'min': float(np.min(episode_rewards)),
                'max': float(np.max(episode_rewards)),
                'median': float(np.median(episode_rewards)),
                'all': episode_rewards
            },
            'lengths': {
                'mean': float(np.mean(episode_lengths)),
                'std': float(np.std(episode_lengths)),
                'min': int(np.min(episode_lengths)),
                'max': int(np.max(episode_lengths)),
                'all': episode_lengths
            }
        }

        # Action analysis
        if analyze_actions:
            action_counts = Counter(all_actions)
            total_actions = len(all_actions)
            action_distribution = {
                int(action): {
                    'count': count,
                    'percentage': 100.0 * count / total_actions
                }
                for action, count in action_counts.items()
            }
            results['actions'] = action_distribution

        # Q-value analysis
        if analyze_q_values:
            all_q_values = np.array(all_q_values)
            results['q_values'] = {
                'mean': float(np.mean(all_q_values)),
                'std': float(np.std(all_q_values)),
                'min': float(np.min(all_q_values)),
                'max': float(np.max(all_q_values)),
                'per_action_mean': [float(x) for x in np.mean(all_q_values, axis=0)]
            }

        # Print summary
        self._print_results(results)

        return results

    def _print_results(self, results: Dict):
        """Print evaluation results summary."""
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")

        # Rewards
        print(f"\nRewards:")
        print(f"  Mean: {results['rewards']['mean']:.2f} ± {results['rewards']['std']:.2f}")
        print(f"  Min: {results['rewards']['min']:.2f}")
        print(f"  Max: {results['rewards']['max']:.2f}")
        print(f"  Median: {results['rewards']['median']:.2f}")

        # Lengths
        print(f"\nEpisode Lengths:")
        print(f"  Mean: {results['lengths']['mean']:.1f} ± {results['lengths']['std']:.1f}")
        print(f"  Min: {results['lengths']['min']}")
        print(f"  Max: {results['lengths']['max']}")

        # Actions
        if 'actions' in results:
            print(f"\nAction Distribution:")
            sorted_actions = sorted(
                results['actions'].items(),
                key=lambda x: x[1]['percentage'],
                reverse=True
            )
            for action, stats in sorted_actions[:5]:  # Top 5 actions
                print(f"  Action {action}: {stats['percentage']:.1f}% "
                      f"({stats['count']} times)")

        # Q-values
        if 'q_values' in results:
            print(f"\nQ-Values:")
            print(f"  Mean: {results['q_values']['mean']:.2f}")
            print(f"  Std: {results['q_values']['std']:.2f}")
            print(f"  Range: [{results['q_values']['min']:.2f}, "
                  f"{results['q_values']['max']:.2f}]")

        print(f"\n{'='*60}")

    def compare_with_baseline(
        self,
        num_episodes: int = 100
    ) -> Dict:
        """
        Compare agent with random baseline.

        Args:
            num_episodes: Number of episodes for comparison

        Returns:
            Comparison results
        """
        print(f"\n{'='*60}")
        print("BASELINE COMPARISON")
        print(f"{'='*60}")

        # Evaluate trained agent
        print("\nEvaluating trained agent...")
        agent_results = self.evaluate(
            num_episodes=num_episodes,
            render=False,
            record_video=False,
            analyze_actions=False,
            analyze_q_values=False
        )

        # Evaluate random baseline
        print("\nEvaluating random baseline...")
        env = gym.make(self.env_name)
        random_rewards = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            random_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{num_episodes}")

        env.close()

        # Comparison
        random_mean = np.mean(random_rewards)
        random_std = np.std(random_rewards)
        agent_mean = agent_results['rewards']['mean']
        agent_std = agent_results['rewards']['std']

        improvement = ((agent_mean - random_mean) / random_mean) * 100

        comparison = {
            'random': {
                'mean': float(random_mean),
                'std': float(random_std)
            },
            'agent': {
                'mean': float(agent_mean),
                'std': float(agent_std)
            },
            'improvement': float(improvement)
        }

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"\nRandom Baseline:")
        print(f"  Mean: {random_mean:.2f} ± {random_std:.2f}")
        print(f"\nTrained Agent:")
        print(f"  Mean: {agent_mean:.2f} ± {agent_std:.2f}")
        print(f"\nImprovement: {improvement:+.1f}%")
        print(f"{'='*60}")

        return comparison


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_agent_from_checkpoint(
    checkpoint_path: str,
    config_path: str = "config.yaml"
) -> DQNAgent:
    """
    Create agent and load from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        config_path: Path to config file

    Returns:
        Loaded DQNAgent
    """
    # Load config
    config = load_config(config_path)

    # Extract parameters
    preprocessing = config['preprocessing']
    network = config['network']

    # State shape
    state_shape = (
        preprocessing['frame_stack'],
        preprocessing['frame_height'],
        preprocessing['frame_width']
    )

    # Get number of actions
    env = gym.make(config['environment']['name'])
    n_actions = env.action_space.n
    env.close()

    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        network_type=network['type'],
        hidden_size=network['fc_hidden_size'],
        device='auto'
    )

    # Load checkpoint
    agent.load_checkpoint(checkpoint_path)

    return agent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN agent"
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )

    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes to screen'
    )

    parser.add_argument(
        '--record',
        action='store_true',
        help='Record videos of episodes'
    )

    parser.add_argument(
        '--video-dir',
        type=str,
        default='videos',
        help='Directory to save videos'
    )

    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Compare with random baseline'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output file for results'
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    print("="*60)
    print("BREAKOUT DQN EVALUATION")
    print("="*60)

    # Parse arguments
    args = parse_args()

    # Load agent
    print(f"\nLoading agent from: {args.checkpoint}")
    agent = create_agent_from_checkpoint(args.checkpoint, args.config)

    # Load config
    config = load_config(args.config)

    # Create evaluator
    evaluator = Evaluator(
        agent=agent,
        env_name=config['environment']['name'],
        frame_height=config['preprocessing']['frame_height'],
        frame_width=config['preprocessing']['frame_width'],
        num_frames=config['preprocessing']['frame_stack'],
        crop_top=config['preprocessing']['crop_top']
    )

    # Run evaluation
    results = evaluator.evaluate(
        num_episodes=args.episodes,
        render=args.render,
        record_video=args.record,
        video_dir=args.video_dir
    )

    # Baseline comparison
    if args.baseline:
        comparison = evaluator.compare_with_baseline(num_episodes=args.episodes)
        results['baseline_comparison'] = comparison

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")

    if args.record:
        print(f"✓ Videos saved to: {args.video_dir}")


if __name__ == "__main__":
    main()
