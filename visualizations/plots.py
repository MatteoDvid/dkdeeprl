"""
Visualization Functions for DQN Training

Creates publication-quality plots for analysis and reporting.

Includes:
- Training curves (rewards, loss, Q-values)
- Moving averages
- Comparison plots
- Action distributions
- Heatmaps
- State space visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
import os


# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_curves(
    metrics: Dict,
    save_path: str = "training_curves.png",
    show: bool = False
):
    """
    Plot comprehensive training curves.

    Args:
        metrics: Dictionary with training metrics
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DQN Training Progress', fontsize=16, fontweight='bold')

    # Episode rewards
    ax = axes[0, 0]
    rewards = metrics['episode_rewards']
    ax.plot(rewards, alpha=0.3, label='Episode Reward')

    # Moving average
    if len(rewards) >= 100:
        ma_100 = moving_average(rewards, window=100)
        ax.plot(range(len(ma_100)), ma_100, linewidth=2, label='MA-100')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode lengths
    ax = axes[0, 1]
    lengths = metrics['episode_lengths']
    ax.plot(lengths, alpha=0.3, label='Episode Length')

    if len(lengths) >= 100:
        ma_100 = moving_average(lengths, window=100)
        ax.plot(range(len(ma_100)), ma_100, linewidth=2, label='MA-100')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1, 0]
    if metrics.get('losses'):
        losses = metrics['losses']
        # Downsample for readability
        if len(losses) > 1000:
            step = len(losses) // 1000
            losses = losses[::step]

        ax.plot(losses, alpha=0.5)
        if len(losses) >= 100:
            ma = moving_average(losses, window=100)
            ax.plot(range(len(ma)), ma, linewidth=2, color='red', label='MA-100')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Q-values and Epsilon
    ax = axes[1, 1]
    if metrics.get('q_values') and metrics.get('epsilons'):
        q_values = metrics['q_values']
        epsilons = metrics['epsilons']

        # Downsample
        if len(q_values) > 1000:
            step = len(q_values) // 1000
            q_values = q_values[::step]
            epsilons = epsilons[::step]

        ax2 = ax.twinx()

        # Q-values
        ax.plot(q_values, color='blue', alpha=0.5, label='Q-value')
        if len(q_values) >= 100:
            ma = moving_average(q_values, window=100)
            ax.plot(range(len(ma)), ma, linewidth=2, color='blue', label='Q-value MA-100')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Q-value', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # Epsilon
        ax2.plot(epsilons, color='red', alpha=0.7, linewidth=2, label='Epsilon')
        ax2.set_ylabel('Epsilon', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title('Q-values and Exploration Rate')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_evaluation_results(
    metrics: Dict,
    save_path: str = "evaluation_results.png",
    show: bool = False
):
    """
    Plot evaluation results over training.

    Args:
        metrics: Dictionary with metrics
        save_path: Path to save figure
        show: Whether to display plot
    """
    if not metrics.get('eval_rewards') or not metrics.get('eval_episodes'):
        print("No evaluation data found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    eval_episodes = metrics['eval_episodes']
    eval_rewards = metrics['eval_rewards']

    ax.plot(eval_episodes, eval_rewards, marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Evaluation Reward', fontsize=12)
    ax.set_title('Evaluation Performance Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add best performance line
    best_reward = max(eval_rewards)
    best_episode = eval_episodes[eval_rewards.index(best_reward)]
    ax.axhline(y=best_reward, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_reward:.2f}')
    ax.axvline(x=best_episode, color='r', linestyle='--', alpha=0.5)

    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Evaluation results saved to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_comparison(
    comparison_data: Dict[str, List[float]],
    labels: List[str],
    save_path: str = "comparison.png",
    show: bool = False
):
    """
    Plot comparison between multiple algorithms/agents.

    Args:
        comparison_data: Dictionary mapping algorithm name to rewards
        labels: List of algorithm names
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')

    # Bar plot of mean rewards
    ax = axes[0]
    means = [np.mean(comparison_data[label]) for label in labels]
    stds = [np.std(comparison_data[label]) for label in labels]

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Average Reward')
    ax.set_title('Mean Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Box plot
    ax = axes[1]
    data = [comparison_data[label] for label in labels]
    ax.boxplot(data, labels=labels)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Reward Distribution')
    ax.set_title('Performance Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_action_distribution(
    action_counts: Dict[int, int],
    save_path: str = "action_distribution.png",
    show: bool = False
):
    """
    Plot distribution of actions taken.

    Args:
        action_counts: Dictionary mapping action to count
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    actions = sorted(action_counts.keys())
    counts = [action_counts[a] for a in actions]
    total = sum(counts)
    percentages = [100 * c / total for c in counts]

    ax.bar(actions, percentages, alpha=0.7)
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Action Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(actions)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for i, (action, pct) in enumerate(zip(actions, percentages)):
        ax.text(action, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Action distribution saved to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_reward_heatmap(
    rewards: List[float],
    episodes_per_row: int = 50,
    save_path: str = "reward_heatmap.png",
    show: bool = False
):
    """
    Plot heatmap of rewards over training.

    Args:
        rewards: List of episode rewards
        episodes_per_row: Number of episodes per row in heatmap
        save_path: Path to save figure
        show: Whether to display plot
    """
    # Reshape rewards into 2D array
    n_episodes = len(rewards)
    n_rows = n_episodes // episodes_per_row
    rewards_trimmed = rewards[:n_rows * episodes_per_row]
    reward_matrix = np.array(rewards_trimmed).reshape(n_rows, episodes_per_row)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(reward_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')

    ax.set_xlabel(f'Episode (within batch of {episodes_per_row})', fontsize=12)
    ax.set_ylabel('Batch Number', fontsize=12)
    ax.set_title('Reward Evolution Heatmap', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Reward', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Reward heatmap saved to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_learning_curve(
    rewards: List[float],
    window: int = 100,
    threshold: Optional[float] = None,
    save_path: str = "learning_curve.png",
    show: bool = False
):
    """
    Plot learning curve with moving average.

    Args:
        rewards: Episode rewards
        window: Window size for moving average
        threshold: Target performance threshold
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    episodes = range(len(rewards))

    # Raw rewards
    ax.plot(episodes, rewards, alpha=0.2, color='blue', label='Episode Reward')

    # Moving average
    if len(rewards) >= window:
        ma = moving_average(rewards, window=window)
        ma_episodes = range(window - 1, len(rewards))
        ax.plot(ma_episodes, ma, linewidth=2, color='blue', label=f'MA-{window}')

    # Threshold line
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Target: {threshold}')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Learning curve saved to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_all_metrics(
    metrics_path: str = "logs/metrics.json",
    output_dir: str = "results/figures",
    show: bool = False
):
    """
    Generate all plots from metrics file.

    Args:
        metrics_path: Path to metrics JSON file
        output_dir: Directory to save plots
        show: Whether to display plots
    """
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}\n")

    # Training curves
    plot_training_curves(
        metrics,
        save_path=os.path.join(output_dir, "training_curves.png"),
        show=show
    )

    # Evaluation results
    if metrics.get('eval_rewards'):
        plot_evaluation_results(
            metrics,
            save_path=os.path.join(output_dir, "evaluation_results.png"),
            show=show
        )

    # Learning curve
    if metrics.get('episode_rewards'):
        plot_learning_curve(
            metrics['episode_rewards'],
            window=100,
            save_path=os.path.join(output_dir, "learning_curve.png"),
            show=show
        )

    # Reward heatmap
    if metrics.get('episode_rewards') and len(metrics['episode_rewards']) >= 50:
        plot_reward_heatmap(
            metrics['episode_rewards'],
            episodes_per_row=50,
            save_path=os.path.join(output_dir, "reward_heatmap.png"),
            show=show
        )

    print(f"\n✓ All plots saved to {output_dir}")


def moving_average(data: List[float], window: int) -> np.ndarray:
    """
    Calculate moving average.

    Args:
        data: Input data
        window: Window size

    Returns:
        Moving average
    """
    return np.convolve(data, np.ones(window) / window, mode='valid')


def create_report_figures(
    metrics_path: str = "logs/metrics.json",
    eval_results_path: str = "evaluation_results.json",
    output_dir: str = "report/figures"
):
    """
    Create all figures needed for the final report.

    Args:
        metrics_path: Path to training metrics
        eval_results_path: Path to evaluation results
        output_dir: Directory for report figures
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("CREATING REPORT FIGURES")
    print(f"{'='*60}\n")

    # Load data
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Main training curves
    plot_training_curves(
        metrics,
        save_path=os.path.join(output_dir, "fig1_training_curves.png")
    )

    # Learning curve
    plot_learning_curve(
        metrics['episode_rewards'],
        window=100,
        save_path=os.path.join(output_dir, "fig2_learning_curve.png")
    )

    # Evaluation results
    if metrics.get('eval_rewards'):
        plot_evaluation_results(
            metrics,
            save_path=os.path.join(output_dir, "fig3_evaluation.png")
        )

    # Load and plot evaluation results if available
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)

        # Action distribution
        if 'actions' in eval_results:
            action_counts = {int(k): v['count'] for k, v in eval_results['actions'].items()}
            plot_action_distribution(
                action_counts,
                save_path=os.path.join(output_dir, "fig4_action_distribution.png")
            )

        # Comparison if baseline data exists
        if 'baseline_comparison' in eval_results:
            comparison = eval_results['baseline_comparison']
            # Create dummy data for comparison plot
            comparison_data = {
                'Random': [comparison['random']['mean']] * 10,
                'DQN Agent': [comparison['agent']['mean']] * 10
            }
            plot_comparison(
                comparison_data,
                labels=['Random', 'DQN Agent'],
                save_path=os.path.join(output_dir, "fig5_comparison.png")
            )

    print(f"\n✓ Report figures created in {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Visualization module")
    print("Use plot_all_metrics() to generate plots from training metrics")
