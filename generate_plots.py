"""
Generate all training plots
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Load metrics
print("Loading metrics...")
with open('logs/metrics.json', 'r') as f:
    metrics = json.load(f)

# Create output directory
os.makedirs('results/figures', exist_ok=True)

print(f"Episodes: {len(metrics['episode_rewards'])}")
print(f"Eval runs: {len(metrics['eval_rewards'])}")

# 1. REWARD PROGRESSION
print("\n1. Plotting reward progression...")
fig, ax = plt.subplots(figsize=(12, 6))
episodes = range(1, len(metrics['episode_rewards']) + 1)
rewards = metrics['episode_rewards']

# Raw rewards
ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')

# Moving average (100 episodes)
window = 100
if len(rewards) >= window:
    ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(rewards) + 1), ma, color='red', linewidth=2, label=f'Moving Average ({window})')

ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Training Reward Progression')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/reward_progression.png', dpi=150)
print("   Saved: results/figures/reward_progression.png")
plt.close()

# 2. EPISODE LENGTH
print("\n2. Plotting episode lengths...")
fig, ax = plt.subplots(figsize=(12, 6))
lengths = metrics['episode_lengths']
ax.plot(episodes, lengths, alpha=0.3, color='green', label='Episode Length')

if len(lengths) >= window:
    ma_length = np.convolve(lengths, np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(lengths) + 1), ma_length, color='darkgreen', linewidth=2, label=f'Moving Average ({window})')

ax.set_xlabel('Episode')
ax.set_ylabel('Steps')
ax.set_title('Episode Length Progression')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/episode_length.png', dpi=150)
print("   Saved: results/figures/episode_length.png")
plt.close()

# 3. LOSS AND Q-VALUES
print("\n3. Plotting loss and Q-values...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Loss
losses = metrics['losses']
train_steps = range(1, len(losses) + 1)
ax1.plot(train_steps, losses, alpha=0.5, color='orange')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# Q-values
q_values = metrics['q_values']
ax2.plot(train_steps, q_values, alpha=0.5, color='purple')
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Mean Q-value')
ax2.set_title('Q-value Evolution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/loss_qvalues.png', dpi=150)
print("   Saved: results/figures/loss_qvalues.png")
plt.close()

# 4. EPSILON DECAY
print("\n4. Plotting epsilon decay...")
fig, ax = plt.subplots(figsize=(12, 6))
epsilons = metrics['epsilons']
ax.plot(train_steps, epsilons, color='red', linewidth=2)
ax.set_xlabel('Training Step')
ax.set_ylabel('Epsilon')
ax.set_title('Epsilon Decay')
ax.grid(True, alpha=0.3)
ax.axhline(y=0.01, color='gray', linestyle='--', label='Minimum (0.01)')
ax.legend()
plt.tight_layout()
plt.savefig('results/figures/epsilon_decay.png', dpi=150)
print("   Saved: results/figures/epsilon_decay.png")
plt.close()

# 5. EVALUATION RESULTS
if len(metrics['eval_rewards']) > 0:
    print("\n5. Plotting evaluation results...")
    fig, ax = plt.subplots(figsize=(12, 6))
    eval_episodes = metrics['eval_episodes']
    eval_rewards = metrics['eval_rewards']

    ax.plot(eval_episodes, eval_rewards, marker='o', color='green', linewidth=2, markersize=8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Evaluation Reward')
    ax.set_title('Evaluation Performance (Greedy Policy)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/evaluation_performance.png', dpi=150)
    print("   Saved: results/figures/evaluation_performance.png")
    plt.close()

# 6. SUMMARY PLOT
print("\n6. Creating summary plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Reward
ax = axes[0, 0]
ax.plot(episodes, rewards, alpha=0.3, color='blue')
if len(rewards) >= window:
    ax.plot(range(window, len(rewards) + 1), ma, color='red', linewidth=2)
ax.set_title('Reward Progression', fontsize=14, fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.grid(True, alpha=0.3)

# Length
ax = axes[0, 1]
ax.plot(episodes, lengths, alpha=0.3, color='green')
if len(lengths) >= window:
    ax.plot(range(window, len(lengths) + 1), ma_length, color='darkgreen', linewidth=2)
ax.set_title('Episode Length', fontsize=14, fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Steps')
ax.grid(True, alpha=0.3)

# Q-values
ax = axes[1, 0]
ax.plot(train_steps, q_values, alpha=0.5, color='purple')
ax.set_title('Q-value Evolution', fontsize=14, fontweight='bold')
ax.set_xlabel('Training Step')
ax.set_ylabel('Mean Q-value')
ax.grid(True, alpha=0.3)

# Epsilon
ax = axes[1, 1]
ax.plot(train_steps, epsilons, color='red', linewidth=2)
ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Epsilon Decay', fontsize=14, fontweight='bold')
ax.set_xlabel('Training Step')
ax.set_ylabel('Epsilon')
ax.grid(True, alpha=0.3)

plt.suptitle('Breakout DQN Training Summary', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/training_summary.png', dpi=150)
print("   Saved: results/figures/training_summary.png")
plt.close()

print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"\nLocation: results/figures/")
print(f"Files created:")
print("  - reward_progression.png")
print("  - episode_length.png")
print("  - loss_qvalues.png")
print("  - epsilon_decay.png")
if len(metrics['eval_rewards']) > 0:
    print("  - evaluation_performance.png")
print("  - training_summary.png")
print("\nFinal Stats:")
print(f"  Episodes trained: {len(rewards)}")
print(f"  Final reward: {rewards[-1]:.2f}")
print(f"  Best reward: {max(rewards):.2f}")
if len(eval_rewards) > 0:
    print(f"  Best eval: {max(eval_rewards):.2f}")
