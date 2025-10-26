"""
DQN Agent Implementation

Implements Double DQN with:
- Target network for stable learning
- Epsilon-greedy or Noisy Networks exploration
- Support for standard or prioritized replay
- Huber loss for robust training
- Gradient clipping for stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import os

from .networks import DQN, DuelingDQN
from .replay_buffer import ReplayBuffer, EfficientReplayBuffer
from .prioritized_replay import PrioritizedReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent with Double DQN algorithm.

    Features:
    - Double DQN: decouples action selection and evaluation
    - Target network: periodic updates for stability
    - Epsilon-greedy exploration
    - Gradient clipping
    - Huber loss (optional)
    - Prioritized Experience Replay (optional)
    """

    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        n_actions: int,
        # Network parameters
        network_type: str = "dueling_dqn",
        hidden_size: int = 512,
        # Algorithm parameters
        gamma: float = 0.99,
        learning_rate: float = 0.00025,
        adam_epsilon: float = 1e-4,
        # Exploration
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,
        # Replay buffer
        buffer_type: str = "prioritized",
        buffer_capacity: int = 100000,
        batch_size: int = 32,
        # Prioritized replay parameters
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 100000,
        per_epsilon: float = 1e-6,
        # Training parameters
        target_update_freq: int = 1000,
        gradient_clip_norm: float = 10.0,
        use_huber_loss: bool = True,
        # Device
        device: str = "auto"
    ):
        """
        Initialize DQN Agent.

        Args:
            state_shape: Shape of state (C, H, W)
            n_actions: Number of actions
            network_type: "dqn" or "dueling_dqn"
            hidden_size: Hidden layer size
            gamma: Discount factor
            learning_rate: Learning rate for Adam
            adam_epsilon: Epsilon for Adam optimizer
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            buffer_type: "standard", "efficient", or "prioritized"
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            per_alpha: Prioritization exponent
            per_beta_start: Initial IS weight
            per_beta_end: Final IS weight
            per_beta_steps: Beta annealing steps
            per_epsilon: Small constant for priorities
            target_update_freq: Steps between target network updates
            gradient_clip_norm: Max gradient norm
            use_huber_loss: Use Huber loss instead of MSE
            device: "cuda", "cpu", or "auto"
        """
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Store parameters
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gradient_clip_norm = gradient_clip_norm
        self.use_huber_loss = use_huber_loss

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        # Networks
        input_channels = state_shape[0]
        NetworkClass = DuelingDQN if network_type == "dueling_dqn" else DQN

        self.online_network = NetworkClass(
            input_channels=input_channels,
            n_actions=n_actions,
            hidden_size=hidden_size
        ).to(self.device)

        self.target_network = NetworkClass(
            input_channels=input_channels,
            n_actions=n_actions,
            hidden_size=hidden_size
        ).to(self.device)

        # Initialize target network with online network weights
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
            eps=adam_epsilon
        )

        # Replay buffer
        self.buffer_type = buffer_type
        if buffer_type == "prioritized":
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_end=per_beta_end,
                beta_annealing_steps=per_beta_steps,
                epsilon=per_epsilon
            )
        elif buffer_type == "efficient":
            self.replay_buffer = EfficientReplayBuffer(
                capacity=buffer_capacity,
                state_shape=state_shape
            )
        else:  # standard
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training step counter
        self.train_step_count = 0

        # Statistics
        self.stats = {
            'loss': [],
            'q_values': [],
            'epsilon': []
        }

        print(f"Initialized {network_type.upper()} Agent with {buffer_type} replay")
        print(f"Network parameters: {self._count_parameters():,}")

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.online_network.parameters() if p.requires_grad)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state (C, H, W) numpy array
            training: If True, uses epsilon-greedy; if False, greedy only

        Returns:
            Action index
        """
        # Exploration (epsilon-greedy)
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Exploitation (greedy)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step (if buffer is ready).

        Returns:
            Dictionary with training metrics, or None if buffer not ready
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample batch
        if self.buffer_type == "prioritized":
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1))
        current_q_values = current_q_values.squeeze(1)

        # Compute target Q-values using Double DQN
        with torch.no_grad():
            # Double DQN: use online network to select action
            next_actions = self.online_network(next_states).argmax(dim=1)

            # Use target network to evaluate action
            next_q_values = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            # Target Q = r + γ * Q_target(s', a*)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute TD-error
        td_error = target_q_values - current_q_values

        # Loss (weighted for prioritized replay)
        if self.use_huber_loss:
            # Huber loss is more robust to outliers
            loss = F.smooth_l1_loss(
                current_q_values,
                target_q_values,
                reduction='none'
            )
        else:
            # MSE loss
            loss = F.mse_loss(
                current_q_values,
                target_q_values,
                reduction='none'
            )

        # Apply importance sampling weights
        loss = (loss * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.gradient_clip_norm
        )

        self.optimizer.step()

        # Update priorities in prioritized replay
        if self.buffer_type == "prioritized":
            self.replay_buffer.update_priorities(
                indices,
                td_error.detach().cpu().numpy()
            )

        # Update target network
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        # Decay epsilon
        self._decay_epsilon()

        # Statistics
        metrics = {
            'loss': loss.item(),
            'q_value_mean': current_q_values.mean().item(),
            'q_value_std': current_q_values.std().item(),
            'epsilon': self.epsilon
        }

        self.stats['loss'].append(metrics['loss'])
        self.stats['q_values'].append(metrics['q_value_mean'])
        self.stats['epsilon'].append(metrics['epsilon'])

        return metrics

    def _decay_epsilon(self):
        """Linearly decay epsilon."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - self.epsilon_decay
        )

    def soft_update_target(self, tau: float = 0.001):
        """
        Soft update of target network (Polyak averaging).

        Alternative to hard updates every N steps.

        θ_target = τ * θ_online + (1 - τ) * θ_target

        Args:
            tau: Interpolation parameter
        """
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.online_network.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1 - tau) * target_param.data
            )

    def save_checkpoint(self, filepath: str):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step_count': self.train_step_count,
            'stats': self.stats
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        Load agent checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step_count = checkpoint['train_step_count']
        self.stats = checkpoint['stats']

        print(f"Checkpoint loaded from {filepath}")
        print(f"  Epsilon: {self.epsilon:.4f}")
        print(f"  Train steps: {self.train_step_count}")

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a state (for analysis).

        Args:
            state: State (C, H, W)

        Returns:
            Q-values for all actions
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state_tensor)
            return q_values.cpu().numpy()[0]

    def set_train_mode(self):
        """Set networks to training mode."""
        self.online_network.train()

    def set_eval_mode(self):
        """Set networks to evaluation mode."""
        self.online_network.eval()


def test_agent():
    """Test DQN Agent."""
    print("Testing DQN Agent...")

    # Create dummy agent
    agent = DQNAgent(
        state_shape=(4, 84, 84),
        n_actions=18,
        network_type="dueling_dqn",
        buffer_type="prioritized",
        buffer_capacity=1000,
        batch_size=32,
        device="cpu"
    )

    print(f"\nAgent epsilon: {agent.epsilon:.3f}")
    print(f"Train step count: {agent.train_step_count}")

    # Dummy state and transition
    state = np.random.rand(4, 84, 84).astype(np.float32)
    action = agent.select_action(state, training=True)
    print(f"\nSelected action (training): {action}")

    action = agent.select_action(state, training=False)
    print(f"Selected action (eval): {action}")

    # Store some transitions
    print("\nStoring 100 transitions...")
    for _ in range(100):
        next_state = np.random.rand(4, 84, 84).astype(np.float32)
        reward = np.random.random()
        done = np.random.random() < 0.1

        agent.store_transition(state, action, reward, next_state, done)
        state = next_state

    print(f"Replay buffer size: {len(agent.replay_buffer)}")

    # Train
    print("\nPerforming 10 training steps...")
    for i in range(10):
        metrics = agent.train_step()
        if metrics:
            print(f"  Step {i+1}: Loss={metrics['loss']:.4f}, "
                  f"Q={metrics['q_value_mean']:.2f}, "
                  f"ε={metrics['epsilon']:.4f}")

    # Test checkpoint saving/loading
    print("\nTesting checkpoint save/load...")
    checkpoint_path = "test_checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Create new agent and load
    agent2 = DQNAgent(
        state_shape=(4, 84, 84),
        n_actions=18,
        device="cpu"
    )
    agent2.load_checkpoint(checkpoint_path)

    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n✓ All agent tests passed!")


if __name__ == "__main__":
    test_agent()
