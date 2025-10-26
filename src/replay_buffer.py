"""
Experience Replay Buffer for DQN

Stores and samples transitions for training.
"""

import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional


# Define transition structure
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done')
)


class ReplayBuffer:
    """
    Standard uniform experience replay buffer.

    Stores transitions and samples uniformly for training.
    Breaks correlation between consecutive samples.
    """

    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state (C, H, W)
            action: Action taken
            reward: Reward received
            next_state: Next state (C, H, W)
            done: Episode termination flag
        """
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each as numpy array
        """
        # Sample random indices
        batch = random.sample(self.buffer, batch_size)

        # Unpack transitions
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.position = 0

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size


class EfficientReplayBuffer:
    """
    Memory-efficient replay buffer using pre-allocated numpy arrays.

    More efficient than deque for large buffers, especially on GPU.
    """

    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, int, int],
        state_dtype: np.dtype = np.float32
    ):
        """
        Args:
            capacity: Maximum buffer size
            state_shape: Shape of state (C, H, W)
            state_dtype: Data type for states (float32 for normalized frames)
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.position = 0
        self.size = 0

        # Pre-allocate arrays
        self.states = np.zeros((capacity, *state_shape), dtype=state_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=state_dtype)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample random batch."""
        # Random indices
        indices = np.random.choice(self.size, batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


class MultiStepBuffer:
    """
    N-step replay buffer for multi-step learning.

    Computes n-step returns: R_t = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1}
    """

    def __init__(
        self,
        capacity: int,
        n_steps: int = 3,
        gamma: float = 0.99
    ):
        """
        Args:
            capacity: Buffer capacity
            n_steps: Number of steps for n-step returns
            gamma: Discount factor
        """
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma

        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_steps)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition with n-step return calculation."""
        # Add to n-step buffer
        self.n_step_buffer.append(Transition(state, action, reward, next_state, done))

        # Only create transition if we have n steps (or episode ended)
        if len(self.n_step_buffer) == self.n_steps or done:
            # Calculate n-step return
            n_step_return = 0.0
            n_step_state = self.n_step_buffer[0].state
            n_step_action = self.n_step_buffer[0].action

            for i, transition in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * transition.reward
                if transition.done:
                    break

            # Last state and done flag
            last_transition = self.n_step_buffer[-1]
            n_step_next_state = last_transition.next_state
            n_step_done = last_transition.done

            # Store n-step transition
            n_step_transition = Transition(
                n_step_state,
                n_step_action,
                n_step_return,
                n_step_next_state,
                n_step_done
            )
            self.buffer.append(n_step_transition)

            # Clear n-step buffer if episode ended
            if done:
                self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch of n-step transitions."""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


def test_replay_buffer():
    """Test replay buffer implementations."""
    print("Testing Standard ReplayBuffer...")

    buffer = ReplayBuffer(capacity=1000)

    # Dummy state (4 frames, 84x84)
    state = np.random.rand(4, 84, 84).astype(np.float32)
    action = 5
    reward = 1.0
    next_state = np.random.rand(4, 84, 84).astype(np.float32)
    done = False

    # Add transitions
    for i in range(100):
        buffer.push(state, action, reward, next_state, done)

    print(f"  Buffer size: {len(buffer)}")
    print(f"  Is ready (batch_size=32): {buffer.is_ready(32)}")

    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"  Sampled states shape: {states.shape}")
    print(f"  Sampled actions shape: {actions.shape}")

    print("\nTesting EfficientReplayBuffer...")

    efficient_buffer = EfficientReplayBuffer(
        capacity=1000,
        state_shape=(4, 84, 84)
    )

    # Add transitions
    for i in range(100):
        efficient_buffer.push(state, action, reward, next_state, done)

    print(f"  Buffer size: {len(efficient_buffer)}")

    # Sample batch
    states, actions, rewards, next_states, dones = efficient_buffer.sample(32)
    print(f"  Sampled states shape: {states.shape}")

    print("\nTesting MultiStepBuffer...")

    multistep_buffer = MultiStepBuffer(
        capacity=1000,
        n_steps=3,
        gamma=0.99
    )

    # Add transitions
    for i in range(100):
        multistep_buffer.push(state, action, reward, next_state, done)

    print(f"  Buffer size: {len(multistep_buffer)}")

    if len(multistep_buffer) >= 32:
        states, actions, rewards, next_states, dones = multistep_buffer.sample(32)
        print(f"  Sampled states shape: {states.shape}")
        print(f"  Sample n-step reward: {rewards[0]:.3f}")

    print("\n✓ All replay buffer tests passed!")


if __name__ == "__main__":
    test_replay_buffer()
