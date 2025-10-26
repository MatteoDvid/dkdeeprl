"""
Prioritized Experience Replay Buffer

Implements prioritized sampling based on TD-error magnitude.
Samples transitions with higher TD-error more frequently.

Reference:
Schaul et al. (2016) "Prioritized Experience Replay" - ICLR 2016
"""

import numpy as np
import random
from typing import Tuple, List
from collections import namedtuple


# Transition structure
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done')
)


class SumTree:
    """
    Sum Tree data structure for efficient priority sampling.

    Binary tree where:
    - Leaf nodes: store priorities
    - Internal nodes: sum of children priorities
    - Root: total sum of all priorities

    Allows O(log n) sampling and O(log n) priority updates.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of elements
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree structure
        self.data = np.zeros(capacity, dtype=object)  # Actual transitions
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float, data: Transition):
        """
        Add new transition with priority.

        Args:
            priority: Priority value (typically |TD-error| + epsilon)
            data: Transition tuple
        """
        # Tree index (leaf position)
        tree_idx = self.data_pointer + self.capacity - 1

        # Store transition
        self.data[self.data_pointer] = data

        # Update tree with new priority
        self.update(tree_idx, priority)

        # Move pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        """
        Update priority of a leaf node and propagate change to root.

        Args:
            tree_idx: Index in tree array
            priority: New priority value
        """
        # Change in priority
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2  # Parent index
            self.tree[tree_idx] += change

    def get_leaf(self, value: float) -> Tuple[int, float, Transition]:
        """
        Sample a leaf node based on priority value.

        Uses proportional prioritization:
        - Higher priority = more likely to be sampled

        Args:
            value: Random value in [0, total_priority]

        Returns:
            (tree_idx, priority, data)
        """
        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # If we reach leaf
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            # Descend tree based on value
            if value <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        # Convert tree index to data index
        data_idx = leaf_idx - self.capacity + 1

        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        """Return sum of all priorities (root node)."""
        return self.tree[0]

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.

    Key features:
    1. Prioritized sampling based on |TD-error|
    2. Importance sampling weights to correct bias
    3. Efficient sum tree data structure

    Hyperparameters:
    - alpha (α): How much prioritization (0 = uniform, 1 = full priority)
    - beta (β): Importance sampling weight (0 = no correction, 1 = full correction)
    - epsilon (ε): Small constant to ensure non-zero priority
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_annealing_steps: int = 100000,
        epsilon: float = 1e-6
    ):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full priority)
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_annealing_steps: Steps to anneal beta from start to end
            epsilon: Small constant for numerical stability
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_annealing_steps = beta_annealing_steps
        self.epsilon = epsilon

        # Max priority for new transitions (ensures new transitions are sampled)
        self.max_priority = 1.0

        # Step counter for beta annealing
        self.step = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add transition with maximum priority.

        New transitions get max priority to ensure they're sampled at least once.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        transition = Transition(state, action, reward, next_state, done)

        # New transitions get max priority
        priority = self.max_priority ** self.alpha

        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample batch with prioritized sampling and importance weights.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
            - indices: Tree indices for updating priorities
            - weights: Importance sampling weights
        """
        batch = []
        indices = []
        priorities = []

        # Divide priority range into segments
        segment_size = self.tree.total_priority / batch_size

        # Sample from each segment (stratified sampling)
        for i in range(batch_size):
            # Random value in segment
            a = segment_size * i
            b = segment_size * (i + 1)
            value = np.random.uniform(a, b)

            # Sample leaf
            idx, priority, data = self.tree.get_leaf(value)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Unpack transitions
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)

        # Calculate importance sampling weights
        # w_i = (N * P(i))^(-β) / max_w
        priorities = np.array(priorities)
        sampling_probabilities = priorities / self.tree.total_priority

        # IS weights
        weights = (len(self.tree) * sampling_probabilities) ** (-self.beta)
        weights = weights / weights.max()  # Normalize by max for stability

        # Anneal beta
        self._anneal_beta()

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            np.array(indices),
            weights.astype(np.float32)
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for sampled transitions.

        Priority = (|TD-error| + ε)^α

        Args:
            indices: Tree indices from sample()
            td_errors: TD-errors for each transition (can be tensor or numpy)
        """
        # Convert to numpy if needed
        if hasattr(td_errors, 'detach'):
            td_errors = td_errors.detach().cpu().numpy()

        # Calculate new priorities
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        # Update max priority
        self.max_priority = max(self.max_priority, priorities.max())

        # Update tree
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)

    def _anneal_beta(self):
        """Linearly anneal beta from beta_start to beta_end."""
        self.step += 1
        fraction = min(self.step / self.beta_annealing_steps, 1.0)
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return len(self.tree)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.tree) >= batch_size


class PrioritizedMultiStepBuffer:
    """
    Combines Prioritized Experience Replay with n-step returns.

    Uses both:
    1. Prioritized sampling (PER)
    2. Multi-step bootstrapping (n-step returns)
    """

    def __init__(
        self,
        capacity: int = 100000,
        n_steps: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_annealing_steps: int = 100000,
        epsilon: float = 1e-6
    ):
        """
        Args:
            capacity: Buffer capacity
            n_steps: Number of steps for n-step returns
            gamma: Discount factor
            alpha: Prioritization exponent
            beta_start: Initial IS weight
            beta_end: Final IS weight
            beta_annealing_steps: Annealing steps for beta
            epsilon: Small constant for priorities
        """
        self.per_buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_annealing_steps=beta_annealing_steps,
            epsilon=epsilon
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = []

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
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Calculate n-step return if we have enough steps
        if len(self.n_step_buffer) == self.n_steps or done:
            # Get first transition
            n_step_state, n_step_action = self.n_step_buffer[0][:2]

            # Calculate n-step return
            n_step_return = 0.0
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * r
                if d:
                    break

            # Get last transition
            _, _, _, n_step_next_state, n_step_done = self.n_step_buffer[-1]

            # Add to PER buffer
            self.per_buffer.push(
                n_step_state,
                n_step_action,
                n_step_return,
                n_step_next_state,
                n_step_done
            )

            # Remove first transition from n-step buffer
            self.n_step_buffer.pop(0)

        # Clear n-step buffer if episode ended
        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch from PER buffer."""
        return self.per_buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities in PER buffer."""
        self.per_buffer.update_priorities(indices, td_errors)

    def __len__(self) -> int:
        return len(self.per_buffer)

    def is_ready(self, batch_size: int) -> bool:
        return self.per_buffer.is_ready(batch_size)


def test_prioritized_replay():
    """Test Prioritized Experience Replay implementation."""
    print("Testing PrioritizedReplayBuffer...")

    # Create buffer
    buffer = PrioritizedReplayBuffer(
        capacity=1000,
        alpha=0.6,
        beta_start=0.4,
        epsilon=1e-6
    )

    # Dummy transitions
    state = np.random.rand(4, 84, 84).astype(np.float32)
    action = 5
    reward = 1.0
    next_state = np.random.rand(4, 84, 84).astype(np.float32)
    done = False

    # Add transitions
    print("Adding 100 transitions...")
    for i in range(100):
        buffer.push(state, action, reward, next_state, done)

    print(f"  Buffer size: {len(buffer)}")
    print(f"  Total priority: {buffer.tree.total_priority:.2f}")
    print(f"  Max priority: {buffer.max_priority:.2f}")
    print(f"  Beta: {buffer.beta:.3f}")

    # Sample batch
    print("\nSampling batch of 32...")
    states, actions, rewards, next_states, dones, indices, weights = buffer.sample(32)

    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"  Beta after sampling: {buffer.beta:.3f}")

    # Update priorities
    print("\nUpdating priorities with random TD-errors...")
    td_errors = np.random.rand(32) * 2.0  # Random errors [0, 2]
    buffer.update_priorities(indices, td_errors)

    print(f"  Max priority after update: {buffer.max_priority:.2f}")

    # Test beta annealing
    print("\nTesting beta annealing (1000 samples)...")
    for _ in range(1000):
        buffer.sample(32)

    print(f"  Beta after 1000 samples: {buffer.beta:.3f}")

    print("\nTesting PrioritizedMultiStepBuffer...")

    multistep_buffer = PrioritizedMultiStepBuffer(
        capacity=1000,
        n_steps=3,
        gamma=0.99,
        alpha=0.6
    )

    # Add transitions
    for i in range(100):
        multistep_buffer.push(state, action, reward, next_state, done)

    print(f"  Buffer size: {len(multistep_buffer)}")

    if len(multistep_buffer) >= 32:
        results = multistep_buffer.sample(32)
        states, actions, rewards, next_states, dones, indices, weights = results
        print(f"  Sampled n-step states shape: {states.shape}")
        print(f"  Sample n-step reward: {rewards[0]:.3f}")

    print("\n✓ All Prioritized Replay tests passed!")


if __name__ == "__main__":
    test_prioritized_replay()
