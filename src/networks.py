"""
Neural Network Architectures for DQN

Implements:
1. Standard DQN (CNN + FC)
2. Dueling DQN (separate value and advantage streams)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DQN(nn.Module):
    """
    Standard Deep Q-Network with CNN feature extractor.

    Architecture:
        Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
        Conv1: 32 filters, 8x8, stride 4 -> (batch, 32, 20, 20)
        Conv2: 64 filters, 4x4, stride 2 -> (batch, 64, 9, 9)
        Conv3: 64 filters, 3x3, stride 1 -> (batch, 64, 7, 7)
        Flatten -> (batch, 3136)
        FC1: 512 neurons
        FC2: n_actions outputs (Q-values)
    """

    def __init__(
        self,
        input_channels: int = 4,
        n_actions: int = 18,
        hidden_size: int = 512
    ):
        """
        Args:
            input_channels: Number of stacked frames (default 4)
            n_actions: Number of possible actions
            hidden_size: Size of hidden fully-connected layer
        """
        super(DQN, self).__init__()

        self.input_channels = input_channels
        self.n_actions = n_actions

        # Convolutional layers (feature extraction)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate size of flattened features
        # For 84x84 input: 84 -> 20 -> 9 -> 7
        self.feature_size = 64 * 7 * 7  # 3136

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input state (batch, channels, height, width)

        Returns:
            Q-values for each action (batch, n_actions)
        """
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract convolutional features (useful for visualization).

        Args:
            x: Input state

        Returns:
            Flattened features after conv layers
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture.

    Separates the Q-value into:
    - V(s): Value of being in state s
    - A(s,a): Advantage of taking action a in state s

    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))

    This architecture learns more efficiently by decoupling value and advantage.
    Particularly effective when many actions have similar Q-values.

    Reference:
    Wang et al. (2016) "Dueling Network Architectures for Deep RL"
    """

    def __init__(
        self,
        input_channels: int = 4,
        n_actions: int = 18,
        hidden_size: int = 512
    ):
        """
        Args:
            input_channels: Number of stacked frames
            n_actions: Number of possible actions
            hidden_size: Size of hidden layers in value/advantage streams
        """
        super(DuelingDQN, self).__init__()

        self.input_channels = input_channels
        self.n_actions = n_actions

        # Shared convolutional layers (feature extraction)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Feature size after convolutions
        self.feature_size = 64 * 7 * 7  # 3136

        # Value stream: V(s)
        self.value_fc1 = nn.Linear(self.feature_size, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)

        # Advantage stream: A(s, a)
        self.advantage_fc1 = nn.Linear(self.feature_size, hidden_size)
        self.advantage_fc2 = nn.Linear(hidden_size, n_actions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture.

        Args:
            x: Input state (batch, channels, height, width)

        Returns:
            Q-values for each action (batch, n_actions)
        """
        # Shared convolutional feature extraction
        features = self._get_features(x)

        # Value stream: V(s)
        value = F.relu(self.value_fc1(features))
        value = self.value_fc2(value)  # (batch, 1)

        # Advantage stream: A(s, a)
        advantage = F.relu(self.advantage_fc1(features))
        advantage = self.advantage_fc2(advantage)  # (batch, n_actions)

        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # Subtracting mean ensures identifiability (unique V and A)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from convolutional layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x

    def get_value_advantage(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get separate value and advantage estimates (for analysis).

        Args:
            x: Input state

        Returns:
            (value, advantage) tensors
        """
        features = self._get_features(x)

        # Value stream
        value = F.relu(self.value_fc1(features))
        value = self.value_fc2(value)

        # Advantage stream
        advantage = F.relu(self.advantage_fc1(features))
        advantage = self.advantage_fc2(advantage)

        return value, advantage


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for NoisyNet-DQN.

    Adds parametric noise to weights and biases for exploration.
    Alternative to epsilon-greedy that learns to explore.

    Reference:
    Fortunato et al. (2018) "Noisy Networks for Exploration"
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            sigma_init: Initial value for noise standard deviation
        """
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters for weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        # Learnable parameters for biases
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learnable, resampled each forward)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        # Initialize parameters
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))

    def reset_noise(self):
        """Sample new noise values."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Factorized Gaussian noise (more efficient)."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


def test_networks():
    """Test network architectures."""
    batch_size = 4
    input_channels = 4
    n_actions = 18

    # Create dummy input (4 stacked 84x84 frames)
    dummy_input = torch.randn(batch_size, input_channels, 84, 84)

    print("Testing Standard DQN...")
    dqn = DQN(input_channels=input_channels, n_actions=n_actions)
    q_values = dqn(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {q_values.shape}")
    print(f"  Number of parameters: {sum(p.numel() for p in dqn.parameters()):,}")

    print("\nTesting Dueling DQN...")
    dueling_dqn = DuelingDQN(input_channels=input_channels, n_actions=n_actions)
    q_values = dueling_dqn(dummy_input)
    value, advantage = dueling_dqn.get_value_advantage(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {q_values.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Advantage shape: {advantage.shape}")
    print(f"  Number of parameters: {sum(p.numel() for p in dueling_dqn.parameters()):,}")

    print("\nTesting NoisyLinear...")
    noisy = NoisyLinear(512, 18)
    x = torch.randn(batch_size, 512)
    output = noisy(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    print("\n✓ All network tests passed!")


if __name__ == "__main__":
    test_networks()
