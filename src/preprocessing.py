"""
Frame Preprocessing Module
Handles frame transformation and stacking for Atari environments
"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple


class FramePreprocessor:
    """
    Preprocesses raw Atari frames into network-ready format.

    Transformations:
    1. RGB to Grayscale (weighted average)
    2. Crop (remove score area)
    3. Resize to target size (84x84)
    4. Normalize to [0, 1]
    """

    def __init__(
        self,
        frame_height: int = 84,
        frame_width: int = 84,
        crop_top: int = 30,
        grayscale: bool = True,
        normalize: bool = True
    ):
        """
        Args:
            frame_height: Target height after resize
            frame_width: Target width after resize
            crop_top: Number of pixels to crop from top (removes score)
            grayscale: Convert to grayscale if True
            normalize: Normalize to [0, 1] if True
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.crop_top = crop_top
        self.grayscale = grayscale
        self.normalize = normalize

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.

        Args:
            frame: Raw frame from environment (H, W, C) uint8

        Returns:
            Preprocessed frame (H, W) float32
        """
        # Convert to grayscale using weighted average (perceptual luminance)
        if self.grayscale and len(frame.shape) == 3:
            frame = self._rgb_to_gray(frame)

        # Crop top (remove score/UI)
        if self.crop_top > 0:
            frame = frame[self.crop_top:, :]

        # Resize to target size
        frame = cv2.resize(
            frame,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_AREA
        )

        # Normalize to [0, 1]
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0

        return frame

    @staticmethod
    def _rgb_to_gray(frame: np.ndarray) -> np.ndarray:
        """
        Convert RGB frame to grayscale using perceptual luminance weights.

        Args:
            frame: RGB frame (H, W, 3)

        Returns:
            Grayscale frame (H, W)
        """
        # ITU-R BT.601 luma coefficients
        weights = np.array([0.299, 0.587, 0.114])
        gray = np.dot(frame, weights)
        return gray.astype(np.uint8)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Allow calling instance as function."""
        return self.preprocess(frame)


class FrameStack:
    """
    Stacks multiple consecutive frames to provide temporal information.

    This allows the agent to infer velocity and direction from static frames.
    """

    def __init__(
        self,
        num_frames: int = 4,
        frame_shape: Tuple[int, int] = (84, 84)
    ):
        """
        Args:
            num_frames: Number of frames to stack
            frame_shape: Shape of each preprocessed frame (H, W)
        """
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.frames = deque(maxlen=num_frames)

    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        """
        Reset the frame stack with copies of the initial frame.

        Args:
            initial_frame: First frame of episode (H, W)

        Returns:
            Stacked frames (num_frames, H, W)
        """
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(initial_frame)
        return self.get_state()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """
        Add a new frame to the stack.

        Args:
            frame: New preprocessed frame (H, W)

        Returns:
            Updated stacked frames (num_frames, H, W)
        """
        self.frames.append(frame)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Get the current stacked state.

        Returns:
            Stacked frames (num_frames, H, W) as numpy array
        """
        return np.stack(list(self.frames), axis=0)

    def __len__(self) -> int:
        """Return number of frames currently in stack."""
        return len(self.frames)


class AtariWrapper:
    """
    Combines frame preprocessing and stacking into a single wrapper.

    This provides a clean interface for the agent to interact with.
    """

    def __init__(
        self,
        frame_height: int = 84,
        frame_width: int = 84,
        num_frames: int = 4,
        crop_top: int = 30,
        grayscale: bool = True,
        normalize: bool = True
    ):
        """
        Args:
            frame_height: Target height after resize
            frame_width: Target width after resize
            num_frames: Number of frames to stack
            crop_top: Pixels to crop from top
            grayscale: Convert to grayscale
            normalize: Normalize to [0, 1]
        """
        self.preprocessor = FramePreprocessor(
            frame_height=frame_height,
            frame_width=frame_width,
            crop_top=crop_top,
            grayscale=grayscale,
            normalize=normalize
        )

        self.frame_stack = FrameStack(
            num_frames=num_frames,
            frame_shape=(frame_height, frame_width)
        )

    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        """
        Reset with a new episode's first frame.

        Args:
            initial_frame: Raw frame from environment

        Returns:
            Stacked state ready for network (num_frames, H, W)
        """
        processed_frame = self.preprocessor(initial_frame)
        return self.frame_stack.reset(processed_frame)

    def step(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a new frame and update the stack.

        Args:
            frame: Raw frame from environment

        Returns:
            Updated stacked state (num_frames, H, W)
        """
        processed_frame = self.preprocessor(frame)
        return self.frame_stack.push(processed_frame)

    @property
    def state_shape(self) -> Tuple[int, int, int]:
        """Return the shape of the stacked state."""
        return (
            self.frame_stack.num_frames,
            self.preprocessor.frame_height,
            self.preprocessor.frame_width
        )


class RewardShaper:
    """
    Optional reward shaping to provide denser rewards.

    WARNING: Reward shaping can change the optimal policy!
    Document clearly if used in experiments.
    """

    def __init__(
        self,
        clip_rewards: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        custom_shaping: bool = False
    ):
        """
        Args:
            clip_rewards: Clip rewards to range
            clip_range: (min, max) range for clipping
            custom_shaping: Enable custom reward shaping logic
        """
        self.clip_rewards = clip_rewards
        self.clip_range = clip_range
        self.custom_shaping = custom_shaping
        self.prev_info = {}

    def shape_reward(
        self,
        raw_reward: float,
        info: dict,
        done: bool
    ) -> float:
        """
        Apply reward shaping.

        Args:
            raw_reward: Original reward from environment
            info: Info dict from environment
            done: Episode termination flag

        Returns:
            Shaped reward
        """
        shaped_reward = raw_reward

        # Custom shaping (game specific)
        if self.custom_shaping:
            # Bonus for vertical progress (climbing)
            if 'y_position' in info and 'y_position' in self.prev_info:
                delta_y = info['y_position'] - self.prev_info['y_position']
                shaped_reward += 0.01 * delta_y

            # Small penalty for inactivity (encourages exploration)
            if raw_reward == 0 and not done:
                shaped_reward -= 0.001

        # Clip rewards for stability
        if self.clip_rewards:
            shaped_reward = np.clip(
                shaped_reward,
                self.clip_range[0],
                self.clip_range[1]
            )

        # Store info for next step
        self.prev_info = info.copy()

        return shaped_reward

    def reset(self):
        """Reset internal state at episode start."""
        self.prev_info = {}


# Utility functions

def visualize_preprocessing(
    raw_frame: np.ndarray,
    preprocessor: FramePreprocessor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visualize the preprocessing pipeline (useful for debugging).

    Args:
        raw_frame: Original frame from environment
        preprocessor: FramePreprocessor instance

    Returns:
        (raw_frame, preprocessed_frame) for side-by-side comparison
    """
    processed = preprocessor(raw_frame)
    return raw_frame, processed


def test_preprocessing_pipeline():
    """
    Test the preprocessing pipeline with a dummy frame.
    """
    # Create dummy RGB frame (simulating Atari)
    dummy_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)

    # Initialize preprocessor
    preprocessor = FramePreprocessor(
        frame_height=84,
        frame_width=84,
        crop_top=30,
        grayscale=True,
        normalize=True
    )

    # Process frame
    processed = preprocessor(dummy_frame)

    print(f"Original shape: {dummy_frame.shape}, dtype: {dummy_frame.dtype}")
    print(f"Processed shape: {processed.shape}, dtype: {processed.dtype}")
    print(f"Processed range: [{processed.min():.3f}, {processed.max():.3f}]")

    # Test frame stack
    frame_stack = FrameStack(num_frames=4, frame_shape=(84, 84))
    stacked = frame_stack.reset(processed)

    print(f"Stacked shape: {stacked.shape}")

    # Test wrapper
    wrapper = AtariWrapper(num_frames=4)
    state = wrapper.reset(dummy_frame)

    print(f"Wrapper state shape: {state.shape}")
    print(f"Wrapper state_shape property: {wrapper.state_shape}")

    print("✓ Preprocessing pipeline test passed!")


if __name__ == "__main__":
    # Run tests when executed directly
    test_preprocessing_pipeline()
