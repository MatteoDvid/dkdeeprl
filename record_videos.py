"""
Record videos of the trained agent playing Breakout
"""
import gymnasium as gym
import ale_py
import yaml
import numpy as np
from src.agent import DQNAgent
from src.preprocessing import AtariWrapper
import os

# Register ALE
gym.register_envs(ale_py)

print("="*60)
print("RECORDING AGENT VIDEOS")
print("="*60)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load best model
print("\nLoading best model...")
checkpoint_path = "checkpoints/best_model.pth"

# Create agent
agent = DQNAgent(
    state_shape=(4, 84, 84),
    n_actions=4,
    device='cpu'  # Use CPU for video recording
)

agent.load_checkpoint(checkpoint_path)
agent.set_eval_mode()
print("Agent loaded!")

# Create wrapper
wrapper = AtariWrapper(
    frame_height=84,
    frame_width=84,
    num_frames=4,
    crop_top=32
)

# Create video directory
os.makedirs('videos', exist_ok=True)

# Record videos
num_videos = 3
print(f"\nRecording {num_videos} videos...")

for i in range(num_videos):
    print(f"\nVideo {i+1}/{num_videos}:")

    # Create environment with video recording
    env = gym.make(
        'ALE/Breakout-v5',
        render_mode='rgb_array'
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder='videos',
        episode_trigger=lambda x: True,
        name_prefix=f"breakout_best_agent_ep{i+1}"
    )

    # Play one episode
    obs, _ = env.reset()
    state = wrapper.reset(obs)

    episode_reward = 0
    episode_length = 0
    done = False

    while not done and episode_length < 10000:
        # Greedy action (no exploration)
        action = agent.select_action(state, training=False)

        # Step
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Process frame
        state = wrapper.step(obs)

        episode_reward += reward
        episode_length += 1

    env.close()

    print(f"  Reward: {episode_reward:.2f}")
    print(f"  Length: {episode_length} steps")

print("\n" + "="*60)
print("VIDEOS RECORDED SUCCESSFULLY!")
print("="*60)
print(f"\nLocation: videos/")
print(f"Files: breakout_best_agent_ep1.mp4, ep2.mp4, ep3.mp4")
