"""
Test Environment Setup

Verifies that:
1. All dependencies are installed
2. Breakout environment works
3. Preprocessing pipeline works
4. Network architectures work
5. Agent can be created and run
"""

import sys
import os


def test_imports():
    """Test all required imports."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)

    modules = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'gymnasium': 'Gymnasium',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm'
    }

    all_ok = True
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError as e:
            print(f"✗ {name} NOT installed: {e}")
            all_ok = False

    return all_ok


def test_atari_roms():
    """Test if Atari ROMs are installed."""
    print("\n" + "="*60)
    print("TESTING ATARI ROMS")
    print("="*60)

    try:
        import gymnasium as gym
        import ale_py

        # Try to create Breakout environment
        env = gym.make('ALE/Breakout-v5')
        print("✓ Breakout environment available")

        # Get environment info
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")

        # Test reset
        obs, info = env.reset()
        print(f"  Initial observation shape: {obs.shape}")

        # Test step
        obs, reward, terminated, truncated, info = env.step(0)
        print(f"  Step successful")

        env.close()
        return True

    except Exception as e:
        print(f"✗ Breakout environment error: {e}")
        print("\nTo install Atari ROMs, run:")
        print("  pip install 'gymnasium[atari]' ale-py")
        print("  pip install autorom[accept-rom-license]")
        print("  Or manually: python -m atari_py.import_roms <path_to_roms>")
        return False


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\n" + "="*60)
    print("TESTING PREPROCESSING")
    print("="*60)

    try:
        from src.preprocessing import FramePreprocessor, FrameStack, AtariWrapper
        import numpy as np

        # Create dummy frame (simulating Atari output)
        dummy_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)

        # Test preprocessor
        preprocessor = FramePreprocessor()
        processed = preprocessor(dummy_frame)
        print(f"✓ FramePreprocessor works")
        print(f"  Input shape: {dummy_frame.shape}")
        print(f"  Output shape: {processed.shape}")

        # Test frame stack
        frame_stack = FrameStack(num_frames=4)
        stacked = frame_stack.reset(processed)
        print(f"✓ FrameStack works")
        print(f"  Stacked shape: {stacked.shape}")

        # Test wrapper
        wrapper = AtariWrapper()
        state = wrapper.reset(dummy_frame)
        print(f"✓ AtariWrapper works")
        print(f"  State shape: {state.shape}")

        return True

    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_networks():
    """Test network architectures."""
    print("\n" + "="*60)
    print("TESTING NETWORKS")
    print("="*60)

    try:
        import torch
        from src.networks import DQN, DuelingDQN

        # Test input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 4, 84, 84)

        # Test DQN
        dqn = DQN(input_channels=4, n_actions=18)
        output = dqn(dummy_input)
        print(f"✓ DQN works")
        print(f"  Input: {dummy_input.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in dqn.parameters()):,}")

        # Test Dueling DQN
        dueling_dqn = DuelingDQN(input_channels=4, n_actions=18)
        output = dueling_dqn(dummy_input)
        value, advantage = dueling_dqn.get_value_advantage(dummy_input)
        print(f"✓ DuelingDQN works")
        print(f"  Output: {output.shape}")
        print(f"  Value: {value.shape}")
        print(f"  Advantage: {advantage.shape}")
        print(f"  Parameters: {sum(p.numel() for p in dueling_dqn.parameters()):,}")

        return True

    except Exception as e:
        print(f"✗ Network error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_replay_buffers():
    """Test replay buffers."""
    print("\n" + "="*60)
    print("TESTING REPLAY BUFFERS")
    print("="*60)

    try:
        import numpy as np
        from src.replay_buffer import ReplayBuffer
        from src.prioritized_replay import PrioritizedReplayBuffer

        # Test standard buffer
        buffer = ReplayBuffer(capacity=1000)
        state = np.random.rand(4, 84, 84).astype(np.float32)

        for i in range(100):
            buffer.push(state, 0, 1.0, state, False)

        batch = buffer.sample(32)
        print(f"✓ ReplayBuffer works")
        print(f"  Buffer size: {len(buffer)}")
        print(f"  Batch size: {len(batch)}")

        # Test prioritized buffer
        per_buffer = PrioritizedReplayBuffer(capacity=1000)

        for i in range(100):
            per_buffer.push(state, 0, 1.0, state, False)

        batch = per_buffer.sample(32)
        print(f"✓ PrioritizedReplayBuffer works")
        print(f"  Buffer size: {len(per_buffer)}")
        print(f"  Batch components: {len(batch)}")  # Should be 7 (includes indices and weights)

        return True

    except Exception as e:
        print(f"✗ Replay buffer error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent():
    """Test DQN Agent."""
    print("\n" + "="*60)
    print("TESTING AGENT")
    print("="*60)

    try:
        import numpy as np
        from src.agent import DQNAgent

        # Create agent
        agent = DQNAgent(
            state_shape=(4, 84, 84),
            n_actions=18,
            network_type="dueling_dqn",
            buffer_type="prioritized",
            buffer_capacity=1000,
            device="cpu"
        )

        print(f"✓ Agent created")
        print(f"  Network type: dueling_dqn")
        print(f"  Buffer type: prioritized")
        print(f"  Device: {agent.device}")

        # Test action selection
        state = np.random.rand(4, 84, 84).astype(np.float32)
        action = agent.select_action(state, training=True)
        print(f"✓ Action selection works")
        print(f"  Selected action: {action}")

        # Test storing transitions
        for i in range(100):
            next_state = np.random.rand(4, 84, 84).astype(np.float32)
            agent.store_transition(state, action, 1.0, next_state, False)
            state = next_state

        print(f"✓ Transition storage works")
        print(f"  Buffer size: {len(agent.replay_buffer)}")

        # Test training
        metrics = agent.train_step()
        if metrics:
            print(f"✓ Training step works")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Q-value: {metrics['q_value_mean']:.2f}")

        return True

    except Exception as e:
        print(f"✗ Agent error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration file."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)

    try:
        import yaml

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        print(f"✓ Configuration file loaded")
        print(f"  Environment: {config['environment']['name']}")
        print(f"  Network: {config['network']['type']}")
        print(f"  Episodes: {config['training']['num_episodes']}")

        return True

    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_full_integration():
    """Test full integration with actual Breakout environment."""
    print("\n" + "="*60)
    print("TESTING FULL INTEGRATION")
    print("="*60)

    try:
        import gymnasium as gym
        import numpy as np
        from src.agent import DQNAgent
        from src.preprocessing import AtariWrapper

        # Create environment
        env = gym.make('ALE/Breakout-v5')
        print(f"✓ Environment created")

        # Create wrapper
        wrapper = AtariWrapper()

        # Create agent
        agent = DQNAgent(
            state_shape=(4, 84, 84),
            n_actions=env.action_space.n,
            buffer_capacity=1000,
            device="cpu"
        )
        print(f"✓ Agent created")

        # Run one episode
        obs, _ = env.reset()
        state = wrapper.reset(obs)
        print(f"✓ Episode reset successful")
        print(f"  State shape: {state.shape}")

        total_reward = 0
        for step in range(100):  # Short test episode
            action = agent.select_action(state, training=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = wrapper.step(obs)
            agent.store_transition(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"✓ Episode completed")
        print(f"  Steps: {step + 1}")
        print(f"  Reward: {total_reward}")
        print(f"  Buffer size: {len(agent.replay_buffer)}")

        # Test training
        if len(agent.replay_buffer) >= agent.batch_size:
            metrics = agent.train_step()
            print(f"✓ Training step successful")

        env.close()
        return True

    except Exception as e:
        print(f"✗ Integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*15 + "BREAKOUT DQN - ENVIRONMENT TESTS")
    print("="*70)

    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['atari_roms'] = test_atari_roms()
    results['preprocessing'] = test_preprocessing()
    results['networks'] = test_networks()
    results['replay_buffers'] = test_replay_buffers()
    results['agent'] = test_agent()
    results['config'] = test_config()
    results['integration'] = test_full_integration()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nYou can now start training with:")
        print("  python train.py")
        print("\nOr run a quick test with:")
        print("  python train.py --episodes 10 --no-warmup")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease fix the issues before training.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Install Atari ROMs: pip install autorom[accept-rom-license]")
        print("  - Check that all files are in the correct directories")
        return 1


if __name__ == "__main__":
    exit(main())
