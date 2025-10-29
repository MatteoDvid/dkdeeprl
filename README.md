# Breakout Deep Reinforcement Learning

Implementation of Double DQN with Dueling Architecture and Prioritized Experience Replay for Atari Breakout.

## Authors

- Matteo David
- Kirsten Chang

---

## Features

### Advanced Algorithms
- **Double DQN**: Reduces Q-value overestimation
- **Dueling Architecture**: Separates value and advantage streams
- **Prioritized Experience Replay**: Samples important transitions more frequently
- **Huber Loss**: Robust training
- **Gradient Clipping**: Training stability
- **Target Network**: Periodic updates

### Complete Pipeline
- **Frame Preprocessing**: Grayscale, resize, normalize, crop
- **Frame Stacking**: 4 frames for temporal information
- **Reward Shaping**: Optional custom rewards
- **Training Loop**: Warmup, periodic evaluation, checkpointing
- **Evaluation**: Statistics, video recording, baseline comparison

### Visualization & Analysis
- **Training Curves**: Rewards, loss, Q-values, epsilon
- **Learning Curves**: With moving averages
- **Heatmaps**: Reward evolution visualization
- **Action Distribution**: Analysis of agent behavior
- **Comparison Plots**: Agent vs baseline

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone repository
cd projetintermediatedeepl

# Install dependencies
pip install -r requirements.txt

# Install Atari ROMs (required)
pip install autorom[accept-rom-license]

# Or manually import ROMs
# python -m atari_py.import_roms <path_to_roms>
```

### Verify Installation

```bash
python test_environment.py
```

This will test:
- All dependencies
- Breakout environment
- Preprocessing pipeline
- Network architectures
- Agent functionality
- Full integration

---

## Quick Start

### Training

```bash
# Default training (uses config.yaml)
python train.py

# Custom parameters
python train.py --episodes 1000 --device cuda

# Resume from checkpoint
python train.py --checkpoint checkpoints/best_model.pth

# Quick test (10 episodes, no warmup)
python train.py --episodes 10 --no-warmup
```

### Evaluation

```bash
# Evaluate trained agent
python evaluate.py --checkpoint checkpoints/best_model.pth --episodes 100

# With video recording
python evaluate.py --checkpoint checkpoints/best_model.pth --record

# With rendering (watch agent play)
python evaluate.py --checkpoint checkpoints/best_model.pth --render

# Compare with random baseline
python evaluate.py --checkpoint checkpoints/best_model.pth --baseline
```

### Visualization

```python
from visualizations.plots import plot_all_metrics, create_report_figures

# Generate all plots
plot_all_metrics(metrics_path="logs/metrics.json", output_dir="results/figures")

# Create report figures
create_report_figures(
    metrics_path="logs/metrics.json",
    eval_results_path="evaluation_results.json",
    output_dir="report/figures"
)
```

---

## Project Structure

```
projetintermediatedeepl/
   README.md                    # This file
   plan.md                      # Detailed project plan
   summary.md                   # Code analysis and patterns
   config.yaml                  # Hyperparameters configuration
   requirements.txt             # Dependencies

   train.py                     # Main training script
   evaluate.py                  # Evaluation script
   test_environment.py          # Environment tests
   main.py                      # Entry point

   src/                         # Core implementation
      __init__.py
      preprocessing.py         # Frame preprocessing & stacking
      networks.py              # DQN & Dueling DQN architectures
      replay_buffer.py         # Standard replay buffers
      prioritized_replay.py    # Prioritized Experience Replay
      agent.py                 # DQN Agent (Double DQN)
      trainer.py               # Training loop manager

   visualizations/              # Plotting & analysis
      __init__.py
      plots.py                 # All visualization functions

   checkpoints/                 # Saved models (created during training)
   logs/                        # Training logs & metrics
   results/                     # Evaluation results & figures
   report/                      # Report figures
```

---

## Configuration

Edit `config.yaml` to customize:

### Environment
```yaml
environment:
  name: "ALE/Breakout-v5"
  frame_skip: 4
```

### Network
```yaml
network:
  type: "dueling_dqn"  # or "dqn"
  fc_hidden_size: 512
```

### Training
```yaml
training:
  num_episodes: 2000
  warmup_steps: 50000
  train_frequency: 4
  target_network_update_freq: 1000
```

### Replay Buffer
```yaml
replay_buffer:
  type: "prioritized"  # or "standard", "efficient"
  capacity: 100000
  batch_size: 32
  per_alpha: 0.6
```

See `config.yaml` for all options.

---

## Results & Metrics

### During Training

The trainer automatically:
- Saves checkpoints every N episodes
- Evaluates every N episodes
- Logs metrics to JSON
- Tracks: rewards, lengths, loss, Q-values, epsilon

### After Training

Generated files:
- `checkpoints/best_model.pth` - Best performing model
- `checkpoints/final_model.pth` - Final model
- `logs/metrics.json` - All training metrics
- `evaluation_results.json` - Evaluation statistics

### Visualization

```bash
# Generate all plots
python -c "from visualizations.plots import plot_all_metrics; plot_all_metrics()"

# Create report figures
python -c "from visualizations.plots import create_report_figures; create_report_figures()"
```

---

## Testing Individual Components

```bash
# Test preprocessing
python -c "from src.preprocessing import test_preprocessing_pipeline; test_preprocessing_pipeline()"

# Test networks
python -c "from src.networks import test_networks; test_networks()"

# Test replay buffers
python -c "from src.replay_buffer import test_replay_buffer; test_replay_buffer()"

# Test prioritized replay
python -c "from src.prioritized_replay import test_prioritized_replay; test_prioritized_replay()"

# Test agent
python -c "from src.agent import test_agent; test_agent()"
```

---

## Technical Details

### Double DQN Algorithm

Standard DQN suffers from Q-value overestimation. Double DQN fixes this by:

1. **Action Selection**: Use online network
   ```
   a* = argmax_a Q_online(s', a)
   ```

2. **Q-value Evaluation**: Use target network
   ```
   y = r + gamma * Q_target(s', a*)
   ```

### Dueling Architecture

Separates Q-value into:
- **V(s)**: Value of state s
- **A(s,a)**: Advantage of action a

```
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
```

Benefits:
- Learns value function independently
- More efficient learning
- Better performance when many actions have similar Q-values

### Prioritized Experience Replay

Samples transitions based on TD-error magnitude:

```
Priority = (|TD-error| + epsilon)^alpha
P(i) = priority_i / sum_k(priority_k)
```

Importance sampling weights correct for bias:
```
w_i = (N * P(i))^(-beta)
```

---

## Performance Expectations

### Random Agent
- Average reward: ~50-100
- Mostly random movement

### Trained DQN
- Average reward: 800-1500 (after 1000 episodes)
- Average reward: 1500-2500 (after 2000 episodes)

### Human Expert
- Average reward: 2000-8000

Note: Results vary based on hyperparameters and training duration.

---

## Troubleshooting

### Atari ROMs Not Found
```bash
pip install autorom[accept-rom-license]
# Or manually:
# python -m atari_py.import_roms <path>
```

### CUDA Out of Memory
- Reduce batch size in `config.yaml`
- Use smaller replay buffer
- Disable mixed precision training

### Slow Training
- Enable GPU: Set `device: cuda` in config
- Reduce warmup steps for testing
- Use fewer evaluation episodes

### Import Errors
```bash
pip install -r requirements.txt
```

---

## References

### Papers
1. **DQN**: Mnih et al. (2015) - "Human-level control through deep RL"
2. **Double DQN**: van Hasselt et al. (2016) - "Deep RL with Double Q-learning"
3. **Dueling DQN**: Wang et al. (2016) - "Dueling Network Architectures"
4. **PER**: Schaul et al. (2016) - "Prioritized Experience Replay"
5. **Rainbow**: Hessel et al. (2018) - "Rainbow: Combining Improvements in Deep RL"

### Resources
- Gymnasium Docs: https://gymnasium.farama.org/
- Atari Learning Environment: https://github.com/mgbellemare/Arcade-Learning-Environment
- PyTorch RL Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

---

## License

This project is for educational purposes.
