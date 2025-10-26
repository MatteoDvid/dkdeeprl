# 🚀 GPU Setup & Training Guide

## 📊 Your GPU Configuration

**Detected GPU**: NVIDIA GeForce RTX 2060
**CUDA Version**: 12.9
**GPU Memory**: 6GB
**Status**: ✅ Ready for Training

---

## ✅ What Has Been Configured

### 1. PyTorch with CUDA Support
- ✅ PyTorch 2.5.1+cu121 installed
- ✅ CUDA 12.1 support enabled
- ✅ cuDNN enabled for acceleration

### 2. Optimized Configuration
File: `config.yaml`
- ✅ `device: "cuda"` - GPU enabled by default
- ✅ `batch_size: 64` - Optimized for GPU (was 32)
- ✅ All other settings optimized for GPU training

### 3. Helper Scripts Created
- ✅ `check_gpu.py` - Verify GPU setup
- ✅ `quick_gpu_test.py` - Quick training test

---

## 🎯 Quick Start (3 Steps)

### Step 1: Verify GPU Works
```bash
python check_gpu.py
```

**Expected output:**
```
🚀 GPU VERIFICATION 🚀
✅ CUDA available: True
📊 GPU 0:
   Name: NVIDIA GeForce RTX 2060
   Memory: 6.00 GB
✅ GPU is ready for training!
```

### Step 2: Quick Test (2-5 minutes)
```bash
python quick_gpu_test.py
```

This runs 10 episodes to verify everything works.

### Step 3: Full Training (6-12 hours)
```bash
python train.py
```

That's it! The training will run on GPU automatically.

---

## 📈 Expected Performance

### Your RTX 2060 Performance:
- **10 episodes**: ~2-5 minutes
- **100 episodes**: ~20-50 minutes
- **1000 episodes**: ~3-6 hours
- **2000 episodes (full)**: ~6-12 hours

### Comparison (CPU vs GPU):
| Task | CPU | GPU (RTX 2060) | Speedup |
|------|-----|----------------|---------|
| 10 episodes | 10-15 min | 2-5 min | **3-4x faster** |
| 2000 episodes | 24-48 hours | 6-12 hours | **3-4x faster** |

---

## 🎮 Training Commands

### Interactive Menu (Recommended)
```bash
python main.py
# Select option 2 (Train Agent)
# Choose your training mode
```

### Direct Commands
```bash
# Quick test (verify GPU works)
python quick_gpu_test.py

# Full training (2000 episodes)
python train.py

# Custom episodes
python train.py --episodes 500

# Resume from checkpoint
python train.py --checkpoint checkpoints/best_model.pth
```

---

## 📊 Monitoring GPU During Training

### Windows Task Manager
1. Open Task Manager (Ctrl+Shift+Esc)
2. Go to "Performance" tab
3. Select "GPU 0"
4. Watch utilization (should be 60-90%)

### NVIDIA SMI (Detailed)
```bash
# One-time check
nvidia-smi

# Continuous monitoring (every 2 seconds)
nvidia-smi -l 2
```

**What to look for:**
- GPU Utilization: 60-90% (good)
- Memory Usage: ~2-4GB (normal for this project)
- Temperature: <85°C (safe, RTX 2060 can handle up to 88°C)

---

## 🔥 Optimizations for Your RTX 2060

### Already Applied:
✅ Batch size increased to 64 (from 32)
✅ Device set to CUDA
✅ Efficient replay buffer

### Optional (if needed):

#### If Running Out of Memory:
Edit `config.yaml`:
```yaml
replay_buffer:
  batch_size: 32          # Reduce if out of memory
  capacity: 50000         # Reduce from 100000

advanced:
  use_mixed_precision: true  # Enable to save memory
```

#### To Train Faster:
```yaml
training:
  warmup_steps: 25000     # Reduce from 50000 (faster start)
  train_frequency: 2      # Train more often (from 4)
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1** - Reduce batch size:
```yaml
replay_buffer:
  batch_size: 32  # or even 16
```

**Solution 2** - Enable mixed precision:
```yaml
advanced:
  use_mixed_precision: true
```

**Solution 3** - Close other GPU applications
```bash
# Check what's using GPU
nvidia-smi
# Close unnecessary programs
```

### Issue: GPU not being used (stays at 0%)

**Check 1** - Verify PyTorch CUDA:
```bash
python check_gpu.py
```

**Check 2** - Verify config:
```bash
# Make sure config.yaml has:
device: "cuda"  # NOT "cpu" or "auto"
```

**Check 3** - Force GPU:
```bash
python train.py --device cuda
```

### Issue: Training is slow even on GPU

**Possible causes:**
1. **Warmup phase** - First 50k steps are slow (filling replay buffer)
   - This is normal! Just wait
2. **Small batch size** - Increase to 64 in config
3. **Other programs using GPU** - Close them

---

## 📁 Generated Files

### During Training:
```
checkpoints/
  ├── checkpoint_ep100.pth    # Every 100 episodes
  ├── checkpoint_ep200.pth
  ├── best_model.pth          # Best performing model
  └── final_model.pth         # End of training

logs/
  └── metrics.json            # All training metrics
```

### After Evaluation:
```
videos/
  └── donkey_kong_eval_*.mp4  # Gameplay videos

results/
  └── figures/                 # All plots
```

---

## ⚡ Power Users: Advanced Tips

### 1. Parallel Training (Experimental)
```yaml
advanced:
  parallel_envs: 2  # Run 2 environments in parallel
```

### 2. Model Compilation (PyTorch 2.0+)
```yaml
advanced:
  compile_model: true  # Extra 10-20% speedup
```

### 3. Custom Learning Rate Schedule
Edit `config.yaml`:
```yaml
algorithm:
  learning_rate: 0.0005  # Try different values
```

---

## 🎉 Ready to Train!

Your system is fully configured for GPU training!

**Next steps:**
1. ✅ Run `python check_gpu.py` to verify
2. ✅ Run `python quick_gpu_test.py` for quick test
3. ✅ Run `python train.py` for full training

**Estimated total time: ~6-12 hours for 2000 episodes**

---

## 📞 Need Help?

**Common checks:**
```bash
# 1. Verify GPU
python check_gpu.py

# 2. Check config
cat config.yaml | grep device

# 3. Test training
python quick_gpu_test.py

# 4. Monitor GPU
nvidia-smi -l 2
```

---

**Happy Training! 🚀🎮**

Your RTX 2060 is ready to train a world-class Donkey Kong agent!
