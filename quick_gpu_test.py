"""
Quick GPU Training Test

Runs a very short training session (10 episodes, no warmup) to verify:
- GPU is being used
- Training loop works
- Agent can learn
"""

import sys
import os


def main():
    print("\n" + "="*70)
    print(" "*15 + "🎮 QUICK GPU TRAINING TEST 🎮")
    print("="*70 + "\n")

    # Check GPU first
    print("Step 1: Checking GPU availability...")
    import torch
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   Please run check_gpu.py for details.")
        return 1

    print(f"✅ GPU found: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Run quick training test
    print("Step 2: Running quick training test (10 episodes, no warmup)...")
    print("This will take approximately 2-5 minutes.\n")

    # Run training with minimal episodes
    exit_code = os.system("python train.py --device cuda --episodes 10 --no-warmup")

    if exit_code == 0:
        print("\n" + "="*70)
        print("✅ GPU TRAINING TEST SUCCESSFUL!")
        print("="*70)
        print("\nYour system is ready for full training.")
        print("\nTo start full training (2000 episodes), run:")
        print("  python train.py")
        print("\nOr use the interactive menu:")
        print("  python main.py")
        print("="*70 + "\n")
        return 0
    else:
        print("\n" + "="*70)
        print("❌ GPU TRAINING TEST FAILED")
        print("="*70)
        print("\nPlease check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
