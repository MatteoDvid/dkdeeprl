"""
Quick GPU Check Script

Verifies CUDA is working and provides GPU information.
"""

import torch
import sys


def check_gpu():
    """Check GPU availability and details."""
    print("\n" + "="*70)
    print(" "*20 + "🚀 GPU VERIFICATION 🚀")
    print("="*70 + "\n")

    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        print("\n❌ CUDA is NOT available!")
        print("   PyTorch was installed without CUDA support.")
        print("\n   To fix this, run:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

    # CUDA details
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # GPU details
    for i in range(torch.cuda.device_count()):
        print(f"\n📊 GPU {i}:")
        print(f"   Name: {torch.cuda.get_device_name(i)}")
        print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(i)}")

    # Test tensor operations
    print("\n🧪 Testing GPU operations...")
    try:
        # Create tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()

        # Perform operation
        z = torch.matmul(x, y)

        print("✅ GPU tensor operations successful!")
        print(f"   Tensor device: {z.device}")
        print(f"   Tensor shape: {z.shape}")

    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        return False

    # Memory info
    print("\n💾 GPU Memory Status:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9

        print(f"   GPU {i}:")
        print(f"      Allocated: {allocated:.2f} GB")
        print(f"      Reserved:  {reserved:.2f} GB")
        print(f"      Total:     {total:.2f} GB")
        print(f"      Free:      {total - reserved:.2f} GB")

    print("\n" + "="*70)
    print("✅ GPU is ready for training!")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
