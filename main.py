"""
Breakout DQN - Main Entry Point

Provides a simple menu interface for:
- Testing the environment
- Training the agent
- Evaluating a trained agent
- Generating visualizations
"""

import sys
import os


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print(" " * 15 + "🧱 BREAKOUT DEEP REINFORCEMENT LEARNING 🧱")
    print("="*70)
    print("\n  Implementation: Double DQN + Dueling + Prioritized Replay")
    print("  Environment: ALE/Breakout-v5")
    print("="*70 + "\n")


def print_menu():
    """Print main menu."""
    print("\nWhat would you like to do?\n")
    print("  1. Test Environment Setup")
    print("     └─ Verify all dependencies and environment")
    print()
    print("  2. Train Agent")
    print("     └─ Start training (full or quick test)")
    print()
    print("  3. Evaluate Agent")
    print("     └─ Evaluate trained agent and generate statistics")
    print()
    print("  4. Generate Visualizations")
    print("     └─ Create plots from training metrics")
    print()
    print("  5. Show Project Info")
    print("     └─ Display project structure and documentation")
    print()
    print("  0. Exit")
    print()


def test_environment():
    """Run environment tests."""
    print("\n" + "="*70)
    print("Running environment tests...")
    print("="*70 + "\n")

    import test_environment
    return test_environment.main()


def train_agent():
    """Train agent with options."""
    print("\n" + "="*70)
    print("TRAINING OPTIONS")
    print("="*70 + "\n")

    print("1. Full training (2000 episodes, ~12-24 hours)")
    print("2. Quick test (10 episodes, ~5-10 minutes)")
    print("3. Resume from checkpoint")
    print("4. Custom parameters")
    print("0. Back to main menu")

    choice = input("\nSelect option: ").strip()

    if choice == "1":
        os.system("python train.py")
    elif choice == "2":
        os.system("python train.py --episodes 10 --no-warmup")
    elif choice == "3":
        checkpoint = input("Enter checkpoint path: ").strip()
        os.system(f"python train.py --checkpoint {checkpoint}")
    elif choice == "4":
        episodes = input("Number of episodes (default 2000): ").strip() or "2000"
        device = input("Device (cuda/cpu/auto, default auto): ").strip() or "auto"
        os.system(f"python train.py --episodes {episodes} --device {device}")


def evaluate_agent():
    """Evaluate trained agent."""
    print("\n" + "="*70)
    print("EVALUATION OPTIONS")
    print("="*70 + "\n")

    # Check for available checkpoints
    if os.path.exists("checkpoints"):
        checkpoints = [f for f in os.listdir("checkpoints") if f.endswith('.pth')]
        if checkpoints:
            print("Available checkpoints:")
            for i, cp in enumerate(checkpoints, 1):
                print(f"  {i}. {cp}")
            print()

    checkpoint = input("Enter checkpoint path (e.g., checkpoints/best_model.pth): ").strip()

    if not checkpoint or not os.path.exists(checkpoint):
        print(f"Error: Checkpoint '{checkpoint}' not found!")
        return

    print("\n1. Basic evaluation (100 episodes)")
    print("2. With video recording")
    print("3. With rendering (watch agent play)")
    print("4. Compare with baseline")
    print("5. Full evaluation (all features)")

    choice = input("\nSelect option: ").strip()

    if choice == "1":
        os.system(f"python evaluate.py --checkpoint {checkpoint} --episodes 100")
    elif choice == "2":
        os.system(f"python evaluate.py --checkpoint {checkpoint} --record")
    elif choice == "3":
        os.system(f"python evaluate.py --checkpoint {checkpoint} --render")
    elif choice == "4":
        os.system(f"python evaluate.py --checkpoint {checkpoint} --baseline")
    elif choice == "5":
        os.system(f"python evaluate.py --checkpoint {checkpoint} --record --baseline")


def generate_visualizations():
    """Generate plots and visualizations."""
    print("\n" + "="*70)
    print("VISUALIZATION OPTIONS")
    print("="*70 + "\n")

    if not os.path.exists("logs/metrics.json"):
        print("Error: No training metrics found!")
        print("Please train the agent first (option 2 in main menu)")
        return

    print("1. Generate all plots")
    print("2. Create report figures")
    print("3. Both")

    choice = input("\nSelect option: ").strip()

    if choice in ["1", "3"]:
        print("\nGenerating all plots...")
        os.system("python -c \"from visualizations.plots import plot_all_metrics; plot_all_metrics()\"")

    if choice in ["2", "3"]:
        print("\nCreating report figures...")
        os.system("python -c \"from visualizations.plots import create_report_figures; create_report_figures()\"")

    print("\n✓ Visualizations generated!")
    print("  Check results/figures/ for output")


def show_info():
    """Display project information."""
    print("\n" + "="*70)
    print("PROJECT INFORMATION")
    print("="*70 + "\n")

    print("📁 Project Structure:")
    print("  src/             - Core implementation")
    print("  visualizations/  - Plotting tools")
    print("  checkpoints/     - Saved models")
    print("  logs/            - Training logs")
    print("  results/         - Evaluation results")
    print()
    print("📄 Documentation:")
    print("  README.md   - User guide and quick start")
    print("  plan.md     - Detailed implementation plan")
    print("  summary.md  - Code analysis and patterns")
    print("  config.yaml - Hyperparameters configuration")
    print()
    print("🔧 Key Scripts:")
    print("  train.py             - Main training script")
    print("  evaluate.py          - Evaluation script")
    print("  test_environment.py  - Environment tests")
    print()
    print("💡 Quick Commands:")
    print("  python train.py")
    print("  python evaluate.py --checkpoint checkpoints/best_model.pth")
    print("  python test_environment.py")
    print()
    print("📚 Algorithm:")
    print("  Double DQN + Dueling Architecture + Prioritized Experience Replay")
    print("  State-of-the-art Deep RL for Atari environments")
    print()

    input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    print_banner()

    # Check if user wants direct command
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "test":
            return test_environment()
        elif command == "train":
            os.system("python train.py " + " ".join(sys.argv[2:]))
            return
        elif command == "eval":
            os.system("python evaluate.py " + " ".join(sys.argv[2:]))
            return
        elif command == "viz":
            generate_visualizations()
            return
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  python main.py test   - Test environment")
            print("  python main.py train  - Train agent")
            print("  python main.py eval   - Evaluate agent")
            print("  python main.py viz    - Generate visualizations")
            return

    # Interactive menu
    while True:
        print_menu()

        try:
            choice = input("Select option: ").strip()

            if choice == "1":
                test_environment()
            elif choice == "2":
                train_agent()
            elif choice == "3":
                evaluate_agent()
            elif choice == "4":
                generate_visualizations()
            elif choice == "5":
                show_info()
            elif choice == "0":
                print("\n👋 Goodbye! Happy training!\n")
                break
            else:
                print("\n❌ Invalid option. Please try again.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
