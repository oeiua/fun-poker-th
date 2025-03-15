import sys
from trainer import train_poker_ai
from console_ui import play_vs_ai

def print_header():
    header = """
    ╔═════════════════════════════════════════════╗
    ║            POKER AI TRAINING SYSTEM         ║
    ╚═════════════════════════════════════════════╝
    """
    print(header)

def main():
    print_header()
    print("Please select an option:")
    print("1: Train Poker AI")
    print("2: Play against AI")
    print("3: Exit")
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                print("Starting Poker AI training...")
                train_poker_ai()
                print("Training complete!")
                break
            elif choice == "2":
                print("Loading game against AI...")
                play_vs_ai()
                break
            elif choice == "3":
                print("Goodbye!")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError as e:
            print(f"Error: {e}")
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            print("Please try again.")

if __name__ == "__main__":
    main()