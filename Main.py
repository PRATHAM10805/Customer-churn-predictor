import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction System - Command Center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Main.py train  # Start model training
  python Main.py ui     # Launch Streamlit dashboard
  python Main.py api    # Start FastAPI backend
        """
    )
    
    parser.add_argument(
        "mode", 
        choices=["train", "ui", "api"], 
        help="The component of the system you want to run."
    )

    args = parser.parse_args()

    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    try:
        if args.mode == "train":
            print("\n[INFO] Starting Model Training Pipeline...")
            subprocess.run([sys.executable, "-m", "src.Train"], check=True)
        
        elif args.mode == "ui":
            print("\n[INFO] Launching Streamlit Dashboard...")
            # Note: Streamlit usually needs to be run as a module or via its own CLI
            subprocess.run(["streamlit", "run", "app/app.py"], check=True)
        
        elif args.mode == "api":
            print("\n[INFO] Starting FastAPI Backend via Uvicorn...")
            subprocess.run(["uvicorn", "app.api:app", "--reload"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with return code {e.returncode}")
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
