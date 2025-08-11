"""
Demo script to launch the F1 Race Prediction Streamlit application.
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit application."""
    print("🏎️  F1 Race Prediction Web Application Demo")
    print("=" * 50)
    
    # Check if we're in the right directory
    app_path = Path("web/app.py")
    if not app_path.exists():
        print("❌ Error: web/app.py not found!")
        print("Please run this script from the project root directory.")
        return
    
    print("✅ Found Streamlit application")
    print("🚀 Launching F1 Race Predictor...")
    print()
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web/app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check that all dependencies are installed")
        print("3. Ensure you're running from the project root directory")

if __name__ == "__main__":
    main()