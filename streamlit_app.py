# Main entry point for Streamlit Cloud deployment
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'web'))

# Import and run the main app
from web.app import main

if __name__ == "__main__":
    main()