# Streamlit Cloud entry point
# This file redirects to the main app
import subprocess
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main app
if __name__ == "__main__":
    import streamlit as st
    from app.app import main
    
    main()
