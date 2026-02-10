import streamlit as st
import sys
from pathlib import Path

# Add project root to path so imports work
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import main app
# Note: dashboard.app contains the st.set_page_config call at module level,
# so simply importing it will configure the page.
from dashboard.app import main

if __name__ == "__main__":
    main()
