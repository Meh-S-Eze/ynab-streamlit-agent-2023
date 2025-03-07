#!/usr/bin/env python3
"""
YNAB Financial Assistant Streamlit runner.
"""
import streamlit.web.bootstrap as bootstrap
from streamlit.web.cli import _main_run_clexpr

if __name__ == "__main__":
    bootstrap.run(_main_run_clexpr, 'src.streamlit.app', 'main', [], {}) 