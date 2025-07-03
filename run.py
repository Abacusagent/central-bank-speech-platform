# run.py

"""
Universal entry point for the Central Bank Speech Analysis Platform.

Usage (from project root):
    python run.py collect --institution FED --start 2023-01-01 --end 2023-12-31
    python run.py analyze --institution FED

This script simply delegates to the CLI application defined in tools/cli.py.
"""

from tools.cli import app

if __name__ == "__main__":
    app()
