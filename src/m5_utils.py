"""
m5_utils.py — Shared utilities for M5 quantum trajectory scripts.

Output directory convention
---------------------------
Set the M5_OUTPUT environment variable to control where figures and data
files are written:

    export M5_OUTPUT=/kaggle/working       # Kaggle GPU notebook
    export M5_OUTPUT=/mnt/user-data/outputs # Claude container
    export M5_OUTPUT=~/results              # local machine

If M5_OUTPUT is not set, files are written to ./output (created
automatically).
"""

import os

_OUTPUT_DIR = None          # resolved lazily on first call


def output_dir():
    """Return (and create if needed) the output directory."""
    global _OUTPUT_DIR
    if _OUTPUT_DIR is None:
        _OUTPUT_DIR = os.environ.get(
            "M5_OUTPUT",
            os.path.join(os.getcwd(), "output")
        )
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR


def output_path(filename):
    """Return the full path for an output file.

    >>> output_path("fig1.png")
    '/kaggle/working/fig1.png'   # if M5_OUTPUT=/kaggle/working
    """
    return os.path.join(output_dir(), filename)
