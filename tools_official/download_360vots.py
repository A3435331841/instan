#!/usr/bin/env python3
"""Download 360VOTS dataset from HuggingFace mirror to D:/instan/pano360/data360/official."""
import os
import sys
from pathlib import Path

# Use HuggingFace mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

# Target directory on D drive
TARGET_DIR = Path('D:/instan/pano360/data360/official')
TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading 360VOTS dataset to: {TARGET_DIR}")
print(f"Using mirror: {os.environ['HF_ENDPOINT']}")
print("This will download the full dataset (~120GB). Please be patient.")
print()

try:
    snapshot_download(
        repo_id="xuyzshaun/360VOTS",
        repo_type="dataset",
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"\nDownload completed! Files saved to: {TARGET_DIR}")
except Exception as e:
    print(f"\nDownload failed: {e}")
    sys.exit(1)
