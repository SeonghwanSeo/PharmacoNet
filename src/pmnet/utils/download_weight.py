import os
from pathlib import Path


def download_pretrained_model(weight_path, verbose):
    if not os.path.exists(weight_path):
        weight_path = Path(weight_path)
        weight_path.parent.mkdir(exist_ok=True)
        try:
            import gdown
        except ImportError:
            import subprocess
            import sys

            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        if verbose:
            print(f"Download pre-trained model... (path: {weight_path})")
        gdown.download(
            "https://drive.google.com/uc?id=1gzjdM7bD3jPm23LBcDXtkSk18nETL04p",
            str(weight_path),
            quiet=False,
        )
        if verbose:
            print("Download pre-trained model finish")
