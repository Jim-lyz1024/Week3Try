import subprocess
import sys

libraries = [
    "ftfy",
    "regex",
    "tqdm",
    "yacs",
    "gdown",
    "tensorboard",
    "future",
    "scipy",
    "scikit-learn",
    "packaging",
    "wilds",
    "tabulate",
    "openai-clip",
]

for lib in libraries:
    try:
        __import__(lib)
        print(f"{lib} is already installed.")
    except ImportError:
        print(f"{lib} is not installed, installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

print("All libraries have been checked and installed if necessary.")
