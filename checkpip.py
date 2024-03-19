import subprocess
import sys

# 列出所有需要检查的库
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

# 对每个库进行检查和安装（如果需要）
for lib in libraries:
    try:
        # 尝试导入库来检查它是否已安装
        __import__(lib)
        print(f"{lib} is already installed.")
    except ImportError:
        # 如果库未安装，则安装它
        print(f"{lib} is not installed, installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

print("All libraries have been checked and installed if necessary.")
