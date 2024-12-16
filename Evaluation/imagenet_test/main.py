import logging
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

import config as config
from engine import train

logging.basicConfig(level=logging.INFO)
cfg = SimpleNamespace(**vars(config))


def set_seed(seed):
    """设置随机种子以确保可重复性。"""

    # 1. 设置 PYTHONHASHSEED 环境变量，以便使 Python 中的哈希函数可重复
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. 设置 Python 内置随机数生成器的种子
    random.seed(seed)

    # 3. 设置 NumPy 随机数生成器的种子
    np.random.seed(seed)

    # 4. 设置 PyTorch 随机数生成器的种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

    # 5. 设置 PyTorch 后端为确定性算法，以避免潜在的随机性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_config(file_path):
    print("Current config.py settings:")
    with open(file_path, "r") as file:
        print(file.read())


if __name__ == "__main__":
    set_seed(cfg.seed)
    train()
