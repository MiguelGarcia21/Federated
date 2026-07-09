import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds.

    This keeps the same behavior as the Kaggle notebook:
    cuDNN deterministic mode is not forced, and benchmark mode is enabled.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True