import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Make experiments as reproducible as possible on the same machine.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Prefer CUDA if available.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")