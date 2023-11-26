import numpy as np
import torch

def amax(x):
    return torch.argmax(x, dim=1)


def multi_label(x):
    return (x > 0)