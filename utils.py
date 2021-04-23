import torch
import torch.nn as nn
import os
import torch.nn.functional as F


def to_img(x):
    x = x.clamp(0, 1)
    imgs = x.reshape(x.shape[0], 1, 28, 28)
    return imgs


def to_one_hot(labels: torch.Tensor, num_class: int):
    y = torch.zeros(labels.shape[0], num_class)
    for i, label in enumerate(labels):
        y[i, label] = 1
    return y


def save_model(model: nn.Module, path):
    torch.save(model.state_dict(), path)
    print("save model..........")


def load_model(model: nn.Module, path):
    model.load_state_dict(torch.load(path))
    print("load model..........")


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)