import torch
import torch.nn as nn
import os


def to_img(x):
    x = x.clamp(0, 1)
    imgs = x.reshape(x.shape[0], 1, 28, 28)
    return imgs


def save_model(model: nn.Module, path):
    torch.save(model.state_dict(), path)
    print("save model..........")


def load_model(model: nn.Module, path):
    model.load_state_dict(torch.load(path))
    print("load model..........")


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
