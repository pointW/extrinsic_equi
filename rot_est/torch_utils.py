import torch
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
import numpy as np


def randomCrop(imgs, out=64):
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = torch.empty((n, c, out, out), dtype=imgs.dtype).to(imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped

def centerCrop(imgs, out=64):
    n, c, h, w = imgs.shape
    top = (h - out) // 2
    left = (w - out) // 2

    imgs = imgs[:, :, top:top + out, left:left + out]
    return imgs
