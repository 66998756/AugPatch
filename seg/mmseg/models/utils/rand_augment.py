# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image

import torchvision
import torchvision.transforms.functional as F

from mmseg.models.utils.dacs_transforms import denorm, renorm, color_jitter

# -------------------------------------------------------------------------
# UniMatch-like aug
# -------------------------------------------------------------------------
import kornia

from torchvision import transforms

class basic_aug:
    def __init__(self):
        pass

    def unimatch_blur(self, blur, data=None, target=None):
        if not (data is None):
            if data.shape[1] == 3:
                if blur > 0.5:
                    sigma = np.random.uniform(0.1, 2.0)
                    kernel_size_y = int(
                        np.floor(
                            np.ceil(0.1 * data.shape[2]) - 0.5 +
                            np.ceil(0.1 * data.shape[2]) % 2))
                    kernel_size_x = int(
                        np.floor(
                            np.ceil(0.1 * data.shape[3]) - 0.5 +
                            np.ceil(0.1 * data.shape[3]) % 2))
                    kernel_size = (kernel_size_y, kernel_size_x)
                    seq = torch.nn.Sequential(
                        kornia.filters.GaussianBlur2d(
                            kernel_size=kernel_size, sigma=(sigma, sigma)))
                    data = seq(data)
        return data, target


    def unimatch_gray_scale(self, data):
        data = transforms.RandomGrayscale(p=0.2)(data)
        return data

    @torch.no_grad()
    def apply_basic_aug(self, img, param):
        # unimatch like augment
        img, _ = color_jitter(
            color_jitter=param['color_jitter'],
            s=param['color_jitter_s'],
            p=param['color_jitter_p'],
            mean=param['mean'],
            std=param['std'],
            data=img,
            target=None)
        img = self.unimatch_gray_scale(img)
        img, _ = self.unimatch_blur(blur=param['blur'], data=img, target=None)
        return img

# -------------------------------------------------------------------------
# Original RandAugment component
# -------------------------------------------------------------------------

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Identity(img, v):
    return img

def augment_list(ignore_identity):  # 16 oeprations and their ranges
    # 這裡把所有變形的全註解
    l = [
        (Identity, 0., 1.0),
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 110),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
    ]

    if ignore_identity:
        l = [
            (AutoContrast, 0, 1),  # 5
            (Invert, 0, 1),  # 6
            (Equalize, 0, 1),  # 7
            (Solarize, 0, 110),  # 8
            (Posterize, 4, 8),  # 9
            (Contrast, 0.1, 1.9),  # 10
            (Color, 0.1, 1.9),  # 11
            (Brightness, 0.1, 1.9),  # 12
            (Sharpness, 0.1, 1.9),  # 13
        ]

    return l


class RandAugment:
    def __init__(self, n, m, ignore_identity, seed):
        self.n = n
        self.m = m      # [0, 30]
        if self.m == 'random':
            self.m = random.randint(10, 20)
        self.seed = seed
        self.augment_list = augment_list(ignore_identity)

    def __call__(self, img, means, stds, basic_aug_param=None):
        # random.seed(self.seed)
        d = img.device

        if basic_aug_param:
            aug = basic_aug()
            img = aug.apply_basic_aug(img, basic_aug_param)

        img = denorm(img, means[0].unsqueeze(0), stds[0].unsqueeze(0))
        img = F.to_pil_image(img.squeeze())
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        
        img = F.to_tensor(img).to(d).unsqueeze(dim=0)
        img = renorm(img, means[0].unsqueeze(0), stds[0].unsqueeze(0))
        return img