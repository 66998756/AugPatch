# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import kornia.augmentation
import kornia.augmentation
import kornia.augmentation
import kornia.augmentation
import kornia.augmentation
import kornia.augmentation
import kornia.augmentation
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
import numpy as np
import torch.nn as nn

from torchvision import transforms

"""
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
                    seq = nn.Sequential(
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
"""

### kornia based augment function

def ColorJitter(img, v):
    # unimatch like
    img = kornia.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.25, p=1.)(img)
    return torch.clamp(img, 0, 1)

""" 由於 kornia equalize壞了 所以直接手動實現
def Equalize(img, v):
    # img = kornia.augmentation.RandomEqualize(p=1.)(img)
    # kornia 0.5.8 的Equalize有bug，徒手實現
    img = img.squeeze()
    c, h, w = img.shape
    image_equalized = torch.zeros_like(img)

    for i in range(c):
        channel = img[i]  # 获取单个颜色通道

        # 展平通道
        channel_flattened = channel.flatten()
        
        # 计算每个像素值的直方图
        histogram = torch.histc(channel_flattened, bins=256, min=0, max=1)
        
        # 计算累积分布函数（CDF），注意dim=0
        cdf = histogram.cumsum(dim=0)
        cdf = cdf / cdf[-1]  # 归一化

        # 生成均匀的原始值范围（0到1）
        original_values = torch.linspace(0, 1, steps=256, device=img.device)
        
        # 映射原始像素值到新的像素值
        image_equalized[i] = torch.interp(channel_flattened, original_values, cdf).reshape(h, w)
    
    return image_equalized.unsqueeze(dim=0)
    # return torch.clamp(img, 0, 1)
"""

def manual_interpolation(channel_flattened, bins, cdf):
    # 将像素值归一化到 [0, 255]
    scaled_pixels = (channel_flattened * (bins - 1)).long()

    # 直接使用 CDF 值进行映射
    return cdf[scaled_pixels]

def Equalize(image, v):
    # 假设image是一个RGB图像，形状为(C, H, W)
    image = image.squeeze()
    c, h, w = image.shape
    image_equalized = torch.zeros_like(image)
    bins = 256

    for i in range(c):
        channel = image[i]  # 获取单个颜色通道

        # 展平通道
        channel_flattened = channel.flatten()

        # 计算直方图
        histogram = torch.histc(channel_flattened, bins=bins, min=0, max=1)

        # 计算累积分布函数（CDF）
        cdf = histogram.cumsum(0)
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())  # 归一化CDF

        # 手动插值
        channel_equalized = manual_interpolation(channel_flattened, bins, cdf).view(h, w)
        image_equalized[i] = channel_equalized
    
    # return image_equalized
    return image_equalized.unsqueeze(dim=0)

def GaussianBlur(img, v):
    img = kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.)(img)
    return torch.clamp(img, 0, 1)

def Invert(img, v):
    img = kornia.augmentation.RandomInvert(p=1.)(img)
    return torch.clamp(img, 0, 1)

def Posterize(img, _):
    # kornia 0.5.8 的 posterize 有 bug,
    # 這邊直接從source code改了, 見
    # korinia.enhance.adjust, line 544:
    # bits = (8 - bits) ->
    #   bits = (8 - bits).to(device=input.device)
    img = kornia.augmentation.RandomPosterize(3, p=1.)(img)
    return torch.clamp(img, 0, 1)

def Sharpness(img, _):
    img = kornia.augmentation.RandomSharpness(0.5, p=1.)(img)
    return torch.clamp(img, 0, 1)

def Solarize(img, _):
    img = kornia.augmentation.RandomSolarize(0.1, 0.1, p=1.)(img)
    return torch.clamp(img, 0, 1)

def Identity(img, v):
    return img

def augment_list():  # 16 oeprations and their ranges
    # PIL based method
    # l = [
    #     (Identity, 0., 1.0),
    #     # (ShearX, 0., 0.3),  # 0
    #     # (ShearY, 0., 0.3),  # 1
    #     # (TranslateX, 0., 0.33),  # 2
    #     # (TranslateY, 0., 0.33),  # 3
    #     # (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # kornia based transform method, 
    # hardness are fix in function.
    l = [
        (Identity, 0.1, 1.0),
        (ColorJitter, 0.1, 1.0),
        (Equalize, 0.1, 1.0),
        (GaussianBlur, 0.1, 1.0),
        (Invert, 0.1, 1.0),
        (Posterize, 0.1, 1.0),
        (Sharpness, 0.1, 1.0),
        (Solarize, 0.1, 1.0)
    ]

    return l


class RandAugment:
    def __init__(self, n, m, seed):
        self.n = n
        self.m = m      # [0, 30]
        self.seed = seed
        self.augment_list = augment_list()

    def __call__(self, img, basic_aug_param=None):
        random.seed(self.seed)
        d = img.device

        # if basic_aug_param:
        #     aug = basic_aug()
        # img = aug.apply_basic_aug(img, basic_aug_param)
        # img = F.to_pil_image(img)
        # img = img.squeeze()
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        # img = F.to_tensor(img).to(d)
        return img
