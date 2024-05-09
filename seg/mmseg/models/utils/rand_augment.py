# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image

import torchvision
import torchvision.transforms.functional as F
# from torchvision import transforms as F

from mmseg.models.utils.dacs_transforms import denorm, renorm, color_jitter

# -------------------------------------------------------------------------
# UniMatch-like aug
# -------------------------------------------------------------------------
import kornia
import numpy as np
import torch.nn as nn

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
def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


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


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img

### torch implemtation
def posterize(img, v):  # [0.25, 0.875]
    assert 0.1 <= v <= 0.9
    bits = int((1 - v) * 8)
    return img.mul(2 ** bits).div(2 ** bits)

def solarize(img, v):  # [0.25, 0.875]
    assert 0.1 <= v <= 0.9
    threshold = int(v * 255)
    return torch.where(img < threshold / 255, img, 1 - img)

def auto_contrast(img, _):
    # Compute per-channel minimum and maximum values
    min_vals, _ = torch.min(img, dim=-1, keepdim=True)
    max_vals, _ = torch.max(img, dim=-1, keepdim=True)
    
    # Compute per-channel range
    range_vals = max_vals - min_vals
    
    # Apply autocontrast normalization
    normalized_img = (img - min_vals) / (range_vals + 1e-6)  # Add a small epsilon to avoid division by zero
    normalized_img = torch.clamp(normalized_img, 0, 1)  # Ensure values are in [0, 1] range
    
    return normalized_img

def invert(img, _):
    return 1 - img

# def equalize(img, _):
#     # Convert the image to grayscale
#     gray_img = img.mean(dim=0, keepdim=True)
    
#     # Compute the cumulative distribution function (CDF)
#     cdf = gray_img.cumsum(dim=-1).cumsum(dim=-2)
    
#     # Normalize the CDF to the range [0, 1]
#     cdf_normalized = cdf / cdf.max()
    
#     # Interpolate the values using the normalized CDF
#     equalized_img = F.interpolate(gray_img.unsqueeze(dim=0), size=img.shape[-2:], mode='bicubic', align_corners=False)    
#     # Stack the equalized grayscale image across channels
#     # equalized_img = equalized_img.expand_as(img)

    
#     return equalized_img
def equalize(image, v):
    """Implements Equalize function from PIL using PyTorch ops based on:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352"""
    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[c, :, :]

        im = im * 255
        # Compute the histogram of the image channel.
        histo = torch.histc(im, bins=256, min=0, max=255)#.type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1).cuda(), lut[:-1]]) 
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
            result = result.reshape_as(im)
        result = result / 255
        return result.type(torch.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = torch.stack([s1, s2, s3], 2)
    return image

def contrast(img, v):  # [0.1, 1.9]
    assert 0.1 <= v <= 0.9
    mean_value = img.mean()
    return torch.clamp((img - mean_value) * v + mean_value, 0, 1)

def color(img, v):  # [0.1, 1.9]
    assert 0.1 <= v <= 0.9
    gray_img = img.mean(dim=0, keepdim=True)
    return torch.lerp(gray_img, img, v)

def brightness(img, v):  # [0.1, 1.9]
    assert 0.1 <= v <= 0.9
    return torch.clamp(img * v, 0, 1)


def sharpness(img, alpha=1.0):
    # Define a sharpening kernel
    kernel = torch.tensor([
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    ], dtype=torch.float32).unsqueeze(0)  # 將卷積核的深度由1擴展為3，並將其轉換為4D張量


    # Apply convolution with the sharpening kernel
    sharpened_img = F.conv2d(img.unsqueeze(0), kernel.to(device=img.device), padding=1, stride=1)

    # Apply scaling factor alpha to the sharpened image and add to the original image
    sharpened_img = alpha * sharpened_img + (1 - alpha) * img

    return sharpened_img.squeeze()


def augment_list():  # 16 oeprations and their ranges
    # 這裡把所有變形的全註解
    l = [
        (Identity, 0., 1.0), # 強制一定要aug
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

    # l = [
    #     (Identity, 0., 1.0), # 強制一定要aug
    #     (auto_contrast, 0, 1),  # 5
    #     (invert, 0, 1),  # 6
    #     (equalize, 0.1, 0.9),  # 7
    #     (solarize, 0.1, 0.9),  # 8
    #     (posterize, 0.1, 0.9),  # 9
    #     (contrast, 0.1, 0.9),  # 10
    #     (color, 0.1, 0.9),  # 11
    #     (brightness, 0.1, 0.9),  # 12
    #     (sharpness, 0.1, 0.9),  # 13
    # ]

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

        if basic_aug_param:
            aug = basic_aug()
        img = aug.apply_basic_aug(img, basic_aug_param)
        img = F.to_pil_image(img)
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        # img.save("/home/Bill0041/paper/PGMIC/seg/debug.jpg") 
        return F.to_tensor(img).to(d)