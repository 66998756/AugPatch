import random
import torchvision.transforms.functional as TF

import torch

class RandomTransform:
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, img):
        func = self.rng.choice([
            self.ShearX, self.ShearY, self.TranslateX, self.TranslateXabs,
            self.TranslateY, self.TranslateYabs, self.Rotate
        ])
        return func(img)

    def ShearX(self, img):
        v = self.rng.uniform(-0.3, 0.3)
        if self.rng.random() > 0.5:
            v = -v
        return TF.affine(img, angle=0, translate=(0, 0), scale=1, shear=(v, 0))

    def ShearY(self, img):
        v = self.rng.uniform(-0.3, 0.3)
        if self.rng.random() > 0.5:
            v = -v
        return TF.affine(img, angle=0, translate=(0, 0), scale=1, shear=(0, v))

    def TranslateX(self, img):
        v = self.rng.uniform(-0.45, 0.45)
        if self.rng.random() > 0.5:
            v = -v
        v = v * img.size[0]
        return TF.affine(img, angle=0, translate=(v, 0), scale=1, shear=0)

    def TranslateXabs(self, img):
        v = self.rng.uniform(0, 0.45)
        if self.rng.random() > 0.5:
            v = -v
        v = v * img.size[0]
        return TF.affine(img, angle=0, translate=(v, 0), scale=1, shear=0)

    def TranslateY(self, img):
        v = self.rng.uniform(-0.45, 0.45)
        if self.rng.random() > 0.5:
            v = -v
        v = v * img.size[1]
        return TF.affine(img, angle=0, translate=(0, v), scale=1, shear=0)

    def TranslateYabs(self, img):
        v = self.rng.uniform(0, 0.45)
        if self.rng.random() > 0.5:
            v = -v
        v = v * img.size[1]
        return TF.affine(img, angle=0, translate=(0, v), scale=1, shear=0)

    def Rotate(self, img):
        v = self.rng.uniform(-30, 30)
        if self.rng.random() > 0.5:
            v = -v
        return TF.rotate(img, angle=v)
    

class GeometricPerturb:
    def __init__(self, patch_size):
        self.patch_size = patch_size[0]
        self.perturb = RandomTransform()
    
    @torch.no_grad()
    def generate_perturb(self, imgs):
        B, C, H, W = imgs.shape
        unfolded_imgs = torch.nn.functional.unfold(imgs, 
            kernel_size=self.patch_size, stride=self.patch_size)
        perturbed_unfolded_imgs = self.perturb(unfolded_imgs)
        perturbed_imgs = torch.nn.functional.fold(perturbed_unfolded_imgs, 
            output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return perturbed_imgs
