import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CutMix:
    def __init__(self, cut_rate=0.5):
        self.cut_rate = cut_rate

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, src_img, tgt_img, src_lbls, tgt_lbls, alpha=1.0):
        mixed_src_img = src_img.clone()
        mixed_src_lbls = src_lbls.clone()

        for i in range(src_img.size(0)):
            # lam = np.random.beta(alpha, alpha)
            # manually set cutmix region to 50%
            lam = self.cut_rate
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(src_img.size(), lam)

            mixed_src_img[i, :, bbx1:bbx2, bby1:bby2] = tgt_img[i, :, bbx1:bbx2, bby1:bby2]
            mixed_src_lbls[i, :, bbx1:bbx2, bby1:bby2] = tgt_lbls[i, :, bbx1:bbx2, bby1:bby2]

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (src_img.size()[-1] * src_img.size()[-2]))

        return mixed_src_img, mixed_src_lbls


class MixingGenerator(nn.Module):
    def __init__(self, cfg):
        super(MixingGenerator, self).__init__()
        self.aug_block_size = cfg['aug_block_size']
        self.mixing_ratio = cfg['mixing_ratio']
        self.mixing_type = cfg['mixing_type']

        if self.mixing_type == 'cutmix':
            self.mixing = CutMix(cut_rate=self.mixing_ratio)
        # elif self.mixing_type == 'cowmix':
        #     self.mixing = self.CowMix()
        # elif self.mixing_type == 'classmix':
        #     self.mixing = self.ClassMix()

    @torch.no_grad()
    def mixing_img_and_lbl(self, src_imgs, tgt_imgs, src_lbls, tgt_lbls):
        mixed_imgs, mixed_lbls = self.mixing(
            src_imgs, tgt_imgs, src_lbls, tgt_lbls)
        return mixed_imgs, mixed_lbls
    
    # @torch.no_grad()
    # def CutMix(self, src_imgs, tgt_imgs, src_lbls, tgt_lbls):
    #     pass

    # # TODO
    # @torch.no_grad()
    # def CowMix(self, src_imgs, tgt_imgs, src_lbls, tgt_lbls):
    #     pass

    # # TODO
    # @torch.no_grad()
    # def ClassMix(self, src_imgs, tgt_imgs, src_lbls, tgt_lbls):
    #     pass
