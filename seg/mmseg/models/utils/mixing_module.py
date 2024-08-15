import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)


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
    

class ClassMix:
    def __init__(self, cut_rate=0.5):
        self.cut_rate = cut_rate
        self.strong_parameters = {
            'mix': None,
            'color_jitter': 0,
            'color_jitter_s': 0,
            'color_jitter_p': 1,
            'blur': 0,
            'mean': 0,
            'std': 0
        }


    def __call__(self, 
                 src_img, 
                 tgt_img, 
                 src_lbls, 
                 tgt_lbls, 
                 alpha=1.0):
        mixed_img = src_img.clone()
        mixed_lbls = src_lbls.clone()

        mix_mask = get_class_masks(src_lbls)
        for i in range(src_img.size(0)):
            self.strong_parameters['mix'] = mix_mask[i]
            mixed_img[i], mixed_lbls[i] = strong_transform(
                self.strong_parameters,
                data=torch.stack((src_img[i], tgt_img[i])),
                target=torch.stack((src_lbls[i], tgt_lbls[i]))
            )

        return mixed_img, mixed_lbls
    

class MixingGenerator(nn.Module):
    def __init__(self, cfg):
        super(MixingGenerator, self).__init__()
        self.mixing_type = cfg['mixing_type']

        self.strong_parameters = {
            'mix': None,
            'color_jitter': 0,
            'color_jitter_s': 0,
            'color_jitter_p': 1,
            'blur': 0,
            'mean': 0,
            'std': 0
        }

        if self.mixing_type == 'cutmix':
            self.mixing = CutMix(cut_rate=0.5)
        elif self.mixing_type == 'classmix':
            self.mixing = ClassMix(ignore_label=255)
        # elif self.mixing_type == 'cowmix':
        #     self.mixing = self.CowMix()

    @torch.no_grad()
    def mixing_img_and_lbl(self, 
                           src_imgs, 
                           tgt_imgs, 
                           src_lbls, 
                           tgt_lbls,
                           pseudo_weight):
        # mixed_imgs, mixed_lbls = self.mixing(
        #     src_imgs, tgt_imgs, src_lbls, tgt_lbls)
        # return mixed_imgs, mixed_lbls
        # Apply mixing
        
        batch_size = src_imgs.shape[0]
        mix_idx = torch.max(pseudo_weight[0]) > torch.max(pseudo_weight[1])
        gt_pixel_weight = torch.ones((pseudo_weight.shape), 
                                     device=src_imgs.device)
        
        # src_imgs: [CS[0], ACDC[0]]
        # tgt_imgs: [ACDC[0], ACDC[1]]
        # src_lbls: [CS[0], ACDC[0]]
        # tgt_lbls: [ACDC[0], ACDC[1]]
        # pseudo_weight: [auged_pweight[0], auged_pweight[1]]
        # gt_pixel_weight: [1, 1]
        if mix_idx:
            gt_pixel_weight[1] = pseudo_weight[0].clone()
        else:
            # image 互換
            src_imgs[1] = tgt_imgs[1].clone()
            tgt_imgs[1] = tgt_imgs[0].clone()
            tgt_imgs[0] = src_imgs[1].clone()

            # label 互換
            src_lbls[1] = tgt_lbls[1].clone()
            tgt_lbls[1] = tgt_lbls[0].clone()
            tgt_lbls[0] = src_lbls[1].clone()

            # weight 互換
            gt_pixel_weight[1] = pseudo_weight[1].clone()
            tmp = pseudo_weight[0].clone()
            pseudo_weight[0] = pseudo_weight[1].clone()
            pseudo_weight[1] = tmp
            
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mixed_seg_weight = pseudo_weight.clone()
        mix_masks = get_class_masks(src_lbls)
        for i in range(batch_size):
            self.strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                self.strong_parameters,
                data=src_imgs if mix_idx else torch.stack((src_imgs[i], tgt_imgs[i])),
                target=torch.stack(
                    (src_lbls[i][0], tgt_lbls[i])))
            _, mixed_seg_weight[i] = strong_transform(
                self.strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        del gt_pixel_weight
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # mix_tgt = torch.stack([mix_tgt[0][0], mix_tgt[0][1]])
        # auged_img, auged_lbl = self.mixing.mixing_img_and_lbl(
        #     auged_img, mix_tgt, auged_lbl, mix_lbl)

        return mixed_img, mixed_lbl, mixed_seg_weight
