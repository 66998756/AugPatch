# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import random

import torch
from torch.nn import Module

from mmseg.models.uda.teacher_module import EMATeacher
from mmseg.models.utils.dacs_transforms import get_mean_std, strong_transform
# from mmseg.models.utils.rand_augment import aug_generator

from mmseg.models.utils.augment_patch import Augmentations
from mmseg.models.utils.class_masking import ClassMaskGenerator
from mmseg.models.utils.geometric_perturb import GeometricPerturb


class AugPatchConsistencyModule(Module):

    def __init__(self, require_teacher, cfg):
        super(AugPatchConsistencyModule, self).__init__()

        self.source_only = cfg.get('source_only', False)
        self.max_iters = cfg['max_iters']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']

        self.aug_mode = cfg['aug_mode']
        self.aug_alpha = cfg['aug_alpha']
        self.aug_pseudo_threshold = cfg['aug_pseudo_threshold']
        self.aug_lambda = cfg['aug_lambda']
        self.transforms = Augmentations(cfg['aug_generator'])

        self.geometric_perturb = cfg['geometric_perturb']
        if self.geometric_perturb:
            self.perturb = GeometricPerturb(cfg['aug_generator']['patch_size'])

        # class masking config
        if cfg['cls_mask'] == 'Random':
            self.cls_mask = ClassMaskGenerator(
                'Random', cfg['aug_generator']['patch_size'])
        else:
            self.cls_mask = None

        assert self.aug_mode in [
            'separate', 'separatesrc', 'separatetrg', 'separateaug',
            'separatesrcaug', 'separatetrgaug'
        ]

        self.teacher = None
        if require_teacher or \
                self.aug_alpha != 'same' or \
                self.aug_pseudo_threshold != 'same':
            self.teacher = EMATeacher(use_mask_params=True, cfg=cfg)

        self.debug = False
        self.debug_output = {}

        
    def update_weights(self, model, iter):
        if self.teacher is not None:
            self.teacher.update_weights(model, iter)

    def update_debug_state(self):
        if self.teacher is not None:
            self.teacher.debug = self.debug

    def __call__(self,
                 model,
                 img,
                 img_metas,
                 gt_semantic_seg,
                 target_img,
                 target_img_metas,
                 valid_pseudo_mask,
                 pseudo_label=None,
                 pseudo_weight=None,
                 loss_adjustment=False):
        self.update_debug_state()
        self.debug_output = {}
        model.debug_output = {}
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        if not self.source_only:
            # Share the pseudo labels with the host UDA method
            if self.teacher is None:
                assert self.aug_alpha == 'same'
                assert self.aug_pseudo_threshold == 'same'
                assert pseudo_label is not None
                assert pseudo_weight is not None
                auged_plabel = pseudo_label
                auged_pweight = pseudo_weight
            # Use a separate EMA teacher for MIC
            else:
                auged_plabel, auged_pweight = \
                    self.teacher(
                        target_img, target_img_metas, valid_pseudo_mask)
                if self.debug:
                    self.debug_output['Mask Teacher'] = {
                        'Img': target_img.detach(),
                        'Pseudo Label': auged_plabel.cpu().numpy(),
                        'Pseudo Weight': auged_pweight.cpu().numpy(),
                    }
        # Don't use target images at all
        if self.source_only:
            auged_img = img
            auged_lbl = gt_semantic_seg
            b, _, h, w = gt_semantic_seg.shape
            auged_seg_weight = None
        # Use 1x source image and 1x target image for MIC
        elif self.aug_mode in ['separate', 'separateaug']:
            assert img.shape[0] == 2
            auged_img = torch.stack([img[0], target_img[0]])
            auged_lbl = torch.stack(
                [gt_semantic_seg[0], auged_plabel[0].unsqueeze(0)])
            gt_pixel_weight = torch.ones(auged_pweight[0].shape, device=dev)
            auged_seg_weight = torch.stack(
                [gt_pixel_weight, auged_pweight[0]])
        # Use only source images for MIC
        elif self.aug_mode in ['separatesrc', 'separatesrcaug']:
            auged_img = img
            auged_lbl = gt_semantic_seg
            auged_seg_weight = None
        # Use only target images for MIC
        elif self.aug_mode in ['separatetrg', 'separatetrgaug', 'separatetrgaugAugPatch']:
            auged_img = target_img
            auged_lbl = auged_plabel.unsqueeze(1)
            auged_seg_weight = auged_pweight
        else:
            raise NotImplementedError(self.aug_mode)

        # completed augment
        if 'aug' in self.aug_mode:
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': {
                    'brightness': 0.5,
                    'contrast': 0.5,
                    'saturation': 0.5,
                    'hue': 0.25,
                },
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1),
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)
            }
        else:
            strong_parameters = None

        # Apply AugPatch to image
        auged_img = self.transforms.generate_augpatch(
            auged_img.clone(), means, stds, strong_parameters)
        
        # 因為 valid_pseudo_mask可能是None，所以用pseudo_mask當作range
        if valid_pseudo_mask is None:
            if torch.max(pseudo_weight):
                valid_mask_region = pseudo_weight.clone().bool()
            else:
                valid_mask_region = (~ pseudo_weight.clone().bool())
        else:
            valid_mask_region = valid_pseudo_mask.clone()
        # Apply class masking to auged image
        if self.cls_mask:
            auged_img, mask_targets = self.cls_mask.mask_image(
                auged_img, auged_lbl, valid_mask_region.unsqueeze(dim=1))
        # Apply random patch geometric perturb
        if self.geometric_perturb:
            auged_img = self.perturb.generate_perturb(auged_img)

        # Train on masked images
        auged_loss = model.forward_train(
            auged_img,
            img_metas,
            auged_lbl,
            seg_weight=auged_seg_weight,
        )
        if self.aug_lambda != 1:
            auged_loss['decode.loss_seg'] *= self.aug_lambda

        # Dynamic Loss Adjustment
        # L_aug = L_aug + alpha * L_aug
        # alpha = (current_iter / total_iter)
        if loss_adjustment:
            auged_loss['decode.loss_seg'] += auged_loss['decode.loss_seg'] * \
                (loss_adjustment / self.max_iters)

        if self.debug:
            self.debug_output['Auged'] = model.debug_output
            if auged_seg_weight is not None:
                self.debug_output['Auged']['PL Weight'] = \
                    auged_seg_weight.cpu().numpy()

        return auged_loss, mask_targets
