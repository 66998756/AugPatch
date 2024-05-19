# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import random
import queue

import torch
from torch.nn import Module

from mmseg.models.uda.teacher_module import EMATeacher
from mmseg.models.utils.dacs_transforms import get_mean_std, strong_transform
from mmseg.models.utils.masking_transforms import build_mask_generator

from mmseg.models.utils.dacs_transforms import denorm, renorm

from mmseg.models.utils.augment_patch import Augmentations

# from mmseg.models.utils.masking_transforms import DualMaskGenerator, ClassMaskGenerator, SceneMaskGenerator


class AdaptivePseudoLabelRefinement(Module):

    def __init__(self, require_teacher, cfg):
        super(AdaptivePseudoLabelRefinement, self).__init__()

        # self.source_only = cfg.get('source_only', False)
        self.start_iters = cfg['start_iters']
        # self.color_jitter_s = cfg['color_jitter_strength'] # 0.2
        # self.color_jitter_p = cfg['color_jitter_probability'] # 0.2

        # self.refine_mode = cfg['refine_mode']
        # self.refine_alpha = cfg['refine_alpha']
        self.max_bank_size = cfg['max_bank_size']

        # Augmentation Setup
        self.transforms = Augmentations(cfg['refine_aug'])
        self.k = cfg['k']

        # self.teacher = None
        # if require_teacher or self.refine_alpha != 0.0:
        #     self.teacher = EMATeacher(use_mask_params=True, cfg=cfg)

        # 每個item有四個不同等級的特徵，詳見 SegFormer
        self.source_queue = torch.zeros(self.max_bank_size, 512, 16, 16)
        self.source_queue_meta = queue.Queue(maxsize=self.max_bank_size)
        # self.target_memoqueue = torch.zeros(self.max_bank_size, 512, 16)

        self.debug = False
        self.debug_output = {}

    def init_centroid(self, source_ft, target_ft):
        mean = [0 for _ in range(4)]
        for i in range(4):
            for j in range(self.max_bank_size):
                mean[i] += self.source_queue[j]

    def update_memo_queue(self, queue, new_feature, img):
        """
        将新特征添加到队列中。如果队列已满，移除最旧的特征。

        :param queue: 当前的内存队列，形状 [max_queue_size, 512, 16]
        :param new_feature: 新的特征，形状 [512, 16] 或 [B, 512, 16]
        """
        # 检查new_feature是否是单个特征或一个batch
        if new_feature.dim() == 2:
            new_feature = new_feature.unsqueeze(0)  # 从 [512, 16] 转换为 [1, 512, 16]

        # 计算新特征加入后总大小
        total_size = queue.size(0) + new_feature.size(0)
        
        # 如果加入新特征后超过了队列大小
        if total_size > self.max_bank_size:
            # 计算超出的数量
            excess = total_size - self.max_bank_size
            # 移除最旧的特征，并添加新特征
            queue = torch.cat((queue, new_feature), dim=0)[excess:]
        else:
            # 直接添加新特征
            queue = torch.cat((queue, new_feature), dim=0)

        if self.source_queue_meta.full():
            _ = self.source_queue_meta.get()
        self.source_queue_meta.put(img)
        
        return queue

    def ema_centroid(self, iter, source_ft, target_ft):
        alpha_centroid = min(1 - 1 / (iter + 1), self.refine_alpha)
        for i in range(4):
            self.source_centroid[i] = alpha_centroid * \
                self.source_centroid[i] + (1 - alpha_centroid) * source_ft[i]
            self.target_centroid[i] = alpha_centroid * \
                self.target_centroid[i] + (1 - alpha_centroid) * target_ft[i]

    def update_debug_state(self, model):
        # if self.teacher is not None:
        #     self.teacher.debug = self.debug
        model.debug = self.debug

    def __call__(self,
                 model,
                 img,
                 img_metas,
                 target_img,
                 target_img_metas,
                 pseudo_label=None,
                 local_iter=None):
        self.update_debug_state(model)
        self.debug_output = {}
        model.debug_output = {}
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        """
        if not self.source_only:
            # Share the pseudo labels with the host UDA method
            if self.teacher is None:
                assert self.mask_alpha == 'same'
                assert self.mask_pseudo_threshold == 'same'
                assert pseudo_label is not None
                assert pseudo_weight is not None
                masked_plabel = pseudo_label
                masked_pweight = pseudo_weight
            # Use a separate EMA teacher for MIC
            else:
                masked_plabel, masked_pweight = \
                    self.teacher(
                        target_img, target_img_metas, valid_pseudo_mask)
                if self.debug:
                    self.debug_output['Mask Teacher'] = {
                        'Img': target_img.detach(),
                        'Pseudo Label': masked_plabel.cpu().numpy(),
                        'Pseudo Weight': masked_pweight.cpu().numpy(),
                    }
        # Don't use target images at all
        if self.source_only:
            masked_img = img
            masked_lbl = gt_semantic_seg
            b, _, h, w = gt_semantic_seg.shape
            masked_seg_weight = None
        # Use 1x source image and 1x target image for MIC
        elif self.mask_mode in ['separate', 'separateaug']:
            assert img.shape[0] == 2
            masked_img = torch.stack([img[0], target_img[0]])
            masked_lbl = torch.stack(
                [gt_semantic_seg[0], masked_plabel[0].unsqueeze(0)])
            gt_pixel_weight = torch.ones(masked_pweight[0].shape, device=dev)
            masked_seg_weight = torch.stack(
                [gt_pixel_weight, masked_pweight[0]])
        # Use only source images for MIC
        elif self.mask_mode in ['separatesrc', 'separatesrcaug']:
            masked_img = img
            masked_lbl = gt_semantic_seg
            masked_seg_weight = None
        # Use only target images for MIC
        elif self.mask_mode in ['separatetrg', 'separatetrgaug']:
            masked_img = target_img
            masked_lbl = masked_plabel.unsqueeze(1)
            masked_seg_weight = masked_pweight
        else:
            raise NotImplementedError(self.mask_mode)
        """

        # self.source_queue = self.source_queue.to(device=dev)
        ### Self-volting memory bank setup
        src_feat = model.extract_feat(img)
        # 暫時只取最高級特徵
        # print(src_feat[3].shape)
        # self.source_queue = self.update_memo_queue(
        #     self.source_queue, src_feat[3].cpu().detach(), img.cpu().detach())
        self.source_queue = self.update_memo_queue(
            self.source_queue.to(device=dev), 
            src_feat[3], 
            img.cpu().detach().clone()
        ).to(device='cpu')
        
        # Warm up
        if local_iter < self.start_iters:
            return pseudo_label, None

        # 計算距離最近的src feature
        tgt_soft_label, tgt_feat = model.generate_pseudo_label(
            target_img, target_img_metas, return_feat=True)
        
        # 生成N個擴增影像
        target_img = denorm(
            target_img, means[0].unsqueeze(0), stds[0].unsqueeze(0))
        auged_imgs = self.transforms.apply_transforms(target_img, None)

        debug_imgs = {
            'source_indice': [],
            'topk_imgs': [],
            'auged_imgs': [],
        }
        pseudo_soft_labels = torch.zeros_like(
            pseudo_label, device=dev)
        for B in range(tgt_feat[3].shape[0]):
            # 找距離最近的src feature和距離值
            queue_flat = self.source_queue.view(
                self.source_queue.shape[0], -1).to(device=dev)
            target_flat = tgt_feat[3][B].view(-1)
            dist = torch.norm(queue_flat - target_flat, dim=1)
            min_distance_idx = torch.argmin(dist)
            closest_feature, closest_distance = \
                self.source_queue[min_distance_idx].to(device=dev), \
                dist[min_distance_idx].to(device=dev)

            # 計算擴增影像的特徵和 pseudo label
            auged_img = torch.stack(auged_imgs[B]).to(device=dev)
            auged_img = renorm(
                auged_img, means[0].unsqueeze(0), stds[0].unsqueeze(0))
            logits, auged_feat = model.generate_pseudo_label(
                auged_img, target_img_metas, return_feat=True)
            auged_soft_label = torch.softmax(logits.detach(), dim=1)
            del logits
            
            auged_flat = auged_feat[3].view(auged_feat[3].shape[0], -1)  # [size, H*W]
            closest_feature_flat = closest_feature.view(-1)  # [H*W]

            distances = torch.norm(
                auged_flat - closest_feature_flat, dim=1)

            # 过滤操作
            mask = distances <= closest_distance
            filtered_features = auged_feat[3][mask]
            filtered_soft_labels = auged_soft_label[mask]
            filtered_distances = distances[mask]
            del auged_feat

            # 找 topk 最近的特征
            values = None
            if len(filtered_distances) > 0:  # 确保有特征满足过滤条件
                values, indices = torch.topk(filtered_distances, 
                    min(self.k, len(filtered_distances)), largest=False)
                topk_closest_features = filtered_features[indices]
                topk_closest_soft_label = filtered_soft_labels[indices]

                # apply soft-volting
                summed_labels = topk_closest_soft_label.sum(dim=0)
                soft_labels = summed_labels / topk_closest_soft_label.shape[0]
            else:
                soft_labels = pseudo_label[B]

            pseudo_soft_labels[B] = torch.max(soft_labels, dim=0)[1]

            # debug
            if self.debug:
                debug_imgs['auged_imgs'].append(auged_img)
                selected_auged_imgs = auged_img[mask][indices]  # 使用 mask 后再索引以保证正确性
                debug_imgs['topk_imgs'].append(selected_auged_imgs)
                debug_imgs['source_indice'].append(min_distance_idx)


        # transform to one-hot
        # refined_pseudo_label = torch.max(pseudo_soft_labels, dim=1)

        if self.debug:
            self.debug_output['Refine'] = {
                'Referenced_source': # for batchsize = 2
                    torch.stack(
                        [self.source_queue_meta[debug_imgs['source_indice'][0]],
                        self.source_queue_meta[debug_imgs['source_indice'][1]]]),
                'auged_imgs': debug_imgs['auged_imgs'],
                'choised_topk_imgs': debug_imgs['topk_imgs'],
                'org_pseudo_label': pseudo_label,
                'refined_pseudo_label': pseudo_soft_labels
            }

        return pseudo_soft_labels, self.debug_output