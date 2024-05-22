# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import random
from collections import deque

import torch
from torch.nn import Module

from mmseg.models.uda.teacher_module import EMATeacher
from mmseg.models.utils.dacs_transforms import get_mean_std, strong_transform
from mmseg.models.utils.masking_transforms import build_mask_generator

from mmseg.models.utils.dacs_transforms import denorm, renorm

from mmseg.models.utils.augment_patch import Augmentations

# from mmseg.models.utils.masking_transforms import DualMaskGenerator, ClassMaskGenerator, SceneMaskGenerator


class IntrinsicAugmentRefine(Module):

    def __init__(self, require_teacher, cfg):
        super(IntrinsicAugmentRefine, self).__init__()

        self.start_iters = cfg['start_iters']
        self.max_bank_size = cfg['max_bank_size']

        # Augmentation Setup
        cfg['refine_aug'].update({'ignore_identity': True})
        self.transforms = Augmentations(cfg['refine_aug'])
        self.k = cfg['k']
        self.refine_conf = cfg['refine_conf']

        # org MIC setup
        self.strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': {
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.2,
                    'hue': 0.2,
                },
                'color_jitter_p': 0.5,
                'blur': 0,
            }

        # max size for cityscapes
        self.source_dataset = {
            'imgs': [],
            'imgs_meta': [],
            'imgs_feat': [],
            'imgs_gt': [],
            'len': 0
        }

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

        for idx in range(img.shape[0]):
            self.source_queue_meta.append(img[idx])
        
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
                 gt_semantic_seg,
                 target_img,
                 target_img_metas,
                 src_feat,
                 pseudo_label=None, # [2, 512, 512]
                 local_iter=None,
                 debug=False):
        # self.update_debug_state(model)
        self.debug = debug
        self.debug_output = {}
        model.debug_output = {}
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        # cityscapes dataset size
        if  self.source_dataset['len'] < 2975:
            self.source_dataset['imgs'].append(img.clone().cpu())
            self.source_dataset['imgs_meta'].append(img_metas)
            self.source_dataset['imgs_feat'].append(src_feat[3].clone().cpu())
            self.source_dataset['imgs_gt'].append(gt_semantic_seg.clone().cpu())
            self.source_dataset['len'] += img.shape[0]

        ### Self-volting memory bank setup
        # src_feat = model.extract_feat(img)
        # 暫時只取最高級特徵
        # print(src_feat[3].shape)
        # self.source_queue = self.update_memo_queue(
        #     self.source_queue.to(device=dev), 
        #     src_feat[3], 
        #     img.cpu().detach().clone()
        # ).to(device='cpu')
        # Warm up
        if local_iter < self.start_iters:
            return pseudo_label, None
        

        # 將list中的tensors堆疊成一個新的tensor
        stacked_tensors = torch.stack(self.source_dataset['imgs_gt'], dim=0).to(device=dev, dtype=pseudo_label.dtype)  # Shape: [Size, B, 1, H, W]

        # 將target調整形狀以匹配stacked_tensors的維度
        expanded_target = pseudo_label.unsqueeze(0).unsqueeze(2)  # Shape: [1, B, 1, H, W]

        # 計算mask來忽略數值為255的位置
        mask = (stacked_tensors != 255)  # Shape: [Size, B, 1, H, W]

        # 執行條件式計算：只有當mask為True時才計算差異
        differences = torch.where(mask, stacked_tensors - expanded_target, 
                                  torch.tensor(0, dtype=int).to(dev))

        # 計算每個tensor與目標的Hamming距離
        hamming_distances = differences.bool().abs().sum(dim=(1, 2, 3, 4))  # Shape: [Size]

        # 計算有效的數值計數來進行正規化
        valid_counts = mask.float().sum(dim=(1, 2, 3, 4))  # Shape: [Size]

        # 進行正規化計算
        normalized_distances = hamming_distances / valid_counts

        # 找到最小距離的索引
        min_distance_idx = torch.argmin(normalized_distances)

        # 取得對應的最近的tensor
        # closest_tensor = tensor_list[min_distance_idx]
        # print(min_distance_idx)

        anchor_imgs = self.source_dataset['imgs'] \
            [min_distance_idx.item()].to(device=dev)
        # anchor_metas = self.source_dataset['imgs_meta'][min_distance_idx.item()]
        anchor_feat = self.source_dataset['imgs_feat'] \
            [min_distance_idx.item()].to(device=dev)

        # 計算target img 的feature 和 predict 以製作mask
        # HRDA: tgt_feat = [feature, boxes], d=[4, dict]
        tgt_logits, tgt_feat = model.generate_pseudo_label(
            target_img, target_img_metas, return_feat=True)
        tgt_soft_label = torch.softmax(tgt_logits.detach(), dim=1)
        max_probs, _ = torch.max(tgt_soft_label, dim=1, keepdim=True)
        pseudo_mask = max_probs < self.refine_conf
        # Dummy Fool-proof
        hrda_backbone = False
        if len(tgt_feat) != 4:
            hrda_backbone = True
            tgt_feat = tgt_feat[0]

        # 生成N個擴增影像
        self.strong_parameters.update({
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)})
        auged_imgs = self.transforms.apply_transforms(
            target_img, means, stds, self.strong_parameters)

        debug_imgs = {
            'source_indice': [],
            'topk_imgs': [],
            'topk_preds': [],
            'auged_imgs': [],
        }
        pseudo_soft_labels = torch.zeros_like(
            pseudo_label, device=dev)
        for B in range(tgt_feat[3].shape[0]):
            # 找距離最近的src feature和距離值
            anchor_flat = anchor_feat[B].view(1, -1)
            target_flat = tgt_feat[3][B].view(1, -1)
            dist_threshold = torch.norm(anchor_flat - target_flat,p=2, dim=1)
            # dist_threshold = anchor_dist.mean()

            # 計算擴增影像的特徵和 pseudo label
            auged_img = torch.stack(auged_imgs[B]).to(device=dev).squeeze()
            # print(auged_img.shape)
            # hrda 直接塞A40會OOM，改用mini-batch
            if hrda_backbone:
                logits, auged_feat = [], []
                bsize = 4
                for mini_b in range(0, auged_img.shape[0], bsize):
                    mini_batch = auged_img[mini_b:mini_b + bsize]
                    logit, feat = model.generate_pseudo_label(
                        mini_batch, target_img_metas, return_feat=True)
                    logits.append(logit)
                    auged_feat.append(feat[0][3])
                logits = torch.cat(logits, dim=0)
                auged_feat = torch.cat(auged_feat, dim=0)
            else:
                logits, auged_feat = model.generate_pseudo_label(
                    auged_img, target_img_metas, return_feat=True)
                auged_feat = auged_feat[3]
            auged_soft_label = torch.softmax(logits.detach(), dim=1)
            del logits
            
            # 計算距離
            anchor_flat = anchor_feat[B].unsqueeze(dim=0)
            anchor_flat = anchor_flat.view(1, -1)
            auged_flat = auged_feat.view(auged_feat.shape[0], -1)  # [size, H*W]

            distances = torch.norm(anchor_flat - auged_flat, p=2, dim=1)  # 计算沿最后一个维度

            # 过滤操作
            mask = distances <= dist_threshold
            # filtered_features = auged_feat[mask]
            filtered_soft_labels = auged_soft_label[mask]
            filtered_distances = distances[mask]
            filtered_imgs = auged_img[mask]
            del auged_feat

            # 找 topk 最近的特征
            values, indices = None, None
            if len(filtered_distances) > 0:  # 确保有特征满足过滤条件
                values, indices = torch.topk(filtered_distances, 
                    min(self.k, len(filtered_distances)), largest=False)
                # topk_closest_features = filtered_features[indices]
                topk_closest_soft_label = filtered_soft_labels[indices]
                topk_auged_imgs = filtered_imgs[indices]

                # apply soft-volting
                summed_labels = topk_closest_soft_label.sum(dim=0)
                soft_labels = summed_labels / topk_closest_soft_label.shape[0]
            else:
                soft_labels = pseudo_label[B].unsqueeze(0)

            pseudo_soft_labels[B] = torch.max(soft_labels, dim=0)[1]

            # debug
            if self.debug:
                debug_imgs['auged_imgs'].append(auged_img)
                if indices is not None:
                    debug_imgs['topk_imgs'].append(topk_auged_imgs)
                    topk_pred = torch.max(topk_closest_soft_label, dim=1)[1]
                    debug_imgs['topk_preds'].append(topk_pred)
                debug_imgs['source_indice'].append(min_distance_idx)

        # 依照threshold 貼上 refine pixel
        refined_pseudo_label = torch.where(
            pseudo_mask.squeeze(), pseudo_soft_labels, pseudo_label)
        # print(soft_labels)
        # print(refined_pseudo_label)

        if self.debug:
            self.debug_output = {
                'Referenced_source': anchor_imgs,
                'auged_imgs': debug_imgs['auged_imgs'],
                'choised_topk_imgs': debug_imgs['topk_imgs'],
                'choised_topk_preds': debug_imgs['topk_preds'],
                'org_pseudo_label': pseudo_label,
                'refined_pseudo_label': refined_pseudo_label,
                'not_pass_pixel': pseudo_mask
            }

        return refined_pseudo_label, self.debug_output