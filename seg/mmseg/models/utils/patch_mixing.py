import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMixingGenerator(nn.Module):
    def __init__(self, cfg):
        super(PatchMixingGenerator, self).__init__()
        self.aug_block_size = cfg['aug_block_size']
        self.mixing_ratio = cfg['mixing_ratio']

    @torch.no_grad()
    def mixing_img_and_lbl(self, src_imgs, tgt_imgs, src_lbls, tgt_lbls):
        B, C, H, W = src_imgs.shape

        # 生成随机遮罩
        mshape = B, 1, H // self.aug_block_size, W // self.aug_block_size
        input_mask = torch.rand(mshape, device=src_imgs.device)
        input_mask = (input_mask > self.mixing_ratio).float()
        input_mask = F.interpolate(input_mask, size=(H, W), mode='nearest')

        # 找到src_imgs完全为零的patch位置
        # zero_mask = F.adaptive_avg_pool2d((src_imgs == 0).float(), (H // self.aug_block_size, W // self.aug_block_size))
        # zero_mask = (zero_mask.sum(dim=1, keepdim=True) == C).float()  # 每个patch中所有通道全为零的位置
        zero_mask = src_imgs.bool()
        input_mask = torch.sum(input_mask * zero_mask, 1).unsqueeze(1).bool()


        # 将完全为零的位置从input_mask中去除
        # input_mask = input_mask * (1 - F.interpolate(zero_mask, size=(H, W), mode='nearest'))

        # 使用遮罩替换patch
        # print(tgt_lbls.shape)
        src_imgs = src_imgs * ~(input_mask) + tgt_imgs.squeeze() * input_mask
        src_lbls = src_lbls * ~(input_mask) + tgt_lbls.unsqueeze(1) * input_mask
        # src_imgs[~input_mask] = tgt_imgs[input_mask]
        # src_lbls[~input_mask] = tgt_lbls[input_mask]
        # print(src_imgs.shape)
        # print(src_lbls.shape)

        return src_imgs, src_lbls