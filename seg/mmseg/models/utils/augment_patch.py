import torch

from mmseg.ops import resize
from mmseg.models.utils.rand_augment import RandAugment
from mmseg.models.utils.dacs_transforms import denorm, renorm, color_jitter


class Augmentations:
    def __init__(self, cfg):
        self.aug_type = cfg['type']
        self.augment_setup = cfg['augment_setup']
        self.num_diff_aug = cfg['num_diff_aug']
        self.patch_size = cfg['patch_size'][0]
        
        self.transforms = []
        for i in range(self.num_diff_aug):
            if self.aug_type == "RandAugment":
                augmenter = RandAugment(
                    self.augment_setup['n'], 
                    self.augment_setup['m'],
                    seed=i
                )

            self.transforms.append(augmenter)

    @torch.no_grad()
    def apply_transforms(self, imgs, basic_aug_param):
        auged_imgs = []
        for batch_idx in range(imgs.shape[0]):
            # 由於 RandAug 是 PIL based 的所以只能迴圈處理
            batched_auged_img = []
            for transform in self.transforms:    
                auged_img = transform(imgs[batch_idx], basic_aug_param)
                batched_auged_img.append(auged_img)
            auged_imgs.append(batched_auged_img)
        return auged_imgs
    
    @torch.no_grad()
    def generate_augpatch(self, imgs, means, stds, basic_aug_param=None):
        imgs = denorm(imgs, means[0].unsqueeze(0), stds[0].unsqueeze(0))
        auged_imgs = self.apply_transforms(imgs, basic_aug_param)

        B, C, H, W = imgs.shape
        mshap = 1, 1, round(H / self.patch_size), round(W / self.patch_size)
        for batch in range(B):
            
            augment_indice = torch.randint(
                high=self.num_diff_aug, size=mshap, device=imgs.device)
            augment_indice = resize(
                augment_indice.float(), size=(H, W))[0].expand((C, H, W))

            current_aug_set = auged_imgs[batch]
            for target, auged_img in enumerate(current_aug_set):
                imgs[batch][augment_indice == target] = \
                    auged_img[augment_indice == target]

        imgs = renorm(imgs, means[0].unsqueeze(0), stds[0].unsqueeze(0))
        return imgs
    