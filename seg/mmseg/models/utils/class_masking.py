import torch


class ClassMaskGenerator:

    def __init__(self, mask_type, mask_block_size):
        # self.mask_ratio = mask_ratio
        # self.mask_block_size = mask_block_size
        # self.hint_ratio = hint_ratio

        self.mask_type = mask_type
        self.mask_block_size = int(mask_block_size[0])

    @torch.no_grad()
    def DHA(self, unfolded_block_mask, hint_ratio):
        # print(unfolded_block_mask.shape)
        hint_patch_num = int(torch.sum(unfolded_block_mask[0, :]) * hint_ratio)

        ones_indices = torch.nonzero(unfolded_block_mask[0, :])
        
        ones_indices = torch.where(unfolded_block_mask == True)[1]
        random_indices = torch.randperm(len(ones_indices))[:hint_patch_num]
        # selected_indices = torch.randperm(ones_indices.size(0))[:hint_patch_num]

        hint_block_mask = unfolded_block_mask.clone()
        hint_block_mask[:, ones_indices[random_indices]] = 0

        return hint_block_mask, hint_patch_num


    @torch.no_grad()
    def generate_mask(self, imgs, lbls, pseudo_label_region):
        B, C, H, W = imgs.shape
        ### Implement of Class Masking ###
        input_mask = torch.zeros((B, 1, H, W), device=imgs.device)

        mask_targets, hint_patch_nums = [], []
        for batch in range(B):
            # ignore non-reliable region
            valid_mask = (pseudo_label_region[batch] == True)
            current_classes = torch.unique(lbls[batch][valid_mask])
            # ignore "void" class
            void_mask = current_classes != 255
            current_classes = current_classes[void_mask]
            
            mask_target = current_classes[torch.randint(0, len(
                current_classes), (1,)).item()]
            mask_targets.append(mask_target.item())
            
            class_mask = (lbls[batch] == mask_target).float()
            # class_mask = torch.logical_and(class_mask, pseudo_label_region[batch]).float()

            unfolded_mask = torch.nn.functional.unfold(class_mask.unsqueeze(dim=0), 
                kernel_size=self.mask_block_size, stride=self.mask_block_size)
            # True for mask patch, False for appear patch
            unfolded_block_mask = torch.any(unfolded_mask.bool(), dim=1)

            # # dynamc hint adjustment
            # if self.hint_ratio > 0.0:
            #     hint_block_mask, hint_patch_num = self.DHA(unfolded_block_mask, local_hint_ratio)
            #     hint_patch_nums.append(hint_patch_num)
            # else:
            #     hint_block_mask = unfolded_block_mask

            # # add remain mask block
            # operation_mask = torch.zeros(unfolded_block_mask.shape, dtype=bool, device=imgs.device)
            # if torch.sum(hint_block_mask) / unfolded_block_mask.numel() < self.mask_ratio:
            #     remain_mask_times = int(
            #         self.mask_ratio * unfolded_block_mask.numel() - torch.sum(hint_block_mask))
                
            #     zero_indices = torch.where(unfolded_block_mask == False)[1]
            #     random_indices = torch.randperm(len(zero_indices))[:remain_mask_times]

            #     operation_mask[:, zero_indices[random_indices]] = True

            # # final mask
            # final_block_mask = torch.logical_or(operation_mask, hint_block_mask)
            # block_mask = (~final_block_mask.expand(unfolded_mask.shape)).float()

            block_mask = (~unfolded_block_mask.expand(unfolded_mask.shape)).float()
            
            input_mask[batch, :, :, :] = torch.nn.functional.fold(block_mask, lbls.shape[2:],
                kernel_size=self.mask_block_size, stride=self.mask_block_size)

        return input_mask, mask_targets

    @torch.no_grad()
    def mask_image(self, imgs, lbls, valid_pseudo_mask):
        # return_imgs = []
        masks, mask_targets = self.generate_mask(
            imgs, lbls, valid_pseudo_mask)
        return imgs * masks, mask_targets