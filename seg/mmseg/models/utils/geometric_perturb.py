import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.geometry.transform as T

# class RandomTransform:
#     def __init__(self, perturb_range, perturb_prob):
#         # Define the transformations: random rotate, translate, and shear
#         self.transforms = torch.nn.Sequential(
#             # K.RandomRotation(degrees=30),
#             K.RandomAffine(degrees=(-15, 15), translate=(0.15, 0.15), shear=(-15, 15), p=0.3)
#         )
    
#     def __call__(self, patches):
#         # Apply transformations to the batch of patches
#         # Expect patches of shape [B*N, C, aug_block_size, aug_block_size]
#         # where B is batch size, N is number of patches per batch
#         return self.transforms(patches)
    

# class GeometricPerturb:
#     def __init__(self, block_size, cfg):
#         self.aug_block_size = block_size
#         self.perturb = RandomTransform(cfg['perturb_range'], cfg['perturb_prob'])
    
#     @torch.no_grad()
#     def perturb_img_and_lbl(self, imgs):
#         B, C, H, W = imgs.shape
        
#         # Unfold the image into patches
#         patches = imgs.unfold(2, self.aug_block_size, self.aug_block_size).unfold(3, self.aug_block_size, self.aug_block_size)
#         patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # Reshape to [B, nH, nW, C, H_patch, W_patch]
#         patches = patches.view(-1, C, self.aug_block_size, self.aug_block_size)  # Flatten patches
        
#         # Apply random geometric perturbations
#         perturbed_patches = self.perturb(patches)
        
#         # Fold the perturbed patches back into an image
#         perturbed_patches = perturbed_patches.view(B, H // self.aug_block_size, W // self.aug_block_size, C, self.aug_block_size, self.aug_block_size)
#         perturbed_patches = perturbed_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
#         perturbed_patches = perturbed_patches.view(B, C, H, W)

#         # Mix two imgs
#         mask = perturbed_patches == 0
#         perturbed_patches[mask] = imgs[mask]
        
#         return perturbed_patches
    


class RandomTransform(nn.Module):
    def __init__(self, perturb_range, perturb_prob):
        super(RandomTransform, self).__init__()
        self.transforms = K.RandomAffine(
            degrees=(-perturb_range[0], perturb_range[0]),
            translate=(perturb_range[1] / 100, perturb_range[1] / 100),
            shear=(-perturb_range[2], perturb_range[2]),
            p=perturb_prob,
            return_transform=True)

    def forward(self, patches):
        # Apply transformations to the batch of patches
        # Expect patches of shape [B*N, C, aug_block_size, aug_block_size]
        # where B is batch size, N is number of patches per batch
        transformed_patches, transform_matrices = self.transforms(patches)
        # Adjust transform matrices to [B*N, 2, 3]
        transform_matrices = transform_matrices[:, :2, :]
        return transformed_patches, transform_matrices
    
class GeometricPerturb(nn.Module):
    def __init__(self, block_size, cfg):
        super(GeometricPerturb, self).__init__()
        self.aug_block_size = block_size
        self.perturb = RandomTransform(cfg['perturb_range'], cfg['perturb_prob'])

    @torch.no_grad()
    def perturb_img_and_lbl(self, imgs, lbls):
        B, C, H, W = imgs.shape
        _, _, H_mask, W_mask = lbls.shape
        assert H == H_mask and W == W_mask, "Image and label dimensions must match"

        # Preprocess labels: change 0 to -1
        lbls[lbls == 0] = -1

        # Unfold the images into patches
        img_patches = imgs.unfold(2, self.aug_block_size, self.aug_block_size).unfold(3, self.aug_block_size, self.aug_block_size)
        img_patches = img_patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # Reshape to [B, nH, nW, C, H_patch, W_patch]
        img_patches = img_patches.view(-1, C, self.aug_block_size, self.aug_block_size)  # Flatten patches

        # Unfold the labels into patches
        lbl_patches = lbls.unfold(2, self.aug_block_size, self.aug_block_size).unfold(3, self.aug_block_size, self.aug_block_size)
        lbl_patches = lbl_patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # Reshape to [B, nH, nW, 1, H_patch, W_patch]
        lbl_patches = lbl_patches.view(-1, 1, self.aug_block_size, self.aug_block_size)  # Flatten patches
        
        # Apply random geometric perturbations to both image and label patches
        perturbed_img_patches, transform_matrices = self.perturb(img_patches)
        perturbed_lbl_patches = T.warp_affine(lbl_patches.float(), transform_matrices, dsize=(self.aug_block_size, self.aug_block_size), mode='nearest', padding_mode='zeros').long()

        # Reshape back to original shape for combining
        perturbed_img_patches = perturbed_img_patches.view(B, H // self.aug_block_size, W // self.aug_block_size, C, self.aug_block_size, self.aug_block_size)
        perturbed_img_patches = perturbed_img_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        perturbed_img = perturbed_img_patches.view(B, C, H, W)

        perturbed_lbl_patches = perturbed_lbl_patches.view(B, H // self.aug_block_size, W // self.aug_block_size, 1, self.aug_block_size, self.aug_block_size)
        perturbed_lbl_patches = perturbed_lbl_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        perturbed_lbl = perturbed_lbl_patches.view(B, 1, H, W)

        # Replace padding values with original image and label values
        perturbed_img[perturbed_img == 0] = imgs[perturbed_img == 0]
        perturbed_lbl[perturbed_lbl == 0] = lbls[perturbed_lbl == 0]

        # Postprocess labels: change -1 back to 0
        perturbed_lbl[perturbed_lbl == -1] = 0

        return perturbed_img, perturbed_lbl