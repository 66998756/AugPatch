import torch
import kornia
import kornia.augmentation as K

class RandomTransform:
    def __init__(self):
        # Define the transformations: random rotate, translate, and shear
        self.transforms = torch.nn.Sequential(
            K.RandomRotation(degrees=30),
            K.RandomAffine(degrees=0, translate=(0.25, 0.25), shear=(30, 30))
        )
    
    def __call__(self, patches):
        # Apply transformations to the batch of patches
        # Expect patches of shape [B*N, C, aug_block_size, aug_block_size]
        # where B is batch size, N is number of patches per batch
        return self.transforms(patches)
    

class GeometricPerturb:
    def __init__(self, aug_block_size):
        self.aug_block_size = aug_block_size
        self.perturb = RandomTransform()
    
    @torch.no_grad()
    def perturb_img(self, imgs):
        B, C, H, W = imgs.shape
        
        # Unfold the image into patches
        patches = imgs.unfold(2, self.aug_block_size, self.aug_block_size).unfold(3, self.aug_block_size, self.aug_block_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # Reshape to [B, nH, nW, C, H_patch, W_patch]
        patches = patches.view(-1, C, self.aug_block_size, self.aug_block_size)  # Flatten patches
        
        # Apply random geometric perturbations
        perturbed_patches = self.perturb(patches)
        
        # Fold the perturbed patches back into an image
        perturbed_patches = perturbed_patches.view(B, H // self.aug_block_size, W // self.aug_block_size, C, self.aug_block_size, self.aug_block_size)
        perturbed_patches = perturbed_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        perturbed_patches = perturbed_patches.view(B, C, H, W)
        
        return perturbed_patches