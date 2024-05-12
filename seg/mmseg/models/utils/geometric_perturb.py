import torch
import kornia
import kornia.augmentation as K

class RandomTransform:
    def __init__(self):
        # Define the transformations: random rotate, translate, and shear
        self.transforms = torch.nn.Sequential(
            K.RandomRotation(degrees=30),
            K.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(10, 10))
        )
    
    def __call__(self, patches):
        # Apply transformations to the batch of patches
        # Expect patches of shape [B*N, C, patch_size, patch_size]
        # where B is batch size, N is number of patches per batch
        return self.transforms(patches)
    

class GeometricPerturb:
    def __init__(self, patch_size):
        self.patch_size = patch_size[0]
        self.perturb = RandomTransform()
    
    @torch.no_grad()
    def perturb_img(self, imgs):
        B, C, H, W = imgs.shape
        
        # Unfold the image into patches
        patches = imgs.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(
            -1, C, self.patch_size, self.patch_size)  # Reshape to [total_patches, C, H_patch, W_patch]
        
        # Apply random geometric perturbations
        perturbed_patches = self.perturb(patches)

        # Fold the perturbed patches back into an image
        perturbed_patches = perturbed_patches.view(
            B, C, H // self.patch_size, W // self.patch_size, self.patch_size, self.patch_size)
        perturbed_patches = perturbed_patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        perturbed_patches = perturbed_patches.view(B, H, W, C)
        
        return perturbed_patches.permute(0, 3, 1, 2)