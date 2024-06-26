import numpy as np
import torch
import cv2
import albumentations as al
import torchvision.transforms as transforms
from PIL import Image

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


class PixMatch:
    def __init__(self) -> None:
        pass

    def get_augmentation(self):
        return al.Compose([
            al.RandomResizedCrop(512, 512, scale=(0.2, 1.)),
            al.Compose([
                al.RandomBrightnessContrast(p=1),
                al.HueSaturationValue(p=1),
            ], p=0.8),
            al.ToGray(p=0.2),
            al.GaussianBlur(5, p=0.5),
        ])

    def augment(self, images, labels, aug):
        labels_are_3d = (len(labels.shape) == 4)
        if labels_are_3d:
            labels = labels.permute(0, 2, 3, 1)

        aug_images, aug_labels = [], []
        for image, label in zip(images, labels):
            image = image.numpy().transpose(1, 2, 0)  # C, H, W -> H, W, C
            image = (image * 255.0 + IMG_MEAN).astype(np.uint8)
            label = label.numpy()

            data = aug(image=image, mask=label)
            image, label = data['image'], data['mask']

            image = torch.from_numpy(((image.astype(np.float32) - IMG_MEAN) / 255.0).transpose(2, 0, 1))
            label = torch.from_numpy(label)
            if not labels_are_3d:
                label = label.long()

            aug_images.append(image)
            aug_labels.append(label)

        images = torch.stack(aug_images, dim=0)
        labels = torch.stack(aug_labels, dim=0)

        if labels_are_3d:
            labels = labels.permute(0, 3, 1, 2)
        return images, labels


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    image = transform(image)
    return image


def save_image(image, output_path):
    transform = transforms.ToPILImage()
    image = transform(image)
    image.save(output_path)


if __name__ == "__main__":
    img_path = 'data/cityscapes/leftImg8bit/train/stuttgart/stuttgart_000177_000019_leftImg8bit.png'
    output_path = './augmented_image.png'

    # Load the image
    img = load_image(img_path)

    # Normalize the image
    img = img * 255.0  # Scale to [0, 255]
    img = img - torch.tensor(IMG_MEAN).view(3, 1, 1)
    img = img / 255.0  # Scale back to [0, 1]
    img = img.unsqueeze(0)  # Add batch dimension

    # Placeholder for label (dummy data)
    label = torch.zeros((1, img.shape[2], img.shape[3]), dtype=torch.long)

    # Initialize PixMatch and get augmentation
    pixmatch = PixMatch()
    aug = pixmatch.get_augmentation()

    # Apply augmentation
    augmented_img, _ = pixmatch.augment(img, label, aug)

    # Remove batch dimension and denormalize
    augmented_img = augmented_img.squeeze(0)
    augmented_img = augmented_img * 255.0 + torch.tensor(IMG_MEAN).view(3, 1, 1)
    augmented_img = augmented_img.byte().numpy().transpose(1, 2, 0)

    # Save the augmented image
    save_image(augmented_img, output_path)
    print(f'Augmented image saved to {output_path}')
