import os
import torch
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class DatasetGenerate(Dataset):
    def __init__(self, image_folder, mask_folder, phase='train', transform=None):
        self.image_files = sorted([image_folder +
                                   file_name for file_name in os.listdir(image_folder)])
        self.mask_files = sorted([mask_folder +
                                  file_name for file_name in os.listdir(mask_folder)])
        self.transform = transform

        train_imgs, val_imgs, train_gts, val_gts = train_test_split(
            self.image_files, self.mask_files, test_size=0.05, random_state=42)

        if phase == 'train':
            self.image_files = train_imgs
            self.mask_files = train_gts
        elif phase == 'val':
            self.image_files = val_imgs
            self.mask_files = val_gts
        else:
            pass

    def __getitem__(self, idx):
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_files[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            augmented = self.transform(image=image, masks=[mask])
            image = augmented['image']
            mask = np.expand_dims(augmented['masks'][0], axis=0) / 255.0

        return image, mask

    def __len__(self):
        return len(self.image_files)


class Test_DatasetGenerate(Dataset):
    def __init__(self, image_folder, mask_folder=None, transform=None):
        self.image_files = sorted([image_folder +
                                   file_name for file_name in os.listdir(image_folder)])
        self.mask_files = sorted([mask_folder +
                                  file_name for file_name in os.listdir(mask_folder)]) if mask_folder is not None else None
        self.image_names = sorted(os.listdir(image_folder))
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.mask_files is not None:
            return image, self.mask_files[idx], original_size, image_name
        else:
            return image, original_size, image_name

    def __len__(self):
        return len(self.image_files)


def get_train_augmentation(img_size):
    transforms = albu.Compose([
        albu.OneOf([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.RandomRotate90(),
        ], p=0.5),
        albu.OneOf([
            albu.RandomContrast(),
            albu.RandomGamma(),
            albu.RandomBrightness(),
        ], p=0.5),
        albu.OneOf([
            albu.MotionBlur(blur_limit=5),
            albu.MedianBlur(blur_limit=5),
            albu.GaussianBlur(blur_limit=5),
            albu.GaussNoise(var_limit=(5.0, 20.0)),
        ], p=0.5),
        albu.Resize(img_size, img_size, always_apply=True),
        albu.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transforms


def get_test_augmentation(img_size):
    transforms = albu.Compose([
        albu.Resize(img_size, img_size, always_apply=True),
        albu.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transforms


def get_loader(img_folder, gt_folder, phase: str, batch_size, shuffle, num_workers, transform):
    if phase == 'test':
        dataset = Test_DatasetGenerate(img_folder, gt_folder, transform)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        dataset = DatasetGenerate(img_folder, gt_folder, phase, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers, drop_last=True)

    print(f'{phase} length : {len(dataset)}')

    return data_loader


def gt_to_tensor(gt):
    gt = cv2.imread(gt)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) / 255.0
    gt = np.where(gt > 0.5, 1.0, 0.0)
    gt = torch.tensor(gt, device='cuda', dtype=torch.float32)
    gt = gt.unsqueeze(0).unsqueeze(1)

    return gt


if __name__ == '__main__':
    train_loader = get_loader('data/DUTS-TR/images/', 'data/DUTS-TR/masks/',
                              'all', 1, True, 1, get_train_augmentation(352))
    image, mask = next(iter(train_loader))

    plt.subplot(121)
    plt.imshow(image[0])
    plt.subplot(122)
    plt.imshow(mask[0][0]*255, cmap='gray')
    plt.show()
