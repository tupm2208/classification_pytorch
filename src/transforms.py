import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def transform(image_size=512, is_training=True):
    
    if is_training:
        return  A.Compose([
                    A.Resize(width=image_size, height=image_size,),
                    A.RandomContrast(limit=0.2, p=0.4),
                    A.ColorJitter(),
                    A.Rotate(40),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.Normalize(
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ])
    
    return  A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.Normalize(
                    mean=[0, 0, 0],
                    std=[1, 1, 1],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ])