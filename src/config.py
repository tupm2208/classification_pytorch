import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 1
PIN_MEMORY = True
LOAD_MODEL = False
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
SIZE = 416

train_transforms = A.Compose([
    A.Resize(width=SIZE, height=SIZE,),
    # A.RandomCrop(width=SIZE, height=SIZE),
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

val_transforms = A.Compose([
    A.Resize(height=SIZE, width=SIZE),
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])