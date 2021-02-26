import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2


class DataFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        super(DataFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index]*len(files)))

    def __len__(self):
        return len(self.data)

    def __num_class__(self):
        return len(self.class_names)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        # image = np.array(Image.open().convert("RGB"))
        image = cv2.imread(os.path.join(root_and_dir, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        return image, label

