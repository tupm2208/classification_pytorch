import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import pandas as pd


class DataFolder(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False, csv_path=None):
        super(DataFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        
        self.is_test = is_test

        self.df = pd.read_csv(csv_path)
        self.class_names = self.df['class_name'].unique().tolist()

        print(self.root_dir)
        print("number of data:", self.__len__())
        print("number of classes:", self.__num_class__())
        print()

    def __len__(self):
        # return len(self.data)
        return self.df.shape[0]

    def __num_class__(self):
        return len(self.class_names)

    def __getitem__(self, index):
        img_file, classname, label = self.df.iloc[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image is None:
            print(img_path)
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        
        if self.is_test:
            return image, label, img_path
        
        return image, label


