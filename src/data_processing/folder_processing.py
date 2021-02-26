import os
import numpy as np
import shutil
from tqdm import tqdm


def copy(source, des, folder, image_list, dtype='train'):
    for image_name in image_list:
        image_src = os.path.join(source, folder, image_name)
        image_des = os.path.join(des, dtype, folder, image_name)

        os.makedirs(os.path.dirname(image_des), exist_ok=True)

        shutil.copy(image_src, image_des)
    
def split_train_test(source, des, p=0.7):
    folders = os.listdir(source)

    for folder in tqdm(folders):
        image_list = os.listdir(os.path.join(source, folder))[:1500]
        num_train = int(p*len(image_list))
        copy(source, des, folder, image_list[:num_train], dtype='train')
        copy(source, des, folder, image_list[num_train:], dtype='val')

if __name__ == "__main__":
    split_train_test("/home/tupm/datasets/complete_data", "/home/tupm/projects/classification_pytorch/datasets")
        
