import os
import numpy as np
import shutil
from tqdm import tqdm
import cv2

exclusive = ["detective and mystery book", "historical fiction", "graphic novel", "non-fiction", "literary fiction"]

def copy(source, des, folder, image_list, dtype='train'):
    for image_name in image_list:
        image_src = os.path.join(source, folder, image_name)
        image_des = os.path.join(des, dtype, folder, image_name)

        os.makedirs(os.path.dirname(image_des), exist_ok=True)

        shutil.copy(image_src, image_des)
    
def split_train_test(source, des, p=0.7):
    folders = os.listdir(source)

    for folder in tqdm(folders):
        if folder in exclusive:
            continue
        image_list = os.listdir(os.path.join(source, folder))[:1500]
        num_train = int(p*len(image_list))
        copy(source, des, folder, image_list[:num_train], dtype='train')
        copy(source, des, folder, image_list[num_train:], dtype='val')


def remove_error_file(source):
    folders = os.listdir(source)

    for folder in tqdm(folders):
        image_list = [os.path.join(source, folder, e) for e in os.listdir(os.path.join(source, folder))]

        for src in image_list:
            img = cv2.imread(src)

            if img is None:
                os.remove(src)

if __name__ == "__main__":
    split_train_test("/home/tupm/SSD/CAT_datasets/download", "/home/tupm/SSD/CAT_datasets/datasets", p=0.9)
    # remove_error_file("/home/tupm/SSD/CAT_datasets/download")
        
