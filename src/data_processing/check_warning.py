import warnings
import cv2
import pandas as pd
import os
from tqdm import tqdm

warnings.filterwarnings("error")

df = pd.read_csv("../datasets/val_fix_2.csv")

for i in tqdm(range(df.shape[0])):
    path, name, label = df.iloc[i].values

    image_path = os.path.join("/home/tupm/SSD/CAT_datasets/workspace/smartcare/topics_v3", path)

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(image_path)
    except:
        print(image_path)