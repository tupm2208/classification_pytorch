import pandas as pd
import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np



def check_image(df):
    root_dir = "/home/tupm/SSD/CAT_datasets/workspace/smartcare/topics_v3"
    idxs = []
    for i in tqdm(range(df.shape[0])):
        img_file, label_name, label = df.iloc[i].values
        img_path = os.path.join(root_dir, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(img_path)
            idxs.append(i)
    return df.loc[~df.index.isin(idxs)], df.loc[df.index.isin(idxs)]


def check_file(input_file, output_file, n_cores=8):

    colnames=[ 'image_path', 'label_name', 'label'] 
    df = pd.read_csv(input_file, names=colnames, header=None)
    
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    arr = pool.map(check_image, df_split)
    ps, nps = [], []
    for e1, e2 in arr:
        ps.append(e1)
        nps.append(e2)
    new_df = pd.concat(ps)
    n_df = pd.concat(nps)
    pool.close()
    pool.join()
    
    new_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    check_file("../datasets/test_fix_1.csv", "../datasets/test_fix_2.csv", 12)    