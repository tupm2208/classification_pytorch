import pandas as pd
import numpy as np
from multiprocessing import Pool



num_sample = 1700


def handle_upsample(group):
    global num_sample
    k, df = group

    l = df.shape[0]
    nrp = num_sample - l
    if nrp <= 0:
        return df[:num_sample]

    extended = df.iloc[np.random.choice(l, nrp)].reset_index()

    return pd.concat((df, extended)).reset_index()


def upsampling(csv_path, out_path='datasets/train_balance.csv', n_cores=8):
    df = pd.read_csv(csv_path)

    group = df.groupby('class_id')
    pool = Pool(n_cores)
    arr = pool.map(handle_upsample, group)
    new_df = pd.concat(arr)

    pool.close()
    pool.join()

    new_df = new_df.drop(['level_0', 'index'], axis=1)
    print(new_df.head())
    new_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    upsampling('/home/tupm/projects/classification_pytorch/datasets/train_fix_1.csv')