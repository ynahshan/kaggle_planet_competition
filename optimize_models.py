import argparse
import os
from utils.data_utils import *
from glob import glob
import pandas as pd
import numpy as np
import shutil


pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.colheader_justify', 'left')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ml', '--max_loss', type=float, help='(int) Maximal loss for models to be evaluated and dumped',
                        default='0.1')
    args = parser.parse_args()

    # Currently support only single loss folder
    res_dir = os.path.join('predictions', '0.09')
    df_l = pd.read_csv(os.path.join(res_dir, 'models.csv'))
    # print(df_l)

    preds = []
    for npz_path in df_l.res_path:
        p = npz_to_ndarray(np.load(npz_path))
        # print(npz_path)