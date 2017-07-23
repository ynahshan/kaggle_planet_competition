import argparse
import os
from utils.data_utils import *
from glob import glob
import pandas as pd
import numpy as np
import shutil

class ModelDescription:
    def __init__(self, m_dir, h5_file):
        f_name, f_ext = os.path.splitext(h5_file)
        ns = f_name.split('_')
        self.loss = float(ns[-1])
        self.epoch = int(ns[-2])
        self.name = '_'.join(ns[:-2])
        self.model_path = os.path.join(m_dir, h5_file)
        self.fname = f_name


pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.colheader_justify', 'left')

def load_models_to_dataframe():
    models_by_type = []
    models_dir = 'models'
    for dir_ in os.listdir(models_dir):
        models = {}
        m_dir = os.path.join(models_dir, dir_)
        for f in os.listdir(m_dir):
            _, f_ext = os.path.splitext(f)
            if f_ext == '.h5':
                md = ModelDescription(m_dir, f)
                models_by_type.append(md)

    df = pd.DataFrame(columns=['model', 'model_name', 'loss', 'epoch', 'path'], index=range(len(models_by_type)))
    for i, md in enumerate(models_by_type):
        df.loc[i].model = md.fname
        df.loc[i].model_name = md.name
        df.loc[i].loss = md.loss
        df.loc[i].epoch = md.epoch
        df.loc[i].path = md.model_path

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-all', '--all_sorted',
                        help='(String) print all models sorted by loss.',
                        action='store_true', default=False)
    args = parser.parse_args()

    df = load_models_to_dataframe()

    if args.all_sorted:
        print(df.sort_values(by='loss'))
        print("\n")
