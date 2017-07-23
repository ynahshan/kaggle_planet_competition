import cv2
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from sklearn.metrics import fbeta_score
import glob
from sklearn.model_selection import train_test_split

def get_labels():
    labels = [
        'slash_burn',
        'clear',
        'blooming',
        'primary',
        'cloudy',
        'conventional_mine',
        'water',
        'haze',
        'cultivation',
        'partly_cloudy',
        'artisinal_mine',
        'habitation',
        'bare_ground',
        'blow_down',
        'agriculture',
        'road',
        'selective_logging'
    ]

    l_map = {l: i for i, l in enumerate(labels)}
    inv_map = {i: l for l, i in l_map.items()}
    return labels, l_map, inv_map


def npz_to_ndarray(npz_container):
    np_arr = None
    # Assume one array in container
    for arr in npz_container:
        np_arr = npz_container[arr]
        break
    return np_arr


def load_jpg_data(df_csv, data_dir, label_map, img_size=None, subset_size=None):
    X = []
    Y = []

    # for f, tags in tqdm(df_train.sample(subset_size).values, miniters=1000):
    if subset_size is not None:
        data_progress = tqdm(df_csv.sample(subset_size).values, miniters=1000)
    else:
        data_progress = tqdm(df_csv.values, miniters=1000)

    for f, tags in data_progress:
        f_name = '{}.jpg'.format(f)
        img = cv2.imread(os.path.join(data_dir, f_name))
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        if img_size is not None:
            X.append(cv2.resize(img, (img_size, img_size)))
        else:
            X.append(img)
        Y.append(targets)
    print("Creating numpy array for data...")
    X = np.array(X, np.float16) / 255.
    Y = np.array(Y, np.uint8)
    return X, Y

def to_tagging(one_hot_data, inv_label_map):
    res = pd.DataFrame(index=range(len(one_hot_data)), columns=['tags'])
    for j in range(len(one_hot_data)):
        tags = []
        for i in range(17):
            if one_hot_data[j][i] == 1:
                tags.append(inv_label_map[i])
        res['tags'][j] = ' '.join(sorted(tags))
    return res

import copy
def find_ratios(y_true, y_pred, default=0.2):
    threshold = [0.2]*17
    best_score = fbeta_score(y_true, (y_pred > threshold).astype(int), beta=2, average='samples')
    step = 0.025
    n = int(1/step)
    for j in range(17):
        temp_threshold = threshold[:] # [0.2]*17
        r = step
        for _ in range(n):
            temp_threshold[j] = r
            score = fbeta_score(y_true, (y_pred > temp_threshold).astype(int), beta=2, average='samples')
            if score > best_score:
                best_score = score
                threshold[j] = r
            r += step

    return threshold

class ModelSelector:
    def __init__(self, model_per_tag):
        self.models_per_tag = model_per_tag

    def select_model(self, tag_id):
        model_prop = self.models_per_tag[tag_id]
        matcher = os.path.join(model_prop['dir'], model_prop['name']) + '_' + str(model_prop['epoch']) + '*.h5'
        m_paths = glob.glob(matcher)
        if len(m_paths) != 1:
            print("Error: Invalid number of files compatible with matcher %s" % str(matcher))
            print(m_paths)
            return None

        return m_paths[0]

    def get_input_sizes(self):
        sizes = []
        for p in self.models_per_tag:
            if p['in_size'] not in sizes:
                sizes.append(p['in_size'])
        return sizes

    def get_input_size(self, tag_id):
        model_prop = self.models_per_tag[tag_id]
        return model_prop['in_size']

class Data:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_train = None
        self.X_valid = None
        self.Y_train = None
        self.Y_valid = None

    def train_test_split(self):
        if self.X_train is None or self.X_valid is None or self.Y_train is None or self.Y_valid is None:
            X_train, X_valid, Y_train, Y_valid = train_test_split(self.X, self.Y, test_size=0.2, random_state=0)
        return X_train, X_valid, Y_train, Y_valid

class InputManager:
    def __init__(self, df_train, train_data_dir):
        self.df_train = df_train
        self.train_data_dir = train_data_dir

        labels, label_map, inv_label_map = get_labels()
        self.labels = labels
        self.label_map = label_map
        self.inv_label_map = inv_label_map
        self.data = {}

    def load_inputs(self, in_sizes):
        for in_size in in_sizes:
            print("Loading data and resizing to %d" % in_size)
            X, Y = load_jpg_data(self.df_train, self.train_data_dir, self.label_map, img_size=in_size)
            self.data[in_size] = Data(X, Y)
            print("Done. Shape of X: %s, shape of Y: %s" % (X.shape, Y.shape))

    def get_train_test(self, in_size):
        data = self.data[in_size]
        return data.train_test_split()

    def get_data(self, in_size):
        data = self.data[in_size]
        return data.X, data.Y

