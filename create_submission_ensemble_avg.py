import argparse
import os
from utils.data_utils import *
from glob import glob
from keras.models import load_model
import pandas as pd
import numpy as np
import keras.backend as K
import shutil
from evaluate_ensemble import get_weight

def get_test_df():
    work_dir = os.path.dirname(os.path.realpath(__file__))
    inputs_dir = os.path.join(work_dir, 'inputs')
    print("Read inputs from: %s" % inputs_dir)
    df_test = pd.read_csv(os.path.join(inputs_dir, 'sample_submission_v2.csv'))
    return df_test

def create_submission(pred, type, sufix):
    df_test = get_test_df()
    p_tags = to_tagging(pred, inv_label_map)
    df_test.tags = p_tags.tags
    file_name = 'submission_' + type + '_of_' + sufix + '.csv'
    df_test.to_csv(os.path.join(submission_dir, file_name), index=False)
    print("%s created" % os.path.join(submission_dir, file_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', '--ensemble_csv', help='(String) Path to list of models to use for ensemble.', default=None, required=True)
    parser.add_argument('-i', '--in_size', type=int, help='(int) The size of the image', default='128')
    parser.add_argument('-dev', '--dev',
                        help='(String) Specify if run in development mode to use small network and subset of data.',
                        action='store_true', default=False)
    parser.add_argument('-tag', '--tag',
                        help='(String) Specify tag to be added to the directory.', required=True)
    args = parser.parse_args()


    print("Loading models...\n")
    if not os.path.exists(args.ensemble_csv):
        print("Error: file not exist %s" % args.ensemble_csv)
        exit(-1)

    df_ensemble = pd.read_csv(args.ensemble_csv)

    work_dir = os.path.dirname(os.path.realpath(__file__))
    inputs_dir = os.path.join(work_dir, 'inputs')
    print("Read inputs from: %s" % inputs_dir)
    df_test = pd.read_csv(os.path.join(inputs_dir, 'sample_submission_v2.csv'))

    test_data_dir = os.path.join(inputs_dir, 'test-jpg')
    if not os.path.exists(test_data_dir):
        print("Error: Mising data folder: %s" % test_data_dir)
        exit(-1)

    labels, label_map, inv_label_map = get_labels()
    print("Loading test data")
    if args.dev:
        img_size = args.in_size
        X, _ = load_data(df_test, test_data_dir, label_map, img_size=img_size, subset_size=100)
    else:
        img_size = args.in_size
        X, _ = load_data(df_test, test_data_dir, label_map, img_size=img_size)

    print("Evaluate models")
    preds = []
    M_Num = len(df_ensemble)
    for i, m in enumerate(df_ensemble.path):
        model = load_model(m)
        print("Predicting model %s" % m)
        preds.append(model.predict(X))
        del model
        K.clear_session()

    preds = np.array(preds)
    avg_pred = preds.mean(axis=0)
    avg_pred_final = (avg_pred > 0.2).astype(int)

    sufix = args.tag
    submission_dir = os.path.join('submissions', 'ensemble_avg_' + sufix)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    print("\nCreating submission files for ensemble model...")
    create_submission(avg_pred_final, 'avg', sufix)
    print('Done\n')
