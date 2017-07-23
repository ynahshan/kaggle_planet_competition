import argparse
import os
from utils.data_utils import *
from glob import glob
from keras.models import load_model
import pandas as pd
import numpy as np
import keras.backend as K
import shutil

def vote(pred):
    return (pred.mean(axis=0) > 0.5).astype(int)

def get_test_df():
    work_dir = os.path.dirname(os.path.realpath(__file__))
    inputs_dir = os.path.join(work_dir, 'inputs')
    print("Read inputs from: %s" % inputs_dir)
    df_test = pd.read_csv(os.path.join(inputs_dir, 'sample_submission_v2.csv'))
    return df_test

all_names = []
all_scores = []
all_preds = []


def add_model(Y_true, pred, name):
    score = fbeta_score(Y_true, pred, beta=2, average='samples')
    print("%s model score %f" % (name, score))
    all_names.append(name)
    all_scores.append(score)
    all_preds.append(pred)


def get_weight(model):
    s = os.path.splitext(model.split('_')[-1])[0]
    if s.startswith('w'):
        return int(s[1:])
    else:
        return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, help='(int) The size of the batch', default='128')
    parser.add_argument('-ad', '--augmented_dir', help='(String) Path to augmented data folder', default=None, required=True)
    parser.add_argument('-md', '--model_dir', help='(String) Path to models directory.', default=None, required=True)
    parser.add_argument('-ex', '--extra',
                        help='(String) Evaluate on extra augmented data.',
                        action='store_true', default=False)
    parser.add_argument('-ot', '--opt_th',
                        help='(String) Specify if to optimize thresholds per class.',
                        action='store_true', default=False)
    args = parser.parse_args()

    print("Loading models...\n")
    if not os.path.exists(args.model_dir):
        print("Error: directory not exist %s" % args.model_dir)
        exit(-1)

    models_list = glob(os.path.join(args.model_dir, '*.h5'))

    print("\nLoading data...")
    if not os.path.exists(args.augmented_dir):
        print("Error: directory not exist %s" % args.augmented_dir)
        exit(-1)

    if args.extra:
        x_name = 'X_valid_augmented.npz'
        y_name = 'Y_valid_augmented.npz'
    else:
        x_name = 'X_valid.npz'
        y_name = 'Y_valid.npz'

    print("Loading %s" % x_name)
    X_valid = npz_to_ndarray(np.load(os.path.join(args.augmented_dir, x_name)))
    print("X_valid shape %s" % str(X_valid.shape))
    print("Loading %s" % y_name)
    Y_valid = npz_to_ndarray(np.load(os.path.join(args.augmented_dir, y_name)))
    print("Y_valid shape %s" % str(Y_valid.shape))

    print("Evaluate models")
    preds = []
    M_Num = len(models_list)
    weights = np.ones(M_Num)
    for i, m in enumerate(models_list):
        weights[i] = get_weight(m)
        model = load_model(m)
        print("Predicting model %s" % models_list[i])
        preds.append(model.predict(X_valid))
        del model
        K.clear_session()

    preds = np.array(preds)

    print()
    # Calculate regular score per model
    for i in range(M_Num):
        add_model(Y_valid, (preds[i] > 0.2).astype(int), "model%d" % i)

    ratios = []
    if args.opt_th:
        for i in range(M_Num):
            t = find_ratios(Y_valid, preds[i])
            ratios.append(t)
            print("Opt threshold: %s" % t)
        ratios = np.array(ratios)

        print("\nModels scores with optimized ratios")
        for i in range(M_Num):
            add_model(Y_valid, (preds[i] > ratios[i]).astype(int), "opt_model%d" % i)

    max_pred_soft = preds.max(axis=0)
    max_pred = (max_pred_soft > 0.2).astype(int)
    add_model(Y_valid, max_pred, "max_model")

    avg_pred_soft = preds.mean(axis=0)
    avg_pred = (avg_pred_soft > 0.2).astype(int)
    add_model(Y_valid, avg_pred, "avg_model")

    vote_pred = vote((preds > 0.2).astype(int))
    add_model(Y_valid, vote_pred, "vote_model")

    if args.opt_th:
        opt_preds = preds.copy()
        for i in range(M_Num):
            opt_preds[i] = (preds[i] > ratios[i]).astype(int)
        opt_vote_pred = vote(opt_preds)
        add_model(Y_valid, opt_vote_pred, "opt_vote_model")

        max_ratios = find_ratios(Y_valid, max_pred_soft)
        opt_max = (max_pred_soft > max_ratios).astype(int)
        add_model(Y_valid, opt_max, "opt_max_model")
        print("Max model thresholds")
        print(max_ratios)

        avg_ratios = find_ratios(Y_valid, avg_pred_soft)
        opt_avg = (avg_pred_soft > avg_ratios).astype(int)
        add_model(Y_valid, opt_avg, "opt_avg_model")
        print("Avg model thresholds")
        print(avg_ratios)

    df = pd.DataFrame(columns=['name', 'score'])
    df.name = all_names
    df.score = all_scores
    print(df)

    print("\nModels sorted by higher score:")
    print(df.sort_values(by='score', ascending=False))

    print("Saving thresholds...")
    np.save(os.path.join(args.model_dir, 'thresholds.npy'), ratios)
    np.save(os.path.join(args.model_dir, 'avg_model_thresholds.npy'), avg_ratios)
    np.save(os.path.join(args.model_dir, 'max_model_thresholds.npy'), max_ratios)

    m_names = [os.path.splitext(os.path.split(m)[-1])[0] for m in models_list]

    th_df = pd.DataFrame(columns=['model', 'th_id'])
    th_df.th_id = list(range(len(models_list)))
    th_df.model = m_names
    th_df.to_csv(os.path.join(args.model_dir, 'models_order.csv'), index=False)

    print('Done\n')