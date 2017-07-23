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

def vote(pred):
    return (pred.mean(axis=0) > 0.5).astype(int)

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
    parser.add_argument('-md', '--model_dir', help='(String) Path to models directory.', default=None, required=True)
    parser.add_argument('-i', '--in_size', type=int, help='(int) The size of the image', default='128')
    parser.add_argument('-dev', '--dev',
                        help='(String) Specify if run in development mode to use small network and subset of data.',
                        action='store_true', default=False)
    parser.add_argument('-tag', '--tag',
                        help='(String) Specify tag to be added to the directory.', required=True)
    args = parser.parse_args()


    print("Loading models...\n")
    if not os.path.exists(args.model_dir):
        print("Error: directory not exist %s" % args.model_dir)
        exit(-1)

    models_list = glob(os.path.join(args.model_dir, '*.h5'))

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
        X, _ = load_jpg_data(df_test, test_data_dir, label_map, img_size=img_size, subset_size=100)
    else:
        img_size = args.in_size
        X, _ = load_jpg_data(df_test, test_data_dir, label_map, img_size=img_size)

    print("Evaluate models")
    preds = []
    M_Num = len(models_list)
    for i, m in enumerate(models_list):
        model = load_model(m)
        print("Predicting model %s" % models_list[i])
        preds.append(model.predict(X))
        del model
        K.clear_session()

    preds = np.array(preds)

    avg_pred = preds.mean(axis=0)
    avg_pred_final = (avg_pred > 0.2).astype(int)
    max_pred = preds.max(axis=0)
    max_pred_final = (max_pred > 0.2).astype(int)
    vote_pred_final = vote((preds > 0.2).astype(int))

    sufix = args.tag
    submission_dir = os.path.join('submissions', 'ensamble_' + sufix)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    print("\nCreating submission files for ensamble model...")
    create_submission(avg_pred_final, 'avg', sufix)
    create_submission(max_pred_final, 'max', sufix)
    create_submission(vote_pred_final, 'vote', sufix)

    if os.path.exists(os.path.join(args.model_dir, 'models_order.csv')):
        print("\nCreating submission files for ensamble model with optimized thresholds...")
        df_order = pd.read_csv(os.path.join(args.model_dir, 'models_order.csv'))
        thresholds = np.load(os.path.join(args.model_dir, 'thresholds.npy'))

        m_names = [os.path.splitext(os.path.split(m)[-1])[0] for m in models_list]
        opt_preds = preds.copy()
        for i in range(M_Num):
            th_id = df_order[df_order.model == m_names[i]].th_id
            opt_preds[i] = (preds[i] > thresholds[th_id]).astype(int)
        opt_vote_pred = vote(opt_preds)

        max_model_threshold = np.load(os.path.join(args.model_dir, 'max_model_thresholds.npy'))
        opt_max_pred = (max_pred > max_model_threshold).astype(int)

        create_submission(opt_max_pred, 'opt_max', sufix)
        create_submission(opt_vote_pred, 'opt_vote', sufix)

    if os.path.exists(os.path.join(args.model_dir, 'avg_model_thresholds.npy')):
        avg_model_threshold = np.load(os.path.join(args.model_dir, 'avg_model_thresholds.npy'))
        opt_avg_pred = (avg_pred > avg_model_threshold).astype(int)
        create_submission(opt_avg_pred, 'opt_avg', sufix)

    print('Done\n')
