import argparse
import os
from utils.data_utils import *
from glob import glob
import pandas as pd
import numpy as np
import shutil
from print_models import load_models_to_dataframe
from keras.models import load_model
import keras.backend as K
from datetime import datetime
from model.resnet_152_keras import *

pd.options.mode.chained_assignment = None


def get_res_path(model_path, model_name, dir_name):
    prefix = os.path.split(os.path.split(model_path)[0])[1]
    npz_name = prefix + '_' + model_name + '.npz'
    npz_path = os.path.join(dir_name, npz_name)
    return npz_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ml', '--max_loss', type=float, help='(float) Maximal loss for models to be evaluated and dumped',
                        default='0.1')
    parser.add_argument('-ad', '--augmented_dir', help='(String) Path to augmented data folder', default=None, required=True)
    parser.add_argument('-ex', '--extra',
                        help='(String) Evaluate on extra augmented data.',
                        action='store_true', default=False)
    parser.add_argument('-t', '--train',
                        help='(String) Evaluate on training.',
                        action='store_true', default=False)
    parser.add_argument('-dfo', '--dataframe_only',
                        help='(String) Only modify dataframe for predictions if required.',
                        action='store_true', default=False)
    args = parser.parse_args()

    df = load_models_to_dataframe()

    unique = df.loss.drop_duplicates()
    interesting_losses = np.sort(unique[unique <= args.max_loss].values)

    print("\nLoading data...")
    if not os.path.exists(args.augmented_dir):
        print("Error: directory not exist %s" % args.augmented_dir)
        exit(-1)

    if args.train:
        if args.extra:
            x_name = 'X_train_augmented.npz'
            y_name = 'Y_train_augmented.npz'
            data_dir = 'train_ex'
        else:
            x_name = 'X_train.npz'
            y_name = 'Y_train.npz'
            data_dir = 'train'
    else:
        if args.extra:
            x_name = 'X_valid_augmented.npz'
            y_name = 'Y_valid_augmented.npz'
            data_dir = 'valid_ex'
        else:
            x_name = 'X_valid.npz'
            y_name = 'Y_valid.npz'
            data_dir = 'valid'

    if not args.dataframe_only:
        print("Loading %s" % x_name)
        X_valid = npz_to_ndarray(np.load(os.path.join(args.augmented_dir, x_name)))
        print("X_valid shape %s" % str(X_valid.shape))
    print("Loading %s" % y_name)
    Y_valid = npz_to_ndarray(np.load(os.path.join(args.augmented_dir, y_name)))
    print("Y_valid shape %s" % str(Y_valid.shape))

    pred_dir = 'predictions'
    for l in interesting_losses:
        loss_dir = os.path.join(pred_dir, data_dir, str(l))
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)

        print("\nDumping models with loss %f" % l)

        if os.path.exists(os.path.join(loss_dir, 'models.csv')):
            df_l_existing = pd.read_csv(os.path.join(loss_dir, 'models.csv'))
            df_l = df[df.loss == l].copy()
            for p in df_l_existing.path:
                df_l = df_l[df_l.path != p]
        else:
            df_l_existing = None
            df_l = df[df.loss == l]
        # Evaluate models and dump. Add relevant columns to df.
        for i, row in enumerate(df_l.iterrows()):
            pred_path = get_res_path(row[1].path, row[1].model, loss_dir)
            if not os.path.exists(pred_path):
                if args.dataframe_only:
                    df_l = df_l[df_l.path != row[1].path]
                    continue

                print("\nModel %d from %d" % (i+1, len(df_l)))
                print("%s" % str(datetime.now()))
                print("Loading model %s" % row[1].model)
                model = load_model(row[1].path, custom_objects={"Scale": Scale})
                print("Predicting model %s" % row[1].model_name)
                pred = model.predict(X_valid)
                np.savez_compressed(pred_path, pred)
                del model
                K.clear_session()
            else:
                print("Found existing prediction %s" % pred_path)
                pred = npz_to_ndarray(np.load(pred_path))

            pred_final = (pred>0.2).astype(int)

            score = fbeta_score(Y_valid, pred_final, beta=2, average='samples')
            print("%s model score %f" % (row[1].model_name, score))
            df_l.loc[row[0], 'score'] = score
            df_l.loc[row[0], 'res_path'] = pred_path

        print("\n")
        if len(df_l) > 0:
            if df_l_existing is not None:
                df_l = pd.concat([df_l_existing, df_l])
            print(df_l)
            df_l.to_csv(os.path.join(loss_dir, 'models.csv'), index=False)
