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
from model.resnet_152_keras import Scale

pd.options.mode.chained_assignment = None

#
# def save_result(model_path, model_name, dir_name, pred):
#     prefix = os.path.split(os.path.split(model_path)[0])[1]
#     npz_name = prefix + '_' + model_name + '.npz'
#     npz_path = os.path.join(dir_name, npz_name)
#     np.savez_compressed(npz_path, pred)
#     return npz_path


def get_res_path(model_path, model_name, dir_name):
    prefix = os.path.split(os.path.split(model_path)[0])[1]
    npz_name = prefix + '_' + model_name + '.npz'
    npz_path = os.path.join(dir_name, npz_name)
    return npz_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ml', '--max_loss', type=float, help='(float) Maximal loss for models to be evaluated and dumped',
                        default='0.1')
    parser.add_argument('-i', '--in_size', type=int, help='(int) The size of the image', default='128')
    parser.add_argument('-dev', '--dev',
                        help='(String) Specify if run in development mode to use small network and subset of data.',
                        action='store_true', default=False)
    parser.add_argument('-df', '--data_frame', help='(String) Use models from data frame', default=None)
    parser.add_argument('-dfo', '--dataframe_only',
                        help='(String) Only modify dataframe for predictions if required.',
                        action='store_true', default=False)
    args = parser.parse_args()

    if args.data_frame is not None:
        df = pd.read_csv(args.data_frame)
    else:
        df = load_models_to_dataframe()

    unique_loss = df.loss.drop_duplicates()
    interesting_losses = np.sort(unique_loss[unique_loss <= args.max_loss].values)

    work_dir = os.path.dirname(os.path.realpath(__file__))
    inputs_dir = os.path.join(work_dir, 'inputs')
    print("Read inputs from: %s" % inputs_dir)
    df_test = pd.read_csv(os.path.join(inputs_dir, 'sample_submission_v2.csv'))

    test_data_dir = os.path.join(inputs_dir, 'test-jpg')
    if not os.path.exists(test_data_dir):
        print("Error: Mising data folder: %s" % test_data_dir)
        exit(-1)

    labels, label_map, inv_label_map = get_labels()

    if not args.dataframe_only:
        print("Loading test data")
        if args.dev:
            img_size = args.in_size
            X, _ = load_jpg_data(df_test, test_data_dir, label_map, img_size=img_size, subset_size=100)
        else:
            img_size = args.in_size
            X, _ = load_jpg_data(df_test, test_data_dir, label_map, img_size=img_size)

    pred_dir = 'test_predictions'
    for l in interesting_losses:
        loss_dir = os.path.join(pred_dir, str(l))
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)

        print("\nDumping models with loss %f" % l)

        if os.path.exists(os.path.join(loss_dir, 'models.csv')):
            df_l_existing = pd.read_csv(os.path.join(loss_dir, 'models.csv'))
            df_l_existing = df_l_existing.dropna()
            df_l = df[df.loss == l].copy()
            for p in df_l_existing.path:
                if os.path.exists(p):
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
                if model.input.shape[1] != img_size:
                    print("\n***********************************************************************")
                    print("Warning: Model input size is %d incompatible with data size %d" % (int(model.input.shape[1]), img_size))
                    print("Skipping model %s" % row[1].model)
                    print("***********************************************************************\n")
                    df_l = df_l[df_l.path != row[1].path]
                else:
                    print("Predicting model %s" % row[1].model_name)
                    pred = model.predict(X)
                    print("Saving %s" % pred_path)
                    np.savez_compressed(pred_path, pred)
                del model
                K.clear_session()
            else:
                print("Found existing prediction %s" % pred_path)
                pred = npz_to_ndarray(np.load(pred_path))

            df_l.loc[row[0], 'res_path'] = pred_path

        print("\n")
        if len(df_l) > 0:
            if df_l_existing is not None:
                df_l = pd.concat([df_l_existing, df_l])
            print(df_l)
            df_l.to_csv(os.path.join(loss_dir, 'models.csv'), index=False)
