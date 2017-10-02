import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from keras.models import load_model
from keras.layers import Flatten
from keras.models import Model
import shutil

from utils.data_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='(String) Path to the model file', required=True)
    parser.add_argument('-i', '--in_size', type=int, help='(int) The size of the image', default='128')
    parser.add_argument('-b', '--batch_size', type=int, help='(int) The size of the batch', default='128')
    parser.add_argument('-dev', '--dev',
                        help='(String) Specify if run in development mode to use small network and subset of data.',
                        action='store_true', default=None)
    args = parser.parse_args()

    print (args.model)

    # Parse labels
    work_dir = os.path.dirname(os.path.realpath(__file__))
    inputs_dir = os.path.join(work_dir, 'inputs')
    print("Read inputs from: %s" % inputs_dir)
    df_train = pd.read_csv(os.path.join(inputs_dir, 'train_v2.csv'))

    model_path = os.path.join(work_dir, args.model)
    if not os.path.exists(model_path):
        print ("Error: Model file not found. %s" % model_path)

    train_data_dir = os.path.join(inputs_dir, 'train-jpg')
    if not os.path.exists(train_data_dir):
        print("Error: Mising data folder: %s" % train_data_dir)
        exit(-1)

    # Take image size from args
    img_size = args.in_size
    labels, label_map, inv_label_map = get_labels()
    print(labels)

    # Assume all networks requires same input size, don't check it
    if args.dev is not None:
        X, Y = load_data(df_train, train_data_dir, label_map, img_size=img_size, subset_size=1000)
    else:
        X, Y = load_data(df_train, train_data_dir, label_map, img_size=img_size)

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=0)
    print("Loading model...")
    model = load_model(model_path)

    features_layer = None
    for l in model.layers:
        if type(l) == Flatten:
            features_layer = l.name

    feature_extractor = Model(inputs=model.input,
                              outputs=model.get_layer(features_layer).output)

    print("Extracting features...")
    features_train = feature_extractor.predict(X_train)
    features_valid = feature_extractor.predict(X_valid)

    model_dir = os.path.splitext(os.path.basename(args.model))[0]
    features_dir = 'features'
    work_dir = os.path.join(os.path.abspath(features_dir), model_dir)
    print("Saving features to %s" % work_dir)
    # Create directory for features
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    np.save(os.path.join(work_dir, 'train_features.npy'), features_train)
    np.save(os.path.join(work_dir, 'val_features.npy'), features_valid)
    np.save(os.path.join(work_dir, 'train_labels.npy'), Y_train)
    np.save(os.path.join(work_dir, 'val_labels.npy'), Y_valid)
    # features_restored = np.load('test_feature.npy')
    print("Done")
