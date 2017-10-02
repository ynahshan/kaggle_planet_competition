import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from keras.models import load_model

from utils.data_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='(String) Path to the model file', required=True)
    parser.add_argument('-i', '--in_size', type=int, help='(int) The size of the image', default='128')
    parser.add_argument('-b', '--batch_size', type=int, help='(int) The size of the batch', default='128')
    parser.add_argument('-ot', '--opt_th',
                        help='(String) Specify if to optimize thresholds per class.',
                        action='store_true', default=False)
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
    X, Y = load_data(df_train, train_data_dir, label_map, img_size=img_size)
    print("Sape of X: %s, shape of Y: %s" % (X.shape, Y.shape))

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=0)
    print ("Loading model...")
    model = load_model(model_path)

    batch_size = args.batch_size
    print ("Evaluating model...")
    p_valid = model.predict(X_valid, batch_size=batch_size, verbose=2)
    score = fbeta_score(Y_valid, (p_valid > 0.2).astype(int), beta=2, average='samples')
    print ("Model score %f" % score)

    print ("Score per class:")
    labels, label_map, inv_label_map = get_labels()
    for i in range(17):
        print("%s: %f" % (inv_label_map[i], fbeta_score(Y_valid[:, i], (p_valid[:, i] > 0.2).astype(int), beta=2)))

    if args.opt_th:
        ratios = find_ratios(Y_valid, p_valid)
        print(ratios)
        score = fbeta_score(Y_valid, (p_valid > ratios).astype(int), beta=2, average='samples')
        print("Model score with opt ratios %f" % score)