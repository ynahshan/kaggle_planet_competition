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
                        action='store_true', default=None)
    parser.add_argument('-dev', '--dev',
                        help='(String) Specify if run in development mode to use small network and subset of data.',
                        action='store_true', default=False)

    args = parser.parse_args()

    # Parse labels
    work_dir = os.path.dirname(os.path.realpath(__file__))
    inputs_dir = os.path.join(work_dir, 'inputs')
    print("Read inputs from: %s" % inputs_dir)
    df_test = pd.read_csv(os.path.join(inputs_dir, 'sample_submission_v2.csv'))
    df_train = pd.read_csv(os.path.join(inputs_dir, 'train_v2.csv'))

    model_path = os.path.join(work_dir, args.model)
    if not os.path.exists(model_path):
        print ("Error: Model file not found. %s" % model_path)
        exit(-1)

    test_data_dir = os.path.join(inputs_dir, 'test-jpg')
    if not os.path.exists(test_data_dir):
        print("Error: Mising data folder: %s" % test_data_dir)
        exit(-1)

    # Take image size from args
    img_size = args.in_size
    labels, label_map, inv_label_map = get_labels()
    print(labels)

    print ("Loading model...")
    model = load_model(model_path)
    batch_size = args.batch_size

    if args.opt_th is not None:
        train_data_dir = os.path.join(inputs_dir, 'train-jpg')
        if not os.path.exists(train_data_dir):
            print("Error: Mising data folder: %s" % train_data_dir)
            exit(-1)

        print("Optimizing thresholds per class on Validation set")
        if args.dev is not None:
            img_size = args.in_size
            X, Y = load_jpg_data(df_train, train_data_dir, label_map, img_size=img_size, subset_size=1000)
        else:
            img_size = args.in_size
            X, Y = load_jpg_data(df_train, train_data_dir, label_map, img_size=img_size)

        _, X_valid, _, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=0)
        p_valid = model.predict(X_valid, batch_size=batch_size, verbose=2)
        ratios = find_ratios(Y_valid, p_valid)
        print("Optimal ratios per class:")
        print(ratios)

    print("Loading test data")
    if args.dev is not None:
        img_size = args.in_size
        X, _ = load_jpg_data(df_test, test_data_dir, label_map, img_size=img_size, subset_size=1000)
    else:
        img_size = args.in_size
        X, _ = load_jpg_data(df_test, test_data_dir, label_map, img_size=img_size)

    print ("Evaluating model...")
    p_test = model.predict(X, batch_size=batch_size, verbose=2)

    if args.opt_th is not None:
        p_test = (p_test > ratios).astype(int)
    else:
        p_test = (p_test > 0.2).astype(int)

    p_tags = to_tagging(p_test, inv_label_map)
    df_test.tags = p_tags.tags
    file_name = 'submission_nydevyura.csv'
    df_test.to_csv(file_name, index=False)
    if os.path.exists(file_name):
        print("Submission file %s created successfully" % file_name)
    else:
        print("Error: Failed to create submission")