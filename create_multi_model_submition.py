import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from keras.models import load_model

from utils.data_utils import *
from evaluate_multi_topology import model_per_tag
import keras.backend as K

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, help='(int) The size of the batch', default='128')
    parser.add_argument('-ot', '--opt_th',
                        help='(String) Specify if to optimize thresholds per class.',
                        action='store_true', default=None)
    parser.add_argument('-dev', '--dev',
                        help='(String) Specify if run in development mode to use small network and subset of data.',
                        action='store_true', default=None)

    args = parser.parse_args()

    # Parse labels
    work_dir = os.path.dirname(os.path.realpath(__file__))
    inputs_dir = os.path.join(work_dir, 'inputs')
    print("Read inputs from: %s" % inputs_dir)
    df_test = pd.read_csv(os.path.join(inputs_dir, 'sample_submission_v2.csv'))
    df_train = pd.read_csv(os.path.join(inputs_dir, 'train_v2.csv'))

    test_data_dir = os.path.join(inputs_dir, 'test-jpg')
    if not os.path.exists(test_data_dir):
        print("Error: Mising data folder: %s" % test_data_dir)
        exit(-1)

    batch_size = args.batch_size

    model_selector = ModelSelector(model_per_tag)
    in_sizes = model_selector.get_input_sizes()

    train_data_manager = InputManager(df_train, "")
    input_manager = InputManager(df_test, test_data_dir)
    input_manager.load_inputs(in_sizes)

    _, Y = input_manager.get_data(in_sizes[0])
    Y_final = np.empty(shape=Y.shape)
    for i in range(17):
        print("Evaluating tag %s" % train_data_manager.inv_label_map[i])
        model_path = model_selector.select_model(i)
        print ("Loading model %s" % model_path)
        model = load_model(model_path)

        in_size = model_selector.get_input_size(i)
        X, _ = input_manager.get_data(in_size)

        print ("Evaluating model on %s" % train_data_manager.inv_label_map[i])
        p_test = model.predict(X, batch_size=batch_size, verbose=2)
        Y_final[:,i] = p_test[:,i].copy()
        del model
        K.clear_session()


    p_test = (Y_final > 0.2).astype(int)

    p_tags = to_tagging(p_test, train_data_manager.inv_label_map)
    df_test.tags = p_tags.tags
    file_name = 'submission_nydevyura.csv'
    df_test.to_csv(file_name, index=False)
    if os.path.exists(file_name):
        print("Submission file %s created successfully" % file_name)
    else:
        print("Error: Failed to create submission")