import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from keras.models import load_model
import keras.backend as K
from utils.data_utils import *
import glob


# model_per_tag = [
#     {'dir': 'models/starter256_256_binary_crossentropy_rmsprop', 'name': 'starter256', 'epoch': 63, 'in_size': 256},# slash_burn
#     {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 50, 'in_size': 128},          # clear
#     {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 57, 'in_size': 128},    # blooming
#     {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 85, 'in_size': 128},          # primary
#     {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 81, 'in_size': 128},          # cloudy
#     {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 193, 'in_size': 128},         # conventional_min
#     {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 82, 'in_size': 128},          # water
#     {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 61, 'in_size': 128},          # haze
#     {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 72, 'in_size': 128},          # cultivation
#     {'dir': 'models/starter256_256_binary_crossentropy_rmsprop', 'name': 'starter256', 'epoch': 57, 'in_size': 256},# partly_cloudy
#     {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 124, 'in_size': 128},         # artisinal_mine
#     {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 41, 'in_size': 128},    # habitation
#     {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 47, 'in_size': 128},    # bare_ground
#     {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 158, 'in_size': 128},         # blow_down
#     {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 52, 'in_size': 128},          # agriculture
#     {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 46, 'in_size': 128},    # road
#     {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 56, 'in_size': 128}     # selective_logging
# ]
model_per_tag = [
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 173, 'in_size': 128},         # slash_burn
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 50, 'in_size': 128},          # clear
    {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 57, 'in_size': 128},    # blooming
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 85, 'in_size': 128},          # primary
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 81, 'in_size': 128},          # cloudy
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 193, 'in_size': 128},         # conventional_min
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 82, 'in_size': 128},          # water
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 61, 'in_size': 128},          # haze
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 72, 'in_size': 128},          # cultivation
    {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 29, 'in_size': 128},    # partly_cloudy
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 124, 'in_size': 128},         # artisinal_mine
    {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 41, 'in_size': 128},    # habitation
    {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 47, 'in_size': 128},    # bare_ground
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 158, 'in_size': 128},         # blow_down
    {'dir': 'models/best1_128_binary_crossentropy_rmsprop', 'name': 'best1', 'epoch': 52, 'in_size': 128},          # agriculture
    {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 46, 'in_size': 128},    # road
    {'dir': 'models/starter1_128_binary_crossentropy_rmsprop', 'name': 'starter1', 'epoch': 56, 'in_size': 128}     # selective_logging
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', help='(String) Path to the model file', required=True)
    parser.add_argument('-i', '--in_size', type=int, help='(int) The size of the image', default='128')
    parser.add_argument('-b', '--batch_size', type=int, help='(int) The size of the batch', default='128')
    parser.add_argument('-ot', '--opt_th',
                        help='(String) Specify if to optimize thresholds per class.',
                        action='store_true', default=None)
    args = parser.parse_args()

    # Parse labels
    work_dir = os.path.dirname(os.path.realpath(__file__))
    inputs_dir = os.path.join(work_dir, 'inputs')
    print("Read inputs from: %s" % inputs_dir)
    df_train = pd.read_csv(os.path.join(inputs_dir, 'train_v2.csv'))

    train_data_dir = os.path.join(inputs_dir, 'train-jpg')
    if not os.path.exists(train_data_dir):
        print("Error: Mising data folder: %s" % train_data_dir)
        exit(-1)

    batch_size = args.batch_size

    model_selector = ModelSelector(model_per_tag)
    in_sizes = model_selector.get_input_sizes()

    input_manager = InputManager(df_train, train_data_dir)
    input_manager.load_inputs(in_sizes)

    scores = []
    _, _, _, Y_valid = input_manager.get_train_test(in_sizes[0])
    Y_final = np.empty(shape=Y_valid.shape)
    for i in range(17):
        print("Evaluating tag %s" % input_manager.inv_label_map[i])
        model_path = model_selector.select_model(i)
        print ("Loading model %s" % model_path)
        model = load_model(model_path)

        in_size = model_selector.get_input_size(i)
        _, X_valid, _, Y_valid = input_manager.get_train_test(in_size)

        print ("Evaluating model on %s" % input_manager.inv_label_map[i])
        p_valid = model.predict(X_valid, batch_size=batch_size, verbose=2)
        score = fbeta_score(Y_valid[:,i], (p_valid[:,i] > 0.2).astype(int), beta=2)
        scores.append(score)
        print ("%s score is %f" % (input_manager.inv_label_map[i], score))
        Y_final[:,i] = p_valid[:,i].copy()
        del model
        K.clear_session()

    print("Final score is %f" % fbeta_score(Y_valid, (Y_final > 0.2).astype(int), beta=2, average='samples'))
    print ("Score per class:")

    for i in range(17):
        print("%s: %f" % (input_manager.inv_label_map[i], scores[i]))

    if args.opt_th is not None:
        print("Optimizing thresholds per class on Validation set")
        ratios = find_ratios(Y_valid, Y_final)
        print(ratios)
        print("Final score with optimized thresholds: %f" % fbeta_score(Y_valid, (Y_final > ratios).astype(int), beta=2, average='samples'))
        print("Score per class with optimized throsholds:")
        for i in range(17):
            score = fbeta_score(Y_valid[:,i], (Y_final[:,i] > ratios[i]).astype(int), beta=2)
            print("%s: %f" % (input_manager.inv_label_map[i], scores[i]))
    print('Done')
    