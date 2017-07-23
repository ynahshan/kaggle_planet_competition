import argparse
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil
import math
from utils.data_utils import *


def data_augmentation(x_data, y_data, augmentation_multiplier=4):
    import random
    def randomTranspose(img, u=0.5):
        if random.random() < u:
            img = img.transpose(1, 0, 2)  # cv2.transpose(img)
        return img

    batch_size = 128 if len(x_data) > 128 else min(4, len(x_data))
    augmented_data_size = int(augmentation_multiplier * len(x_data))
    X_augmented_shape = (augmented_data_size, x_data.shape[1], x_data.shape[2], x_data.shape[3])
    Y_augmented_shape = (augmented_data_size, y_data.shape[1])

    X_augmented = np.empty(X_augmented_shape)
    Y_augmented = np.empty(Y_augmented_shape)
    counter = 0

    datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, preprocessing_function=randomTranspose)
    print(len(X_augmented))
    for x_batch, y_batch in datagen.flow(x_data, y_data, batch_size=batch_size, seed=0):
        X_augmented[counter:counter + len(x_batch)] = x_batch
        Y_augmented[counter:counter + len(y_batch)] = y_batch
        counter += len(x_batch)
        if counter >= augmented_data_size:
            print("break counter %d" % counter)
            break

    return X_augmented, Y_augmented

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_size', type=int, help='(int) The size of the image', default='128')
    parser.add_argument('-mult', '--augmentation_multiplier', type=int, help='(int) Multiplier for augmented data generation',
                        default='4')
    parser.add_argument('-seed', '--random_seed', type=int,
                        help='(int) Random seed for train test split',
                        default='0')
    parser.add_argument('-dev', '--dev',
                        help='(String) Specify if run in development mode to use small network and subset of data.',
                        action='store_true', default=False)
    parser.add_argument('-val', '--validation_only',
                        help='(String) Only create augmented data for validation set.',
                        action='store_true', default=False)
    args = parser.parse_args()

    work_dir = os.path.dirname(os.path.realpath(__file__))
    inputs_dir = os.path.join(work_dir, 'inputs')
    print("Read inputs from: %s" % inputs_dir)
    df_train = pd.read_csv(os.path.join(inputs_dir, 'train_v2.csv'))

    labels, label_map, inv_label_map = get_labels()
    print(labels)

    train_data_dir = os.path.join(inputs_dir, 'train-jpg')
    if not os.path.exists(train_data_dir):
        print("Error: Mising data folder: %s" % train_data_dir)
        exit(-1)

    # Assume all networks requires same input size, don't check it
    if args.dev:
        X, Y = load_jpg_data(df_train, train_data_dir, label_map, img_size = args.in_size, subset_size=200)
    else:
        X, Y = load_jpg_data(df_train, train_data_dir, label_map, img_size = args.in_size)
    print("Sape of X: %s, shape of Y: %s" % (X.shape, Y.shape))

    print('Splitting to train and test...')
    data_split_seed = args.random_seed
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=data_split_seed)

    data_dir = os.path.join('augmented',
                            'train_augmented_size_' + str(args.in_size) + '_mult_' + str(args.augmentation_multiplier) + '_seed_' + str(data_split_seed))
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    if not args.validation_only:
        print("Create augmented training data")
        X_train_augmented, Y_train_augmented = data_augmentation(X_train, Y_train, augmentation_multiplier=args.augmentation_multiplier)

        print("Saving X_train_augmented.npz of shape %s" % str(X_train_augmented.shape))
        np.savez_compressed(os.path.join(data_dir, 'X_train_augmented.npz'), X_train_augmented)
        print("Saving Y_train_augmented.npz of shape %s" % str(Y_train_augmented.shape))
        np.savez_compressed(os.path.join(data_dir, 'Y_train_augmented.npz'), Y_train_augmented)
        print("Saving X_train.npz of shape %s" % str(X_train.shape))
        np.savez_compressed(os.path.join(data_dir, 'X_train.npz'), X_train)
        print("Saving Y_train.npz of shape %s" % str(Y_train.shape))
        np.savez_compressed(os.path.join(data_dir, 'Y_train.npz'), Y_train)

        # Free memory
        X_train_augmented = None
        Y_train_augmented = None
        X_train = None
        Y_train = None


    print("Create augmented validation data")
    X_valid_augmented, Y_valid_augmented = data_augmentation(X_valid, Y_valid, augmentation_multiplier=args.augmentation_multiplier)

    print("Saving X_valid_augmented.npz of shape %s" % str(X_valid_augmented.shape))
    np.savez_compressed(os.path.join(data_dir, 'X_valid_augmented.npz'), X_valid_augmented)
    print("Saving Y_valid_augmented.npz of shape %s" % str(Y_valid_augmented.shape))
    np.savez_compressed(os.path.join(data_dir, 'Y_valid_augmented.npz'), Y_valid_augmented)
    print("Saving X_valid.npz of shape %s" % str(X_valid.shape))
    np.savez_compressed(os.path.join(data_dir, 'X_valid.npz'), X_valid)
    print("Saving Y_valid.npz of shape %s" % str(Y_valid.shape))
    np.savez_compressed(os.path.join(data_dir, 'Y_valid.npz'), Y_valid)
    print("Augmented data saved to %s" % data_dir)