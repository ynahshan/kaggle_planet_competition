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

    print("Length of original data %d" % len(x_data))
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
    parser.add_argument('-mult', '--augmentation_multiplier', type=int, help='(int) Multiplier for augmented data generation',
                        default='4')
    parser.add_argument('-seed', '--random_seed', type=int,
                        help='(int) Random seed for train test split',
                        default='0')
    parser.add_argument('-d', '--data_dir', help='(String) Path to data directory.', default=None, required=True)
    args = parser.parse_args()

    work_dir = os.path.dirname(os.path.realpath(__file__))

    data_dir = args.data_dir
    print("Loading X")
    X = npz_to_ndarray(np.load(os.path.join(data_dir, 'X_test_extra.npz')))
    print("Loading Y")
    Y = npz_to_ndarray(np.load(os.path.join(data_dir, 'Y_test_extra.npz')))

    print("Create augmented training data")
    X_augmented, Y_augmented = data_augmentation(X, Y, augmentation_multiplier=args.augmentation_multiplier)

    print("Saving X_test_augmented.npz of shape %s" % str(X_augmented.shape))
    np.savez_compressed(os.path.join(data_dir, 'X_test_augmented.npz'), X_augmented)
    print("Saving Y_test_augmented.npz of shape %s" % str(Y_augmented.shape))
    np.savez_compressed(os.path.join(data_dir, 'Y_test_augmented.npz'), Y_augmented)
    print('Done')