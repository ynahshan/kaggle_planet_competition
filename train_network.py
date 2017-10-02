import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from utils.data_utils import *
from model.model_factory import ModelFactory
from utils.gpu_utils import to_multi_gpu

import shutil
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import random


def randomTranspose(img, u=0.5):
    if random.random() < u:
        img = img.transpose(1,0,2)  #cv2.transpose(img)
    return img

class TrainingParameters:
    def __init__(self, name, in_size=128, loss='binary_crossentropy', optimizer='rmsprop', params = None):
        self.model_name = name
        self.in_size = in_size
        self.loss = loss
        self.optimizer = optimizer
        self.parameters = params

    def __str__(self):
        s = "%s_%d_%s_%s" % (self.model_name, self.in_size, self.loss, self.optimizer)
        if self.parameters is not None and 'id' in self.parameters:
            s = s + '_' + str(self.parameters['id'])
        return s

# models = [ TrainingParameters('dev', 32) ]
models = [
    TrainingParameters('starter1', 128)
    # TrainingParameters('resnet50_1', 256)
]

def write_log(log_name, line):
    f = open(log_name, 'a')
    f.write("%s\n" % line)
    f.close()

def write_summary(log_name, history_callback, scores, scores_per_class, comb, comb_scores, desc):
    f = open(log_name, 'a')
    f.write("Results for model %s\n\n" % keras_model.name)
    f.write("loss\n")
    f.write("%s\n" % np.array(history_callback.history['loss']))
    f.write("val_loss\n")
    f.write("%s\n" % np.array(history_callback.history['val_loss']))
    f.write("scores\n")
    f.write("%s\n\n" %np.array(scores))
    f.write("Max score %f at epoch %d\n" % (np.max(scores), np.argmax(scores)))
    f.write("Min val_loss %f at epoch %d\n" % (np.min(history_callback.history['val_loss']), np.argmin(history_callback.history['val_loss'])))
    f.write("Scores per class:\n")
    score_pc_np = np.array(scores_per_class)
    for i in range(17):
        f.write("%s - max score %f at epoch %d\n" % (inv_label_map[i], score_pc_np[:,i].max(), score_pc_np[:,i].argmax()))
        if comb:
            idx = int(i+Y_valid.shape[1]/2)
            f.write("%s second - max score %f at epoch %d\n" % (inv_label_map[i], score_pc_np[:, idx].max(), score_pc_np[:, idx].argmax()))

    if comb:
        f.write("%s\n" % np.array(comb_scores))
        f.write("Max score ever %f at pos %d" % (np.array(comb_scores).max(), np.array(comb_scores).argmax()))
        print ("Max score ever %f at pos %d" % (np.array(comb_scores).max(), np.array(comb_scores).argmax()))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_callback.history['loss'])
    plt.plot(history_callback.history['val_loss'])
    plt.legend(labels=['loss', 'val_loss'])
    plt.subplot(1, 2, 2)
    plt.plot(scores)
    plt.legend(labels=['val_fbeta'])
    plt.savefig(os.path.join(model_dir, desc + '_loss_score.png'))
    f.close()

def train_model(keras_model, batch_size, model_dir, epochs, desc, early_stop_patience, comb = False, is_dev = False, data_gen = None):
    scores = []
    scores_per_class = []
    comb_scores = []

    print("\nTrain %s model." % desc)
    if is_dev:
        batch_size = 4

    def get_scores_per_class(_Y_, pred):
        spc = []
        for i in range(_Y_.shape[1]):
            spc.append(fbeta_score(_Y_[:, i], (pred[:, i] > 0.2).astype(int), beta=2))
        return spc

    np.set_printoptions(precision=4)
    log_name = os.path.join(model_dir, desc + '_results.log')
    write_log(log_name, "Train %s model.\n" % desc)

    def my_callback_func(batch, logs):
        print("%s" % str(datetime.now()))
        pred = keras_model.model.predict(X_valid, batch_size=batch_size, verbose=2)
        if not comb:
            score = fbeta_score(Y_valid, (pred > 0.2).astype(int), beta=2, average='samples')
            print("Regular fscore %f" % score)
            write_log(log_name, logs)
            write_log(log_name, "Regular fscore %f\n" % score)
        else:
            pred_comb = pred[:,:17]
            pred_sep = pred[:,17:]
            score = fbeta_score(Y_valid[:,:17], (pred_comb > 0.2).astype(int), beta=2, average='samples')
            print("Regular fscore %f" % score)
            score_sep = fbeta_score(Y_valid[:,:17], (pred_sep > 0.2).astype(int), beta=2, average='samples')
            print("Score on separated features %f" % score_sep)
            pred_max = np.maximum(pred_comb, pred_sep)
            score_max = fbeta_score(Y_valid[:,:17], (pred_max > 0.2).astype(int), beta=2, average='samples')
            print("Score on maximum %f" % score_sep)
            pred_avg = (pred_comb + pred_sep)/2
            score_avg = fbeta_score(Y_valid[:, :17], (pred_avg > 0.2).astype(int), beta=2, average='samples')
            print("Score on average %f" % score_avg)
            comb_scores.append([score, score_sep, score_max, score_avg])
        scores.append(score)
        scores_per_class.append(get_scores_per_class(Y_valid, pred))

    model_path = os.path.join(model_dir, keras_model.name + '_{epoch:02d}_{val_loss:.2f}.h5')
    my_callback = LambdaCallback(on_epoch_end=lambda batch, logs: my_callback_func(batch, logs))
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=0)
    checkpoint_callback = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=False, verbose=0)
    callbacks = [my_callback, early_stop_callback, checkpoint_callback]
    # callbacks = [my_callback]

    if data_gen is not None:
        steps = int(len(X_train) / batch_size)
        print("Fiting data generator with %d steps per epoch" % steps)
        train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, seed=0)
        history_callback = keras_model.model.fit_generator(train_generator, steps_per_epoch=steps, epochs=epochs, verbose=2,
                                               validation_data=(X_valid, Y_valid), callbacks=callbacks)
    else:
        history_callback = keras_model.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                                                 validation_data=(X_valid, Y_valid), callbacks=callbacks)

    print ("Max score %f at epoch %d" % (np.max(scores), np.argmax(scores)))
    write_summary(log_name, history_callback, scores, scores_per_class, comb, comb_scores, desc)

    keras_model.free()
    print("Training of %s finished\n" % desc)
    return np.max(scores)

def preprocess(Y):
    size = Y.shape[1]
    Y_out = np.empty((Y.shape[0], Y.shape[1]*2))
    Y_out[:,:size] = Y
    Y_out[:,size:] = Y
    return Y_out

def create_data_generator(noise_level):
    if noise_level == 'low':
        datagen = ImageDataGenerator(
            vertical_flip=True,
            horizontal_flip=True,
            preprocessing_function=randomTranspose,
            shear_range=0.2,
            zoom_range=0.05,
            rotation_range=2,
        )
    elif noise_level == 'medium':
        datagen = ImageDataGenerator(
            vertical_flip=True,
            horizontal_flip=True,
            preprocessing_function=randomTranspose,
            shear_range=0.3,
            zoom_range=0.1,
            height_shift_range=0.05,
            rotation_range=5,
        )
    else:
        datagen = ImageDataGenerator(
            vertical_flip=True,
            horizontal_flip=True,
            preprocessing_function=randomTranspose,
            shear_range=0.5,
            zoom_range=0.2,
            height_shift_range=0.1,
            rotation_range=10,
        )

    return datagen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, help='(int) The size of the batch', default='128')
    parser.add_argument('-e', '--epoch', type=int, help='(int) The maximal number of epochs to train model', default='45')
    parser.add_argument('-esp', '--early_stop_patience', type=int, help='(int) The early stopping patience factor',
                        default='5')
    parser.add_argument('-ad', '--augmented_dir', help='(String) Path to augmented data folder', default=None)
    parser.add_argument('-ed', '--extra_data_dir', help='(String) Path to extra data folder', default=None)
    parser.add_argument('-n', '--noise', help='(String) Set the level of the noise for data generator', default='medium')
    parser.add_argument('-dev', '--dev',
                        help='(String) Specify if run in development mode to use small network and subset of data.',
                        action='store_true', default=False)
    parser.add_argument('-com', '--comb',
                        help='(String) Specify if use combined mode.',
                        action='store_true', default=False)
    parser.add_argument('-sd', '--synthetic_data',
                        help='(String) If create synthetic data on the fly.',
                        action='store_true', default=True)
    parser.add_argument('-mg', '--multi_gpu',
                        help='(String) If specified try to parallelize network between gpus.',
                        action='store_true', default=False)
    parser.add_argument('-all', '--all_data',
                        help='(String) If specified use whole data set for training.',
                        action='store_true', default=False)

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

    labels, label_map, inv_label_map = get_labels()
    print(labels)

    ed_sufix = ''
    all_sufix = ''
    if args.extra_data_dir is not None:
        ed_sufix = '_ed'
        if not os.path.exists(args.extra_data_dir):
            print("Error: directory not exist %s" % args.extra_data_dir)
            exit(-1)

    if args.augmented_dir is None:
        # Assume all networks requires same input size, don't check it
        if args.dev:
            X, Y = load_data(df_train, train_data_dir, label_map, img_size=models[0].in_size, subset_size=1000)
        else:
            X, Y = load_data(df_train, train_data_dir, label_map, img_size=models[0].in_size)
        print("Sape of X: %s, shape of Y: %s" % (X.shape, Y.shape))

        seed = 0
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=seed)
        if args.extra_data_dir is not None:
            print("Loading extra data...")
            print("Loading X_test_extra")
            X_extra = npz_to_ndarray(np.load(os.path.join(args.extra_data_dir, 'X_test_extra.npz')))
            print("Concatenating X_train and X_extra ...")
            X_train = np.concatenate([X_train, X_extra], axis=0)
            X_extra = None

            print("Loading Y_test_extra")
            Y_extra = npz_to_ndarray(np.load(os.path.join(args.extra_data_dir, 'Y_test_extra.npz')))
            print("Concatenating Y_train and Y_extra ...")
            Y_train = np.concatenate([Y_train, Y_extra], axis=0)
            Y_extra = None
    else:
        print("Loading augmented data")
        if not os.path.exists(args.augmented_dir):
            print("Error: directory not exist %s" % args.augmented_dir)
            exit(-1)

        print ("Loading X_train_augmented")
        X_train = npz_to_ndarray(np.load(os.path.join(args.augmented_dir, 'X_train_augmented.npz')))

        if args.extra_data_dir is not None:
            print("Loading extra data...")
            print("Loading X_test_augmented")
            X_extra = npz_to_ndarray(np.load(os.path.join(args.extra_data_dir, 'X_test_augmented.npz')))
            print("Concatenating X_train and X_extra ...")
            X_train = np.concatenate([X_train, X_extra], axis=0)
            X_extra = None

        print("Loading Y_train_augmented")
        Y_train = npz_to_ndarray(np.load(os.path.join(args.augmented_dir, 'Y_train_augmented.npz')))

        if args.extra_data_dir is not None:
            print("Loading Y_test_augmented")
            Y_extra = npz_to_ndarray(np.load(os.path.join(args.extra_data_dir, 'Y_test_augmented.npz')))
            print("Concatenating Y_train and Y_extra ...")
            Y_train = np.concatenate([Y_train, Y_extra], axis=0)
            Y_extra = None

        print("Loading X_valid")
        X_valid = npz_to_ndarray(np.load(os.path.join(args.augmented_dir, 'X_valid.npz')))
        print("Loading Y_valid")
        Y_valid = npz_to_ndarray(np.load(os.path.join(args.augmented_dir, 'Y_valid.npz')))

    print("Finished loading data")


    if args.all_data:
        all_sufix = '_all'
        print("Loading X_valid_augmented")
        X_valid_augmented = npz_to_ndarray(np.load(os.path.join(args.augmented_dir, 'X_valid_augmented.npz')))
        print("Loading Y_valid_augmented")
        Y_valid_augmented = npz_to_ndarray(np.load(os.path.join(args.augmented_dir, 'Y_valid_augmented.npz')))

        print("Concatenating X_train and X_valid_augmented ...")
        X_train = np.concatenate([X_train, X_valid_augmented], axis=0)
        X_valid_augmented = None
        print("Concatenating Y_train and Y_valid_augmented ...")
        Y_train = np.concatenate([Y_train, Y_valid_augmented], axis=0)
        Y_valid_augmented = None

    synt_sufix = ''
    datagen = None
    if args.synthetic_data:
        synt_sufix = '_synt_' + args.noise
        datagen = create_data_generator(args.noise)
        print(datagen.__dict__)


    if args.dev:
        models.clear()
        models.append(TrainingParameters('dev', 128))
        models.append(TrainingParameters('dev1', 128))

    mf = ModelFactory()
    scores = []
    for m in models:
        # Create directory for model
        model_dir = os.path.join('models', str(m) + ed_sufix + all_sufix + synt_sufix)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        # JPG only
        keras_model = mf.create_model(m.model_name, in_shape=(m.in_size, m.in_size, 3), parameters=m.parameters)
        print ("Model created successfuly. Compiliing")
        keras_model.model.compile(loss=m.loss, optimizer=m.optimizer, metrics=['accuracy'])
        print ("Compilation done")
        print (keras_model.model.summary())
        if args.multi_gpu:
            keras_model.model = to_multi_gpu(keras_model.model)
            keras_model.model.compile(loss=m.loss, optimizer=m.optimizer, metrics=['accuracy'])
        score = train_model(keras_model, args.batch_size, model_dir, args.epoch, str(m), args.early_stop_patience, args.comb, is_dev=args.dev, data_gen=datagen)
        scores.append(score)

    best_model_idx = np.argmax(scores)
    print("Best model %s with max score %f" % (models[best_model_idx], np.max(scores)))
    print("Training session finished.")