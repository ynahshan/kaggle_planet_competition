from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Concatenate, Reshape, Add
from keras.layers import Conv2D, MaxPooling2D, Maximum, Input, AveragePooling2D, LocallyConnected1D, Activation
from keras.layers import Conv2DTranspose, UpSampling2D, GlobalMaxPooling2D
from keras.initializers import glorot_normal, he_normal, orthogonal
import keras.backend as K
from keras.applications import Xception, VGG16, VGG19, InceptionV3, ResNet50
from model.resnet_152_keras import resnet152_model

import sys
# sys.path.append('../DenseNet')
# import densenet

def conv_block(prev_layer, conv_out, dropout = 0.25, kernel_size=3):
    _x_ = Conv2D(conv_out, (1, 1), padding='same', activation='relu')(prev_layer)
    _x_ = Conv2D(conv_out, (kernel_size, kernel_size), activation='relu')(_x_)
    _x_ = AveragePooling2D(pool_size=2)(_x_)
    _x_ = Dropout(dropout)(_x_)

    return _x_

def dense_block(prev_layer, dense_out, dropout = 0.25):
    _x_ = Dense(dense_out, activation='relu')(prev_layer)
    _x_ = BatchNormalization()(_x_)
    _x_ = Dropout(dropout)(_x_)

    return _x_

def make_linear_bn(_x_, out_channels):
    _x_ = Dense(out_channels, activation='relu')(_x_)
    _x_ = BatchNormalization()(_x_)
    return _x_

def make_conv_bn(_x_, out_channels, kernel_size=1, groups=1):
    _x_ = Conv2D(out_channels, (kernel_size, kernel_size), padding='same', activation='relu')(_x_)
    _x_ = BatchNormalization()(_x_)
    return _x_

def preprocess(_x_):
    _x_ = make_conv_bn(_x_, 16, kernel_size=1)
    _x_ = make_conv_bn(_x_, 16, kernel_size=1)
    _x_ = make_conv_bn(_x_, 16, kernel_size=1)
    _x_ = make_conv_bn(_x_, 16, kernel_size=1)
    return _x_

def create_conv_btneck(_x_, sizes=[64,64,64], groups=1):
    _x_ = make_conv_bn(_x_, sizes[0], kernel_size=1)
    _x_ = make_conv_bn(_x_, sizes[1], kernel_size=3) # in original implementation sometimes used groups=16
    _x_ = make_conv_bn(_x_, sizes[2], kernel_size=1)
    return _x_

def create_cls(_x_, num_cls):
    _x_ = make_linear_bn(_x_, 512)
    _x_ = make_linear_bn(_x_, 512)
    _x_ = make_linear_bn(_x_, num_cls)
    return _x_

def my_inception_1(prev_layer, channels):
    tower_1 = Conv2D(channels, (1, 1), padding='same', activation='relu')(prev_layer)
    tower_1 = Conv2D(channels, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(channels, (1, 1), padding='same', activation='relu')(prev_layer)
    tower_2 = Conv2D(channels, (5, 5), padding='same', activation='relu')(tower_2)

    output = Concatenate()([tower_1, tower_2])
    return output


def my_inception_2(prev_layer, channels):
    tower_1 = Conv2D(channels, (1, 1), padding='same', activation='relu')(prev_layer)
    tower_1 = Conv2D(channels, (1, 1), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(channels, (1, 1), padding='same', activation='relu')(prev_layer)
    tower_2 = Conv2D(channels, (3, 3), padding='same', activation='relu')(tower_2)

    output = Concatenate()([tower_1, tower_2])
    return output

def cmsc_inception_layer_4(prev_layer, channels):
    tower_1 = Conv2D(channels, (1, 1), padding='same', activation='relu')(prev_layer)
    tower_1 = BatchNormalization()(tower_1)

    tower_2 = Conv2D(channels, (3, 3), padding='same', activation='relu')(prev_layer)
    tower_2 = BatchNormalization()(tower_2)

    tower_3 = Conv2D(channels, (5, 5), padding='same', activation='relu')(prev_layer)
    tower_3 = BatchNormalization()(tower_3)

    tower_4 = Conv2D(channels, (7, 7), padding='same', activation='relu')(prev_layer)
    tower_4 = BatchNormalization()(tower_4)

    output = Maximum()([tower_1, tower_2, tower_3, tower_4])
    return output


def cmsc_inception_layer_3(prev_layer, channels):
    tower_1 = Conv2D(channels, (1, 1), padding='same', activation='relu')(prev_layer)
    tower_1 = BatchNormalization()(tower_1)

    tower_2 = Conv2D(channels, (3, 3), padding='same', activation='relu')(prev_layer)
    tower_2 = BatchNormalization()(tower_2)

    tower_3 = Conv2D(channels, (5, 5), padding='same', activation='relu')(prev_layer)
    tower_3 = BatchNormalization()(tower_3)

    output = Maximum()([tower_1, tower_2, tower_3])
    return output


def cmsc_inception_layer_2(prev_layer, channels):
    tower_1 = Conv2D(channels, (1, 1), padding='same', activation='relu')(prev_layer)
    tower_1 = BatchNormalization()(tower_1)

    tower_2 = Conv2D(channels, (3, 3), padding='same', activation='relu')(prev_layer)
    tower_2 = BatchNormalization()(tower_2)

    output = Maximum()([tower_1, tower_2])
    return output


def sub_net(prev_layer, conv_out, dense1_out, out_size, dropouts=[0.1, 0.25], kernel_size=3, act='sigmoid'):
    _x_ = Conv2D(conv_out, (kernel_size, kernel_size), padding='same', activation='relu')(prev_layer)
    _x_ = Conv2D(conv_out, (kernel_size, kernel_size), activation='relu')(_x_)
    _x_ = MaxPooling2D(pool_size=2)(_x_)

    _x_ = Dropout(dropouts[0])(_x_)
    _x_ = Flatten()(_x_)

    _x_ = Dense(dense1_out, activation='relu')(_x_)
    _x_ = BatchNormalization()(_x_)
    _x_ = Dropout(dropouts[1])(_x_)
    _x_ = Dense(out_size, activation=act)(_x_)

    return _x_

class KerasDeepModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def free(self):
        del self.model
        K.clear_session()

class ModelFactory:
    def __init__(self):
        pass

    def create_model(self, name, in_shape, parameters = None):
        kdm = None
        if name == 'starter1':
            kdm = KerasDeepModel(name, self.starter1(in_shape, parameters))
        elif name == 'inception1':
            kdm = KerasDeepModel(name, self.inception1(in_shape))
        elif name == 'inception2':
            kdm = KerasDeepModel(name, self.inception2(in_shape))
        elif name == 'inception3':
            kdm = KerasDeepModel(name, self.inception3(in_shape))
        elif name == 'inceptnet1':
            kdm = KerasDeepModel(name, self.inceptnet1(in_shape))
        elif name == 'pyramidnet1':
            kdm = KerasDeepModel(name, self.pyramidnet1(in_shape))
        elif name == 'net1':
            kdm = KerasDeepModel(name, self.net1(in_shape))
        elif name == 'split_net1':
            kdm = KerasDeepModel(name, self.split_net1(in_shape))
        elif name == 'comb_net1':
            kdm = KerasDeepModel(name, self.comb_net1(in_shape))
        elif name == 'comb_net2':
            kdm = KerasDeepModel(name, self.comb_net2(in_shape))
        elif name == 'comb_net3':
            kdm = KerasDeepModel(name, self.comb_net3(in_shape))
        elif name == 'comb_net4':
            kdm = KerasDeepModel(name, self.comb_net4(in_shape))
        elif name == 'comb_net5':
            kdm = KerasDeepModel(name, self.comb_net5(in_shape))
        elif name == 'feat_sep1':
            kdm = KerasDeepModel(name, self.feat_sep1(in_shape))
        elif name == 'feat_sep2':
            kdm = KerasDeepModel(name, self.feat_sep2(in_shape))
        elif name == 'starterBN1':
            kdm = KerasDeepModel(name, self.starterBN1(in_shape))
        elif name == 'starterBN2':
            kdm = KerasDeepModel(name, self.starterBN2(in_shape))
        elif name == 'starterBN3':
            kdm = KerasDeepModel(name, self.starterBN3(in_shape))
        elif name == 'starter2':
            kdm = KerasDeepModel(name, self.starter2(in_shape))
        elif name == 'starter3':
            kdm = KerasDeepModel(name, self.starter3(in_shape))
        elif name == 'starter_no_norm':
            kdm = KerasDeepModel(name, self.starter_no_norm(in_shape))
        elif name == 'starter256':
            kdm = KerasDeepModel(name, self.starter256(in_shape))
        elif name == 'cmscnn1':
            kdm = KerasDeepModel(name, self.cmscnn1(in_shape))
        elif name == 'cmscnn2':
            kdm = KerasDeepModel(name, self.cmscnn2(in_shape))
        elif name == 'cmscnn3':
            kdm = KerasDeepModel(name, self.cmscnn3(in_shape))
        elif name == 'cmscnn4':
            kdm = KerasDeepModel(name, self.cmscnn4(in_shape))
        elif name == 'cmscnn5':
            kdm = KerasDeepModel(name, self.cmscnn5(in_shape))
        elif name == 'cmscnn6':
            kdm = KerasDeepModel(name, self.cmscnn6(in_shape))
        elif name == 'mixnet1':
            kdm = KerasDeepModel(name, self.mixnet1(in_shape))
        elif name == 'mixnet2':
            kdm = KerasDeepModel(name, self.mixnet2(in_shape))
        elif name == 'mixnet3':
            kdm = KerasDeepModel(name, self.mixnet3(in_shape))
        elif name == 'best1':
            kdm = KerasDeepModel(name, self.best1(in_shape))
        elif name == 'dev1':
            kdm = KerasDeepModel(name, self.create_dev1_model(in_shape))
        elif name == 'linear1024_avg_sig':
            kdm = KerasDeepModel(name, self.linear_classifier_1024_avg_sigmoid(in_shape))
        elif name == 'linear1024_avg_05_sig':
            kdm = KerasDeepModel(name, self.linear_classifier_1024_avg_05_sigmoid(in_shape))
        elif name == 'linear1024_max_sig':
            kdm = KerasDeepModel(name, self.linear_classifier_1024_max_sigmoid(in_shape))
        elif name == 'linear1024_max_05_sig':
            kdm = KerasDeepModel(name, self.linear_classifier_1024_max_05_sigmoid(in_shape))
        elif name == 'xception0':
            kdm = KerasDeepModel(name, self.xception0(in_shape))
        elif name == 'xception1':
            kdm = KerasDeepModel(name, self.xception1(in_shape))
        elif name == 'xception2':
            kdm = KerasDeepModel(name, self.xception2(in_shape))
        elif name == 'xception3':
            kdm = KerasDeepModel(name, self.xception3(in_shape))
        elif name == 'xception4':
            kdm = KerasDeepModel(name, self.xception4(in_shape))
        elif name == 'xception5':
            kdm = KerasDeepModel(name, self.xception5(in_shape))
        elif name == 'xception6':
            kdm = KerasDeepModel(name, self.xception6(in_shape))
        elif name == 'xception7':
            kdm = KerasDeepModel(name, self.xception7(in_shape))
        elif name == 'xception8':
            kdm = KerasDeepModel(name, self.xception8(in_shape))
        elif name == 'xception9':
            kdm = KerasDeepModel(name, self.xception9(in_shape))
        elif name == 'xception10':
            kdm = KerasDeepModel(name, self.xception10(in_shape))
        elif name == 'xception11':
            kdm = KerasDeepModel(name, self.xception11(in_shape))
        elif name == 'xception12':
            kdm = KerasDeepModel(name, self.xception12(in_shape))
        elif name == 'vgg16_1':
            kdm = KerasDeepModel(name, self.vgg16_1(in_shape))
        elif name == 'vgg16_2':
            kdm = KerasDeepModel(name, self.vgg16_2(in_shape))
        elif name == 'vgg16_3':
            kdm = KerasDeepModel(name, self.vgg16_3(in_shape))
        elif name == 'vgg19_1':
            kdm = KerasDeepModel(name, self.vgg19_1(in_shape))
        elif name == 'vgg19_2':
            kdm = KerasDeepModel(name, self.vgg19_2(in_shape))
        elif name == 'vgg19_3':
            kdm = KerasDeepModel(name, self.vgg19_3(in_shape))
        elif name == 'inceptionV3_0':
            kdm = KerasDeepModel(name, self.inceptionV3_0(in_shape))
        elif name == 'inceptionV3_1':
            kdm = KerasDeepModel(name, self.inceptionV3_1(in_shape))
        elif name == 'resnet50_0':
            kdm = KerasDeepModel(name, self.resnet50_0(in_shape))
        elif name == 'resnet50_1':
            kdm = KerasDeepModel(name, self.resnet50_1(in_shape))
        elif name == 'resnet50_2':
            kdm = KerasDeepModel(name, self.resnet50_2(in_shape))
        elif name == 'resnet152_0':
            kdm = KerasDeepModel(name, self.resnet152_0(in_shape))
        elif name == 'resnet152_1':
            kdm = KerasDeepModel(name, self.resnet152_1(in_shape))
        else:
            kdm = KerasDeepModel(name, self.create_dev_model(in_shape))
        return kdm

    def create_dev_model(self, in_shape):
        # Create small model for reference
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', strides=2, input_shape=in_shape))
        model.add(Conv2D(64, (2, 2), strides=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (3, 3), strides=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(17, activation='sigmoid'))
        return model

    def create_dev1_model(self, in_shape):
        # Create small model for reference
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', strides=2, input_shape=in_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', strides=2))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(17, activation='relu'))
        model.add(Dense(17, activation='sigmoid'))
        return model

    def vgg16_1(self, in_shape):
        model = VGG16(weights="imagenet", include_top=False, input_shape=in_shape)

        # Freeze conv1 - conv2
        for l in model.layers[:3]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def vgg16_2(self, in_shape):
        model = VGG16(weights="imagenet", include_top=False, input_shape=in_shape)

        # Freeze conv1 - conv4
        for l in model.layers[:6]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def vgg16_3(self, in_shape):
        model = VGG16(weights="imagenet", include_top=False, input_shape=in_shape)

        # Freeze conv1 - conv4
        for l in model.layers[:6]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def vgg19_1(self, in_shape):
        model = VGG19(weights="imagenet", include_top=False, input_shape=in_shape)

        # Freeze conv1 - conv2
        for l in model.layers[:3]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def vgg19_2(self, in_shape):
        model = VGG19(weights="imagenet", include_top=False, input_shape=in_shape)

        # Freeze conv1 - conv4
        for l in model.layers[:6]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def vgg19_3(self, in_shape):
        model = VGG19(weights="imagenet", include_top=False, input_shape=in_shape)

        # Freeze conv1 - conv8
        for l in model.layers[:11]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    # # 512k parameters
    # def densenet40_4(self, in_shape):
    #     input_img = Input(shape=in_shape)
    #     x = BatchNormalization()(input_img)
    #     model = densenet.DenseNet(bottleneck=True, include_top=False, weights=None, input_tensor=x,
    #                               growth_rate=4, reduction=0.5)
    #     x = model.output
    #
    #     x = Dense(512, activation='relu')(x)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.5)(x)
    #
    #     x = Dense(17, activation='sigmoid')(x)
    #
    #     # creating the final model
    #     model_final = Model(inputs=model.input, outputs=x)
    #     return model_final
    #
    # # 512k parameters
    # def densenet100_4(self, in_shape):
    #     input_img = Input(shape=in_shape)
    #     x = BatchNormalization()(input_img)
    #     model = densenet.DenseNet(bottleneck=True, include_top=False, weights=None, input_tensor=x,
    #                               growth_rate=4, depth=100, reduction=0.5)
    #     x = model.output
    #
    #     x = Dense(512, activation='relu')(x)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.5)(x)
    #
    #     x = Dense(17, activation='sigmoid')(x)
    #
    #     # creating the final model
    #     model_final = Model(inputs=model.input, outputs=x)
    #     return model_final

    def inceptionV3_0(self, in_shape):
        model = InceptionV3(include_top=False, weights='imagenet', input_shape = in_shape)

        # Save all concatenate layers
        concats = []
        for i, l in enumerate(model.layers):
            if type(l) is Concatenate:
                concats.append(i)

        # Freeze first inception block
        for l in model.layers[:concats[0]]:
            l.trainable = False

        if in_shape[0] == 256:
            pool_size = 3
        else:
            pool_size = 2

        # Adding classifier
        x = model.output
        x = MaxPooling2D(pool_size=pool_size)(x)
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def inceptionV3_1(self, in_shape):
        model = InceptionV3(include_top=False, weights='imagenet', input_shape = in_shape)

        # Save all concatenate layers
        concats = []
        for i, l in enumerate(model.layers):
            if type(l) is Concatenate:
                concats.append(i)

        # Freeze up to second inception block
        for l in model.layers[:concats[1]]:
            l.trainable = False

        if in_shape[0] == 256:
            pool_size = 3
        else:
            pool_size = 2

        # Adding classifier
        x = model.output
        x = MaxPooling2D(pool_size=pool_size)(x)
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def resnet50_0(self, in_shape):
        model = ResNet50(include_top=False, weights='imagenet', input_shape = in_shape)

        # Save all concatenate layers
        adds = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                adds.append(i)

        # Freeze first 3 blocks
        for l in model.layers[:adds[2]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def resnet50_1(self, in_shape):
        model = ResNet50(include_top=False, weights='imagenet', input_shape = in_shape)

        # Save all concatenate layers
        adds = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                adds.append(i)

        # Freeze first 5 blocks
        for l in model.layers[:adds[4]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def resnet50_2(self, in_shape):
        model = ResNet50(include_top=False, weights='imagenet', input_shape = in_shape)

        # Save all concatenate layers
        adds = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                adds.append(i)

        # Freeze first 7 blocks
        for l in model.layers[:adds[6]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def resnet152_0(self, in_shape):
        model = resnet152_model('resnet152_weights_tf.h5', input_size=in_shape[0], include_top=False)

        # Save all concatenate layers
        adds = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                adds.append(i)

        # Freeze first 7 blocks
        for l in model.layers[:adds[6]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def resnet152_1(self, in_shape):
        model = resnet152_model('resnet152_weights_tf.h5', input_size=in_shape[0], include_top=False)

        # Save all concatenate layers
        adds = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                adds.append(i)

        # Freeze first 5 blocks
        for l in model.layers[:adds[4]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception0(self, in_shape):
        model = Xception(weights=None, include_top=False, input_shape=in_shape, pooling='avg')

        # Adding classifier
        x = model.output
        x = Dense(1024, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception1(self, in_shape):
        model = Xception(weights="imagenet", include_top=False, input_shape=in_shape, pooling='avg')

        # Save all residuals
        residuals = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                residuals.append(i)

        # Freeze first block
        for l in model.layers[:residuals[0]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Dense(1024, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception2(self, in_shape):
        model = Xception(weights=None, include_top=False, input_shape=in_shape)

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception3(self, in_shape):
        model = Xception(weights=None, include_top=False, input_shape=in_shape)

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception4(self, in_shape):
        model = Xception(weights=None, include_top=False, input_shape=in_shape)

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception5(self, in_shape):
        model = Xception(weights="imagenet", include_top=False, input_shape=in_shape, pooling='avg')

        # Save all residuals
        residuals = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                residuals.append(i)

        # Freeze first block
        for l in model.layers[:residuals[0]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception6(self, in_shape):
        model = Xception(weights="imagenet", include_top=False, input_shape=in_shape, pooling='avg')

        # Save all residuals
        residuals = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                residuals.append(i)

        # Freeze first block
        for l in model.layers[:residuals[1]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception7(self, in_shape):
        model = Xception(weights="imagenet", include_top=False, input_shape=in_shape, pooling='avg')

        # Save all residuals
        residuals = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                residuals.append(i)

        # Freeze first block
        for l in model.layers[:residuals[2]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception8(self, in_shape):
        model = Xception(weights="imagenet", include_top=False, input_shape=in_shape, pooling='avg')

        # Save all residuals
        residuals = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                residuals.append(i)

        # Freeze first block
        for l in model.layers[:residuals[0]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception9(self, in_shape):
        model = Xception(weights="imagenet", include_top=False, input_shape=in_shape, pooling='max')

        # Save all residuals
        residuals = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                residuals.append(i)

        # Freeze first block
        for l in model.layers[:residuals[0]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception10(self, in_shape):
        model = Xception(weights="imagenet", include_top=False, input_shape=in_shape)

        # Save all residuals
        residuals = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                residuals.append(i)

        # Freeze first block
        for l in model.layers[:residuals[0]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception11(self, in_shape):
        model = Xception(weights="imagenet", include_top=False, input_shape=in_shape)

        # Save all residuals
        residuals = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                residuals.append(i)

        # Freeze first block
        for l in model.layers[:residuals[0]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def xception12(self, in_shape):
        model = Xception(weights="imagenet", include_top=False, input_shape=in_shape, pooling='avg')

        # Save all residuals
        residuals = []
        for i, l in enumerate(model.layers):
            if type(l) is Add:
                residuals.append(i)

        # Freeze first block
        for l in model.layers[:residuals[0]]:
            l.trainable = False

        # Adding classifier
        x = model.output
        x = Dropout(0.25)(x)
        x = Dense(17, activation="sigmoid")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=x)
        return model_final

    def pyramidnet1(self, in_shape):
        input_img = Input(shape=in_shape)
        num_classes = 17

        x = BatchNormalization()(input_img)  # 128
        x = preprocess(x)

        conv1d = create_conv_btneck(x, [32, 32, 64])
        x = MaxPooling2D(pool_size=2, strides=(2, 2))(conv1d)  # 64

        short2d = Conv2D(128, (1, 1), padding='same')(x)
        conv2d = Add()([create_conv_btneck(x, [64, 64, 128]), short2d])
        x = MaxPooling2D(pool_size=2, strides=(2, 2))(conv2d)  # 32
        logit2d = create_cls(GlobalMaxPooling2D()(x), num_classes)

        short3d = Conv2D(256, (1, 1), padding='same')(x)
        conv3d = Add()([create_conv_btneck(x, [128, 128, 256]), short3d])
        x = MaxPooling2D(pool_size=2, strides=(2, 2))(conv3d)  # 16
        logit3d = create_cls(GlobalMaxPooling2D()(x), num_classes)

        short4d = x
        conv4d = Add()([create_conv_btneck(x, [256, 256, 256]), short4d])
        x = MaxPooling2D(pool_size=2, strides=(2, 2))(conv4d)  # 8
        logit4d = create_cls(GlobalMaxPooling2D()(x), num_classes)

        short5d = x
        conv5d = Add()([create_conv_btneck(x, [256, 256, 256]), short5d])
        logit5d = create_cls(GlobalMaxPooling2D()(x), num_classes)

        x = Conv2DTranspose(256, kernel_size=1, strides=(2, 2))(conv5d)  # 16
        x = Add()([x, conv4d])
        conv4u = create_conv_btneck(x, [256, 256, 256])
        logit4u = create_cls(GlobalMaxPooling2D()(conv4u), num_classes)

        x = Conv2DTranspose(256, kernel_size=1, strides=(2, 2))(x)  # 32
        x = Add()([x, conv3d])
        conv3u = create_conv_btneck(x, [128, 128, 128])
        logit3u = create_cls(GlobalMaxPooling2D()(conv3u), num_classes)

        x = Conv2DTranspose(128, kernel_size=1, strides=(2, 2))(x)  # 64
        x = Add()([x, conv2d])
        conv2u = create_conv_btneck(x, [64, 64, 64])
        logit2u = create_cls(GlobalMaxPooling2D()(conv2u), num_classes)

        x = Conv2DTranspose(64, kernel_size=1, strides=(2, 2))(x)  # 128
        x = Add()([x, conv1d])
        conv1u = create_conv_btneck(x, [64, 64, 64])
        logit1u = create_cls(GlobalMaxPooling2D()(conv1u), num_classes)

        out = Add()([logit2d, logit3d, logit4d, logit5d, logit4u, logit3u, logit2u, logit1u])
        # out = Dropout(0.15)(logit)
        out = Activation('sigmoid')(out)

        model = Model(inputs=input_img, outputs=out)
        return model

    def comb_net1(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        out1 = sub_net(x, 512, dense1_out=512, out_size=17, dropouts=[0.25, 0.5], kernel_size=3)

        channels = []
        for i in range(17):
            channels.append(sub_net(x, 64, dense1_out=128, out_size=1, dropouts=[0.2, 0.2], kernel_size=3))

        channels.insert(0, out1)
        x = Concatenate()(channels)

        model = Model(inputs=input_img, outputs=x)
        return model

    def comb_net2(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        out1 = sub_net(x, 512, dense1_out=512, out_size=17, dropouts=[0.25, 0.5], kernel_size=3)

        channels = []
        for i in range(17):
            channels.append(sub_net(x, 128, dense1_out=256, out_size=1, dropouts=[0.2, 0.2], kernel_size=3))

        channels.insert(0, out1)
        x = Concatenate()(channels)

        model = Model(inputs=input_img, outputs=x)
        return model

    def comb_net3(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        out1 = sub_net(x, 512, dense1_out=512, out_size=17, dropouts=[0.25, 0.5], kernel_size=3)

        channels = []
        for i in range(17):
            channels.append(sub_net(x, 256, dense1_out=512, out_size=1, dropouts=[0.2, 0.2], kernel_size=3))

        channels.insert(0, out1)
        x = Concatenate()(channels)

        model = Model(inputs=input_img, outputs=x)
        return model

    def comb_net4(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        out1 = sub_net(x, 512, dense1_out=512, out_size=17, dropouts=[0.25, 0.5], kernel_size=3)

        channels = []
        for i in range(17):
            channels.append(sub_net(x, 256, dense1_out=512, out_size=1, dropouts=[0.25, 0.5], kernel_size=3))

        channels.insert(0, out1)
        x = Concatenate()(channels)

        model = Model(inputs=input_img, outputs=x)
        return model

    def comb_net5(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        out1 = sub_net(x, 512, dense1_out=512, out_size=17, dropouts=[0.25, 0.5], kernel_size=3)

        channels = []
        for i in range(17):
            channels.append(sub_net(x, 512, dense1_out=512, out_size=1, dropouts=[0.25, 0.2], kernel_size=3))

        channels.insert(0, out1)
        x = Concatenate()(channels)

        model = Model(inputs=input_img, outputs=x)
        return model

    def net1(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        out1 = sub_net(x, 512, dense1_out=512, out_size=17, dropouts=[0.25, 0.5], kernel_size=3, act='linear')

        channels = []
        for i in range(17):
            channels.append(
                sub_net(x, 256, dense1_out=512, out_size=1, dropouts=[0.2, 0.2], kernel_size=3, act='linear'))
        out2 = Concatenate()(channels)
        x = Add()([out1, out2])
        x = Activation('sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    def split_net1(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        out1 = conv_block(x, 256, dropout=0.25, kernel_size=3)

        out1 = conv_block(out1, 512, dropout=0.25, kernel_size=3)
        out1 = Flatten()(out1)
        out1 = dense_block(out1, 512, dropout=0.5)
        out1 = Dense(17, activation='linear')(out1)

        channels = []
        for i in range(17):
            outi = conv_block(x, 256, dropout=0.25, kernel_size=3)
            outi = conv_block(outi, 256, dropout=0.25, kernel_size=3)
            outi = Flatten()(outi)
            outi = dense_block(outi, 512, dropout=0.25)
            outi = Dense(1, activation='linear')(outi)
            channels.append(outi)
        out2 = Concatenate()(channels)
        x = Add()([out1, out2])
        x = Activation('sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    def feat_sep1(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        channels = []
        for i in range(17):
            channels.append(sub_net(x, 256, 512, out_size=1, kernel_size=3))

        x = Concatenate()(channels)

        model = Model(inputs=input_img, outputs=x)
        return model

    def feat_sep2(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        channels = []
        for i in range(17):
            channels.append(sub_net(x, 512, 512, out_size=1, kernel_size=3))

        x = Concatenate()(channels)

        model = Model(inputs=input_img, outputs=x)
        return model

    # 3M parameters. ~40s per epoch. Net training ~26min
    def best1(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (2, 2), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(512, (2, 2), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        model.add(Dense(256, activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(17, activation='sigmoid', kernel_initializer=he_normal(seed=0)))
        return model

    # 4M params
    def linear_classifier_1024_avg_sigmoid(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (2, 2), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(512, (2, 2), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(1024, (1, 1), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(1024, (1, 1), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(AveragePooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(17, activation='sigmoid', kernel_initializer=he_normal(seed=0)))
        return model

    # 4M params
    def linear_classifier_1024_max_sigmoid(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (2, 2), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(512, (2, 2), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(1024, (1, 1), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(1024, (1, 1), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(17, activation='sigmoid', kernel_initializer=he_normal(seed=0)))
        return model

    # 4M params
    def linear_classifier_1024_max_05_sigmoid(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (2, 2), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(512, (2, 2), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(1024, (1, 1), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(1024, (1, 1), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(17, activation='sigmoid', kernel_initializer=he_normal(seed=0)))
        return model

    # 4M params
    def linear_classifier_1024_avg_05_sigmoid(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (2, 2), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(512, (2, 2), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(1024, (1, 1), padding='same', activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(Conv2D(1024, (1, 1), activation='relu', kernel_initializer=he_normal(seed=0)))
        model.add(AveragePooling2D(pool_size=2))
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(17, activation='sigmoid', kernel_initializer=he_normal(seed=0)))
        return model

    # 3M parameters. ~40s per epoch. Net training ~26min
    def starter1(self, in_shape, parameters):
        dropouts = [0.25, 0.25, 0.25, 0.25, 0.25, 0.5]
        if parameters is not None:
            dropouts = parameters['dropouts']
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(dropouts[0]))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(dropouts[1]))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(dropouts[2]))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(dropouts[3]))

        model.add(Conv2D(512, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(512, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(dropouts[4]))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropouts[5]))
        model.add(Dense(17, activation='sigmoid'))
        return model

    # 3M parameters. ~40s per epoch. Net training ~26min
    def starter_no_norm(self, in_shape):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=in_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(512, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))
        return model

    # 3M parameters. ~40s per epoch. Net training ~26min
    def starterBN1(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (2, 2), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))
        return model

    # 3M parameters. ~40s per epoch. Net training ~26min
    def starterBN2(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(512, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        # model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))
        return model

    # 3M parameters. ~40s per epoch. Net training ~26min
    def starterBN3(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        # model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(512, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        # model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))
        return model

    # 13M parameters.
    def starter256(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(1024, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(1024, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))

        return model

    # Appropriate for 256 input size
    def starter2(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))
        model.add(Conv2D(32, (7, 7), strides=2, padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2)) # Maxpool to downsample 256 input size
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (5, 5), strides=2, padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (5, 5), strides=2, padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (5, 5), strides=2, padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(1024, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))

        return model

    # 20 parameters.
    def starter3(self, in_shape):
        model = Sequential()
        model.add(BatchNormalization(input_shape=in_shape))
        model.add(Conv2D(32, (7, 7), strides=2, padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (5, 5), strides=2, padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (5, 5), strides=2, padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (5, 5), strides=2, padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(1024, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))

        return model

    # 17M parameters. ~270s per epoch. Net training ~3hours
    def cmscnn1(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)
        x = cmsc_inception_layer_4(x, 32)
        x = cmsc_inception_layer_2(x, 32)
        x = cmsc_inception_layer_2(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_4(x, 64)
        x = cmsc_inception_layer_2(x, 64)
        x = cmsc_inception_layer_2(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    # 18M parameters. ~280s per epoch. Net training ~3hours
    def cmscnn2(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)
        x = cmsc_inception_layer_4(x, 32)
        x = cmsc_inception_layer_3(x, 32)
        x = cmsc_inception_layer_2(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_4(x, 64)
        x = cmsc_inception_layer_3(x, 64)
        x = cmsc_inception_layer_2(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    # 12M parameters. ~190s per epoch. Net training ~2hours
    def cmscnn3(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)
        x = cmsc_inception_layer_4(x, 32)
        x = cmsc_inception_layer_3(x, 32)
        x = cmsc_inception_layer_2(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_4(x, 64)
        x = cmsc_inception_layer_3(x, 64)
        x = cmsc_inception_layer_2(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 256)
        x = cmsc_inception_layer_2(x, 256)
        x = cmsc_inception_layer_2(x, 256)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    # 17M parameters. ~270s per epoch. Net training ~3hours
    def cmscnn4(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)
        x = cmsc_inception_layer_4(x, 32)
        x = cmsc_inception_layer_3(x, 32)
        x = cmsc_inception_layer_2(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_4(x, 64)
        x = cmsc_inception_layer_3(x, 64)
        x = cmsc_inception_layer_2(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 256)
        x = cmsc_inception_layer_2(x, 256)
        x = cmsc_inception_layer_2(x, 256)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 512)
        x = cmsc_inception_layer_2(x, 512)
        x = cmsc_inception_layer_2(x, 512)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    # 13M parameters. 260s per epoch. Net training ~3hours
    def cmscnn5(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)
        x = cmsc_inception_layer_4(x, 32)
        x = cmsc_inception_layer_3(x, 32)
        x = cmsc_inception_layer_2(x, 32)
        x = MaxPooling2D(pool_size=3)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_4(x, 64)
        x = cmsc_inception_layer_3(x, 64)
        x = cmsc_inception_layer_2(x, 64)
        x = MaxPooling2D(pool_size=3)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 256)
        x = cmsc_inception_layer_2(x, 256)
        x = cmsc_inception_layer_2(x, 256)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 512)
        x = cmsc_inception_layer_2(x, 512)
        x = cmsc_inception_layer_2(x, 512)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    # 54M parameters. Net training ~9hours
    def cmscnn6(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)
        x = cmsc_inception_layer_4(x, 32)
        x = cmsc_inception_layer_3(x, 32)
        x = cmsc_inception_layer_2(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_4(x, 64)
        x = cmsc_inception_layer_3(x, 64)
        x = cmsc_inception_layer_2(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 256)
        x = cmsc_inception_layer_2(x, 256)
        x = cmsc_inception_layer_2(x, 256)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 512)
        x = cmsc_inception_layer_2(x, 512)
        x = cmsc_inception_layer_2(x, 512)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = cmsc_inception_layer_3(x, 1024)
        x = cmsc_inception_layer_2(x, 1024)
        x = cmsc_inception_layer_2(x, 1024)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    # 15M parameters. 80s per epoch. Training time ~45 min
    def mixnet1(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = cmsc_inception_layer_3(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = cmsc_inception_layer_3(x, 256)
        x = cmsc_inception_layer_2(x, 256)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = cmsc_inception_layer_3(x, 512)
        x = cmsc_inception_layer_2(x, 512)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    # 6M parameters. 230s per epoch
    def mixnet2(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = cmsc_inception_layer_4(x, 32)
        x = cmsc_inception_layer_3(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = cmsc_inception_layer_3(x, 64)
        x = cmsc_inception_layer_2(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = cmsc_inception_layer_3(x, 128)
        x = cmsc_inception_layer_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    # 6M parameters. 218s per epoch
    def mixnet3(self, in_shape):
        input_img = Input(shape=in_shape)

        x = BatchNormalization()(input_img)

        x = cmsc_inception_layer_4(x, 32)
        x = cmsc_inception_layer_3(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.3)(x)

        x = cmsc_inception_layer_3(x, 64)
        x = cmsc_inception_layer_2(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    def inception1(self, in_shape):
        input_img = Input(shape=in_shape)
        x = BatchNormalization()(input_img)

        x = my_inception_1(x, 16)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_1(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_1(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(512, (2, 2), activation='relu')(x)
        x = Conv2D(512, (2, 2), activation='relu')(x)
        x = AveragePooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)

        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    def inception2(self, in_shape):
        input_img = Input(shape=in_shape)
        x = BatchNormalization()(input_img)

        x = my_inception_1(x, 16)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_1(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_1(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(512, (2, 2), activation='relu')(x)
        x = Conv2D(512, (2, 2), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)

        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)

        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    def inception3(self, in_shape):
        input_img = Input(shape=in_shape)
        x = BatchNormalization()(input_img)

        x = my_inception_1(x, 8)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_1(x, 16)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_1(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_2(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (2, 2), activation='relu')(x)
        x = Conv2D(256, (2, 2), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)

        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)

        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model

    def inceptnet1(self, in_shape):
        input_img = Input(shape=in_shape)
        x = BatchNormalization()(input_img)

        x = my_inception_1(x, 16)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_1(x, 32)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_1(x, 64)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = my_inception_2(x, 128)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        out1 = sub_net(x, 512, dense1_out=512, out_size=17, dropouts=[0.25, 0.5], kernel_size=2, act='linear')

        channels = []
        for i in range(17):
            channels.append(
                sub_net(x, 256, dense1_out=512, out_size=1, dropouts=[0.25, 0.25], kernel_size=2, act='linear'))
        out2 = Concatenate()(channels)
        x = Add()([out1, out2])
        x = Activation('sigmoid')(x)

        model = Model(inputs=input_img, outputs=x)
        return model