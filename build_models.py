'''
Project name

Description:

Author: Charles Zhou
Date: 2019-10-05
'''

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, Dropout, concatenate
from keras.models import Model, Sequential, load_model
import numpy as np
from global_config import INPUT_SHAPE
import keras.backend as K

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1))

def multi_filter_cnn(output_dim=2):
    # let's define some variables
    filter_sizes = [3, 10]
    conv_filters = []
    input_channels = []
    nb_feature_maps = 10
    dropout = 0.0
    for fs in filter_sizes:
        input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format
        input_channels.append(input_img)
        x = Conv2D(nb_feature_maps, (fs, fs), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((fs//2+1, fs//2+1), padding='same')(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        x = Conv2D(nb_feature_maps, (fs, fs), activation='relu', padding='same')(x)
        x = MaxPooling2D((fs//2+1, fs//2+1), padding='same')(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        x = Conv2D(nb_feature_maps, (fs, fs), activation='relu', padding='same')(x)
        x = MaxPooling2D((fs//2+1, fs//2+1), padding='same')(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        x = Conv2D(nb_feature_maps, (fs, fs), activation='relu', padding='same')(x)
        x = MaxPooling2D((fs//2+1, fs//2+1), padding='same')(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        x = Conv2D(nb_feature_maps, (fs, fs), activation='relu', padding='same')(x)
        x = MaxPooling2D((fs//2+1, fs//2+1), padding='same')(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        flattened = Flatten()(x)
        conv_filters.append(flattened)

    merged = concatenate(conv_filters)
    dense = Dense(50)(merged)
    dense = Dense(15)(dense)
    output = Dense(output_dim, activation='relu')(dense)
    model = Model(inputs=input_channels, outputs=[output])
    model.compile(optimizer='adam', loss='MAE', metrics=['MAE'])
    model.summary()
    return model



def build_center_predictor(epoch=50):
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"euclidean_distance_loss": euclidean_distance_loss})
    # m = cnn_rg()
    train_new = False
    try:
        if train_new:
            m = multi_filter_cnn()
        else:
            m = load_model('c_center.h5')
    except Exception as e:
        print(e)
        m = multi_filter_cnn()
    m.compile(optimizer='adam', loss=euclidean_distance_loss, metrics=['MAE'])
    return_original = True
    from task_env import get_samples
    buffer_size = 50
    bs_buffer = []
    while epoch:
        np.random.seed(None)
        X = []
        X_prime = []
        Y = []
        for obj in get_samples(5000, norm=False, return_original=return_original, noise_lvl=2):
            x, y = obj
            if return_original:
                x, x_prime = x
                x_prime = np.expand_dims(x_prime, -1)
                X_prime.append(x_prime)
            x = np.expand_dims(x, -1)
            # single channel
            X.append(x)
            Y.append(y[:2])
        Y = np.array(Y)
        X_prime = np.array(X_prime)
        # print(np.average(X_prime), np.average(Y))
        history = m.fit([
            X_prime,
            X_prime,
        ], Y, epochs=1, validation_split=0.1, batch_size=32, shuffle=True, verbose=1)
        m.save('c_center.h5')
        # print(history.history.keys())
        for i in range(len(history.history['val_loss'])):
            bs_buffer.insert(0, history.history['val_loss'][i])
            while len(bs_buffer) > buffer_size:
                bs_buffer.pop(-1)
        bs_ret = bs.bootstrap(np.array(bs_buffer), stat_func=bs_stats.mean)
        print(bs_ret)
        epoch -= 1


def build_radius_predictor(epoch=50):
    # m = cnn_rg()
    train_new = False
    try:
        if train_new:
            m = multi_filter_cnn(output_dim=1)
        else:
            m = load_model('c_radius.h5')
    except Exception as e:
        print(e)
        m = multi_filter_cnn(output_dim=1)
    m.compile(optimizer='adam', loss='MSE', metrics=['MAE'])
    return_original = True
    from task_env import get_samples
    buffer_size = 50
    bs_buffer = []
    while epoch:
        np.random.seed(None)
        X = []
        X_prime = []
        Y = []
        for obj in get_samples(5000, norm=False, return_original=return_original, noise_lvl=2):
            x, y = obj
            if return_original:
                x, x_prime = x
                x_prime = np.expand_dims(x_prime, -1)
                X_prime.append(x_prime)
            x = np.expand_dims(x, -1)
            # single channel
            X.append(x)
            Y.append(y[-1:])

        X = np.array(X)
        Y = np.array(Y)
        X_prime = np.array(X_prime)
        # print(X.shape, Y.shape, X_prime.shape)

        # print(np.average(X_prime), np.average(Y))
        history = m.fit([
            X_prime,
            X_prime,
        ], Y, epochs=1, validation_split=0.1, batch_size=32, shuffle=True, verbose=2)
        m.save('c_radius.h5')

        for i in range(len(history.history['val_mean_absolute_error'])):
            bs_buffer.insert(0, history.history['val_mean_absolute_error'][i])
            while len(bs_buffer) > buffer_size:
                bs_buffer.pop(-1)

        bs_ret = bs.bootstrap(np.array(bs_buffer), stat_func=bs_stats.mean)
        print(bs_ret)
        epoch -= 1

if __name__ == "__main__":
    import tensorflow as tf

    build_center_predictor()
    build_radius_predictor()
