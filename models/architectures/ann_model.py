import glob

import keras as k
import keras.layers
from keras import backend
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dropout, Flatten, MaxPool1D, Bidirectional, Concatenate, \
    Dense, concatenate, Conv2DTranspose, UpSampling2D, Conv1DTranspose, BatchNormalization, Input, Conv2D, MaxPool2D, MaxPooling2D, LSTM, Add, ConvLSTM2D, Reshape, GRU, TimeDistributed, AvgPool1D
from keras.optimizers import adam_v2 as Adam
import tensorflow as tf
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
import numpy as np
import model.define_model as dfm
from sklearn.model_selection import train_test_split
from tfrecord.generate_tfrecord import AIData
import json
from datetime import datetime


def ann_old():
    model_input = Input(shape=(256, 1))
    x = model_input
    x = Conv1D(2, 3, padding='same', activation='elu')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(2, 3, padding='valid', activation='elu')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(2, 3,  padding='valid', activation='elu')(x)
    x = MaxPool1D(2)(x)

    x = Flatten()(x)

    y = Dense(32, activation='elu')(x)
    z = Dense(32, activation='elu')(x)


    X_RPTT = Dense(1, activation=dfm.activation, name='RPTT')(y)
    X_HR = Dense(1, activation=dfm.activation, name='HR')(z)

    model_output = [X_RPTT, X_HR]
    model = Model(model_input, model_output)
    # model.summary()

    return model

def ann():
    model_input = Input(shape=(256, 1))
    x = model_input
    x = Conv1D(3, 5, padding='same', activation='elu')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(2, 3, 2, padding='valid', activation='elu')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(2, 3, 2,  padding='valid', activation='elu')(x)
    x = MaxPool1D(2)(x)

    x = Flatten()(x)

    y = Dense(8, activation='elu')(x)
    z = Dense(4, activation='elu')(x)

    X_RPTT = Dense(1, activation=dfm.activation, name='RPTT')(y)
    X_HR = Dense(1, activation=dfm.activation, name='HR')(z)

    model_output = [X_RPTT, X_HR]
    model = Model(model_input, model_output)
    # model.summary()

    return model

def ann_v308():
    model_input = Input(shape=(256, 1))
    x = model_input
    x = Conv1D(2, 3, padding='same', activation='elu')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(2, 3, 2, padding='valid', activation='elu')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(2, 3, 2,  padding='valid', activation='elu')(x)
    x = MaxPool1D(2)(x)

    x = Flatten()(x)

    y = Dense(6, activation='elu')(x)
    z = Dense(6, activation='elu')(x)

    X_RPTT = Dense(1, activation=dfm.activation, name='RPTT')(y)
    X_HR = Dense(1, activation=dfm.activation, name='HR')(z)

    model_output = [X_RPTT, X_HR]
    model = Model(model_input, model_output)
    # model.summary()

    return model


def ann_1in():
    model_input = Input(shape=(dfm.input_length, 1))
    x = model_input
    x = Conv1D(8, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv1D(16, 3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(8, 3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)

    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64, return_sequences=False)(x)
    # x = Flatten()(x)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(x)

    model_output = [SBP, DBP]
    model = Model(model_input, model_output)
    # model.summary()

    return model


def cnn_test():
    model_input = Input(shape=(dfm.input_length, 1))
    model_input_2 = Input(shape=2)

    x = model_input

    x = Conv1D(8, 3,  padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(4, 3,  padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(2, 3,  padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)


    x = Flatten()(x)
    x = Concatenate()([model_input_2, x])
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(x)

    model_output = [SBP, DBP]
    model = Model([model_input, model_input_2], model_output)
    # model.summary()

    return model


def ann_test():
    in1 = Input(shape=(256, 1))
    x = in1
    x = Conv1D(2, 3, padding='same', activation='elu')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(2, 3, 2, padding='valid', activation='elu')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(2, 3, 2, padding='valid', activation='elu')(x)
    x = MaxPool1D(2)(x)
    x = Flatten()(x)

    in2 = Input(shape=(1, 1))
    _x = Flatten()(in2)
    # _x = Dense(4, activation='elu')(_x)

    x = concatenate([x, _x])

    x = Dense(8, activation='elu')(x)
    # x = Dense(4, activation='elu')(x)

    y = Dense(4, activation='elu')(x)
    z = Dense(4, activation='elu')(x)

    # x = Dense(3, activation='elu')(X_RPTT)
    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(y)
    X_DBP = Dense(1, activation=dfm.activation, name='DBP')(z)

    model_input = [in1, in2]
    model_output = [X_SBP, X_DBP]
    model = Model(model_input, model_output)
    # model.summary()

    return model


def ann_calibrated():
    input = Input(shape=(2, ))
    x = input
    # x = Dense(32, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)

    # KA = Dense(1, activation=dfm.activation, name='KA')(x)
    KB = Dense(1, activation=dfm.activation, name='KB')(x)
    # X_DBP = Dense(1, activation=dfm.activation, name='DBP')(z)

    model_input = input
    model_output = KB
    model = Model(model_input, model_output)
    # model.summary()

    return model


def ann_rptt2bp():
    input = Input(shape=(2, ))
    x = input
    # x = Dense(32, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)

    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    # X_DBP = Dense(1, activation=dfm.activation, name='DBP')(z)

    model_input = input
    model_output = X_SBP
    model = Model(model_input, model_output)
    # model.summary()

    return model


def ann_rptt2bp_calibrated():
    input = Input(shape=(5, ))
    x = input
    # x = Dense(32, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='elu')(x)
    x = Dense(32, activation='elu')(x)
    x = Dense(16, activation='elu')(x)
    x = Dense(16, activation='elu')(x)
    x = Dense(8, activation='elu')(x)

    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    # X_DBP = Dense(1, activation=dfm.activation, name='DBP')(z)

    model_input = input
    model_output = X_SBP
    model = Model(model_input, model_output)
    # model.summary()

    return model

def cnn_test_3_25():
    model_input = Input(shape=(dfm.input_length, 1))
    model_input_2 = Input(shape=2)
    x = model_input
    x = Conv1D(16, 3,  padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(8, 3,  padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(8, 3,  padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(4, 3,  padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Flatten()(x)
    x = Concatenate()([model_input_2, x])
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    model_output = [SBP, DBP]
    model = Model([model_input, model_input_2], model_output)
    # model.summary()
    return model


def cnn_test_3_26():
    model_input = Input(shape=(dfm.input_length, 1))
    model_input_2 = Input(shape=2)
    x = model_input
    x = Conv1D(16, 3,  padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(16, 3,  padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(8, 3,  padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(4, 3,  padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Flatten()(x)
    x = Concatenate()([model_input_2, x])
    # x = BatchNormalization()(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    model_output = [SBP, DBP]
    model = Model([model_input, model_input_2], model_output)
    # model.summary()
    return model


def cnn_test_3_25_1():
    model_input = Input(shape=(dfm.input_length, 1))
    model_input_2 = Input(shape=2)
    x = model_input
    x = Conv1D(16, 3,  padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(8, 3,  padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(4, 3,  padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    # x = Conv1D(4, 3,  padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool1D(2)(x)
    x = Flatten()(x)

    y = Dense(64, activation='relu')(model_input_2)
    x = Concatenate()([y, x])
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    model_output = [SBP, DBP]
    model = Model([model_input, model_input_2], model_output)
    # model.summary()
    return model


def cnn_4_0_1():
    model_input = Input(shape=(dfm.input_length, 2, 1))
    x = model_input

    # x = Dense(64, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    x = Conv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    # x = Conv2D(4, kernel_size=(3, 1), strides=(2, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Conv2D(2, kernel_size=(3, 2), strides=(2, 2), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, kernel_size=(3, 1), strides=(2, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    # x = Reshape((31, 1))(x)
    x = Flatten()(x)

    # x = LSTM(8, return_sequences=True)(x)
    # x = Dropout(rate=0.1)(x)
    # x = Bidirectional(LSTM(128, return_sequences=False))(x)
    # x = LSTM(4, return_sequences=False)(x)
    # x = Dropout(rate=0.1)(x)

    x = Dense(8, activation='relu')(x)
    SBP = Dense(1, activation='linear', name='SBP')(x)
    DBP = Dense(1, activation='linear', name='DBP')(x)
    model_output = [SBP, DBP]
    model = Model(model_input, model_output)
    # model.summary()
    return model


def cnn_4_0_2():
    model_input = Input(shape=(dfm.input_length, 1))
    x = model_input

    # x = Dense(64, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    # x = Conv1D(1, 7, 5,  padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(4, kernel_size=(3, 1), strides=(2, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)

    # x = Reshape((32, 1))(x)


    x = LSTM(32, return_sequences=True)(x)
    x = Dropout(rate=0.1)(x)
    #
    # # x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = LSTM(8, return_sequences=False)(x)
    x = Dropout(rate=0.1)(x)
    #
    # x = Dense(8, activation='relu')(x)
    SBP = Dense(1, activation='linear', name='SBP')(x)
    DBP = Dense(1, activation='linear', name='DBP')(x)
    model_output = [SBP, DBP]
    model = Model(model_input, model_output)
    # model.summary()
    return model


def cnn_4_0_3():
    model_input = Input(shape=(dfm.input_length, 1))
    x = model_input
    x = Conv1D(20, 9, activation=dfm.Conv_activation)(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(rate=0.1)(x)

    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(rate=0.1)(x)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    model_output = [SBP, DBP]
    model = Model(model_input, model_output)
    # model.summary()
    return model


def cnn_4_0_4():
    model_input = Input(shape=(dfm.input_length, 1))
    model_input_2 = Input(shape=1)
    x = model_input

    x = Conv1D(16, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(8, 3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(4, 3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    # x = Conv1D(4, 3,  padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool1D(2)(x)
    x = Flatten()(x)

    y = model_input_2

    z = Concatenate()([y, x])
    z = BatchNormalization()(z)
    z = Dense(16, activation='relu')(z)
    z1 = Dense(4, activation='relu')(z)
    z2 = Dense(4, activation='relu')(z)
    SBP = Dense(1, activation=dfm.activation, name='SBP')(z1)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(z2)
    model_output = [SBP, DBP]
    model = Model([model_input, model_input_2], model_output)
    # model.summary()
    return model


def cnn_4_0_5():
    model_input = Input(shape=(dfm.input_length, 1))
    model_input_2 = Input(shape=1)
    x = model_input

    x = Conv1D(16, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(8, 3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(4, 3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    # x = Conv1D(4, 3,  padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool1D(2)(x)
    x = Flatten()(x)

    y = model_input_2

    z = Concatenate()([y, x])
    # z = BatchNormalization()(z)
    z = Dense(16, activation='relu')(z)
    z1 = Dense(4, activation='relu')(z)
    z2 = Dense(4, activation='relu')(z)

    z_s = Concatenate()([z, z1, z2])
    z_d = Concatenate()([z, z1, z2])

    z_s = Dense(4, activation='relu')(z_s)
    z_d = Dense(4, activation='relu')(z_d)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(z_s)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(z_d)
    model_output = [SBP, DBP]
    model = Model([model_input, model_input_2], model_output)
    # model.summary()
    return model


def cnn_4_0_5_2():
    model_input = Input(shape=(dfm.input_length, 1, 1))
    model_input_2 = Input(shape=1)
    x = model_input

    x = Conv2D(8, (3, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)

    # # x = MaxPool2D((2, 1))(x)
    x = Conv2D(4, (3, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)

    x = Conv2D(2, (3, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)

    # x = MaxPool2D((2, 1))(x)
    # x = Conv2D(2, (3, 1), (1, 1), padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool2D((2, 1))(x)
    # x = Conv1D(4, 3,  padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool1D(2)(x)
    x = Flatten()(x)

    y = model_input_2

    z = Concatenate()([y, x])
    # z = BatchNormalization()(z)
    z = Dense(64, activation='relu')(z)
    z = Dense(16, activation='relu')(z)
    z1 = Dense(4, activation='relu')(z)
    z2 = Dense(4, activation='relu')(z)
    #
    # z_s = Concatenate()([z, z1, z2])
    # z_d = Concatenate()([z, z1, z2])

    # z_s = Dense(8, activation='relu')(z_s)
    # z_d = Dense(8, activation='relu')(z_d)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(z1)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(z2)
    model_output = [SBP, DBP]
    model = Model([model_input, model_input_2], model_output)
    model.summary()
    return model


def cnn_4_0_5_3():
    model_input = Input(shape=(dfm.input_length, 2, 1))
    # model_input_2 = Input(shape=1)
    x = model_input

    x = Conv2D(8, (3, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)
    x = Conv2D(8, (3, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)
    x = Conv2D(4, (3, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)

    x = Conv2D(2, (2, 2), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    # x = MaxPool2D((2, 1))(x)
    # x = Conv2D(2, (3, 1), (1, 1), padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool2D((2, 1))(x)
    # x = Conv1D(4, 3,  padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool1D(2)(x)
    x = Flatten()(x)

    # y = model_input_2

    # z = Concatenate()([y, x])
    # z = BatchNormalization()(z)
    z = Dense(32, activation='relu')(x)
    z = Dense(16, activation='relu')(z)
    z1 = Dense(4, activation='relu')(z)
    z2 = Dense(4, activation='relu')(z)
    #
    # z_s = Concatenate()([z, z1, z2])
    # z_d = Concatenate()([z, z1, z2])

    # z_s = Dense(8, activation='relu')(z_s)
    # z_d = Dense(8, activation='relu')(z_d)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(z1)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(z2)
    model_output = [SBP, DBP]
    model = Model(model_input, model_output)
    model.summary()
    return model


def cnn_4_0_5_4():
    model_input = Input(shape=(dfm.input_length, 1, 1))
    # model_input_2 = Input(shape=1)
    x = model_input

    x = Conv2D(8, (3, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)
    x = Conv2D(8, (3, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)
    x = Conv2D(4, (3, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)

    x = Conv2D(2, (2, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    # x = MaxPool2D((2, 1))(x)
    # x = Conv2D(2, (3, 1), (1, 1), padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool2D((2, 1))(x)
    # x = Conv1D(4, 3,  padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool1D(2)(x)
    x = Flatten()(x)

    # y = model_input_2

    # z = Concatenate()([y, x])
    # z = BatchNormalization()(z)
    z = Dense(32, activation='relu')(x)
    z = Dense(16, activation='relu')(z)
    z1 = Dense(4, activation='relu')(z)
    z2 = Dense(4, activation='relu')(z)
    #
    # z_s = Concatenate()([z, z1, z2])
    # z_d = Concatenate()([z, z1, z2])

    # z_s = Dense(8, activation='relu')(z_s)
    # z_d = Dense(8, activation='relu')(z_d)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(z1)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(z2)
    model_output = [SBP, DBP]
    model = Model(model_input, model_output)
    model.summary()
    return model


def cnn_4_0_5_5():
    model_input = Input(shape=(dfm.input_length, 1, 1))
    # model_input_2 = Input(shape=1)
    x = model_input

    x = Conv2D(8, (3, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)

    x = Conv2D(4, (3, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)

    x = Conv2D(2, (3, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    # y = model_input_2

    # z = Concatenate()([y, x])
    # z = BatchNormalization()(z)
    z = Dense(64, activation='relu')(x)
    z = Dense(32, activation='relu')(z)
    z = Dense(16, activation='relu')(z)
    z1 = Dense(4, activation='relu')(z)
    z2 = Dense(4, activation='relu')(z)
    #
    # z_s = Concatenate()([z, z1, z2])
    # z_d = Concatenate()([z, z1, z2])

    # z_s = Dense(8, activation='relu')(z_s)
    # z_d = Dense(8, activation='relu')(z_d)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(z1)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(z2)
    model_output = [SBP, DBP]
    model = Model(model_input, model_output)
    # model.summary()
    return model


def cnn_4_0_5_6():
    model_input = Input(shape=(128*4, 1, 1))
    # model_input_2 = Input(shape=1)
    x = model_input

    x = Conv2D(8, kernel_size=(7, 1), strides=(3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(16, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(8, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Flatten()(x)

    # y = model_input_2

    # z = Concatenate()([y, x])
    # z = BatchNormalization()(z)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(4, activation='relu')(x)

    # z_s = Concatenate()([z, z1, z2])
    # z_d = Concatenate()([z, z1, z2])
    #
    # z_s = Dense(8, activation='relu')(z_s)
    # z_d = Dense(8, activation='relu')(z_d)

    RPTT = Dense(1, activation=dfm.activation, name='RPTT')(x)
    model_output = RPTT
    model = Model(model_input, model_output)
    # model.summary()
    return model

def cnn_4_0_8_0():
    model_input = Input(shape=(128*8, 1, 1))
    model_input_2 = Input(shape=2)
    x = model_input

    x = Conv2D(8, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(16, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)
    #
    # x = Conv2D(8, (3, 2), padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(4, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)


    # x = Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
    # x = Bidirectional(LSTM(32, return_sequences=True, activation='relu'))(x)
    # x = Bidirectional(LSTM(8, return_sequences=False, activation='relu'))(x)
    x = Flatten()(x)

    # y = model_input_2

    # x = Concatenate()([y, x])
    # z = BatchNormalization()(z)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)

    # z_s = Concatenate()([z, z1, z2])
    # z_d = Concatenate()([z, z1, z2])
    #
    x_s = Dense(4, activation='relu')(x)
    x_d = Dense(4, activation='relu')(x)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(x_s)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(x_d)
    model_output = [SBP, DBP]
    model_in = [model_input, model_input_2]
    # model_in = model_input

    model = Model(model_in, model_output)
    # model.summary()
    return model


def cnn_4_0_8_1():
    model_input = Input(shape=(128*8, 1, 1))
    # model_input_2 = Input(shape=1)
    x = model_input

    x = Conv2D(8, kernel_size=(7, 1), strides=(3, 1), padding='same', activation='relu')(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(16, (3, 1), padding='same', activation='relu')(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    # x = BatchNormalization()(x)

    # x = Conv2D(8, (3, 1), padding='same', activation='relu')(x)
    # x = MaxPool2D((3, 1), strides=(2, 1))(x)

    # x = Flatten()(x)
    x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    # x = GRU(64, return_sequences=True)(x)
    # x = GRU(128, return_sequences=True)(x)
    x = LSTM(32, return_sequences=False)(x)
    # y = model_input_2

    # x = Reshape((x.shape[1], 1))(x)
    # x = GlobalSelfAttention(num_heads=1, key_dim=32, dropout=0.1)(x)
    # x = Flatten()(x)

    # z = Concatenate()([y, x])
    # z = BatchNormalization()(z)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)

    # z_s = Concatenate()([z, z1, z2])
    # z_d = Concatenate()([z, z1, z2])
    #
    # z_s = Dense(8, activation='relu')(z_s)
    # z_d = Dense(8, activation='relu')(z_d)

    x_s = Dense(4, activation='relu')(x)
    x_d = Dense(4, activation='relu')(x)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(x_s)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(x_d)
    model_output = [SBP, DBP]

    model = Model(model_input, model_output)
    # model.summary()
    return model

def cnn_4_0_8_2():
    model_input = Input(shape=(dfm.input_length//4, 1))
    # model_input_2 = Input(shape=1)
    x = model_input
    # x = Reshape((dfm.input_length, 2, 1))(x)
    # x = Conv2D(8, kernel_size=(7, 1), strides=(3, 1), padding='same', activation='relu')(x)
    # x = MaxPool2D((3, 1), strides=(2, 1))(x)
    #
    # x = Conv2D(16, (3, 1), padding='same', activation='relu')(x)
    # x = MaxPool2D((3, 1), strides=(2, 1))(x)
    #
    # x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    # x = MaxPool2D((3, 1), strides=(2, 1))(x)
    #
    # x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    # x = MaxPool2D((3, 1), strides=(2, 1))(x)
    #
    # # x = BatchNormalization()(x)
    #
    # x = Conv2D(8, (3, 1), padding='same', activation='relu')(x)
    # # x = BatchNormalization()(x)
    # x = MaxPool2D((3, 1), strides=(2, 1))(x)
    # x = Conv2D(1, (1, 2), padding='valid', activation='relu')(x)
    #
    # # x = Flatten()(x)
    # x = tf.squeeze(x, axis=3)
    x = Conv1D(4, 3, padding='same', activation='relu')(x)
    x = AvgPool1D(2)(x)
    x = Conv1D(8, 3, padding='same', activation='relu')(x)
    x = AvgPool1D(2)(x)
    x = Conv1D(2, 3,  padding='same', activation='relu')(x)
    # x = AvgPool1D(2)(x)
    # x = Conv1D(1, 1,  padding='same', activation='relu')(x)

    # x = Bidirectional(GRU(32,  activation='relu', return_sequences=True))(x)
    # x = Bidirectional(GRU(16,  activation='relu', return_sequences=True))(x)
    # x = Bidirectional(GRU(8,  activation='relu', return_sequences=False))(x)
    # x = Bidirectional(GRU(2,  activation='relu', return_sequences=False))(x)

    x = Flatten()(x)

    # y = model_input_2

    # z = Concatenate()([y, x])
    # z = BatchNormalization()(z)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(4, activation='relu')(x)

    # z_s = Concatenate()([z, z1, z2])
    # z_d = Concatenate()([z, z1, z2])
    #
    # z_s = Dense(8, activation='relu')(z_s)
    # z_d = Dense(8, activation='relu')(z_d)

    RPTT = Dense(1, activation='linear', name='RPTT')(x)
    model_output = RPTT
    model = Model(model_input, model_output)
    # model.summary()
    return model


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


def cnn_4_0_8_3():
    model_input = Input(shape=(128*8, 1))
    # model_input_2 = Input(shape=1)
    x = GlobalSelfAttention(num_heads=1, key_dim=64, dropout=0.1)(model_input)
    x = GRU(64, return_sequences=True)(x)
    x = GRU(128, return_sequences=True)(x)
    x = GRU(32, return_sequences=False)(x)
    # x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    # x = Reshape((x.shape[1], 1))(x)
    # x = GRU(64, return_sequences=True)(x)
    # x = GRU(128, return_sequences=True)(x)
    # x = LSTM(32, return_sequences=False)(x)
    # x = Reshape((x.shape[1], 1))(x)
    # lay = GlobalSelfAttention(num_heads=1, key_dim=32, dropout=0.1)
    x = Flatten()(x)


    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    x_s = Dense(4, activation='relu')(x)
    x_d = Dense(4, activation='relu')(x)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(x_s)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(x_d)
    model_output = [SBP, DBP]

    model = Model(model_input, model_output)
    # model.summary()
    return model


def cnn_4_0_8_4():
    model_input = Input(shape=(dfm.input_length, 1, 1))
    model_input_1 = Input(shape=(dfm.input_length, 1, 1))
    model_input_2 = Input(shape=2)

    # x = tf.concat((model_input, model_input_1), axis=2)
    x = model_input

    x = Conv2D(8, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(16, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)

    # x = Conv2D(8, (3, 2), padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPool2D((3, 1), strides=(2, 1))(x)

    x = Conv2D(1, (3, 1), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPool2D((3, 1), strides=(2, 1))(x)
    x = Flatten()(x)

    # x = Conv2D(1, (1, 2), padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = tf.squeeze(x, axis=3)
    # x_1, x_2 = tf.unstack(x, axis=2)
    # x_3 = tf.keras.layers.Subtract()([x_1, x_2])
    # x_3 = tf.keras.layers.ReLU()(x_3)
    # x_1 = Dense(5, activation='relu')(x_1)
    # x_2 = Dense(5, activation='relu')(x_2)
    # x_3 = Dense(5, activation='relu')(x_3)

    x_1 = model_input_1

    x_1 = Conv2D(8, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(x_1)
    # x_1 = BatchNormalization()(x_1)
    x_1 = MaxPool2D((3, 1), strides=(2, 1))(x_1)

    x_1 = Conv2D(16, (3, 1), padding='same', activation='relu')(x_1)
    # x_1 = BatchNormalization()(x_1)
    x_1 = MaxPool2D((3, 1), strides=(2, 1))(x_1)

    x_1 = Conv2D(32, (3, 1), padding='same', activation='relu')(x_1)
    # x_1 = BatchNormalization()(x_1)
    x_1 = Conv2D(32, (3, 1), padding='same', activation='relu')(x_1)
    # x_1 = BatchNormalization()(x_1)
    x_1 = MaxPool2D((3, 1), strides=(2, 1))(x_1)

    # x_1 = Conv2D(8, (3, 2), padding='valid', activation='relu')(x_1)
    # x_1 = BatchNormalization()(x_1)
    # x_1 = MaxPool2D((3, 1), strides=(2, 1))(x_1)

    x_1 = Conv2D(1, (3, 1), padding='same', activation='relu')(x_1)
    # x_1 = BatchNormalization()(x_1)
    x_1 = MaxPool2D((3, 1), strides=(2, 1))(x_1)
    x_1 = Flatten()(x_1)


    y = model_input_2
    # x = x[:, :, 0, :]

    # y = Dense(5, activation='relu')(y)
    # x_1 = Concatenate()([y, x_1])
    # x_1 = Dense(32, activation='relu')(x_1)
    # x_1 = Dense(16, activation='relu')(x_1)


    # x = Concatenate()([y, x_1, x_3, x_2])
    x = Concatenate()([x_1, x])
    # x = tf.stack([y, x_1, x_3, x_2], axis=1)
    # x = Dense(8, activation='relu')(x)

    # x = Reshape((x.shape[1], 1))(x)
    # x = Bidirectional(GRU(32,  activation='relu', return_sequences=True))(x)
    # x = Bidirectional(GRU(16,  activation='relu', return_sequences=True))(x)
    # x = Bidirectional(GRU(8,  activation='relu', return_sequences=True))(x)
    # x = Bidirectional(GRU(2,  activation='relu', return_sequences=False))(x)
    # x = Dense(2, activation='relu')(x)
    # x = Dense(8, activation='relu')(x)
    # x_1 = Dense(8, activation='relu')(x_1)
    # y = Dense(8, activation='relu')(y)
    #
    # x = tf.keras.layers.Add()([x_1, x, y])
    # x = Concatenate()((y, sub))
    # x = BatchNormalization()(x)

    # x = Concatenate()([y, x[:, :, 0, 0], tf.abs(x[:, :, 1, 0] - x[:, :, 0, 0]),  x[:, :, 1, 0]])
    # x = Concatenate()([y, x])
    # x = Dense(32, activation='relu')(x)
    # x = BatchNormalization()(x)


    # z = BatchNormalization()(z)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(4, activation='relu')(x)

    # x = Reshape((32, 1))(x)
    # x = Reshape((x.shape[1], x.shape[2]))(x)
    #
    # x = Bidirectional(LSTM(32,  activation='tanh', return_sequences=True))(x)
    # x = Bidirectional(LSTM(16,  activation='tanh', return_sequences=True))(x)
    # x = Bidirectional(LSTM(8,  activation='tanh', return_sequences=False))(x)
    # x = Dense(4, activation='relu')(x)
    # x = Concatenate()([y, x])
    # x = Dense(4, activation='relu')(x)



    # x = Bidirectional(LSTM(32,  activation='relu', return_sequences=True))(x)
    # x = Bidirectional(LSTM(16,  activation='relu', return_sequences=True))(x)
    # x = Bidirectional(LSTM(8,  activation='relu', return_sequences=False))(x)

    # x = GRU(16,  activation='tanh', return_sequences=True)(x)
    # x = GRU(8,  activation='tanh', return_sequences=True)(x)
    # x = GRU(4,  activation='tanh', return_sequences=False)(x)

    # x = Dense(2, activation='relu')(x)
    # x = Add()([y, x])
    # x = Dense(4, activation='relu')(x)

    # z_s = Concatenate()([z, z1, z2])
    # z_d = Concatenate()([z, z1, z2])
    #
    x_s = Dense(2, activation='relu')(x)
    x_d = Dense(2, activation='relu')(x)

    SBP = Dense(1, activation=dfm.activation, name='SBP')(x_s)
    DBP = Dense(1, activation=dfm.activation, name='DBP')(x_d)
    model_output = [SBP, DBP]
    model_in = [model_input, model_input_1, model_input_2]
    # model_in = model_input

    model = Model(model_in, model_output)
    model.summary()
    return model


# cnn_test_3_25_1()
# cnn_4_0_5_6()
# cnn_4_0_8_4()
