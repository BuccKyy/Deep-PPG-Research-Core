import glob

import keras as k
from keras import backend
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dropout, Flatten, MaxPool1D, Bidirectional, Concatenate, \
    Dense, concatenate, Conv2DTranspose, UpSampling2D, Conv1DTranspose, BatchNormalization, Input, Conv2D, MaxPool2D, MaxPooling2D, LSTM, Add, ConvLSTM2D, Reshape, GRU
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


def lstm_0():
    model_input = Input(shape=(256, 1))
    X = Conv1D(filters=64, kernel_size=5, strides=1, padding='causal', activation='relu')(model_input)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    X = Bidirectional(LSTM(64, return_sequences=False))(X)
    X = Flatten()(X)
    X = Dense(512, activation='relu')(X)
    X = Dense(256, activation='relu')(X)
    X = Dense(128, activation='relu')(X)

    X_HR = Dense(1, activation=dfm.activation, name='HR')(X)
    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(X)
    X_DBP = Dense(1, activation=dfm.activation, name='DBP')(X)
    # model_output = [X_SBP, X_DBP]
    model_output = [X_SBP, X_DBP, X_HR]

    model = Model(model_input, model_output)
    model.summary()

    return model


def lstm_old():
    model_input = Input(shape=(256, 1))
    # model_input_2 = Input(shape=(1, 1))
    x = Conv1D(20, 9, activation=dfm.Conv_activation)(model_input)
    x = BatchNormalization()(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    # x = Conv1D(20, 9, activation=dfm.Conv_activation)(x)
    # x = BatchNormalization()(x)
    # x = MaxPool1D(4)(x)
    # x = Dropout(rate=0.1)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    # x = LSTM(64, return_sequences=False)(x)
    x = Dropout(rate=0.1)(x)

    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    # x = LSTM(128, return_sequences=False)(x)
    x = Dropout(rate=0.1)(x)

    # x = Dense(16, activation=dfm.activation)(x)
    # y = Dense(4, activation=dfm.activation)(Flatten()(model_input_2))
    # x = concatenate([x, y])
    # x = Dense(units=4, activation=dfm.Conv_activation)(x)

    # x = Flatten()(x)
    # X_HR = Dense(1, activation=dfm.activation, name='HR')(x)
    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    X_DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    # model_output = [X_SBP, X_DBP, X_HR]
    model_output = [X_SBP, X_DBP]
    # model = Model([model_input, model_input_2], model_output)
    model = Model(model_input, model_output)

    # model.summary()

    return model


def lstm_22():
    model_input = Input(shape=(256, 1))
    # x = model_input
    x = tf.reshape(model_input, (-1, 4, 64))
    model_input_2 = Input(shape=(6, 1))
    x = Conv1D(32, 3, padding='same', activation=dfm.Conv_activation)(x)
    x = MaxPool1D(2)(x)
    # x = Dropout(rate=0.1)(x)

    # x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(rate=0.1)(x)

    # x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(6, activation=dfm.Conv_activation)(x)
    y = Flatten()(model_input_2)
    x = concatenate([x, y])
    x = Dense(4, activation=dfm.Conv_activation)(x)

    # x = Dense(units=4, activation=dfm.Conv_activation)(x)

    # x = Flatten()(x)
    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    X_DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    # X_HR = Dense(1, activation=dfm.activation, name='HR')(x)

    model_output = [X_SBP, X_DBP]
    # model_output = X_SBP
    model = Model([model_input, model_input_2], model_output)
    # model = Model(model_input, model_output)

    # model.summary()

    return model


def lstm_25():
    model_input = Input(shape=(256, 1))
    model_input_2 = Input(shape=(6, 1))
    model_input_3 = Input(shape=(256*3, 1))

    # z = tf.reshape(model_input_3, (-1, 3, 64))
    z = Flatten()(model_input_3)
    z = Dense(64, activation=dfm.Conv_activation)(z)
    z = Dense(32, activation=dfm.Conv_activation)(z)

    x = model_input
    # x = tf.reshape(model_input, (-1, 4, 64))
    x = Conv1D(64, 3, padding='same', activation=dfm.Conv_activation)(x)
    # x = MaxPool1D(2)(x)
    x = Dropout(rate=0.1)(x)

    # x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(rate=0.1)(x)

    # x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(rate=0.1)(x)

    x = concatenate([z, x])

    x = Dense(32, activation=dfm.Conv_activation)(x)
    x = Dense(6, activation=dfm.Conv_activation)(x)
    y = Flatten()(model_input_2)
    x = concatenate([x, y])
    x = Dense(4, activation=dfm.Conv_activation)(x)

    # x = Dense(units=4, activation=dfm.Conv_activation)(x)

    # x = Flatten()(x)
    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    X_DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    # X_HR = Dense(1, activation=dfm.activation, name='HR')(x)

    model_output = [X_SBP, X_DBP]
    # model_output = X_SBP
    model = Model([model_input, model_input_2, model_input_3], model_output)
    # model = Model(model_input, model_output)

    # model.summary()

    return model


# 27 44
def lstm_27():
    model_input = Input(shape=(256, 1))
    model_input_2 = Input(shape=(6, 1))
    # model_input_3 = Input(shape=(1, 1))
    model_input_3 = Input(shape=(256*3, 1))

    # z = tf.reshape(model_input_3, (-1, 3, 64))
    # z = Conv1D(20, 25, 3, activation=dfm.Conv_activation)(model_input_3)
    # z = Conv1D(20, 9, 9, activation='relu')(model_input_3)
    # z = Flatten()(model_input_3)
    # z = Dense(2, activation=dfm.Conv_activation)(z)
    # z = Dense(32, activation=dfm.Conv_activation)(z)
    # z = Dense(16, activation=dfm.Conv_activation)(z)

    # x = concatenate((model_input, model_input_3), axis=1)

    x = model_input
    # x = tf.reshape(model_input, (-1, 64, 4))
    # x = Conv1D(64, 3, padding='same', activation=dfm.Conv_activation)(x)
    # x = MaxPool1D(2)(x)
    # x = Dropout(rate=0.1)(x)
    # x = concatenate([model_input_3, x], axis=1)

    x = Conv1D(20, 15, activation='relu')(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Conv1D(20, 9, activation='relu')(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    # x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    x = Dropout(rate=0.1)(x)

    # x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = LSTM(128, activation='relu', return_sequences=False)(x)
    x = Dropout(rate=0.1)(x)

    # y = Flatten()(model_input_2)
    # y = Dense(2, activation=dfm.Conv_activation)(y)
    #
    # x = Flatten()(x)
    # x = concatenate([x, z])
    # x = Dense(16, activation=dfm.Conv_activation)(x)

    # x = concatenate([x, y])
    # x = Dense(128, activation=dfm.Conv_activation)(x)

    # x = concatenate([x, z])
    # x = Dense(4, activation=dfm.Conv_activation)(x)

    # x = Dense(units=16, activation=dfm.Conv_activation)(x)

    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    X_DBP = Dense(1, activation=dfm.activation, name='DBP')(x)

    model_output = [X_SBP, X_DBP]
    # model_output = X_SBP
    model = Model([model_input, model_input_2, model_input_3], model_output)
    # model = Model(model_input, model_output)

    # model.summary()

    return model


def lstm():
    model_input = Input(shape=(dfm.input_length, 1))

    x = model_input
    x = Conv1D(32, 3, padding='same', activation='relu')(x)
    x = Conv1D(16, 3,  activation='relu')(x)
    # x = Conv1D(filters=32, kernel_size=5, strides=1, padding='causal', activation='relu')(x)
    # x = Conv1D(filters=32, kernel_size=5, strides=1, padding='valid', activation='relu')(x)

    # x = LSTM(128, return_sequences=True)(x)
    # x = LSTM(128, return_sequences=True)(x)
    # x = LSTM(64, return_sequences=False)(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(512, activation=dfm.Conv_activation)(x)
    x = Dense(256, activation=dfm.Conv_activation)(x)
    x = Dense(128, activation=dfm.Conv_activation)(x)
    # y = Dense(4, activation=dfm.activation)(Flatten()(model_input_2))
    # x = concatenate([x, y])
    # x = Dense(units=4, activation=dfm.Conv_activation)(x)

    # x = Flatten()(x)
    # X_HR = Dense(1, activation=dfm.activation, name='HR')(x)
    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    X_DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    # model_output = [X_SBP, X_DBP, X_HR]
    model_output = [X_SBP, X_DBP]
    # model = Model([model_input, model_input_2], model_output)
    model = Model(model_input, model_output)

    return model


def lstm_8s():
    model_input = Input(shape=(dfm.input_length, 1))

    x = model_input
    # x = Conv1D(32, 3, padding='same', activation='relu')(x)
    # x = Conv1D(16, 3,  activation='relu')(x)
    c1 = Conv1D(filters=10, kernel_size=9, strides=1, padding='same', activation='elu')(x)
    c1 = BatchNormalization()(c1)
    c2 = Conv1D(filters=10, kernel_size=25, strides=1, padding='same', activation='elu')(x)
    c2 = BatchNormalization()(c2)

    x = Concatenate()([c1, c2])

    x = MaxPool1D(pool_size=9, strides=2)(x)

    x = Conv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=9, strides=2)(x)

    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(128, return_sequences=False)(x)

    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    X_DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    model_output = [X_SBP, X_DBP]
    model = Model(model_input, model_output)
    # model.summary()

    return model

'/mnt/Data_1/C_Project/impact-abpm-firmware-abpm_1/abpm'

lstm_8s()


def lstm_1():
    model_input = Input(shape=(256, 1))

    x = model_input
    x = Conv1D(32, 3, padding='same', activation='relu')(x)
    x = Conv1D(16, 3,  activation='relu')(x)

    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(512, activation=dfm.Conv_activation)(x)
    x = Dense(256, activation=dfm.Conv_activation)(x)
    x = Dense(128, activation=dfm.Conv_activation)(x)

    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    X_DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    model_output = [X_SBP, X_DBP]
    model = Model(model_input, model_output)
    model.summary()
    return model


def lstm_2D():
    model_input = Input(shape=(256, 1, 1))

    x = model_input
    x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    x = Conv2D(16, (3, 1),  activation='relu')(x)
    # output shape x = (254, 1, 16)
    x = Reshape((254, 1*16))(x)

    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128, return_sequences=False)(x)

    x = Dense(512, activation=dfm.Conv_activation)(x)

    model_output = x
    model = Model(model_input, model_output)
    model.summary()
    return model

def gru_1():
    model_input = Input(shape=(1024, 1))

    x = model_input
    # x = Conv1D(32, 3, padding='same', activation='relu')(x)
    # x = Conv1D(16, 3,  activation='relu')(x)
    x = GRU(64, return_sequences=True)(x)
    x = GRU(128, return_sequences=True)(x)
    x = GRU(32, return_sequences=False)(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(8, activation=dfm.Conv_activation)(x)

    X_SBP = Dense(1, activation=dfm.activation, name='SBP')(x)
    X_DBP = Dense(1, activation=dfm.activation, name='DBP')(x)
    model_output = [X_SBP, X_DBP]
    model = Model(model_input, model_output)
    # model.summary()
    return model

