import glob

import keras as k
from keras import backend
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dropout, Flatten, MaxPool1D,\
    Dense, concatenate, Conv2DTranspose, UpSampling2D, Conv1DTranspose, BatchNormalization, Input, LSTM, Conv2D, MaxPool2D, MaxPooling2D
# from keras.optimizers import adam_v2 as Adam
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


class SqeezeUnet1D:
    """
        This is the config instruction:
            input_length:   The length of the signal. Example: 256,512,1024, 2^n..
            num_classes:    The number of channel of the signal
            deconv_ksize:   The deconvolution kernal size
            model_width:    The width of the model.The larger, the deeper the model can be. Example:64,256, 2^n
            dropout:        The dropoutrate
            Conv_activation:The activation for the convolution layer ( leakyreul or relu)
            activation:     The activation for the final layer (sigmoid  for classification or linear for regression)
    """

    def __init__(self, input_length, num_classes, deconv_ksize, model_width, dropout, Conv_activation, activation):
        self.inputs = None
        self.input_length = input_length
        self.model_width = model_width
        self.num_classes = num_classes
        self.deconv_ksize = deconv_ksize
        self.dropout_rate = dropout
        self.Conv_activation = Conv_activation
        self.activation = activation

    def fire_module(self, x, fire_id, squeeze=16, expand=64):
        f_name = "fire{0}/{1}"

        x = Conv1D(squeeze, 1, activation=self.Conv_activation, padding='same', name=f_name.format(fire_id, "squeeze1"))(x)
        x = BatchNormalization(axis=1)(x)

        left = Conv1D(expand, 1, activation=self.Conv_activation, padding='same', name=f_name.format(fire_id, "expand1"))(x)
        right = Conv1D(expand, 3, activation=self.Conv_activation, padding='same', name=f_name.format(fire_id, "expand3"))(x)
        x = concatenate([left, right], axis=-1, name=f_name.format(fire_id, "concat"))
        return x

    def SqueezeUNet(self):
        self.inputs = Input((self.input_length, self.num_classes))
        # CB1
        x01 = Conv1D(self.model_width, 3, strides=2, padding='same', activation=self.Conv_activation, name='conv1')(self.inputs)

        # CB2
        x02 = MaxPooling1D(pool_size=2, strides=2, name='pool1', padding='same')(x01)  # maxpooling 2 1

        # CB3
        x03 = self.fire_module(x02, fire_id=1, squeeze=self.model_width / 4, expand=self.model_width)
        x04 = self.fire_module(x03, fire_id=2, squeeze=self.model_width / 4, expand=self.model_width)
        x05 = MaxPooling1D(pool_size=2, strides=2, name='pool3', padding="same")(x04)

        # CB4
        x06 = self.fire_module(x05, fire_id=3, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)
        x07 = self.fire_module(x06, fire_id=4, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)
        x08 = MaxPooling1D(pool_size=2, strides=2, name='pool5', padding="same")(x07)

        # CB5
        x09 = self.fire_module(x08, fire_id=5, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)
        x10 = self.fire_module(x09, fire_id=6, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)
        x11 = self.fire_module(x10, fire_id=7, squeeze=(self.model_width * 4) / 4, expand=self.model_width * 4)
        x12 = self.fire_module(x11, fire_id=8, squeeze=(self.model_width * 4) / 4, expand=self.model_width * 4)

        up1 = concatenate([
            Conv1DTranspose(self.model_width * 3, self.deconv_ksize, strides=1, padding='same')(x12),
            x10], axis=2)

        up1 = self.fire_module(up1, fire_id=9, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)

        up2 = concatenate([
            Conv1DTranspose(self.model_width * 2, self.deconv_ksize, strides=1, padding='same')(up1),
            x08], axis=2)
        up2 = self.fire_module(up2, fire_id=10, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)

        # EB1
        up3 = concatenate([
            Conv1DTranspose(self.model_width, self.deconv_ksize, strides=2, padding='same')(up2),
            x05, ], axis=2)

        up3 = self.fire_module(up3, fire_id=11, squeeze=self.model_width / 4, expand=self.model_width)

        # EB2
        up4 = concatenate([
            Conv1DTranspose(self.model_width / 2, self.deconv_ksize, strides=2, padding='same')(up3),
            x02], axis=2)
        up4 = self.fire_module(up4, fire_id=12, squeeze=self.model_width / 4, expand=self.model_width / 2)
        up4 = UpSampling1D(size=2)(up4)

        x = concatenate([up4, x01], axis=2)
        x = Conv1D(self.model_width, 3, strides=1, padding='same', activation='relu')(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(1, 3, padding='same', activation=self.activation)(x)
        # x = Flatten()(x)
        # x = LSTM(32, activation='relu', return_sequences=True)(x)
        # x = LSTM(16, activation='relu', return_sequences=True)(x)
        # x = LSTM(1024, activation='relu', return_sequences=False)(x)

        return Model(inputs=self.inputs, outputs=x)

    def SqueezeUNet_test(self):
        inputs = Input((self.input_length, self.num_classes))
        input2 = Input((6, 1))
        y = Dense(self.input_length, activation=self.Conv_activation, name='SBP')(Flatten()(input2))
        y = tf.reshape(y, (-1, self.input_length, 1))
        # y = Conv1D(1, 1, padding='same', activation=self.Conv_activation)(y)
        inp = concatenate([inputs, y], axis=-1)
        # CB1
        x01 = Conv1D(self.model_width, 3, strides=2, padding='same', activation=self.Conv_activation, name='conv1')(inp)

        # CB2
        x02 = MaxPooling1D(pool_size=2, strides=2, name='pool1', padding='same')(x01)  # maxpooling 2 1

        # CB3
        x03 = self.fire_module(x02, fire_id=1, squeeze=self.model_width / 4, expand=self.model_width)
        x04 = self.fire_module(x03, fire_id=2, squeeze=self.model_width / 4, expand=self.model_width)
        x05 = MaxPooling1D(pool_size=2, strides=2, name='pool3', padding="same")(x04)

        # CB4
        x06 = self.fire_module(x05, fire_id=3, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)
        x07 = self.fire_module(x06, fire_id=4, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)
        x08 = MaxPooling1D(pool_size=2, strides=2, name='pool5', padding="same")(x07)

        # CB5
        x09 = self.fire_module(x08, fire_id=5, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)
        x10 = self.fire_module(x09, fire_id=6, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)
        x11 = self.fire_module(x10, fire_id=7, squeeze=(self.model_width * 4) / 4, expand=self.model_width * 4)
        x12 = self.fire_module(x11, fire_id=8, squeeze=(self.model_width * 4) / 4, expand=self.model_width * 4)

        up1 = concatenate([
            Conv1DTranspose(self.model_width * 3, self.deconv_ksize, strides=1, padding='same')(x12),
            x10], axis=2)

        up1 = self.fire_module(up1, fire_id=9, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)

        up2 = concatenate([
            Conv1DTranspose(self.model_width * 2, self.deconv_ksize, strides=1, padding='same')(up1),
            x08], axis=2)
        up2 = self.fire_module(up2, fire_id=10, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)

        # EB1
        up3 = concatenate([
            Conv1DTranspose(self.model_width, self.deconv_ksize, strides=2, padding='same')(up2),
            x05, ], axis=2)

        up3 = self.fire_module(up3, fire_id=11, squeeze=self.model_width / 4, expand=self.model_width)

        # EB2
        up4 = concatenate([
            Conv1DTranspose(self.model_width / 2, self.deconv_ksize, strides=2, padding='same')(up3),
            x02], axis=2)
        up4 = self.fire_module(up4, fire_id=12, squeeze=self.model_width / 4, expand=self.model_width / 2)
        up4 = UpSampling1D(size=2)(up4)

        x = concatenate([up4, x01], axis=2)
        x = Conv1D(self.model_width, 3, strides=1, padding='same', activation='relu')(x)
        x = UpSampling1D(size=2)(x)
        out_0 = Conv1D(1, 3, padding='same', activation=self.activation)(x)

        # X_SBP = Dense(1, activation=self.activation, name='SBP')(dense3)
        # X_DBP = Dense(1, activation=self.activation, name='DBP')(dense3)
        return Model(inputs=[inputs, input2], outputs=out_0)

    def SqueezeUNet_2(self):
        self.inputs = Input((self.input_length, self.num_classes))
        # fix = Conv1D(1, 108, strides=1, padding='valid', activation=self.Conv_activation)(self.inputs)

        # CB1
        x01 = Conv1D(self.model_width, 3, strides=2, padding='same', activation=self.Conv_activation, name='conv1')(self.inputs)

        # CB2
        x02 = MaxPooling1D(pool_size=2, strides=2, name='pool1', padding='same')(x01)  # maxpooling 2 1

        # CB3
        x03 = self.fire_module(x02, fire_id=1, squeeze=self.model_width / 4, expand=self.model_width)
        x04 = self.fire_module(x03, fire_id=2, squeeze=self.model_width / 4, expand=self.model_width)
        x05 = MaxPooling1D(pool_size=2, strides=2, name='pool3', padding="same")(x04)

        # CB4
        x06 = self.fire_module(x05, fire_id=3, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)
        x07 = self.fire_module(x06, fire_id=4, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)
        x08 = MaxPooling1D(pool_size=2, strides=2, name='pool5', padding="same")(x07)

        # CB5
        x09 = self.fire_module(x08, fire_id=5, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)
        x10 = self.fire_module(x09, fire_id=6, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)
        x11 = self.fire_module(x10, fire_id=7, squeeze=(self.model_width * 4) / 4, expand=self.model_width * 4)
        x12 = self.fire_module(x11, fire_id=8, squeeze=(self.model_width * 4) / 4, expand=self.model_width * 4)

        up1 = concatenate([
            Conv1DTranspose(self.model_width * 3, self.deconv_ksize, strides=1, padding='same')(x12),
            x10], axis=2)

        up1 = self.fire_module(up1, fire_id=9, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)

        up2 = concatenate([
            Conv1DTranspose(self.model_width * 2, self.deconv_ksize, strides=1, padding='same')(up1),
            x08], axis=2)
        up2 = self.fire_module(up2, fire_id=10, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)

        # EB1
        up3 = concatenate([
            Conv1DTranspose(self.model_width, self.deconv_ksize, strides=2, padding='same')(up2),
            x05, ], axis=2)

        up3 = self.fire_module(up3, fire_id=11, squeeze=self.model_width / 4, expand=self.model_width)

        # EB2
        up4 = concatenate([
            Conv1DTranspose(self.model_width / 2, self.deconv_ksize, strides=2, padding='same')(up3),
            x02], axis=2)
        up4 = self.fire_module(up4, fire_id=12, squeeze=self.model_width / 4, expand=self.model_width / 2)
        up4 = UpSampling1D(size=2)(up4)

        x = concatenate([up4, x01], axis=2)
        x = Conv1D(self.model_width, 3, strides=1, padding='same', activation='relu')(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(1, 3, padding='same', activation=self.activation)(x)

        x = Flatten()(x)
        x = Dense(2048, activation='relu')(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(2048, activation='relu')(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(128, activation='relu')(x)

        X_SBP = Dense(1, activation='relu', name='SBP')(x)
        X_DBP = Dense(1, activation='relu', name='DBP')(x)

        return Model(inputs=self.inputs, outputs=[X_SBP, X_DBP])

    def SqueezeUNet_3(self):
        self.inputs = Input((self.input_length, 1))
        # CB1
        x01 = Conv1D(self.model_width, 3, strides=2, padding='same', activation=self.Conv_activation, name='conv1')(self.inputs)

        # CB2
        x02 = MaxPooling1D(pool_size=2, strides=2, name='pool1', padding='same')(x01)  # maxpooling 2 1

        # CB3
        x03 = self.fire_module(x02, fire_id=1, squeeze=self.model_width / 4, expand=self.model_width)
        x04 = self.fire_module(x03, fire_id=2, squeeze=self.model_width / 4, expand=self.model_width)
        x05 = MaxPooling1D(pool_size=2, strides=2, name='pool3', padding="same")(x04)

        # CB4
        x06 = self.fire_module(x05, fire_id=3, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)
        x07 = self.fire_module(x06, fire_id=4, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)
        x08 = MaxPooling1D(pool_size=2, strides=2, name='pool5', padding="same")(x07)

        # CB5
        x09 = self.fire_module(x08, fire_id=5, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)
        x10 = self.fire_module(x09, fire_id=6, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)
        x11 = self.fire_module(x10, fire_id=7, squeeze=(self.model_width * 4) / 4, expand=self.model_width * 4)
        x12 = self.fire_module(x11, fire_id=8, squeeze=(self.model_width * 4) / 4, expand=self.model_width * 4)

        up1 = concatenate([
            Conv1DTranspose(self.model_width * 3, self.deconv_ksize, strides=1, padding='same')(x12),
            x10], axis=2)

        up1 = self.fire_module(up1, fire_id=9, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)

        up2 = concatenate([
            Conv1DTranspose(self.model_width * 2, self.deconv_ksize, strides=1, padding='same')(up1),
            x08], axis=2)
        up2 = self.fire_module(up2, fire_id=10, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)

        # EB1
        up3 = concatenate([
            Conv1DTranspose(self.model_width, self.deconv_ksize, strides=2, padding='same')(up2),
            x05, ], axis=2)

        up3 = self.fire_module(up3, fire_id=11, squeeze=self.model_width / 4, expand=self.model_width)

        # EB2
        up4 = concatenate([
            Conv1DTranspose(self.model_width / 2, self.deconv_ksize, strides=2, padding='same')(up3),
            x02], axis=2)
        up4 = self.fire_module(up4, fire_id=12, squeeze=self.model_width / 4, expand=self.model_width / 2)
        up4 = UpSampling1D(size=2)(up4)

        x = concatenate([up4, x01], axis=2)
        x = Conv1D(self.model_width, 3, strides=1, padding='same', activation='relu')(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(1, 3, padding='same', activation=self.activation)(x)
        x = LSTM(64, activation='relu', return_sequences=True)(x)
        x = LSTM(128, activation='relu', return_sequences=False)(x)
        X_SBP = Dense(1, activation=self.activation, name='SBP')(x)
        X_DBP = Dense(1, activation=self.activation, name='DBP')(x)

        return Model(inputs=self.inputs, outputs=[X_SBP, X_DBP])

    def SqueezeUNet_4(self):
        self.inputs = Input((self.input_length, self.num_classes))
        # CB1
        x01 = Conv1D(self.model_width, 3, strides=2, padding='same', activation=self.Conv_activation, name='conv1')(self.inputs)

        # CB2
        x02 = MaxPooling1D(pool_size=2, strides=2, name='pool1', padding='same')(x01)  # maxpooling 2 1

        # CB3
        x03 = self.fire_module(x02, fire_id=1, squeeze=self.model_width / 4, expand=self.model_width)
        x04 = self.fire_module(x03, fire_id=2, squeeze=self.model_width / 4, expand=self.model_width)
        x05 = MaxPooling1D(pool_size=2, strides=2, name='pool3', padding="same")(x04)

        # CB4
        x06 = self.fire_module(x05, fire_id=3, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)
        x07 = self.fire_module(x06, fire_id=4, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)
        x08 = MaxPooling1D(pool_size=2, strides=2, name='pool5', padding="same")(x07)

        # CB5
        x09 = self.fire_module(x08, fire_id=5, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)
        x10 = self.fire_module(x09, fire_id=6, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)
        x11 = self.fire_module(x10, fire_id=7, squeeze=(self.model_width * 4) / 4, expand=self.model_width * 4)
        x12 = self.fire_module(x11, fire_id=8, squeeze=(self.model_width * 4) / 4, expand=self.model_width * 4)

        up1 = concatenate([
            Conv1DTranspose(self.model_width * 3, self.deconv_ksize, strides=1, padding='same')(x12),
            x10], axis=2)

        up1 = self.fire_module(up1, fire_id=9, squeeze=(self.model_width * 3) / 4, expand=self.model_width * 3)

        up2 = concatenate([
            Conv1DTranspose(self.model_width * 2, self.deconv_ksize, strides=1, padding='same')(up1),
            x08], axis=2)
        up2 = self.fire_module(up2, fire_id=10, squeeze=(self.model_width * 2) / 4, expand=self.model_width * 2)

        # EB1
        up3 = concatenate([
            Conv1DTranspose(self.model_width, self.deconv_ksize, strides=2, padding='same')(up2),
            x05, ], axis=2)

        up3 = self.fire_module(up3, fire_id=11, squeeze=self.model_width / 4, expand=self.model_width)

        # EB2
        up4 = concatenate([
            Conv1DTranspose(self.model_width / 2, self.deconv_ksize, strides=2, padding='same')(up3),
            x02], axis=2)
        up4 = self.fire_module(up4, fire_id=12, squeeze=self.model_width / 4, expand=self.model_width / 2)
        up4 = UpSampling1D(size=2)(up4)

        x = concatenate([up4, x01], axis=2)
        x = Conv1D(self.model_width, 3, strides=1, padding='same', activation='relu')(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(1, 3, padding='same', activation=self.activation)(x)

        return Model(inputs=self.inputs, outputs=x)


class CNN:
    def __init__(self, input_length, num_classes, Conv_activation, activation, dropout):
        self.input_length = input_length
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.Conv_activation = Conv_activation
        self.activation = activation

    def cnn_1d(self):
        model_input = Input(shape=(self.input_length, 1))

        conv1 = Conv1D(16, 3, activation=self.Conv_activation)(model_input)
        drop1 = Dropout(rate=0.25)(conv1)
        maxp1 = MaxPool1D()(drop1)

        conv2 = Conv1D(32, 3, activation=self.Conv_activation)(maxp1)
        drop2 = Dropout(rate=0.25)(conv2)
        maxp2 = MaxPool1D()(drop2)

        conv3 = Conv1D(32, 3, activation=self.Conv_activation)(maxp2)
        drop3 = Dropout(rate=0.25)(conv3)
        maxp3 = MaxPool1D()(drop3)

        flatten = Flatten()(maxp3)
        dense1 = Dense(units=64, activation=self.Conv_activation)(flatten)
        dense2 = Dense(units=32, activation=self.Conv_activation)(dense1)
        dense3 = Dense(units=16, activation=self.Conv_activation)(dense2)
        model_output = Dense(2, activation=self.activation)(dense3)
        # X_SBP = Dense(1, activation=self.activation, name='SBP')(dense3)
        # X_DBP = Dense(1, activation=self.activation, name='DBP')(dense3)

        model = Model(model_input, model_output)

        return model

    def ANN(self):
        model_input = Input(shape=(self.input_length, 3))
        x = Flatten()(model_input)
        x = Dense(units=256, activation=self.activation)(x)
        x = Dense(units=64, activation=self.activation)(x)
        x = Dense(units=16, activation=self.activation)(x)

        model_output = Dense(2, activation=self.activation)(x)
        # X_SBP = Dense(1, activation=self.activation, name='SBP')(dense3)
        # X_DBP = Dense(1, activation=self.activation, name='DBP')(dense3)

        model = Model(model_input, model_output)

        return model

    def ANN_new(self):
        # model_input = Input(shape=(self.input_length, 1))
        model_input = Input(shape=(self.input_length, 1))
        # model_input_2 = Input(shape=(1, 1))

        x = Flatten()(model_input)
        x = Dense(units=32, activation=self.Conv_activation)(x)
        x = Dense(units=64, activation=self.Conv_activation)(x)
        x = Dense(units=64, activation=self.Conv_activation)(x)
        x = Dense(units=16, activation=self.Conv_activation)(x)
        x = Dense(units=8, activation=self.Conv_activation)(x)

        # y = Flatten()(model_input_2)
        # y = Dense(units=4, activation=self.Conv_activation)(y)
        # y = Dense(units=8, activation=self.Conv_activation)(y)

        # x = concatenate([x, y])
        # x = Dense(units=4, activation=self.Conv_activation)(x)

        # model_output = Dense(2, activation=self.activation)(x)
        # X_HR = Dense(1, activation=self.activation, name='HR')(x)
        X_SBP = Dense(1, activation=self.activation, name='SBP')(x)
        # X_DBP = Dense(1, activation=self.activation, name='DBP')(x)
        # model_output = [X_SBP, X_DBP, X_HR]
        model_output = X_SBP
        # model = Model([model_input, model_input_2], model_output)
        model = Model(model_input, model_output)

        return model

    def ANN_new_0(self):
        # model_input = Input(shape=(self.input_length, 1))
        model_input = Input(shape=(self.input_length, 3))
        # conv1 = Conv1D(4, 7, activation=self.Conv_activation)(model_input)
        x = Flatten()(model_input)
        x = Dense(units=512, activation=self.Conv_activation)(x)
        x = Dense(units=128, activation=self.Conv_activation)(x)
        x = Dense(units=256, activation=self.Conv_activation)(x)
        x = Dense(units=64, activation=self.Conv_activation)(x)
        x = Dense(units=16, activation=self.Conv_activation)(x)

        model_output = Dense(2, activation=self.activation)(x)
        # X_SBP = Dense(1, activation=self.activation, name='SBP')(dense3)
        # X_DBP = Dense(1, activation=self.activation, name='DBP')(dense3)

        model = Model(model_input, model_output)

        return model


class AIProcess:
    def __init__(self, work_dir):
        # self.model: tf.keras.Model = model
        self.model = None
        self.work_dir = work_dir
        self.callback = []
        self.history = None
        self.work_dir = work_dir
        self.list_test = []

    def set_checkpoint(self, save_weight_only=True, save_best_only=True, monitor='val_loss'):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(self.work_dir, 'checkpoint', '{epoch:02d}'),
                                                        save_weights_only=save_weight_only,
                                                        save_best_only=save_best_only,
                                                        monitor=monitor)
        self.callback.append(checkpoint)

    def set_tensorboard(self, log_dir=dfm.DIR_LOG_TensorBoard, histogram_freq=1):
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq)
        self.callback.append(tensor_board)

    def set_EarlyStopping(self, monitor='loss', min_delta=1e-5, patience=5, restore_best_weights=True):
        EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                         min_delta=min_delta,
                                                         patience=patience,
                                                         restore_best_weights=restore_best_weights)
        self.callback.append(EarlyStopping)

    def compile_model(self,
                      loss_function=tf.keras.losses.MeanSquaredError(),
                      optimizer=Adam(learning_rate=10e-4),
                      metrics=None,
                      sumary=True):
        if metrics is None:
            metrics = ['mse', 'mae', 'mape']
            # metrics = ['mean_squared_error', 'mean_absolute_error']
        # tf.keras.losses.huber
        if sumary:
            self.model.summary()
        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    def train(self, epochs, batch_size=1, dir_tfrecord=dfm.DIR_SAVE_TFRECORD, lists_exist=False):
        # sz = [i for i in glob.glob(dir_tfrecord + '/*.tfrecord') if os.path.getsize(i) < 100000]
        # for f in sz:
        #     os.remove(f)

        if not lists_exist:
            list_db = glob.glob(dir_tfrecord + '/*.tfrecord')
            # list_db = glob.glob(dir_tfrecord + '/AI_data_part_1_*.tfrecord')
            # list_train, list_valid_test = train_test_split(list_db, test_size=0.3, shuffle=True, random_state=None)
            # list_valid, list_test = train_test_split(list_valid_test, test_size=0.5, shuffle=True, random_state=None)
            list_train, list_valid_test = train_test_split(list_db, test_size=0.4, shuffle=False, random_state=None)
            list_valid, list_test = train_test_split(list_valid_test, test_size=0.5, shuffle=False, random_state=None)
            self.list_test = list_test

            self.save_log(list_train, list_valid, list_test)

        else:
            path_json = glob.glob(os.path.join(self.work_dir, 'INFO_train_valid_test') + '/*.json')
            with open(path_json[0], 'r') as openfile:
                rd = json.load(openfile)
            list_train = rd.get('train')
            list_valid = rd.get('valid')
            list_test = rd.get('test')
            self.list_test = list_test

        tf_data = AIData(dir_tfrecord, dfm.input_length)
        # tf_data = AIData(dir_tfrecord, 875)
        train_data = tf_data.get_dataset_from_tfrecord(list_train, batch_size=batch_size)
        valid_data = tf_data.get_dataset_from_tfrecord(list_valid, batch_size=batch_size)

        # loss_function = tf.keras.losses.MeanAbsoluteError()  # tf.keras.losses.MeanAbsolutePercentageError()
        loss_function = dfm.FUNCTION_LOSS
        # optimizer = tf.keras.optimizers.Adadelta(learning_rate=1e-3,
        #                                          rho=0.95,
        #                                          epsilon=1e-07,
        #                                          name='Adadelta')
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        optimizer = tf.keras.optimizers.Adam()
        # optimizer = tf.keras.optimizers.RMSprop()

        self.set_EarlyStopping(min_delta=1e-3, patience=10)
        self.set_checkpoint()
        self.set_tensorboard()

        # # self.limit_GPU(True)
        # arg = (dfm.input_length, dfm.num_classes, dfm.deconv_ksize, dfm.model_width, dfm.dropout, dfm.Conv_activation, dfm.activation)
        # self.model = SqeezeUnet1D(*arg).SqueezeUNet()
        # self.model = SqeezeUnet1D(*arg).SqueezeUNet_test()
        # self.model = SqeezeUnet1D(*arg).SqueezeUNet_3()

        # _arg = (dfm.input_length, 2, dfm.Conv_activation, dfm.activation, dfm.dropout)
        # self.model = CNN(*_arg).ANN_new()


        # from ann_model import cnn_test_3_26, cnn_test_3_25_1, cnn_4_0_1, cnn_4_0_2, cnn_4_0_3, cnn_4_0_4, cnn_4_0_5, cnn_4_0_5_2, cnn_4_0_5_3, cnn_4_0_5_4, cnn_4_0_5_5, cnn_4_0_8_2, cnn_4_0_8_0, cnn_4_0_8_1, cnn_4_0_8_3, cnn_4_0_8_4
        from lstm_model import lstm, lstm_8s, lstm_old, gru_1
        # from unet import UNet
        # self.model = UNet(dfm.input_length, n_channel=1)
        # self.model = cnn_4_0_8_2()
        # self.model = gru_1()

        from ResNet_1DCNN import build_resnet
        self.model = build_resnet()
        # from autoencoder import build_vae
        # self.model = build_vae((dfm.input_length, 1), 1)

        self.compile_model(loss_function=loss_function,
                           optimizer=optimizer,
                           metrics=['mae'])
        self.history = self.model.fit(train_data,
                                      epochs=epochs,
                                      callbacks=self.callback,
                                      batch_size=batch_size,
                                      validation_data=valid_data)

        self.model.save(os.path.join(self.work_dir, 'save_model'))

    def save_log(self, list_train, list_valid, list_test):
        info = dict()

        info["DATE"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        info["VER_MODEL"] = str(dfm.VER_MODEL)
        info["VER_DATA"] = str(dfm.VER_DATA)
        # info["MODEL"] = "Sqeeze U-Net 1D"
        info["MODEL"] = "NABNet"

        info["input_length"] = dfm.input_length
        info["num_classes"] = dfm.num_classes
        info["deconv_ksize"] = dfm.deconv_ksize
        info["model_width"] = dfm.model_width
        info["dropout"] = dfm.dropout
        info["Conv_activation"] = str(dfm.CONV_ACTIVATION)
        info["activation"] = dfm.activation
        info["BATCH_SIZE"] = dfm.BATCH_SIZE
        info["EPOCH"] = dfm.EPOCH

        info["FUNC_LOSS"] = str(dfm.FUNCTION_LOSS)
        info["OPTIMIZER"] = str(dfm.OPTIMIZER)

        info["train"] = list_train
        info["valid"] = list_valid
        info["test"] = list_test

        save_file = json.dumps(info, indent=2)
        os.makedirs(os.path.join(self.work_dir, 'INFO_train_valid_test'), exist_ok=True)
        with open(os.path.join(self.work_dir, 'INFO_train_valid_test') + "/data_info.json", "w") as outfile:
            outfile.write(save_file)

    @staticmethod
    def limit_GPU(flag=True):
        if flag:
            # TensorFlow wizardry
            config = tf.compat.v1.ConfigProto()
            # Don't pre-allocate memory; allocate as-needed
            config.gpu_options.allow_growth = True
            # Only allow a total of half the GPU memory to be allocated
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            # Create a session with the above options specified.
            backend.set_session(tf.compat.v1.Session(config=config))

    def train_dat(self, epochs, batch_size=1, dir_wfdb='/mnt/ai_data/PPG2ABP_data_second/Cuff-Less_Blood_Pressure_Estimation'):
        list_db = list(map(lambda path: path.rstrip('.hea'), glob.glob(dir_wfdb + '/*.hea')))

        list_train, list_test = train_test_split(list_db, test_size=0.2, shuffle=True, random_state=11)
        list_train, list_valid = train_test_split(list_train, test_size=0.25, shuffle=True, random_state=11)
        self.list_test = list_test

        tf_data = AIData(dir_wfdb, 256)

        ppg, abp = tf_data.process_data(list_train[10])
        seg = np.arange(0, 256, 1)[None, :] + np.arange(0, len(ppg) - 256, 256 - 0)[:, None]
        seg = seg.astype(int)
        ppg = ppg[seg]
        abp = abp[seg]
        train_data = (ppg, abp)

        # ppg, abp = tf_data.process_data(list_train[0])
        # seg = np.arange(0, 256, 1)[None, :] + np.arange(0, len(ppg) - 256, 256 - 0)[:, None]
        # seg = seg.astype(int)
        # ppg = ppg[seg]
        # abp = abp[seg]
        valid_data = (ppg, abp)

        # train_data = tf_data.get_dataset_from_tfrecord(list_train, batch_size=batch_size)
        # valid_data = tf_data.get_dataset_from_tfrecord(list_valid, batch_size=batch_size)

        self.compile_model()
        self.set_EarlyStopping()
        self.set_checkpoint()
        self.set_tensorboard()

        self.history = self.model.fit(*train_data,
                                      epochs=epochs,
                                      callbacks=self.callback,
                                      validation_data=valid_data)
        # self.plot_training_result()

    def evaluate(self, batch_size=1, dir_tfrecord=dfm.DIR_SAVE_TFRECORD):
        tf_data = AIData(dir_tfrecord, 256)
        test_data = tf_data.get_dataset_from_tfrecord(self.list_test, batch_size=batch_size)
