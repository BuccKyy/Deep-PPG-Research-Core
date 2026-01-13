import os.path

import numpy as np
import tensorflow as tf
from data_tfrecord import AIData
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dropout, Flatten, MaxPool1D, Bidirectional, Concatenate, \
    Dense, concatenate, Conv2DTranspose, UpSampling2D, Conv1DTranspose, BatchNormalization, Input, Conv2D, MaxPool2D, \
    MaxPooling2D, LSTM, Add, ConvLSTM2D, Reshape
from define import DIR_TFRECORD, DIR_SAVE_MODEL, DIR_MODEL_CUSTOM_LAST_LAYER, DIR_MODEL_RAW, INPUT_SHAPE


def cnn():
    model_input = Input(shape=INPUT_SHAPE)
    x = model_input

    x = Conv2D(4, (3, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)
    x = Conv2D(2, (3, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)
    x = Conv2D(2, (5, 3), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1))(x)

    x = Conv2D(2, (2, 1), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(4, activation='softmax')(x)

    model_output = x
    model = Model(model_input, model_output)
    model.summary()
    return model


def train_model(dataset, model, epoch, batch_size=128, dir_save_model=DIR_SAVE_MODEL):
    pic_path = os.path.join(dir_save_model, '{}.png'.format(os.path.basename(dir_save_model)))
    plot_model(model, to_file=pic_path, show_shapes=True)

    train_data = dataset.load_tf_record('train', batch_size)
    valid_data = dataset.load_tf_record('valid', batch_size)

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = ['accuracy']

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    if train_data:
        history = model.fit(train_data,
                            epochs=epoch,
                            validation_data=valid_data)

        # tf.saved_model.save(model, dir_save_model)
        model.save(dir_save_model)


def block_extend_model(input_model):
    # input_model = Input(input_shape)
    x = Dense(8, activation='relu')(input_model)
    x = Dense(4, activation='softmax')(x)

    model = Model(input_model, x, name='block_extend')
    model.summary()
    return model


def merge_model():
    base_model = load_base_model_from_raw_model(DIR_MODEL_RAW)
    block_extend = block_extend_model(base_model.output)
    input = Input(shape=INPUT_SHAPE)

    x = base_model(input)
    x = block_extend(x)
    model = Model(input, x, name='merge_model')
    model.summary()
    return model


def load_base_model_from_raw_model(dir_model=DIR_MODEL_RAW):
    model = tf.keras.models.load_model(dir_model)
    input = model.input
    output = model.layers[-2].output
    base_model = Model(inputs=input, outputs=output, name='base_model')
    base_model.summary()
    base_model.trainable = False
    return base_model


def load_block_extend(dir_model=DIR_MODEL_CUSTOM_LAST_LAYER):
    model = tf.keras.models.load_model(dir_model)
    input = model.layers[-1].input
    output = model.layers[-1].output
    block_extend = Model(inputs=input, outputs=output, name='base_model')
    block_extend.summary()
    return block_extend


def create_example_model(dataset):
    acc_model = cnn()
    train_model(dataset=dataset,
                model=acc_model,
                epoch=10,
                batch_size=128*8,
                dir_save_model=DIR_MODEL_RAW)

    model = merge_model()
    train_model(dataset=dataset,
                model=model,
                epoch=5,
                batch_size=128 * 8,
                dir_save_model=DIR_MODEL_CUSTOM_LAST_LAYER)




