import glob

import tensorflow as tf
from keras.optimizers import Adam
import os
import model.define_model as dfm
from sklearn.model_selection import train_test_split
from tfrecord.generate_tfrecord import AIData
import json
from datetime import datetime
from keras import callbacks


class CustomCallback(callbacks.Callback):
    def on_batch_end(self, epoch, logs=None):
        cnn_block_weights = self.model.get_layer('CNN_Block').get_weights()
        cnn_block_calibrate = self.model.get_layer('CNN_Block_calibrate')
        cnn_block_calibrate.set_weights(cnn_block_weights)


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

        loss_weights = {'SBP': 1/3, 'DBP': 2/3}
        self.model.compile(loss=loss_function, loss_weights=loss_weights, optimizer=optimizer, metrics=metrics)

    def train(self, epochs, batch_size=1, dir_tfrecord=dfm.DIR_SAVE_TFRECORD, lists_exist=False):
        if not lists_exist:
            list_db = glob.glob(dir_tfrecord + '/*.tfrecord')
            list_train, list_valid_test = train_test_split(list_db, test_size=0.3, shuffle=False, random_state=None)
            list_valid, list_test = train_test_split(list_valid_test, test_size=2/3, shuffle=False, random_state=None)
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

        tf_data = AIData(dir_tfrecord, dfm.input_length, dfm.FS_TARGET)

        train_data = tf_data.get_dataset_from_tfrecord(list_train, batch_size=batch_size)
        valid_data = tf_data.get_dataset_from_tfrecord(list_valid, batch_size=batch_size)

        loss_function = dfm.FUNCTION_LOSS
        optimizer = dfm.OPTIMIZER

        self.set_EarlyStopping(min_delta=1e-2, patience=10)
        self.set_checkpoint()
        self.set_tensorboard()
        self.callback.append(CustomCallback())

        from ann_model import ppg2bp

        self.model = ppg2bp()
        self.compile_model(loss_function=loss_function,
                           optimizer=optimizer,
                           metrics=['mae'])
        self.history = self.model.fit(train_data,
                                      epochs=epochs,
                                      callbacks=self.callback,
                                      batch_size=1,
                                      validation_data=valid_data)

    def save_log(self, list_train, list_valid, list_test):
        info = dict()

        info["DATE"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        info["VER_MODEL"] = str(dfm.VER_MODEL)
        info["VER_DATA"] = str(dfm.VER_DATA)
        info["MODEL"] = "CNN"

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
