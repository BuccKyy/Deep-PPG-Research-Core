import os.path

import numpy as np
import tensorflow as tf
from data_tfrecord import AIData

from tensorflow.lite.python.interpreter import Interpreter

from define import DIR_TFRECORD, DIR_SAVE_MODEL, DIR_MODEL_CUSTOM_LAST_LAYER, DIR_MODEL_RAW, DIR_TFLITE, timeit


class ProcessTFlite:
    def __init__(self, base_model_path, block_extend_path, entire_model_path):
        self.base_model_path = base_model_path
        self.block_extend_path = block_extend_path
        self.entire_model_path = entire_model_path
        self.base_model = None
        self.block_extend = None
        self.entire_model = None
        self.output_entire_model = None
        self.input_entire_model = None
        self.output_block_extend = None
        self.input_block_extend = None
        self.output_base_model = None
        self.input_base_model = None

    def init_model(self):
        self.base_model = Interpreter(model_path=self.base_model_path)
        self.input_base_model = self.base_model.get_input_details()
        self.output_base_model = self.base_model.get_output_details()
        self.base_model.allocate_tensors()

        self.block_extend = Interpreter(model_path=self.block_extend_path)
        self.input_block_extend = self.block_extend.get_input_details()
        self.output_block_extend = self.block_extend.get_output_details()
        self.block_extend.allocate_tensors()

        self.entire_model = Interpreter(model_path=self.entire_model_path)
        self.input_entire_model = self.entire_model.get_input_details()
        self.output_entire_model = self.entire_model.get_output_details()
        self.entire_model.allocate_tensors()

    @timeit
    def get_result_merge_model(self, data):
        """

        """
        in_data = data[0]

        self.base_model.set_tensor(self.input_base_model[0]['index'], in_data)
        self.base_model.invoke()
        result_base_model = self.base_model.get_tensor(self.output_base_model[0]['index'])

        self.block_extend.set_tensor(self.input_block_extend[0]['index'], result_base_model)
        self.block_extend.invoke()
        result_block_extend = self.block_extend.get_tensor(self.output_block_extend[0]['index'])

        return result_block_extend

    @timeit
    def get_result_entire_model(self, data):
        in_data = data[0]
        self.entire_model.set_tensor(self.input_entire_model[0]['index'], in_data)
        self.entire_model.invoke()
        result = self.entire_model.get_tensor(self.output_entire_model[0]['index'])

        return result

    @staticmethod
    def savedmodel_to_tflite(src_model_dir, saved_model_dir=DIR_TFLITE):
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_saved_model(src_model_dir)  # path to the SavedModel directory

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        tflite_path = os.path.join(saved_model_dir, '{}.tflite'.format(os.path.basename(src_model_dir)))
        # Save the model.
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

    @staticmethod
    def keras_to_tflite(model, model_name='', saved_model_dir=DIR_TFLITE):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        tflite_path = os.path.join(saved_model_dir, '{}.tflite'.format(model_name))
        # Save the model.
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
