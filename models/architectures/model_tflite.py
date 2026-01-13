import glob
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import wfdb

import cross_correlation.preprocessing
from Sqeeze_Unet import *
from scipy import stats


def convert_tflite(ai_model, checkpoint_path):
    ai_model.load_weights(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    ai_model_save_dir = os.path.join(dfm.WORKSPACE, 'save_model')
    # tf.saved_model.save(ai_model, ai_model_save_dir)

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(ai_model_save_dir)  # path to the SavedModel directory
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    tflite_model_path = os.path.join(dfm.WORKSPACE, 'ppg2bp_opt.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    # result_predict = ai_model.predict()


def convert_tflite_from_saved_model():
    ai_model_save_dir = os.path.join(dfm.WORKSPACE, 'save_model')

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(ai_model_save_dir)  # path to the SavedModel directory
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        # dataset = AIData(strategize_data(), join(WORKSPACE, 'data'), process_data, (-1,))
        dir_tfrecord = '/mnt/ai_data/PPG2ABP_data_fourth/data_4.0.7.1'
        # dataset = AIData(strategize_data(), _DATA_PATH, process_data, (-1,))
        tf_data = AIData(dir_tfrecord, dfm.input_length)
        list_train = glob.glob(dir_tfrecord+'/*.tfrecord')
        train_data = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_fourth/data_4.0.7.1/AI_data_part_1_3.tfrecord', batch_size=1)

        for sample in train_data.take(1000).as_numpy_iterator():
            # Get sample input data as a numpy array in a method of your choosing.
            yield [sample[0].astype('float32')]

        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    # Save the model
    tflite_model_path = os.path.join(dfm.WORKSPACE, 'ppg2bp_opt.tflite')
    # tflite_model_path = os.path.join(dfm.WORKSPACE, 'ppg2bp.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)


def convert_tflite_from_saved_model_itr():
    ai_model_save_dir = os.path.join(dfm.WORKSPACE, 'save_model_fine_tune')

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(ai_model_save_dir)  # path to the SavedModel directory
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        train_data = np.load('/mnt/Data_1/Project/itr-ai-sensor_annotation-ABPM/Steam_PPG/data_in.npy')
        for sample in train_data:
            # Get sample input data as a numpy array in a method of your choosing.
            yield [sample[None, :, :, :].astype('float32')]

        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    # Save the model
    tflite_model_path = os.path.join(dfm.WORKSPACE, 'ppg2bp_opt.tflite')
    # tflite_model_path = os.path.join(dfm.WORKSPACE, 'ppg2bp.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)



def test_tflite(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    # test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    a = 10

    # interpreter.set_tensor(input_index, test_image)
    # interpreter.invoke()
    # predictions = interpreter.get_tensor(output_index)


if __name__ == "__main__":

    # test_tflite(os.path.join(dfm.WORKSPACE, 'ppg2bp_opt.tflite'))
    # convert_tflite_from_saved_model()
    # convert_tflite_from_saved_model_itr()
    # test_tflite('/mnt/ai_data/PPG2ABP_data_fourth/model_4.0.7.1/ppg2bp_opt.tflite')
    # test_tflite('/mnt/ai_data/PPG2ABP_data_fourth/model_4.0.8.1/ppg2bp_opt.tflite')
    test_tflite('/mnt/ai_data/PPG2ABP_data_fourth/model_4.0.8.3/ppg2bp_opt.tflite')
    exit()



    # from ann_model import cnn_4_0_5_3
    # ai_model = cnn_4_0_5_3()
    # checkpoint_path = os.path.join(dfm.WORKSPACE, 'checkpoint')
    # convert_tflite(ai_model, checkpoint_path)
