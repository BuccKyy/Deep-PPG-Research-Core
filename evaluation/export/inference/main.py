import time

from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
from data_tfrecord import AIData
from architect_model import cnn, train_model, load_base_model_from_raw_model, load_block_extend, merge_model, create_example_model
from convert_tflite import ProcessTFlite
from define import DIR_TFRECORD, DIR_SAVE_MODEL, DIR_MODEL_CUSTOM_LAST_LAYER, DIR_MODEL_RAW, BASE_MODEL_TFLITE_PATH, BLOCK_EXTEND_TFLITE_PATH, ENTIRE_MODEL_TFLITE_PATH


def convert_models_tflite():
    ProcessTFlite.savedmodel_to_tflite(DIR_MODEL_RAW)
    ProcessTFlite.savedmodel_to_tflite(DIR_MODEL_CUSTOM_LAST_LAYER)
    model_base = load_base_model_from_raw_model()
    ProcessTFlite.keras_to_tflite(model_base, 'base_model')
    model_extend = load_block_extend()
    ProcessTFlite.keras_to_tflite(model_extend, 'block_extend')


if __name__ == "__main__":
    df = AIData(DIR_TFRECORD)

    train_model = False
    convert_to_tflite = False

    if train_model:
        create_example_model(df)
    if convert_to_tflite:
        convert_models_tflite()

    pt = ProcessTFlite(base_model_path=BASE_MODEL_TFLITE_PATH,
                       block_extend_path=BLOCK_EXTEND_TFLITE_PATH,
                       entire_model_path=ENTIRE_MODEL_TFLITE_PATH)
    pt.init_model()

    valid_data = df.load_tf_record('valid', batch_size=1)
    for data in valid_data.as_numpy_iterator():
        result_merge_model = pt.get_result_merge_model(data)
        result_entire_model = pt.get_result_entire_model(data)

        # mean abs error
        MAE = np.mean(result_merge_model - result_entire_model)
        print('Compare result from 2 method | MAE: {}'.format(MAE))
        a = 10


