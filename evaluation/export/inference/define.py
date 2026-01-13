import functools
import os.path
import time

INPUT_SHAPE = (125, 3, 1)

DIR_TFRECORD = '/mnt/ai_data/Activity-Detection/datav211222'
DIR_SAVE_MODEL = '/mnt/ai_data/tmp/acc_model_3'
DIR_MODEL_RAW = os.path.join(DIR_SAVE_MODEL, 'model_raw')
DIR_MODEL_CUSTOM_LAST_LAYER = os.path.join(DIR_SAVE_MODEL, 'custom_last_layer')
DIR_TFLITE = os.path.join(DIR_SAVE_MODEL, 'TFLite')

BASE_MODEL_TFLITE_PATH = os.path.join(DIR_TFLITE, 'base_model.tflite')
BLOCK_EXTEND_TFLITE_PATH = os.path.join(DIR_TFLITE, 'block_extend.tflite')
ENTIRE_MODEL_TFLITE_PATH = os.path.join(DIR_TFLITE, 'custom_last_layer.tflite')


os.makedirs(DIR_MODEL_RAW, exist_ok=True)
os.makedirs(DIR_MODEL_CUSTOM_LAST_LAYER, exist_ok=True)
os.makedirs(DIR_TFLITE, exist_ok=True)


def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        method_name = method.__name__
        if '.' in method.__qualname__:
            method_name = method.__qualname__

        tmp = ' '
        if 'file' in method_name:
            try:
                tmp = ' [{}] '.format(os.path.basename(kwargs['s3_file']))
            except (Exception,):
                pass

        print('--@timeit-- [{}]{}executed in {} ms'.format(method_name, tmp, round((te-ts)*1000, 4)))
        return result

    return timed
