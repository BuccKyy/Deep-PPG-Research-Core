from os.path import join
from keras.layers import LeakyReLU

import tensorflow as tf
# from keras.utils import losses_utils
from datetime import datetime

now = datetime.now()  # current date and time
date_time = now.strftime("%Y%m%d%H%M%S")

VER_MODEL = '5.0.0.13'  # note: '3.0.25'
VER_DATA = '5.0.13'   # note: 3.0.21 .22

REMOVE_BASLINE = False
REMOVE_BASLINE_PPG = True
REMOVE_BASLINE_ABP = False

ADD_DC_PPG = False
ADD_DC_ABP = False

# WORKDIR = '/mnt/ai_data/PPG2ABP_data_second/'
# WORKDIR = '/mnt/ai_data/PPG2ABP_data_third/'
WORKDIR = '/mnt/ai_data/PPG2ABP_data_fourth/'
WORKSPACE = join(WORKDIR, 'model_{}'.format(VER_MODEL))

VERSION = 0

TRAIN_PATH = join(WORKSPACE, 'training-v{}'.format(VERSION))
DATA_PATH = join(WORKSPACE, 'dataset')
RESULT_PATH = join(WORKSPACE, 'result')

FS = 125
FS_TARGET = 128
input_length = 128*8  # 256*4
NUM_OVERLAP = 128*6  # 0

num_classes = 1
deconv_ksize = 2
model_width = 16 * 2  # 64
dropout = 0.25

# FUNCTION_LOSS = tf.keras.losses.MeanSquaredError()
# FUNCTION_LOSS = tf.keras.losses.MeanAbsolutePercentageError()
FUNCTION_LOSS = tf.keras.losses.MeanAbsoluteError()
# FUNCTION_LOSS = tf.keras.losses.Huber(
#     delta=1.0,
#     reduction=losses_utils.ReductionV2.AUTO,
#     name='huber_loss'
# )

OPTIMIZER = tf.keras.optimizers.Adadelta()

CONV_ACTIVATION = LeakyReLU
# Conv_activation = CONV_ACTIVATION(alpha=0.01)  # 'relu'
Conv_activation = 'relu'

activation = 'linear'

BATCH_SIZE = 128 * 8
EPOCH = 100
# DIR_SAVE_TFRECORD = join(WORKDIR, 'data')

DIR_SAVE_TFRECORD = join(WORKDIR, 'data_{}'.format(VER_DATA))
DIR_LOG_TensorBoard = join(WORKSPACE, 'log_TensorBoard')

PPG_MIN = -1  # -2.4
PPG_MAX = 1.4  # 2.9
ABP_MIN = 50.33  # 50
ABP_MAX = 192.24  # 190

CLASS_BP = {'Norm': 0, 'PreHypertension': 1, 'Stage1': 2, 'Stage2': 3, 'Ot': 4}


REJECT_ID = [9, 10, 15, 27, 30, 34, 39, 45, 49, 50]
NPY_DATA_DIR = '/mnt/ai_data/ABPM_FW/collect_itr/PPG_ITR/*/*.npy'
LABEL_PATH = '/home/hoang/Downloads/Subjects - Labels.xlsx'

# NPY_DATA_DIR = '/mnt/ai_data/ABPM_FW/Hau/Data_Finger/*.npy'
# LABEL_PATH = '/mnt/ai_data/ABPM_FW/Hau/Subjects - Labels.xlsx'
