import numpy as np
import time
import tensorflow as tf

# Define
ADC_GAIN = 26.164
FS = 25
ACC_CHANNELS = 3

ACC_SEG_TIME = 5  # second
ACC_BATCHSIZE = 3
PADDING_THRES = 1

# Path tflite
CHECKPOINT_ACTIVITY = ''


def activity_classification(buf_acc):
    # 1: lying; 2: sitting / standing; 3: walking / running; 0: others
    buf_len = ACC_SEG_TIME * FS * ACC_BATCHSIZE
    n = np.ceil(len(buf_acc) / buf_len)
    paddings = 0
    if len(buf_acc) < n*buf_len:
        paddings = int(n*buf_len - len(buf_acc))
        buf_acc = np.concatenate((buf_acc, np.zeros((paddings, ACC_CHANNELS))))
    buf_acc = buf_acc.reshape((int(len(buf_acc) / buf_len), ACC_BATCHSIZE,
                               int(ACC_SEG_TIME * FS), ACC_CHANNELS))
    start_time = time.time()

    activities = []
    Interpreter = tf.lite.Interpreter

    for i in range(buf_acc.shape[0]):
        seg = np.array(buf_acc[i, :, :, :], dtype=np.float32)

        acc_interpreter = Interpreter(model_path=CHECKPOINT_ACTIVITY)
        acc_interpreter.allocate_tensors()
        acc_input_details = acc_interpreter.get_input_details()
        acc_output_details = acc_interpreter.get_output_details()

        acc_interpreter.set_tensor(acc_input_details[0]['index'], seg)
        acc_interpreter.invoke()
        _output = np.argmax(acc_interpreter.get_tensor(acc_output_details[0]['index']), axis=1)
        activities.append(_output)

    print('Activity classification time: ', (time.time() - start_time))
    activities = np.array(activities).flatten()
    return activities