import datetime
import os
import time
import serial
from Steam_PPG.utils import butter_bandpass_filter

import matplotlib.pyplot as plt
import numpy as np
from handle import HandlePPG
from define import NUM_CH, PPG_GAIN, CHANNELS_COLOR, CHANNELS_LABEL, SAMPLE_TIME_1S_RAW_DATA, SAMPLE_TIME_1S_PPG_DATA, \
    PORT, BAUD_RATE, FS_128, DIR_SAVE, SUBJECT_ID

# Initialize list to store PPG sensor data
sample_count = []
ppg_sensor_data = []
ppg_sensor_data_store = [[] for _ in range(NUM_CH)]

# Initialize UART connection
ser = serial.Serial(PORT, BAUD_RATE)
print("Start Recording")

# Initialize the plot
plt.ion()
fig = plt.figure(figsize=(15, 8))
ax = []
for i in range(6):
    ax.append(fig.add_subplot(3, 2, i+1))


def plot_sig():
    a = np.load('test_st.npy')
    a = a.reshape((-1, NUM_CH))

    plt.plot(a[:, 0], color='r', label='RED')
    plt.plot(a[:, 1], color='g', label='GREEN')
    plt.plot(a[:, 2], color='k', label='IR')
    # plt.plot(a[:, 3], color='b', label='AMBIENT')

    plt.legend(loc='upper right')
    plt.show()


def plot_sig_realtime(s_id, ppg, seconds):
    ppg = ppg.reshape((-1, NUM_CH)) / PPG_GAIN
    time = - 256 * 10
    ppg_raw = -ppg

    ppg = -butter_bandpass_filter(ppg.T, 0.5, 10, 256, 2).T

    ppg_green = ppg[:, 1]
    hd = HandlePPG(fs=FS_128)

    try:
        # _, percent = hd.calculate_sqi_real_time_green(ppg_green, thr_sqi=20)
        _, percent = hd.calculate_sqi_green(ppg_green)
        mm_ss = str(datetime.timedelta(seconds=seconds))
        fig.suptitle('Subject ID: {}\nDuration time: {}\nPass SQI: {}%'.format(s_id, mm_ss, percent))

    except Exception as error:
        print('{} - Error'.format(error))

    for i_raw in range(0, 6, 2):
        ch = i_raw//2
        ax[i_raw].cla()
        ax[i_raw].plot(ppg_raw[time:, ch], color=CHANNELS_COLOR[ch], label=CHANNELS_LABEL[ch])
        ax[i_raw].legend(loc='upper right')

        i = i_raw + 1
        ax[i].cla()
        ax[i].plot(ppg[time:, ch], color=CHANNELS_COLOR[ch], label=CHANNELS_LABEL[ch])
        ax[i].legend(loc='upper right')
    ax[0].set_title('RAW')
    ax[1].set_title('FILTER')


    fig.canvas.draw()
    fig.canvas.flush_events()


def read_data():
    new_data = b''
    cnt = 0
    epoch_time = int(time.time())

    while 1:
        data = ser.read()
        if data == b'':
            continue
        else:
            new_data += data
            cnt += 1
        # print(cnt)
        if cnt == SAMPLE_TIME_1S_RAW_DATA:

            convert = lambda byte: int.from_bytes(byte, byteorder='big', signed=False)
            signal = [convert(new_data[k:k + 3]) for k in range(0, len(new_data), 3)]

            ppg_sensor_data.extend(signal)
            ppg_npy = np.asarray(ppg_sensor_data)
            seconds = len(ppg_npy) // SAMPLE_TIME_1S_PPG_DATA

            # save data - TBD
            path_save_data = os.path.join(DIR_SAVE, '{}_{}.npy'.format(SUBJECT_ID, epoch_time))
            np.save(path_save_data, ppg_npy)

            try:
                plot_sig_realtime(SUBJECT_ID, ppg_npy, seconds)
            except Exception as error:
                print('{} - Error'.format(error))

            cnt = 0
            new_data = b''


if __name__ == "__main__":
    read_data()
