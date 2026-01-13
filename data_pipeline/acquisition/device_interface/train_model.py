import glob
import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Steam_PPG.handle import HandlePPG
from define import PPG_GAIN
from model.ann_model import cnn_4_0_5_6, cnn_4_0_8_0, cnn_4_0_8_1, cnn_4_0_8_2
import model.define_model as dfm
import tensorflow as tf
from ITR_device.process_ppg_device import HandlePPG as HP




def read_dat(filePath, num_bytes=3, start_byte=512, byteorder='big', signed=False):
    with open(filePath, "rb") as f:
        file = f.read()[start_byte:]
        convert = lambda byte: np.int.from_bytes(byte, byteorder=byteorder, signed=signed)
        signal = [convert(file[i:i + num_bytes]) for i in range(0, len(file), num_bytes)]

        signal = np.asarray(signal)
        # signal_1 = np.bitwise_and(signal, 0x3FFF)
    return signal

def plot_sig():
    a = np.load('test_st_ref.npy')
    a = a.reshape((-1, 3))

    plt.plot(a[:, 0], color='r', label='RED')
    plt.plot(a[:, 1], color='g', label='GREEN')
    plt.plot(a[:, 2], color='k', label='IR')
    # plt.plot(a[:, 3], color='b', label='AMBIENT')

    plt.legend(loc='upper right')
    plt.show()


def test_func():
    # model = tf.keras.models.load_model('/mnt/ai_data/PPG2ABP_data_fourth/model_4.0.5.5/save_model')
    model = cnn_4_0_8_1()
    # for i in range(13):
    #     model.layers[i].trainable = False

    df = pd.read_excel(dfm.LABEL_PATH, index_col=None, header=1)
    # df = pd.read_excel('/mnt/ai_data/ABPM_FW/Hau/Subjects - Labels.xlsx', index_col=None, header=1)
    df = df.fillna(method='ffill')
    key = df.columns.tolist()
    df.rename(columns={'Unnamed: 0': 'ID', 'Unnamed: 1': 'Name', 'Unnamed: 2': 'Gender', 'Unnamed: 3': 'Age', 'Unnamed: 4': 'STT', 'Unnamed: 5': 'Date', 'Unnamed: 6': 'Time'}, inplace=True)
    data = df[['ID', 'SBP', 'DBP', 'PUL', 'Gender', 'Age']]

    # plt.hist(np.round(np.mean(data.iloc[:, 1:3].to_numpy().reshape(-1, 5, 2), axis=1)), bins=15, label=['SBP', 'DBP'])
    plot_hist = False
    if plot_hist:
        sd =data.iloc[:, 1:3].to_numpy()
        sd = np.delete(sd, 191, axis=0)
        sd = sd.astype(int)
        plt.hist(sd, bins=25, label=['SBP', 'DBP'])
        plt.xlabel('Blood Pressure (mmHg)')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.show()
        exit()

    hd = HandlePPG(fs_raw=256, fs=128, num_ch=3, num_overlap=2*128)

    paths = glob.glob(dfm.NPY_DATA_DIR)
    # paths = glob.glob('/mnt/ai_data/ABPM_FW/Hau/Data_Finger/*.npy')
    from natsort import natsorted
    paths = natsorted(paths)
    data_in = None
    label = None
    info_in = None
    # paths = paths[15:]
    paths = paths[:-2]
    for path in paths:
        num = int(os.path.basename(path).rstrip('.npy'))
        if num in dfm.REJECT_ID:
            continue
        sub_label = data.query('ID == {}'.format(num))
        SD_train = np.mean(sub_label.iloc[:, 1:3].to_numpy(), axis=0, dtype=int)

        SD_info = sub_label.iloc[:, 1:3].to_numpy().astype(float)
        SD_info = np.mean(SD_info, axis=0)
        PP = SD_info[0] - SD_info[1]
        MAP = (SD_info[0] + 2*SD_info[1])/3
        SD_info[0] = PP
        SD_info[1] = MAP

        info = sub_label[['Gender', 'Age']].iloc[0, :].to_list()
        if info[0] == 'Male':
            info[0] = 0
        else:
            info[0] = 1

        info = np.asarray(info, dtype=float)
        # info = np.concatenate((info, SD_info))
        # info = np.mean(sub_label.iloc[0:2, 1:3].to_numpy(), axis=0, dtype=int)

        # sub_ppg = hd.extract_data(path)
        sub_ppg, label_rptt = hd.extract_data_rptt(path)
        if len(sub_ppg) == 0:
            continue
        # sub_ppg = sub_ppg[:len(sub_ppg)//3]
        label_file = np.concatenate([SD_train[None, :]] * len(sub_ppg), axis=0)
        label_info = np.concatenate([SD_info[None, :]] * len(sub_ppg), axis=0)

        data_in = np.vstack([data_in, sub_ppg]) if data_in is not None else sub_ppg
        info_in = np.vstack([info_in, label_info]) if info_in is not None else label_info
        # label = np.vstack([label, label_file]) if label is not None else label_file
        label = np.vstack([label, label_rptt]) if label is not None else label_rptt

    label = label.astype(float)
    data_in = data_in / 10

    optimizer = tf.keras.optimizers.Adam()

    model.summary()
    model.compile(loss='mae', optimizer=optimizer, metrics=['mape'])
    # model.fit(x=[data_in, info_in], y=[label[:, 0][:, None], label[:, 1][:, None]], epochs=100, batch_size=128, validation_split=0)
    # model.fit(x=data_in, y=[label[:, 0][:, None], label[:, 1][:, None]], epochs=200, batch_size=128, validation_split=0.2)
    model.fit(x=data_in, y=label, epochs=300, batch_size=32)
    model_save_dir = os.path.join(dfm.WORKSPACE, 'save_model_fine_tune')
    model.save(model_save_dir)


test_func()