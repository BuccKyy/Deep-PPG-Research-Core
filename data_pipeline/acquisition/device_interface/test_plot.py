import glob
import os.path

import numpy as np
import matplotlib.pyplot as plt

from Steam_PPG.handle import HandlePPG
from define import PPG_GAIN
from model.test_model import evalute_result


def read_dat(filePath, num_bytes=3, start_byte=512, byteorder='big', signed=False):
    with open(filePath, "rb") as f:
        file = f.read()[start_byte:]
        convert = lambda byte: np.int.from_bytes(byte, byteorder=byteorder, signed=signed)
        signal = [convert(file[i:i + num_bytes]) for i in range(0, len(file), num_bytes)]

        signal = np.asarray(signal)
        # signal_1 = np.bitwise_and(signal, 0x3FFF)
    return signal


def plot_sig():
    a = np.load('/mnt/ai_data/ABPM_FW/Motion_data/121_230821_111130.npy')
    a = a.reshape((-1, 3))

    plt.plot(a[:, 0], color='r', label='RED')
    plt.plot(a[:, 1], color='g', label='GREEN')
    plt.plot(a[:, 2], color='k', label='IR')
    # plt.plot(a[:, 3], color='b', label='AMBIENT')

    plt.legend(loc='upper right')
    plt.show()


def test_func():
    import pandas as pd

    result = None
    # hd = HandlePPG(fs_raw=256, fs=128, num_ch=3, num_overlap=int(8*128*2/3))
    hd = HandlePPG(fs_raw=256, fs=128, num_ch=3, num_overlap=0)

    hd.analyze('/mnt/ai_data/ABPM_FW/Motion_data/121_230821_111130.npy')
    exit()


    dir_data = '/mnt/ai_data/ABPM_FW/collect_itr/PPG_ITR/*/*.npy'
    paths = glob.glob(dir_data)
    from natsort import natsorted
    paths = natsorted(paths)
    for path in paths:
        # if not '051' in path:
        #     continue
        result_file = hd.analyze(path)
        if isinstance(result_file, int):
            continue
        result = np.vstack([result, result_file]) if result is not None else result_file

    # df = pd.DataFrame(result)
    # df.columns = ['Subject ID', 'SBP_predict', 'DBP_predict', 'SBP_OMROM', 'DBP_OMROM', 'HR', 'HR_OMRON', 'SpO2', 'SpO2_Device']
    # df.to_excel('{}.xlsx'.format('Estimate_BP_HR_SpO2'), index=False)


def evaluate_excel():
    import pandas as pd
    df = pd.read_excel('Estimate_BP_HR_SpO2.xlsx', index_col=None, header=0)
    df = df.fillna(method='ffill')
    sd = df.iloc[:, 1:5].to_numpy()
    evalute_result(*sd.T)

    hr = df.iloc[:, 5:7].to_numpy()
    spo2 = df.iloc[:, 7:9].to_numpy()
    hr_rmse = np.sqrt(np.mean(np.power(hr[:, 0] - hr[:, 1], 2)))
    spo2_rmse = np.sqrt(np.mean(np.power(spo2[:, 0] - spo2[:, 1], 2)))

    print('HR_RMSE: {}\nSpO2_RMSE: {}'.format(hr_rmse, spo2_rmse))

    a = 10

plot_sig()
# evaluate_excel()
# test_func()
