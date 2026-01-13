import os

import numpy as np
import matplotlib.pyplot as plt
import glob

from Steam_PPG.handle import HandlePPG
from define import PPG_GAIN
from model.test_model import evalute_result
import model.define_model as dfm


def test_func():
    import pandas as pd

    result = None
    # hd = HandlePPG(fs_raw=256, fs=128, num_ch=3, num_overlap=int(8*128*2/3))
    hd = HandlePPG(fs_raw=256, fs=128, num_ch=3, len_seg=128*8, num_overlap=0)

    dir_data = dfm.NPY_DATA_DIR
    paths = glob.glob(dir_data)
    from natsort import natsorted
    paths = natsorted(paths)

    # paths = paths[:15]
    for path in paths:
        # if not '037' in path:
        #     continue

        num = int(os.path.basename(path).rstrip('.npy'))
        if num in dfm.REJECT_ID:
            continue

        result_file = hd.analyze(path)
        if isinstance(result_file, int):
            continue
        try:
            result = np.vstack([result, result_file]) if result is not None else result_file
        except:
            continue

    df = pd.DataFrame(result)
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


def evaluate_npy():
    import pandas as pd

    paths = glob.glob('/mnt/ai_data/ABPM_FW/result_npy/*.npy')
    from natsort import natsorted
    paths = natsorted(paths)
    total = []
    for path in paths:
        data = np.load(path)
        id = os.path.basename(path).rstrip('.npy')
        id = int(id)
        data = np.concatenate(([[id]]*len(data), data), axis=1)
        total.append(data)

    total = np.concatenate(total)
    # df = pd.DataFrame(total)
    # df.columns = ['Subject ID', 'SBP_OMROM', 'DBP_OMROM', 'SBP_predict', 'DBP_predict', 'Segment']
    # df.to_excel('/mnt/ai_data/ABPM_FW/{}.xlsx'.format('Estimate_BP'), index=False)

    SBP_error = total[:, 3] -total[:, 1]
    DBP_error = total[:, 4] -total[:, 2]
    error = np.vstack((SBP_error, DBP_error))

    mean = np.mean(error, axis=1)
    std = np.std(error, axis=1)

    print(mean, std)

    a = 10

if __name__ == '__main__':
    # evaluate_excel()
    # test_func()
    evaluate_npy()