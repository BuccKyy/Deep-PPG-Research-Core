import glob
import os
import json
import numpy as np
import tqdm
import wfdb.processing

import model.define_model as dfm
from generate_tfrecord import AIData
import matplotlib.pyplot as plt


def state_analyze(ppg, abp):
    import scipy.signal
    x = 125
    normal, preH, stage1H, stage2H, other = [], [], [], [], []

    for i in range(len(ppg)):
        peaks_ppg, _ = scipy.signal.find_peaks(ppg[i], prominence=0.2, distance=0.1 * x)
        peaks_abp, _ = scipy.signal.find_peaks(abp[i], prominence=0.2 * 140, distance=0.1 * x)
        # troughs, _ = scipy.signal.find_peaks(-abp, distance=50)
        troughs, _ = scipy.signal.find_peaks(-abp[i], prominence=0.2 * 140, distance=0.1 * x)

        # print('SBP: {} | DBP: {}'.format(np.mean(abp[i][peaks_abp]), np.mean(abp[i][troughs])))
        SBP = np.mean(abp[i][peaks_abp])
        DBP = np.mean(abp[i][troughs])

        if SBP < 120 and DBP < 80:
            normal.append(i)
        elif 120 < SBP < 140 or 80 < DBP < 90:
            preH.append(i)
        elif 140 < SBP < 160 or 90 < DBP < 100:
            stage1H.append(i)
        elif 160 < SBP or 100 < DBP:
            stage2H.append(i)
        else:
            other.append(i)

    return normal, preH, stage1H, stage2H, other


def main():
    DIR_SAVE_TFRECORD = dfm.DIR_SAVE_TFRECORD
    # dir_wfdb = '/mnt/ai_data/PPG2ABP_data_second/Cuff-Less_Blood_Pressure_Estimation'
    # dir_wfdb = '/mnt/ai_data/PPG2ABP_data_second/best_data/30'
    dir_wfdb = '/mnt/ai_virtual/dataset/PPG_AI'

    os.makedirs(DIR_SAVE_TFRECORD, exist_ok=True)

    db_tfrecord = AIData(DIR_SAVE_TFRECORD, dfm.input_length)
    db_tfrecord.generate_multiprocess(dir_wfdb)
    # db_tfrecord.debug_multiprocess(dir_wfdb)


if __name__ == '__main__':
    main()
    # view_data()
    # view_data_tfrecord()
    # plot_histogram()
    # handle_data()
