import glob
import json

import numpy as np
import matplotlib.pyplot as plt
import wfdb as wf
from preprocessing import butter_bandpass_filter, baseline_wander_remove, butter_lowpass_filter, lowpass_remez, highpass_remez
from define import *
# from data_input import DataPreparation
from scipy.fft import fft


def compute_corr(sig1, sig2):
    # Pre-allocate correlation array
    corr = (len(sig1) - len(sig2) + 1) * [0]
    # Go through lag components one-by-one
    for l in range(len(corr)):
        corr[l] = sum([sig1[i + l] * sig2[i] for i in range(len(sig2))])
    return corr


def plot_ppg_abp(ppg, abp, seg):
    plt.subplot(311)
    plt.title('PPG')
    plt.plot(ppg[seg], 'r')
    plt.grid()
    plt.subplot(312)
    plt.title('ABP')
    plt.plot(abp[seg], 'k')
    plt.grid()
    plt.subplot(313)
    plt.title('Shifting ABP')

    max_ppg = np.max(ppg[seg])
    min_ppg = np.min(ppg[seg])

    max_abp = np.max(abp[seg])
    min_abp = np.min(abp[seg])

    plt.plot((ppg[seg] - min_ppg) / (max_ppg - min_ppg), 'r', label='PPG')
    plt.plot((abp[seg] - min_abp) / (max_abp - min_abp), 'k--', label='ABP')
    # plt.plot(ppg[seg] / (max_ppg - min_ppg), 'r', label='PPG')
    # plt.plot(abp[seg] / (max_abp - min_abp), 'k--', label='ABP')
    plt.legend(loc='upper right')
    plt.gcf().set_size_inches(18, 8)
    plt.grid()
    plt.show()
    plt.close()


def corr_process(sig_ppg, sig_abp, fs):
    x = 1000
    ppg = butter_bandpass_filter(sig_ppg, 0.5, 8, fs, order=2)
    ppg = baseline_wander_remove(ppg)
    abp = butter_bandpass_filter(sig_abp, 0.5, 8, fs, order=2)
    abp = baseline_wander_remove(abp)

    corr = np.correlate(a=ppg[x:-x], v=abp)
    lag = corr.argmax() - int(len(corr)/2)
    print('lag: {}'.format(lag))

    ppg = ppg[lag:]
    abp = abp[:-lag]
    return ppg, abp


def main():
    dir = '/mnt/ai_data/PPG2ABP_data_second/Cuff-Less_Blood_Pressure_Estimation'
    paths = glob.glob(dir+'/*.dat')
    record = wf.rdsamp(paths[0].rstrip('.dat'))
    fs = record[1]['fs']
    ind_PPG = record[1]['sig_name'].index('PLETH')
    ind_ABP = record[1]['sig_name'].index('ABP')

    # x = 350
    ppg = np.nan_to_num(record[0][:, ind_PPG])
    abp = np.nan_to_num(record[0][:, ind_ABP])

    # plt.subplot(311)
    # plt.plot(ppg)
    # plt.subplot(312)
    # plt.plot(abp)
    # plt.subplot(313)
    # plt.plot(corr)
    # plt.gcf().set_size_inches(18, 8)
    # plt.show()
    # plt.close()

    ppg, abp = corr_process(ppg, abp, fs)
    time_segment = LEN/fs
    seg = np.arange(0, time_segment * fs, 1)[None, :] + np.arange(0, len(ppg) - time_segment * fs, LEN - NUM_OVERLAP)[:, None]
    seg = seg.astype(int)

    for i in range(len(seg)):
        plot_ppg_abp(ppg, abp, seg[i])


def view_data_url():
    JsonPath = '/mnt/Dataset/ECG/PortalData_2/mimic3_ppg_abp/info/'
    list_json = glob.glob(JsonPath + '*.json')

    for file_json in list_json:
        f = open(file_json, 'r')
        content = json.load(f)
        url = content['record_url']
        list_records = content['Files']
        for file in list_records:
            data_process(file, url)


        a=10


def noise_mark(signal):
    len_data = len(signal)


def data_process(file, url):
    record = wf.rdsamp(file, pn_dir=url)
    fs = record[1]['fs']
    ind_PPG = record[1]['sig_name'].index('PLETH')
    ind_ABP = record[1]['sig_name'].index('ABP')

    ppg = np.nan_to_num(record[0][:, ind_PPG])
    abp = np.nan_to_num(record[0][:, ind_ABP])

    ppg_fil = lowpass_remez(ppg)
    ppg_fil = highpass_remez(ppg_fil)

    # abp_fil = highpass_remez(abp)
    abp_fil = lowpass_remez(abp)
    # ppg = butter_bandpass_filter(ppg, 0.5, 8, fs, order=2)
    # ppg = baseline_wander_remove(ppg)
    # abp = butter_bandpass_filter(abp, 0.5, 8, fs, order=2)
    # abp = butter_lowpass_filter(abp, 8, fs, order=2)
    # abp = baseline_wander_remove(abp)

    # corr = np.correlate(a=ppg, v=abp)
    DP = DataPreparation()
    ppg_fil_2, is_fit, skewness = DP.preprocessing(raw_sig=ppg, fs=8, fs_resamp=fs)
    # f_resolution = fs / LEN
    # x = fft(ppg[seg])
    # X_magnitude = np.abs(x)[:LEN // 2]

    plt.subplot(211)
    plt.plot(ppg, 'r')
    plt.plot(ppg_fil, 'ko')
    plt.plot(ppg_fil_2, 'g--')

    plt.subplot(212)
    plt.plot(abp, 'r')
    plt.plot(abp_fil, 'k--')
    # plt.subplot(313)
    # plt.plot(corr)
    plt.gcf().set_size_inches(18, 8)
    plt.show()

    # value = input("{} is good? (Y/N): ".format(file))
    # if value in ['Y', 'y']:
    #     ch_file = open('../mimic3_info/list_chosen_file.txt', 'a')
    #     ch_file.writelines('{}/{}\n'.format(url, file))
    #     ch_file.close()
    plt.close()


if __name__ == '__main__':
    # view_data_url()
    main()
