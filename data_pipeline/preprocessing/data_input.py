import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import signal
import scipy.stats as stats
from scipy.stats import skew, kurtosis

from define import *
from cross_correlation.bp_annotate import BpAnnotate, FindPeak
# from utils.reprocessing import butter_lowpass_filter, butter_highpass_filter
from wfdb.processing import resample_sig

from utils_ai.reprocessing import butter_lowpass_filter, butter_highpass_filter


class DataPreparation:
    def init(self):
        super(DataPreparation, self).__init__()

    def baseline_signal(self, signal):
        x = np.linspace(min(signal), max(signal), len(signal))
        polyfit = np.poly1d(np.polyfit(x, signal, 4))
        baseline = polyfit(x)
        return baseline

    def sqi_skewness(self, raw_sig, fs_resamp, thr_ske, thr_kur):
        wd = 2 * fs_resamp
        data_index = np.arange(wd)[None, :] + \
                     np.arange(0, len(raw_sig) - wd, wd)[:, None]

        seg_raw_sig = raw_sig[data_index]
        skewness = max(skew(seg_raw_sig, axis=1))  # calculate SQI
        kurtosi = max(kurtosis(seg_raw_sig, axis=1))
        is_fit = np.logical_or(skewness > thr_ske, kurtosi > thr_kur)

        return is_fit, skewness, kurtosi
    @staticmethod
    def compute_sqi_skewness(raw_sig, fs_resamp, thr_ske, thr_kur):
        wd = 2 * fs_resamp
        data_index = np.arange(wd)[None, :] + \
                     np.arange(0, len(raw_sig) - wd, wd)[:, None]

        seg_raw_sig = raw_sig[data_index]
        skewness = max(skew(seg_raw_sig, axis=1))  # calculate SQI
        kurtosi = max(kurtosis(seg_raw_sig, axis=1))
        is_fit = np.logical_or(skewness > thr_ske, kurtosi > thr_kur)

        return is_fit, skewness, kurtosi

    @staticmethod
    def compute_sqi(sig, fs):
        # sig = np.diff(sig)

        std = np.std(sig)
        skewness = np.mean(np.power((sig - np.mean(sig)), 3)) / np.power(std, 3)
        kurtosi = np.mean(np.power((sig - np.mean(sig)), 4)) / np.power(std, 4)

        KTE_mean, KTE_variance, KTE_iqr = DataPreparation.compute_KTE(sig)
        return [skewness, kurtosi, std, KTE_variance, KTE_iqr]

    def preprocessing(self, raw_sig, fs, fs_resamp, thr_ske=0.88, thr_kur=-0.93):
        try:
            norm = stats.zscore(raw_sig)
            # norm = np.nan_to_num(norm)
            lp = butter_lowpass_filter(norm, fs, fs_resamp, 6)
            # baseline_org = self.baseline_signal(raw_sig)
            baseline = self.baseline_signal(lp)
            bwr = lp - baseline

            is_fit, skewness, kurtosi = self.sqi_skewness(lp, fs_resamp, thr_ske, thr_kur)

            # plt.subplot(311)
            # plt.plot(raw_sig, 'r')
            # plt.subplot(312)
            # plt.plot(lp)
            # plt.plot(baseline, 'b')

            # plt.show()

            return bwr, is_fit, skewness, kurtosi  # , baseline, baseline_org
        except:
            return None, True, None, None

    @staticmethod
    def compute_KTE(sig):
        """
        Kaiser–Teager energy (KTE) of a signal is a standard tool that can classify anomalous and acceptable signals
        by separating noise, transients, and artifacts from the signal. The mean (KTE_mean), variance (KTE_variance)
        and interquartile range (KTE_iqr) of the KTE of a PPG segment are computed as three features.
        """
        N = len(sig)
        KTE_i = []
        for i in range(1, N-1):
            KTE_i.append(sig[i]*sig[i] - sig[i-1]*sig[i+1])
        KTE_i = np.asarray(KTE_i)
        KTE_mean = KTE_i.mean()
        KTE_variance = np.power(KTE_i - KTE_mean, 2).mean()
        q3, q1 = np.percentile(sig, [75, 25])
        KTE_iqr = q3 - q1
        return KTE_mean, KTE_variance, KTE_iqr

    def preprocessing_KTE(self, raw_sig, fs, fs_resamp, thr_ske=0.88, thr_kur=-0.93):
        try:
            norm = stats.zscore(raw_sig)
            # norm = np.nan_to_num(norm)
            lp = butter_lowpass_filter(norm, fs, fs_resamp, 6)
            lp = butter_highpass_filter(lp, 0.5, fs_resamp, 6)
            KTE_mean, KTE_variance, KTE_iqr = self.compute_KTE(lp)

            return lp, KTE_mean, KTE_variance, KTE_iqr  # , baseline, baseline_org

        except:
            return None, True, None, None

    @staticmethod
    def double_derivative(raw_waveformD, raw_waveformDD, fs, systolic_index):
        waveformD = butter_lowpass_filter(raw_waveformD, 12, fs, 3)
        waveformD = FindPeak().avgfilter(waveformD, 15)

        waveformDD = butter_lowpass_filter(raw_waveformDD, 12, fs, 3)
        waveformDD = FindPeak().avgfilter(waveformDD, 10)
        a1 = FindPeak().findpeaks(waveformD, spacing=70, limit=None)

        if len(a1) > len(systolic_index):
            a1_new = np.zeros(len(systolic_index))
            for i in range(len(a1)):
                tmp = systolic_index - a1[i]
                if all(tmp < 0):
                    continue
                index = np.flatnonzero(tmp < 100)
                if len(index) == 0:
                    continue
                index = np.flatnonzero(tmp < 100)[-1]
                if index.size:
                    a1_new[index] = a1[i]
            a1 = a1_new.astype(int)

        b1 = np.zeros_like(a1)
        for i in range(len(a1)):
            if (a1[i] - 10) < 0:
                a1[i] = np.flatnonzero(raw_waveformD == np.max(raw_waveformD[0: a1[i] + 10]))
            else:
                a1[i] = np.flatnonzero(raw_waveformD == np.max(raw_waveformD[a1[i] - 10: a1[i] + 10]))
            b1[i] = np.flatnonzero(raw_waveformD == np.min(raw_waveformD[a1[i]: a1[i] + 50]))

        a2 = FindPeak().findpeaks(waveformDD, spacing=70, limit=None)

        if len(a2) > len(systolic_index):
            a2_new = np.zeros(len(systolic_index))
            for i in range(len(a2)):
                tmp = systolic_index - a2[i]
                if all(tmp < 0):
                    continue
                index = np.flatnonzero(tmp < 100)
                if len(index) == 0:
                    continue
                index = np.flatnonzero(tmp < 100)[-1]
                if index.size:
                    a2_new[index] = a2[i]
            a2 = a2_new.astype(int)

        b2 = np.zeros_like(a2)
        for i in range(len(a2)):
            if (a2[i] - 10) < 0:
                a2[i] = np.flatnonzero(raw_waveformDD == np.max(raw_waveformDD[0: a2[i] + 10]))
            else:
                a2[i] = np.flatnonzero(raw_waveformDD == np.max(raw_waveformDD[a2[i] - 10: a2[i] + 10]))
            b2[i] = np.flatnonzero(raw_waveformDD == np.min(raw_waveformDD[a2[i]: a2[i] + 50]))

        return a1, b1, a2, b2

    @staticmethod
    def freq_domain_feat(filt_sig, fs, debugmode=False):
        den = np.abs(np.fft.rfft(filt_sig, n=len(filt_sig)))
        freq = np.linspace(0, fs / 2, len(den))
        p = signal.argrelextrema(den, np.greater)
        p = np.asarray(p).flatten()
        three_peaks = p[:3]

        a25 = np.trapz(den[np.flatnonzero((freq >= 2) & (freq < 5))])
        a02 = np.trapz(den[np.flatnonzero((freq >= 0) & (freq < 2))])

        if debugmode:
            plt.plot(den)
            plt.plot(three_peaks, den[three_peaks], 'r*')
            plt.show()
        return a02, a25

    def time_domain_feat(self, prep_sig, FS, debugmode=False):
        footIndex, systolicIndex, notchIndex, dicroticIndex, filt_sig, waveformD, waveformDD, waveformDDPlus = BpAnnotate().process(
            prep_sig, FS, 'au', isClean=True, plot=False)
        if len(dicroticIndex) != len(footIndex) or len(systolicIndex) != len(footIndex):
            print("The number of foot peaks, systolic peaks and diastolic peaks is not similar")
            return [], [], [], [], [], []
        if len(dicroticIndex) == 1 or len(systolicIndex) == 1 or len(footIndex) == 1:
            print("Find only 1 systolic peak in signal")
            return [], [], [], [], [], []

        fs = 200
        a1, b1, a2, b2 = self.double_derivative(waveformD, waveformDD, fs, systolicIndex)

        t1 = (systolicIndex - footIndex) / fs
        t3 = (dicroticIndex - footIndex) / fs
        tpi = np.diff(footIndex) / fs
        tpp = np.diff(systolicIndex) / fs

        if debugmode:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(19.2, 10.8))
            ax1.plot(filt_sig, label='Filtered PPG')
            ax1.plot(footIndex, filt_sig[footIndex], 'r*', label='Foot')
            ax1.plot(systolicIndex, filt_sig[systolicIndex], 'g*', label='Systolic')
            ax1.plot(notchIndex, filt_sig[notchIndex], 'b*', label='Notch')
            ax1.plot(dicroticIndex, filt_sig[dicroticIndex], 'k*', label='Discrotic')
            ax1.legend()

            ax2.plot(waveformD, label='1st derivative')
            ax2.plot(a1, waveformD[a1], 'r*', label='a1')
            ax2.plot(b1, waveformD[b1], 'g*', label='b1')
            ax2.legend()

            ax3.plot(waveformDD, label='2nd derivative')
            ax3.plot(a2, waveformDD[a2], 'b*', label='a2')
            ax3.plot(b2, waveformDD[b2], 'k*', label='b2')
            ax3.legend()
            plt.show()
        return t1, t3, tpi, tpp, filt_sig, fs

    # version 3: feature from user (w,h,age)  no hr
    def feature_extract(self, prep_sig, weight, height, age, fs_resamp):
        t1, t3, tpi, tpp, filt_sig, fs = self.time_domain_feat(prep_sig, fs_resamp)
        if len(t1) == 0 or len(t3) == 0 or len(tpi) == 0 or len(tpp) == 0:
            return [], []
        a02, a25 = self.freq_domain_feat(filt_sig, fs)

        sys_features = np.zeros(10)
        dias_features = np.zeros(9)

        sys_features[0] = int(age)  # age
        sys_features[1] = int(weight)  # weight
        sys_features[2] = int(height)  # height

        # Calculate BMI
        BMI = weight / (height / 100) ** 2
        sys_features[3] = float(BMI)  # bmi

        sys_features[4] = a25  # A2-5
        sys_features[5] = float(BMI) / np.mean(t1)  # BMI/t1
        sys_features[6] = int(weight) / np.mean(tpi)  # Weight/tpi
        sys_features[7] = int(weight) / np.mean(tpp)  # Weight/tpp
        sys_features[8] = int(weight) / np.mean(t1)  # Weight/t1
        sys_features[9] = float(BMI) / np.mean(tpp)  # BMI/tpp

        dias_features[0] = int(age)  # age
        dias_features[1] = int(weight)  # weight
        dias_features[2] = int(height)  # height
        dias_features[3] = float(BMI)  # bmi
        dias_features[4] = float(BMI) / np.mean(t3)  # bmi/t3
        dias_features[5] = int(weight) / np.mean(tpi)  # weight/tpi
        dias_features[6] = np.nanmean(t3)  # t3
        dias_features[7] = float(BMI) / np.mean(tpi)  # bmi/tpi
        dias_features[8] = a02  # A0-2

        return sys_features, dias_features

    # version 3:load data with user input: weight, height, age (no hr)
    def load_data(self, ppg_file, weight, height, age, fs, fs_resamp, time=2.1):
        with open(ppg_file) as csvfile:
            data = list(csv.reader(csvfile, delimiter='\t'))[0]
            raw_sig = np.asarray([int(float(d)) for d in data if len(d) > 0])
            raw_sig, _ = resample_sig(raw_sig, fs, fs_resamp)

            prep_sig, is_fit, skewness = self.preprocessing(raw_sig)
            n = int(len(raw_sig) // (time * fs_resamp))
            segment_prep_sig = np.array_split(prep_sig, n)
            segment_sys_features = []
            segment_dias_features = []

            mode = 0
            for i in range(0, n):
                try:
                    sys_features, dias_features = self.feature_extract(segment_prep_sig[i], weight, height, age)
                    segment_sys_features.append(sys_features)
                    segment_dias_features.append(dias_features)

                except:
                    segment_sys_features.append([])
                    segment_dias_features.append([])
                    mode = -1

                if len(segment_dias_features[0]) == 0:
                    mode = -2

            return segment_sys_features, segment_dias_features, n, skewness, mode

# input (ppg_file, weight, height, age, hr)
# DataPreparation().load_data_2(ppg_file, 70,100,20,100)

# ppg_file = '/home/baoson/Documents/GitHub/ppg2bp/datasets/PPG_BP_Database/0_subject/151_3.txt'
# a = DataPreparation().load_data(ppg_file, 70,100,20)
# print(a)
# DataPreparation().load_data(DATA_JSON, DATA_DIR, DEMOGAPHIC_PATH, SQI_PATH)
