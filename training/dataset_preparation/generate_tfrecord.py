import glob
import os
import math
import json

import wfdb.processing

import model.define_model as dfm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wfdb as wf
from cross_correlation.preprocessing import butter_bandpass_filter, baseline_wander_remove, butter_lowpass_filter, \
    butter_highpass_filter, \
    lowpass_remez, highpass_remez
# from dft_noise.dft_main import detect_noise_2
from functools import partial
from define import *
import sys
from datetime import datetime
import scipy.stats
import scipy.signal

from model.define_model import REMOVE_BASLINE_PPG, ADD_DC_PPG, ADD_DC_ABP, REMOVE_BASLINE_ABP, input_length, VER_DATA, \
    PPG_MAX, PPG_MIN, ABP_MIN, FS_TARGET, ABP_MAX
from scipy.fft import fft


def _calc_num_steps(num_samples, batch_size):
    return (num_samples + batch_size - 1) // batch_size


class AIData:
    def __init__(self, save_data_path, len_seg: int):
        # self.tf_record_path = save_data_path + 'v{}'.format(VERSION)
        self.tf_record_dir = save_data_path
        self.len_seg = len_seg
        self.num_overlap = self.len_seg // 3
        # self.sample_shape = None
        # self.label_shape = None

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _dtype_feature(ndarray):
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. \
                               Instead got {}".format(ndarray.dtype))

    @staticmethod
    def save_tf_record(writer, ppg, abp, fs=125, num_overlap=0):
        for i in range(np.min([len(ppg), len(abp)])):
            feature = {
                'ppg': AIData._float_feature(ppg[i].tolist()),
                'abp': AIData._float_feature(abp[i].tolist())
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    @staticmethod
    def save_tf_record_2(writer, ppg, abp, fs=125, num_overlap=0):
        for i in range(np.min([len(ppg), len(abp)])):
            feature = {
                'ppg': AIData._float_feature(ppg[i].tolist()),
                'abp': AIData._float_feature([np.max(abp[i]), np.min(abp[i])])
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    @staticmethod
    def remove_bad_abp(abp):
        SBP = np.max(abp)
        DBP = np.min(abp)
        if SBP < 80 or SBP > 190:
            # print('SBP')
            return True
        if DBP < 50 or DBP > 120:
            # print('DBP')
            return True
        if SBP - DBP < 30 or SBP - DBP > 120:
            # print('SBP - DBP')
            return True
        return False

    @staticmethod
    def remove_bad_seg(ppg, abp):
        fs_target = dfm.FS_TARGET  # dfm.FS/4
        peaks_ppg, _ = scipy.signal.find_peaks(ppg, prominence=0.2, distance=0.1 * fs_target)
        peaks_abp, _ = scipy.signal.find_peaks(abp, prominence=0.2 * 140, distance=0.1 * fs_target)
        troughs_abp, _ = scipy.signal.find_peaks(-abp, prominence=0.2 * 140, distance=0.1 * fs_target)

        if len(peaks_ppg) < 2 or len(peaks_abp) < 2:
            return True

        SBP = np.mean(abp[peaks_abp])
        DBP = np.mean(abp[troughs_abp])
        if SBP < 80 or SBP > 180:
            # print('SBP')
            return True
        if DBP < 55 or DBP > 100:
            # print('DBP')
            return True
        if SBP - DBP < 35 or SBP - DBP > 100:
            # print('SBP - DBP')
            return True

        # if np.max(ppg) > dfm.PPG_MAX or np.min(ppg) < dfm.PPG_MIN:
        # if np.max(ppg) > 1 or np.min(ppg) < 0:
        #     return True

        amp_peaks_ppg = np.mean(ppg[peaks_ppg])
        if amp_peaks_ppg < 0.5:
            return True

        # std_peaks_ppg = np.std(ppg[peaks_ppg]) * 2.4  # 5.5
        # std_peak_dist_ppg = np.std(peaks_ppg) / dfm.FS
        # std_peaks_abp = np.std(abp[peaks_abp])
        # std_peak_dist_abp = np.std(peaks_abp) / dfm.FS
        # # if (std_peaks_ppg > 0.1) or (std_peak_dist_ppg > 2.28) or (std_peaks_abp > 2.3) or (std_peak_dist_abp > 2.28):
        # if (std_peaks_ppg > 0.16) or (std_peak_dist_ppg > 0.68) or (std_peaks_abp > 3) or (std_peak_dist_abp > 0.68):
        # # if (std_peaks_ppg > 0.16) or (std_peak_dist_ppg > 0.68):
        #     return True

        return False

    @staticmethod
    def init_calibrate(ppg, abp):
        list_features = np.array([])
        bp = []
        for i in range(np.min([len(ppg), len(abp)])):
            try:
                # co_bl = np.polyfit(np.arange(len(abp[i])), abp[i], 20)
                # baseline_drift = np.polyval(co_bl, abp[i])
                # abp[i] = abp[i] - baseline_drift + np.mean(baseline_drift)

                if AIData.remove_bad_seg(ppg[i], abp[i]):
                    # if AIData.remove_bad_abp(abp[i]):
                    continue

                peaks_abp, _ = scipy.signal.find_peaks(abp[i], prominence=0.2 * 140, distance=0.1 * dfm.FS / 4)
                troughs_abp, _ = scipy.signal.find_peaks(-abp[i], prominence=0.2 * 140, distance=0.1 * dfm.FS / 4)

                SBP = np.mean(abp[i][peaks_abp])
                DBP = np.mean(abp[i][troughs_abp])
                bp.append([SBP, DBP])

                RPTT, K_a, K_b = AIData.extract_features(ppg[i], SBP, DBP)
                if RPTT == -1:
                    continue
                features = np.array([RPTT, K_a, K_b])
                list_features = np.vstack([list_features, features]) if list_features.size > 0 else features
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(exc_type, fname, exc_tb.tb_lineno)
                continue
        if len(list_features) == 0:
            return None, None
        list_Ka = list_features[:, 1][~np.isnan(list_features[:, 1])]
        list_Kb = list_features[:, 2][~np.isnan(list_features[:, 2])]

        # if np.std(list_Ka) < 3 and np.std(list_Kb) < 6:
        #     return None, None

        feature_Ka = np.mean(list_Ka, dtype=np.float32)
        feature_Kb = np.mean(list_Kb, dtype=np.float32)

        bp = np.asarray(bp, dtype=int)
        len_bp = len(bp[:, 0])
        feature_Ka, feature_Kb = np.mean(bp[len_bp // 4, 0]), np.mean(bp[len_bp // 4, 1])

        return feature_Ka, feature_Kb

    @staticmethod
    def classification_bp(SBP, DBP):
        if SBP < 120 and DBP < 80:
            return 'Norm'
        elif 120 < SBP < 140 or 80 < DBP < 90:
            return 'PreHypertension'
        elif 140 < SBP < 160 or 90 < DBP < 100:
            return 'Stage1'
        elif 160 < SBP or 100 < DBP:
            return 'Stage2'
        else:
            return 'Ot'

    @staticmethod
    def analyze_ppg(ppg):
        maxima_index = scipy.signal.find_peaks(ppg, prominence=0.2, distance=0.1 * dfm.FS_TARGET)[0]
        minima_index = scipy.signal.find_peaks(-ppg, prominence=0.2, distance=0.1 * dfm.FS_TARGET)[0]
        derivative_1 = np.diff(ppg, n=1) * float(dfm.FS_TARGET)
        derivative_1_maxima_index = scipy.signal.argrelmax(np.array(derivative_1))[0]
        derivative_1_minima_index = scipy.signal.argrelmin(np.array(derivative_1))[0]
        derivative_2 = np.diff(ppg, n=2) * float(dfm.FS_TARGET)
        # derivative_2_maxima_index = scipy.signal.argrelmax(np.array(derivative_2))[0]
        derivative_2_maxima_index = \
            scipy.signal.find_peaks(np.array(derivative_2), prominence=2.5, distance=0.1 * dfm.FS_TARGET)[0]
        derivative_2_minima_index = scipy.signal.argrelmin(np.array(derivative_2))[0]
        # derivative_2_minima_index = scipy.signal.find_peaks(-np.array(derivative_2), prominence=2.5, distance=0.1 * dfm.FS_TARGET)[0]

        # return derivative_2, maxima_index, minima_index
        return derivative_2, derivative_2_maxima_index, derivative_2_minima_index, maxima_index, minima_index

    @staticmethod
    def analyze_ppg_fwbw(ppg):
        maxima_index = scipy.signal.argrelmax(np.array(ppg))[0]
        minima_index = scipy.signal.find_peaks(-ppg, prominence=0.2, distance=0.1 * dfm.FS)[0]
        derivative_1 = np.diff(ppg, n=1) * float(dfm.FS)
        derivative_1_maxima_index = scipy.signal.argrelmax(np.array(derivative_1))[0]
        derivative_1_minima_index = scipy.signal.argrelmin(np.array(derivative_1))[0]
        derivative_2 = np.diff(ppg, n=2) * float(dfm.FS)
        derivative_2_maxima_index = \
        scipy.signal.find_peaks(np.array(derivative_2), prominence=0.2, distance=0.1 * dfm.FS)[0]
        derivative_2_minima_index = \
        scipy.signal.find_peaks(np.array(-derivative_2), prominence=0.2, distance=0.1 * dfm.FS / 2)[0]

        return derivative_2, derivative_2_maxima_index, derivative_2_minima_index, maxima_index, minima_index

    @staticmethod
    def find_rptt(ppg):
        params = AIData.analyze_ppg(ppg)
        [derivative_2, derivative_2_maxima_index, derivative_2_minima_index, maxima_index, minima_index] = params
        s_peak = maxima_index[maxima_index > minima_index[0]][0]
        trough = minima_index[1]
        derivative_2_minima_index -= 2
        # d_peaks = derivative_2_minima_index[(derivative_2_minima_index > s_peak) & (derivative_2_minima_index < trough)]

        d_peaks = scipy.signal.find_peaks(-np.diff(ppg[s_peak:trough], 2) * dfm.FS_TARGET, prominence=0.2,
                                          distance=0.1 * dfm.FS_TARGET)[0]
        if len(d_peaks) < 2:
            return -1
        if d_peaks[0] < 15:
            d_peak = d_peaks[1]
        else:
            d_peak = d_peaks[0]
        d_peak += s_peak + 2
        rptt = (d_peak - s_peak) / dfm.FS_TARGET

        return rptt, derivative_2

    @staticmethod
    def find_rptt_fwbw(ppg):
        _ppg = ppg.copy()
        # _ppg = (_ppg - np.mean(_ppg))/np.std(_ppg)
        _ppg = (_ppg - np.min(_ppg)) / (np.max(_ppg) - np.min(_ppg))
        params = AIData.analyze_ppg_fwbw(_ppg)
        [derivative_2, derivative_2_maxima_index, derivative_2_minima_index, maxima_index, minima_index] = params
        s_peak = maxima_index[maxima_index > minima_index[0]][0]
        trough = minima_index[1]
        foward = np.concatenate((_ppg[minima_index[0]:s_peak], np.flip(_ppg[minima_index[0]:s_peak])))

        if len(foward) > len(_ppg[minima_index[0]:minima_index[0] + len(foward)]):
            return -1
        _ppg[minima_index[0]:minima_index[0] + len(foward)] -= foward - _ppg[minima_index[0]]
        # peak_bw = scipy.signal.find_peaks(_ppg[s_peak:trough], prominence=0.2, distance=0.1 * dfm.FS)[0]
        peak_bw = scipy.signal.find_peaks(_ppg[s_peak:trough], prominence=0.1)[0]

        if len(peak_bw) < 1:
            return -1
        d_peak = peak_bw[0] + s_peak
        rptt = (d_peak - s_peak) / dfm.FS_TARGET

        return rptt, derivative_2

    @staticmethod
    def extract_features(ppg, SBP, DBP):
        # peaks, troughs = find_bp(abp)
        # ppg_second, _, D_pgg, S_ppg, V_PGG = analyze(ppg)
        # RPTT, ppg_2 = AIData.find_rptt(ppg)
        RPTT, ppg_2 = AIData.find_rptt_fwbw(ppg)

        K_a = (SBP - DBP) * RPTT * RPTT
        K_b = DBP + 2 / 0.031 * np.log(RPTT) + K_a / (3 * RPTT * RPTT)

        return RPTT, np.float32(K_a), np.float32(K_b)

    @staticmethod
    def feature_ppg_pir(ppg):
        peaks = scipy.signal.find_peaks(ppg, prominence=0.2, distance=0.1 * dfm.FS)[0]
        troughs = scipy.signal.find_peaks(-ppg, prominence=0.2, distance=0.1 * dfm.FS)[0]

        if len(peaks) < len(troughs):
            return -1

        list_pir = []
        for peak in peaks:
            try:
                sub_pir = peak / troughs[troughs < peak][0]
                list_pir.extend([sub_pir])
            except Exception as e:
                continue
        pir = np.mean(list_pir)

        return pir
    @staticmethod
    def compute_fft(signal, fs=dfm.FS_TARGET):
        x = fft(signal)
        f_resolution = fs / len(signal)
        X_magnitude = np.abs(x)[:len(signal) // 2]
        _x = f_resolution * np.arange(len(x) // 2)
        return _x, X_magnitude

    def calculate_sqi_real_time(self, ppg_filtered):
        ppg = ppg_filtered.copy()
        seg = np.arange(0, self.len_seg, 1)[None, :] + np.arange(0, len(ppg) - self.len_seg,
                                                                 self.len_seg - self.num_overlap)[:, None]
        num_overlap = 0

        seg = seg.astype(int)
        ppg = ppg[seg]
        list_sqi = []
        for i in range(1, len(seg), 1):
            c = np.sum(ppg[i - 1] * ppg[i]) / np.sqrt(np.sum(np.power(ppg[i - 1], 2)) * np.sum(np.power(ppg[i], 2)))
            hat = (50 * (c + 1)) * 8 / 99
            sqi = np.exp(hat) / np.exp(8) * 100
            list_sqi.extend([sqi])

        list_sqi = np.asarray(list_sqi)
        ind_acceptable = np.flatnonzero(list_sqi > 20)  # 50

        return ind_acceptable

    @staticmethod
    # def save_tf_record_3(writer, ppg, ppg_first, ppg_second, abp, abp_raw):
    def save_tf_record_3(writer, ppg, ppg_2sd, abp, abp_dc):
        K_a, K_b = AIData.init_calibrate(ppg, abp)
        # if K_a is None or K_b is None:
        #     return None

        index_calib = (np.min([len(ppg), len(abp)])) // 8
        ppg_cl = ppg[index_calib]
        peaks_ppg, _ = scipy.signal.find_peaks(ppg_cl, prominence=0.2, distance=0.1 * dfm.FS_TARGET )

        seg_peak = np.arange(-128, 128, 1)[None, :] + peaks_ppg[((peaks_ppg > 128) & (peaks_ppg < 1024 - 128))][:, None]

        ppg_calib = ppg_cl[seg_peak]
        ppg_calib = np.mean(ppg_calib, axis=0)

        for i in range(np.min([len(ppg), len(abp)])):
            try:
                # co_bl = np.polyfit(np.arange(len(abp[i])), abp[i], 20)
                # baseline_drift = np.polyval(co_bl, abp[i])
                # abp[i] = abp[i] - baseline_drift + np.mean(baseline_drift)

                if AIData.remove_bad_seg(ppg[i], abp[i]):
                    # if AIData.remove_bad_abp(abp[i]):
                    continue

                fs_target = dfm.FS_TARGET  # dfm.FS/4

                peaks_abp, _ = scipy.signal.find_peaks(abp[i], prominence=0.2 * 140, distance=0.1 * fs_target)
                troughs_abp, _ = scipy.signal.find_peaks(-abp[i], prominence=0.2 * 140, distance=0.1 * fs_target)

                peaks_ppg, _ = scipy.signal.find_peaks(ppg[i], prominence=0.2, distance=0.1 * fs_target)

                seg_peak = np.arange(-128, 128, 1)[None, :] + peaks_ppg[((peaks_ppg>128) & (peaks_ppg<1024-128))][:, None]

                ppg_template = ppg[i][seg_peak]
                ppg_template = np.mean(ppg_template, axis=0)

                ppg_1st = np.diff(ppg_template)
                ppg_2nd = np.diff(ppg_template, 2)

                ppg_1st = np.append(ppg_1st, ppg_1st[-1])
                ppg_2nd = np.append(ppg_2nd, ppg_2nd[-2:])

                ppg_template = (ppg_template - np.min(ppg_template)) / (np.max(ppg_template) - np.min(ppg_template))
                # ppg_calib = (ppg_calib - dfm.PPG_MIN) / (dfm.PPG_MAX - dfm.PPG_MIN)
                # ppg_1st = (ppg_1st - np.min(ppg_1st)) / (np.max(ppg_1st) - np.min(ppg_1st))
                # ppg_2nd = (ppg_2nd - np.min(ppg_2nd)) / (np.max(ppg_2nd) - np.min(ppg_2nd))


                SBP = np.mean(abp[i][peaks_abp])
                DBP = np.mean(abp[i][troughs_abp])

                # HR = 60 * fs_target / np.mean(np.diff(peaks_ppg))
                HR = (SBP - DBP) / (K_a -K_b)


                # RPTT, _, _, pir = 1, 1, 1, 1
                # f, mag = AIData.compute_fft(ppg[i], fs=fs_target)
                # mag = mag[:128]

                # RPTT, K_a, K_b = AIData.extract_features(ppg[i], SBP, DBP)

                # ppg[i] = (ppg[i] - np.min(ppg[i])) / (np.max(ppg[i]) - np.min(ppg[i]))

                feature = {
                    'ppg': AIData._float_feature(ppg[i].tolist()),
                    'abp': AIData._float_feature(abp[i].tolist()),
                    'ppg_template': AIData._float_feature(ppg_template.tolist()),
                    'ppg_1st': AIData._float_feature(ppg_1st.tolist()),
                    'ppg_2nd': AIData._float_feature(ppg_2nd.tolist()),
                    # 'abp_dc': AIData._float_feature(abp_dc[i].tolist()),
                    'ppg_raw': AIData._float_feature(ppg_2sd[i].tolist()),
                    'bp': AIData._float_feature([SBP, DBP]),
                    'HR': AIData._float_feature([HR]),
                    # 'RPTT': AIData._float_feature([RPTT]),
                    'calibrate': AIData._float_feature([K_a, K_b]),
                    # 'ppg_calibrate': AIData._float_feature(ppg[index_calib].tolist())
                    'ppg_calibrate': AIData._float_feature(ppg_calib.tolist())
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except Exception as e:
                continue

    @staticmethod
    def save_tf_record_4(writer, ppg, ppg_first, ppg_second, abp, abp_raw):
        calibrate, ppg_init, abp_int = AIData.init_calibrate(ppg, abp)
        for i in range(3, np.min([len(ppg), len(abp)])):
            # co_bl = np.polyfit(np.arange(len(abp[i])), abp[i], 20)
            # baseline_drift = np.polyval(co_bl, abp[i])
            # abp[i] = abp[i] - baseline_drift + np.mean(baseline_drift)

            if AIData.remove_bad_seg(ppg[i], abp[i]):
                # if AIData.remove_bad_abp(abp[i]):
                continue

            peaks_abp, _ = scipy.signal.find_peaks(abp[i], prominence=0.2 * 140, distance=0.1 * dfm.FS / 4)
            troughs, _ = scipy.signal.find_peaks(-abp[i], prominence=0.2 * 140, distance=0.1 * dfm.FS / 4)

            peaks_ppg, _ = scipy.signal.find_peaks(ppg[i], prominence=0.2, distance=0.1 * dfm.FS / 4)
            HR = 60 * dfm.FS / 4 / np.mean(np.diff(peaks_ppg))

            # def round_5(x, base=5):
            #     return int(base * round(x / base))
            #
            # SBP = round_5(abp[i][peaks_abp[-1]])
            # DBP = round_5(abp[i][troughs[-1]])
            # SBP = abp[i][peaks_abp[-1]]
            # DBP = abp[i][troughs[-1]]
            SBP = np.mean(abp[i][peaks_abp])
            DBP = np.mean(abp[i][troughs])

            # abp_norm = (abp[i] - DBP) / (SBP - DBP)
            abp_norm = (abp[i] - 50) / (190 - 50)
            # abp_norm = (ppg[i] - dfm.PPG_MIN) * (SBP - DBP) / (dfm.PPG_MAX - dfm.PPG_MIN) + DBP
            # SBP = np.max(abp_norm)
            # DBP = np.min(abp_norm)

            # ppg_norm = ppg[i]
            # ppg_norm = (ppg[i] - np.min(ppg[i])) / (np.max(ppg[i]) - np.min(ppg[i]))
            ppg_norm = (ppg[i] - dfm.PPG_MIN) / (dfm.PPG_MAX - dfm.PPG_MIN)
            # ppg_norm = (ppg[i] - dfm.PPG_MIN) * (SBP - DBP) / (dfm.PPG_MAX - dfm.PPG_MIN) + DBP
            # ppg_first_norm = (ppg_first[i] - np.min(ppg_first[i])) / (np.max(ppg_first[i]) - np.min(ppg_first[i]))
            # ppg_second_norm = (ppg_second[i] - np.min(ppg_second[i])) / (np.max(ppg_second[i]) - np.min(ppg_second[i]))
            ppg_first_norm = ppg_first[i]
            ppg_second_norm = ppg_second[i]

            # abp_raw_n = (abp_raw[i] - np.min(abp_raw[i])) / (np.max(abp_raw[i]) - np.min(abp_raw[i]))
            abp_raw_n = abp_raw[i]

            ppg_norm = np.concatenate((ppg_init, ppg_norm), axis=-1)
            ppg_first_norm = np.concatenate((ppg_first[:3].flatten(), ppg_first_norm), axis=-1)
            ppg_second_norm = np.concatenate((ppg_second[:3].flatten(), ppg_second_norm), axis=-1)
            tmp_abp = np.concatenate((abp_int, abp[i]), axis=-1)
            # abp[i] = tmp_abp
            abp_norm = abp_norm
            abp_raw_n = abp_raw_n

            feature = {
                'ppg': AIData._float_feature(ppg_norm.tolist()),
                'ppg_1st': AIData._float_feature(ppg_first_norm.tolist()),
                'ppg_2nd': AIData._float_feature(ppg_second_norm.tolist()),
                'abp': AIData._float_feature(abp[i].tolist()),
                # 'abp': AIData._float_feature(tmp_abp.tolist()),
                'abp_norm': AIData._float_feature(abp_norm.tolist()),
                'bp': AIData._float_feature([SBP, DBP]),
                'abp_raw': AIData._float_feature(abp_raw_n.tolist()),
                'HR': AIData._float_feature([HR]),
                'class': AIData._float_feature([dfm.CLASS_BP[AIData.classification_bp(SBP, DBP)]]),
                'calibrate': AIData._float_feature(calibrate.tolist())
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    @staticmethod
    def norm(sig, s_type='ppg'):
        if s_type == 'ppg':
            x_min = -2.4
            x_max = 2.9
            # x_min = dfm.PPG_MIN
            # x_max = dfm.PPG_MAX
        if s_type == 'abp':
            x_min = 50.33
            x_max = 192.24
        sig = (sig - x_min) / (x_max - x_min)
        # sig = (sig - np.mean(sig)) / np.std(sig)
        return sig

    @staticmethod
    def norm_v2(sig, s_type='ppg'):
        if s_type == 'ppg':
            x_min = -3.3351523864831822
            x_max = 3.6912145073278673
            sig = (sig - x_min) / (x_max - x_min)
            return sig
        if s_type == 'abp':
            x_min = 50.0
            x_max = 199.9776287527556
            sig = (sig - x_min) / (x_max - x_min)
            return sig

    @staticmethod
    def norm_v3(sig, s_type='ppg'):
        if s_type == 'ppg':
            x_min = 1.92
            x_max = -2.32
            sig = (sig - x_min) / (x_max - x_min)
            return sig
        if s_type == 'abp':
            x_min = 30
            x_max = 175
            sig = (sig - x_min) / (x_max - x_min)
            return sig

    @staticmethod
    def norm_v4(sig):
        x_min = np.min(sig)
        x_max = np.max(sig)
        sig = (sig - x_min) / (x_max - x_min)
        return sig

    @staticmethod
    def norm_v5(sig):
        u = np.mean(sig)
        v = np.std(sig)
        sig = (sig - u) / v
        return sig

    @staticmethod
    def norm_v6(sig, s_type='ppg'):
        if s_type == 'ppg':
            x_min = -2.4
            x_max = 2.9
        if s_type == 'abp':
            x_min = -50
            x_max = 100
        sig = (sig - x_min) / (x_max - x_min)
        return sig

    @staticmethod
    def corr_process(sig_ppg, sig_abp):
        x = 1000
        # ppg = butter_bandpass_filter(sig_ppg, 0.5, 8, 125, 2)
        # ppg = lowpass_remez(sig_ppg)
        ppg = butter_lowpass_filter(sig_ppg, cutoff=8, fs=125, order=4)
        ppg = butter_highpass_filter(ppg, cutoff=0.05, fs=125, order=4)

        # abp = butter_bandpass_filter(sig_abp, 0.5, 8, 125, 2)
        # _abp = lowpass_remez(sig_abp)
        # abp = lowpass_remez(sig_abp)
        abp = butter_lowpass_filter(sig_abp, cutoff=6.6, fs=125, order=4)
        # _abp = highpass_remez(abp)

        corr = np.correlate(a=ppg[x:-x], v=abp)
        lag = corr[(len(corr) // 2):].argmax()

        ppg = ppg[lag:]
        abp = sig_abp[:-lag]
        # ppg = highpass_remez(ppg)

        return ppg, abp

    @staticmethod
    def corr_process_2(sig_ppg, sig_abp, sig_ecg=None, fs=125):
        x = 1000

        ppg = butter_lowpass_filter(sig_ppg, cutoff=8, fs=fs, order=4)
        ppg_lp = ppg
        abp = sig_abp
        # abp = butter_lowpass_filter(sig_abp, cutoff=8, fs=125, order=4)

        if REMOVE_BASLINE_PPG:
            ppg = butter_highpass_filter(ppg, cutoff=0.5, fs=fs, order=4)

        if REMOVE_BASLINE_ABP:
            abp = butter_highpass_filter(abp, cutoff=0.5, fs=fs, order=3)

        if ADD_DC_PPG:
            ppg_dc = butter_lowpass_filter(sig_ppg, cutoff=0.5, fs=fs, order=4)
            ppg = ppg + np.mean(ppg_dc)

        if ADD_DC_ABP:
            abp_dc = butter_lowpass_filter(sig_abp, cutoff=0.5, fs=fs, order=4)
            abp = abp + np.mean(abp_dc)
        abp_dc = butter_lowpass_filter(sig_abp, cutoff=0.5, fs=fs, order=4)

        corr = np.correlate(a=ppg[x:-x], v=abp)
        lag = corr[(len(corr) // 2):].argmax()

        ppg2 = ppg[:-lag]
        ppg = ppg[lag:]
        abp = abp[:-lag]
        abp_dc = abp_dc[:-lag]
        ppg_lp = ppg_lp[lag:]

        # print('lag: {}'.format(lag))
        # ppg = highpass_remez(ppg)

        # return ppg, abp, ppg_dc, abp_dc
        # return sig_abp[lag:], abp, ppg_dc, abp_dc
        # return ppg, abp, sig_abp[lag:], sig_ppg[lag:]  # last
        return ppg, abp, abp_dc, ppg_lp
        # return ppg, abp, ppg2

    @staticmethod
    def process_data(path):
        path = path.rstrip('.hea').rstrip('.dat')
        rec = wf.rdsamp(path)
        ppg = rec[0][:, 1]
        abp = rec[0][:, 2]
        ecg = rec[0][:, 0]
        if len(ppg) < 10000 or len(abp) < 10000:
            return None, None
        ppg, abp, abp_raw, ppg_raw = AIData.corr_process_2(ppg, abp, ecg)

        # ppg = AIData.norm(ppg, s_type='ppg')
        # abp = AIData.norm(abp, s_type='abp')
        # ppg = AIData.norm_v6(ppg, s_type='ppg')
        # abp = AIData.norm_v6(abp, s_type='abp')
        # ppg = AIData.norm_v5(ppg)
        # abp /= 189.984

        return ppg, abp, abp_raw, ppg_raw

    def generate_tfrecord(self, file_name, ppg, abp):
        # assert np.array(dataset[0]).shape[0] == np.array(dataset[1]).shape[0]
        # if os.path.exists(path):
        #     os.remove(path)
        if ppg is not None:
            file_name = '{}.tfrecord'.format(file_name)
            with tf.io.TFRecordWriter(os.path.join(self.tf_record_dir, file_name)) as writer:
                # self.save_tf_record_2(writer, ppg, abp)
                self.save_tf_record(writer, ppg, abp)

    def generate_tfrecord_2(self, file_name, ppg, ppg_2sd, abp, abp_raw):
        # assert np.array(dataset[0]).shape[0] == np.array(dataset[1]).shape[0]
        # if os.path.exists(path):
        #     os.remove(path)
        if ppg is not None:
            file_name = '{}.tfrecord'.format(file_name)
            with tf.io.TFRecordWriter(os.path.join(self.tf_record_dir, file_name)) as writer:
                self.save_tf_record_3(writer, ppg, ppg_2sd, abp, abp_raw)

    def generate_tfrecord_3(self, file_name, ppg, ppg_first, ppg_second, abp, abp_raw, ecg):
        # assert np.array(dataset[0]).shape[0] == np.array(dataset[1]).shape[0]
        # if os.path.exists(path):
        #     os.remove(path)
        if ppg is not None:
            file_name = '{}.tfrecord'.format(file_name)
            with tf.io.TFRecordWriter(os.path.join(self.tf_record_dir, file_name)) as writer:
                # self.save_tf_record_2(writer, ppg, abp)
                self.save_tf_record_4(writer, ppg, ppg_first, ppg_second, abp, ecg)

    def multi_generate_tfrecord(self, dir_db):
        from multiprocessing import Pool
        import copy
        if not os.path.exists(self.tf_record_dir):
            os.mkdir(self.tf_record_dir)

        len_each_file = 40000  # 10000
        db = glob.glob(dir_db + '/*.hea')

        ppg = abp = np.empty([0, self.len_seg])
        ind = 0
        ind_file = 0
        num_overlap = 0
        while ind < len(db):
            tmp_ppg = tmp_abp = np.empty([0, self.len_seg])
            while len(ppg) < len_each_file:
                try:
                    # self.generate_tfrecord(db[ind])
                    sub_ppg, sub_abp = self.process_data(db[ind])
                    if sub_ppg is None:
                        print('skip processing {}/{}'.format(ind, len(db)))
                        ind += 1
                        continue
                    seg = np.arange(0, self.len_seg, 1)[None, :] + np.arange(0, len(sub_ppg) - self.len_seg,
                                                                             self.len_seg - num_overlap)[:, None]
                    seg = seg.astype(int)
                    x = [i for i in seg if (0.5 <= np.max(sub_ppg[i]) <= 1
                                            and 0 <= np.min(sub_ppg[i]) <= 0.5
                                            and 0.5 <= np.max(sub_abp[i]) <= 1
                                            and 0 <= np.min(sub_abp[i]) <= 0.5)]
                    seg = np.asarray(x).astype(int)
                    len_add = len_each_file - len(ppg)
                    if len_add >= len(seg):
                        print('processing {}/{}'.format(ind, len(db)))
                        ppg = np.concatenate((ppg, sub_ppg[seg]), axis=0)
                        abp = np.concatenate((abp, sub_abp[seg]), axis=0)
                        ind += 1
                    else:
                        print('processing {}/{}'.format(ind, len(db)))
                        ppg = np.concatenate((ppg, sub_ppg[seg[:len_add]]), axis=0)
                        abp = np.concatenate((abp, sub_abp[seg[:len_add]]), axis=0)
                        tmp_ppg = sub_ppg[seg[len_add:]]
                        tmp_abp = sub_abp[seg[len_add:]]
                except Exception as error:
                    print('--- error at {} - {}'.format(sys.exc_info()[-1].tb_lineno, error))
                    ind += 1
                    if not ind < len(db):
                        break
            self.generate_tfrecord('AI_data_{}'.format(ind_file), ppg, abp)
            ind_file += 1
            ppg = tmp_ppg
            abp = tmp_abp

    @staticmethod
    def derivative(ppg, abp):
        y_first = np.diff(ppg, n=1) * float(dfm.FS)
        y_second = np.diff(ppg, n=2) * float(dfm.FS)
        return ppg[2:], y_first[1:], y_second, abp[2:]

    def multi_generate_tfrecord_2(self, dir_db):
        from multiprocessing import Pool
        import copy
        if not os.path.exists(self.tf_record_dir):
            os.mkdir(self.tf_record_dir)

        # db = glob.glob(dir_db + '/*Part_1*.hea')
        db = glob.glob(dir_db + '/*.hea')
        # db = glob.glob(dir_db + '/*/*.hea')
        # db =glob.glob('/mnt/ai_data/PPG2ABP_data_second/pic_analyze_data' + '/*.png')

        with open(self.tf_record_dir + '/tfrecord_log.txt', 'w') as f:
            f.writelines('REMOVE_BASLINE_PPG: {}\n'.format(REMOVE_BASLINE_PPG))
            f.writelines('REMOVE_BASLINE_ABP: {}\n'.format(REMOVE_BASLINE_ABP))
            f.writelines('ADD_DC_PPG: {}\n'.format(ADD_DC_PPG))
            f.writelines('ADD_DC_ABP: {}\n'.format(ADD_DC_ABP))
            f.writelines('LEN_SAMPLE: {}\n'.format(input_length))
            f.writelines(
                'PPG_MAX | PPG_MIN | ABP_MIN | ABP_MAX: {}  {}  {}  {}\n'.format(PPG_MAX, PPG_MIN, ABP_MIN, ABP_MAX))
            f.writelines('DATE: {}\n'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
            f.writelines('VERSION: {}\n'.format(VER_DATA))

        for i, path in enumerate(db):
            try:
                # path = os.path.join(dir_db, os.path.basename(_path).rstrip('.png'))
                sub_ppg, sub_abp, sub_abp_raw, sub_ppg_raw = self.process_data(path)
                if sub_ppg is None:
                    raise Exception(
                        'skip processing {}/{}: {} - None'.format(i, len(db), os.path.basename(path).strip('.hea')))
                r = scipy.stats.pearsonr(sub_ppg, sub_abp)[0]
                if r < 0.85:
                    raise Exception(
                        'skip processing {}/{}: {} - r < 0.85'.format(i, len(db), os.path.basename(path).strip('.hea')))
                sub_ppg, ppg_first, ppg_second, sub_abp = self.derivative(sub_ppg, sub_abp)

                fs_target = dfm.FS_TARGET  # dfm.FS/4

                sub_ppg, _ = wfdb.processing.resample_sig(sub_ppg, dfm.FS, fs_target)
                sub_ppg_raw, _ = wfdb.processing.resample_sig(sub_ppg_raw, dfm.FS, fs_target)
                sub_abp, _ = wfdb.processing.resample_sig(sub_abp, dfm.FS, fs_target)
                sub_abp_raw, _ = wfdb.processing.resample_sig(sub_abp_raw, dfm.FS, fs_target)
                ppg_second, _ = wfdb.processing.resample_sig(ppg_second, dfm.FS, fs_target)
                # ppg_first, _ = wfdb.processing.resample_sig(ppg_first, dfm.FS, fs_target)

                # self.len_seg = self.len_seg//4
                # num_overlap = 192
                num_overlap = dfm.NUM_OVERLAP
                seg = np.arange(0, self.len_seg, 1)[None, :] + np.arange(0, len(sub_ppg) - self.len_seg,
                                                                         self.len_seg - num_overlap)[:, None]
                seg = seg.astype(int)
                # x = [i for i in seg if (0.5 <= np.max(sub_ppg[i]) <= 1
                #                         and 0 <= np.min(sub_ppg[i]) <= 0.5)]
                # seg = np.asarray(x).astype(int)
                # self.len_seg = self.len_seg * 4
                # if len(seg) < (30000 // 16) // self.len_seg:
                if len(seg) < (3000 // self.len_seg):
                    raise Exception(
                        'skip processing {}/{}: {} - Len ppg after process <30000'.format(i, len(db),
                                                                                          os.path.basename(path).strip(
                                                                                              '.hea')))
                print('processing {}/{}: {}'.format(i, len(db), os.path.basename(path).strip('.hea')))
                name_rd = os.path.basename(path).strip('.hea').split('_')
                self.generate_tfrecord_2('AI_data_part_{}_{}'.format(name_rd[-2], name_rd[-1]), sub_ppg[seg],
                                         sub_ppg_raw[seg],
                                         sub_abp[seg], sub_abp_raw[2:][seg])

            except Exception as error:
                print('--- {}'.format(error))
                # exc_type, exc_obj, exc_tb = sys.exc_info()
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(exc_type, fname, exc_tb.tb_lineno)

    def process_work(self, path, num, len_path):
        try:
            # sub_ppg_raw is sub_abp_dc
            sub_ppg, sub_abp, sub_abp_raw, sub_ppg_raw = self.process_data(path)
            if sub_ppg is None:
                raise Exception(
                    'skip processing {}/{}: {} - None'.format(num, len_path, os.path.basename(path).strip('.hea')))
            r = scipy.stats.pearsonr(sub_ppg, sub_abp)[0]
            if r < 0.6:  # 0.85:
                raise Exception(
                    'skip processing {}/{}: {} - r < 0.85'.format(num, len_path, os.path.basename(path).strip('.hea')))
            sub_ppg, ppg_first, ppg_second, sub_abp = self.derivative(sub_ppg, sub_abp)

            fs_target = dfm.FS_TARGET
            sub_ppg, _ = wfdb.processing.resample_sig(sub_ppg, dfm.FS, fs_target)
            sub_ppg_raw, _ = wfdb.processing.resample_sig(sub_ppg_raw, dfm.FS, fs_target)
            sub_abp, _ = wfdb.processing.resample_sig(sub_abp, dfm.FS, fs_target)
            sub_abp_raw, _ = wfdb.processing.resample_sig(sub_abp_raw, dfm.FS, fs_target)
            ppg_second, _ = wfdb.processing.resample_sig(ppg_second, dfm.FS, fs_target)

            # self.len_seg = self.len_seg//4
            # num_overlap = 192
            num_overlap = dfm.NUM_OVERLAP
            seg = np.arange(0, self.len_seg, 1)[None, :] + np.arange(0, len(sub_ppg) - self.len_seg,
                                                                     self.len_seg - num_overlap)[:, None]
            seg = seg.astype(int)

            if len(sub_ppg) < fs_target*8*60:
                raise Exception(
                    'skip processing {}/{}: {} - Len ppg after process < 8m'.format(num, len_path,
                                                                                      os.path.basename(path).strip(
                                                                                          '.hea')))
            print('processing {}/{}: {}'.format(num, len_path, os.path.basename(path).strip('.hea')))
            name_rd = os.path.basename(path).strip('.hea').split('_')
            # self.generate_tfrecord_2('AI_data_part_{}_{}'.format(name_rd[-2], name_rd[-1]), sub_ppg[seg],
            #                          sub_ppg_raw[seg],
            #                          sub_abp[seg], sub_abp_raw[seg])
            self.generate_tfrecord_2('AI_data_part_{}'.format(name_rd[0]), sub_ppg[seg],
                                     sub_ppg_raw[seg],
                                     sub_abp[seg], sub_abp_raw[seg])

        except Exception as error:
            # print('--- {}'.format(error))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def generate_multiprocess(self, dir_db):
        from multiprocessing import Pool
        os.makedirs(self.tf_record_dir, exist_ok=True)
        # db = glob.glob(dir_db + '/*Part_1*.hea')
        db = glob.glob(dir_db + '/*.hea')

        args = []
        for i, path in enumerate(db):
            args.append([path, i, len(db)])
        with Pool(os.cpu_count() - 4) as pool:
            pool.starmap(self.process_work, args)

    def debug_multiprocess(self, dir_db):
        os.makedirs(self.tf_record_dir, exist_ok=True)
        # db = glob.glob(dir_db + '/*Part_1*.hea')
        db = glob.glob(dir_db + '/*.hea')

        args = []
        for i, path in enumerate(db):
            args.append([path, i, len(db)])
            self.process_work(*args[i])

    @staticmethod
    def _parse_function(record_batch, len_seg):
        feature = {
            'ppg': tf.io.FixedLenFeature([len_seg, 1], tf.float32),
            'abp': tf.io.FixedLenFeature([len_seg, 1], tf.float32)
            # 'abp': tf.io.FixedLenFeature([2, 1], tf.float32)
        }
        example = tf.io.parse_example(record_batch, feature)
        example['ppg'] = tf.cast(example['ppg'], tf.float32)
        example['abp'] = tf.cast(example['abp'], tf.float32)
        return example['ppg'], example['abp']

    @staticmethod
    def _parse_function_2(record_batch, len_seg):
        feature = {
            'ppg': tf.io.FixedLenFeature([len_seg, 1], tf.float32),
            # 'abp': tf.io.FixedLenFeature([len_seg, 1], tf.float32)
            'abp': tf.io.FixedLenFeature([2, 1], tf.float32)
        }
        example = tf.io.parse_example(record_batch, feature)
        example['ppg'] = tf.cast(example['ppg'], tf.float32)
        example['abp'] = tf.cast(example['abp'], tf.float32)
        return example['ppg'][:768], (example['abp'][0], example['abp'][1])

    @staticmethod
    def _parse_function_3(record_batch, len_seg):
        feature = {
            'ppg': tf.io.FixedLenFeature([len_seg, 1], tf.float32),
            'abp': tf.io.FixedLenFeature([len_seg, 1], tf.float32),
            'ppg_raw': tf.io.FixedLenFeature([len_seg, 1], tf.float32),
            'ppg_template': tf.io.FixedLenFeature([256, 1], tf.float32),
            'ppg_1st': tf.io.FixedLenFeature([256, 1], tf.float32),
            'ppg_2nd': tf.io.FixedLenFeature([256, 1], tf.float32),

            'bp': tf.io.FixedLenFeature([2, 1], tf.float32),
            'HR': tf.io.FixedLenFeature([1, 1], tf.float32),
            # 'RPTT': tf.io.FixedLenFeature([1, 1], tf.float32),
            # 'PIR': tf.io.FixedLenFeature([1, 1], tf.float32),
            'calibrate': tf.io.FixedLenFeature([2, 1], tf.float32),
            'ppg_calibrate': tf.io.FixedLenFeature([256, 1], tf.float32),

        }
        example = tf.io.parse_example(record_batch, feature)
        example['ppg'] = tf.cast(example['ppg'], tf.float32)
        example['ppg_calibrate'] = tf.cast(example['ppg_calibrate'], tf.float32)
        example['abp'] = tf.cast(example['abp'], tf.float32)
        example['ppg_raw'] = tf.cast(example['ppg_raw'], tf.float32)
        example['bp'] = tf.cast(example['bp'], tf.float32)
        example['HR'] = tf.cast(example['HR'], tf.float32)
        # example['RPTT'] = tf.cast(example['RPTT'], tf.float32)
        # example['PIR'] = tf.cast(example['PIR'], tf.float32)
        example['calibrate'] = tf.cast(example['calibrate'], tf.float32)
        # custom = tf.concat([example['ppg'], example['ppg_calib']], axis=1)
        # data = (tf.reshape(custom, (256, 2, 1)), example['HR'])
        # data = (example['ppg'], example['HR'])
        # data = tf.concat([example['ppg'], example['ppg_calib']], axis=1)
        # data = tf.reshape(custom, (256, 2, 1))

        # data = example['ppg']
        # data = example['ppg_template']
        # data = (example['ppg_template'], example['ppg_calibrate'], tf.reshape(example['calibrate'], (2, )))

        data = tf.concat([example['ppg_template'], example['ppg_1st'], example['ppg_2nd']], axis=1)

        # data = tf.reshape(data, (len_seg, 1))
        # data = tf.reshape(data, (len_seg, 1, 1))
        # data = tf.concat([example['ppg_calibrate'], example['ppg']], axis=1)
        # data = tf.concat([example['ppg_calibrate'], example['ppg_template']], axis=0)
        # data = (tf.concat([example['ppg_calibrate'], example['ppg']], axis=1), tf.reshape(example['calibrate'], (2, )))
        # data_1 = tf.reshape(tf.concat([example['ppg_calibrate'], example['ppg']], axis=1), (len_seg, 2, 1))
        # data = (data_1, tf.reshape(example['calibrate'], (2, )))
        # data = (example['ppg'], example['ppg_calibrate'], tf.reshape(example['calibrate'], (2, )))
        # data = (tf.reshape(example['ppg'], (len_seg, 1, 1)),tf.reshape(example['ppg_calibrate'], (len_seg, 1, 1)), tf.reshape(example['calibrate'], (2, )))

        # data = (example['ppg'], tf.reshape(example['calibrate'], (2, )))
        # data = example['ppg']
        # data = tf.reshape(tf.concat([example['RPTT'], example['HR'], example['PIR'], example['calibrate']], axis=0), (5,))
        # data = (example['ppg'], tf.reshape(tf.concat([example['RPTT'], example['HR']], axis=0), (2,)))
        # data = tf.reshape(tf.concat([example['RPTT'], example['HR'], example['calibrate']], axis=0), (4,))
        # output = example['bp']
        # output = (example['bp'][0], example['bp'][1])
        # output = example['calibrate'][1]
        # output = example['RPTT']
        # output = example['abp']
        # output = example['HR']
        output = example['bp'][0]
        # output = tf.reshape(output, (len_seg, 1))

        # output = example['bp'][0] - example['calibrate'][0] + 120


        # tmp_sub = tf.abs(tf.subtract(example['bp'], example['calibrate']))
        # tmp_sub = tf.subtract(example['bp'], example['calibrate'])
        # output = (tmp_sub[0], tmp_sub[1])

        # tmp_sub = tf.divide(example['bp'], example['calibrate'])
        # output = tmp_sub

        return data, output

    @staticmethod
    def _parse_function_test_result(record_batch, len_seg):
        feature = {
            'ppg': tf.io.FixedLenFeature([len_seg, 1], tf.float32),
            # 'ppg_raw': tf.io.FixedLenFeature([len_seg, 1], tf.float32),
            'bp': tf.io.FixedLenFeature([2], tf.float32),
            'HR': tf.io.FixedLenFeature([1], tf.float32),
            'mag': tf.io.FixedLenFeature([128, 1], tf.float32)

        }
        example = tf.io.parse_example(record_batch, feature)
        example['ppg'] = tf.cast(example['ppg'], tf.float32)
        # example['ppg_raw'] = tf.cast(example['ppg_raw'], tf.float32)
        example['bp'] = tf.cast(example['bp'], tf.float32)
        example['HR'] = tf.cast(example['HR'], tf.float32)
        example['mag'] = tf.cast(example['mag'], tf.float32)

        # return (example['ppg'], example['calibrate'], example['ppg_cali']), (example['bp'][0], example['bp'][1])
        # return example['ppg'], example['calibrate'], example['HR'], example['bp']


        return example['ppg'], (example['bp'][0], example['bp'][1]), example['HR'], example['mag']

    @staticmethod
    def _parse_function_4(record_batch, len_seg):
        feature = {
            'ppg': tf.io.FixedLenFeature([len_seg, 1], tf.float32),
            'abp': tf.io.FixedLenFeature([len_seg, 1], tf.float32),
            'ppg_raw': tf.io.FixedLenFeature([len_seg, 1], tf.float32),
            'ppg_template': tf.io.FixedLenFeature([256, 1], tf.float32),
            'ppg_1st': tf.io.FixedLenFeature([256, 1], tf.float32),
            'ppg_2nd': tf.io.FixedLenFeature([256, 1], tf.float32),

            'bp': tf.io.FixedLenFeature([2, 1], tf.float32),
            'HR': tf.io.FixedLenFeature([1, 1], tf.float32),
            # 'RPTT': tf.io.FixedLenFeature([1, 1], tf.float32),
            # 'PIR': tf.io.FixedLenFeature([1, 1], tf.float32),
            'calibrate': tf.io.FixedLenFeature([2, 1], tf.float32),
            'ppg_calibrate': tf.io.FixedLenFeature([256, 1], tf.float32),

        }
        example = tf.io.parse_example(record_batch, feature)
        example['ppg'] = tf.cast(example['ppg'], tf.float32)
        example['abp'] = tf.cast(example['abp'], tf.float32)
        # example['ppg_raw'] = tf.cast(example['ppg_raw'], tf.float32)
        example['bp'] = tf.cast(example['bp'], tf.float32)
        example['HR'] = tf.cast(example['HR'], tf.float32)
        # example['RPTT'] = tf.cast(example['RPTT'], tf.float32)
        example['calibrate'] = tf.cast(example['calibrate'], tf.float32)

        # return (example['ppg'], example['calibrate'], example['ppg_cali']), (example['bp'][0], example['bp'][1])
        # return example['ppg'], example['calibrate'], example['HR'], example['bp']
        # data_1 = tf.reshape(tf.concat([example['ppg_calibrate'], example['ppg']], axis=1), (len_seg, 2, 1))
        # data = (data_1, tf.reshape(example['calibrate'], (2, )))
        # data = (tf.reshape(example['ppg'], (len_seg, 1, 1)),tf.reshape(example['ppg_calibrate'], (len_seg, 1, 1)), tf.reshape(example['calibrate'], (2, )))
        # data = tf.concat([example['ppg_calibrate'], example['ppg']], axis=1)
        # data = (tf.concat([example['ppg_calibrate'], example['ppg']], axis=1), tf.reshape(example['calibrate'], (2, )))

        # data = tf.concat([example['ppg_calibrate'], example['ppg_template']], axis=0)
        # data = example['ppg_template']
        # data = example['ppg_template.'] * (example['calibrate'][0] - example['calibrate'][1]) + example['calibrate'][1]
        data = tf.concat([example['ppg_template'], example['ppg_1st'], example['ppg_2nd']], axis=1)
        # data = (example['ppg_template'], example['ppg_calibrate'], tf.reshape(example['calibrate'], (2, )))

        # tmp_sub = tf.subtract(example['bp'], example['calibrate'])
        # output = (tmp_sub[0], tmp_sub[1])
        # tmp_sub = tf.divide(example['bp'], example['calibrate'])
        # output = (tmp_sub[0], tmp_sub[1])

        example['HR'] = example['bp'][0]

        return example['ppg'], (example['bp'][0], example['bp'][1]), example['abp'], tf.reshape(example['calibrate'], (2, )), data, example['HR']
        # return example['ppg'], output, example['abp'], tf.reshape(example['calibrate'], (2, )), data


    def get_dataset_from_tfrecord(self, files, batch_size):
        ds = tf.data.TFRecordDataset(files)
        map_function = partial(self._parse_function_3, len_seg=self.len_seg)
        ds = ds.map(map_function, num_parallel_calls=os.cpu_count())
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def get_dataset_from_tfrecord_test_result(self, files, batch_size):
        ds = tf.data.TFRecordDataset(files)
        map_function = partial(self._parse_function_test_result, len_seg=self.len_seg)
        ds = ds.map(map_function, num_parallel_calls=os.cpu_count())
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def get_dataset_from_tfrecord_test_result_4(self, files, batch_size):
        ds = tf.data.TFRecordDataset(files)
        map_function = partial(self._parse_function_4, len_seg=self.len_seg)
        ds = ds.map(map_function, num_parallel_calls=os.cpu_count())
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def get_numpy_from_tfrecord(self, files, batch_size):
        ppg, abp = [], []
        ds = self.get_dataset_from_tfrecord(files, batch_size)
        for index, batch in enumerate(ds):
            list_tensors = [i for i in batch]
            ppg.extend([tensor.numpy().flatten() for tensor in list_tensors[0]])
            abp.extend([tensor.numpy().flatten() for tensor in list_tensors[1]])
        ppg = np.asarray(ppg)
        abp = np.asarray(abp)
        return ppg, abp

    def get_numpy_from_tfrecord_2(self, files, batch_size):
        ppg, abp = [], []
        ds = self.get_dataset_from_tfrecord(files, batch_size)
        for index, batch in enumerate(ds):
            list_tensors = [i for i in batch]
            ppg.extend([tensor.numpy() for tensor in list_tensors[0]])
            abp.extend([tensor.numpy().flatten() for tensor in list_tensors[1]])
        ppg = np.asarray(ppg)
        abp = np.asarray(abp)
        return ppg, abp
