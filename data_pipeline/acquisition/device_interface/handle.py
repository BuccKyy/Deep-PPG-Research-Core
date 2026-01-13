import datetime
import glob
import os
import numpy as np
import wfdb
import wfdb.processing
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from cross_correlation.preprocessing import smooth
import model.define_model as dfm
import scipy.signal

# from ITR_device.test_ppg_device import read_dat
from Steam_PPG.utils import butter_bandpass_filter, butter_highpass_filter, butter_lowpass_filter
from Steam_PPG.define import PPG_GAIN, FS_DEVICE, FS_PROCESS, START_BYTE, LEN_SEG, NUM_OVERLAP, NUM_CH, THR_SQI, \
    THR_STD, THR_KURTOSIS, THR_SKEWNESS, CHANNELS_COLOR, CHANNELS_LABEL, THR_TOTAL_SQI
from ITR_device.process_ppg_device import HandlePPG as HP

def read_dat(filePath, num_bytes=3, start_byte=512, byteorder='big', signed=False):
    with open(filePath, "rb") as f:
        file = f.read()[start_byte:]
        convert = lambda byte: np.int.from_bytes(byte, byteorder=byteorder, signed=signed)
        signal = [convert(file[i:i + num_bytes]) for i in range(0, len(file), num_bytes)]

        signal = np.asarray(signal)
        # signal_1 = np.bitwise_and(signal, 0x3FFF)
    return signal

class HandlePPG:
    def __init__(self, num_ch=NUM_CH, fs_raw=FS_DEVICE, fs=FS_PROCESS, len_seg=LEN_SEG, num_overlap=NUM_OVERLAP):
        self.num_ch = num_ch
        self.fs_raw = fs_raw
        self.fs = fs
        self.len_seg = len_seg
        self.num_overlap = num_overlap

    def detect_good_sqi(self, ppg):
        ppg, _ = wfdb.processing.resample_sig(ppg, self.fs_raw, self.fs)
        std = np.std(ppg)
        skewness = np.mean(np.power((ppg - np.mean(ppg)), 3)) / np.power(std, 3)
        kurt = np.mean(np.power((ppg - np.mean(ppg)), 4)) / np.power(std, 4)
        title = 'skew: {} | kurt: {} | std: {}'.format(round(skewness, 3), round(kurt, 3), round(std, 3))
        print(title)

        if (std > THR_STD) or (skewness > THR_SKEWNESS) or (kurt > THR_KURTOSIS):
            return False
        return True

    def analyze_sqi(self, file):
        # define thr: skew: <-0.5   | kurt: <2.2 | std: <0.015

        ppg = np.load(file).reshape((-1, self.num_ch)) / PPG_GAIN
        ppg = -butter_bandpass_filter(ppg.T, 0.5, 10, self.fs_raw, 2).T

        time_seg = 8 * self.fs
        ppg_green = ppg[:, 1]
        ppg_green, _ = wfdb.processing.resample_sig(ppg_green, self.fs_raw, self.fs)
        segs = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_green) - time_seg,
                                                             time_seg - self.num_overlap)[:, None]
        ppg_segs = ppg_green[segs]
        result_sqi = None
        for num, ppg_seg in enumerate(ppg_segs):
            std = np.std(ppg_seg)
            skewness = np.mean(np.power((ppg_seg - np.mean(ppg_seg)), 3)) / np.power(std, 3)
            kurtoris = np.mean(np.power((ppg_seg - np.mean(ppg_seg)), 4)) / np.power(std, 4)
            sqis = np.asarray([num, round(skewness, 3), round(kurtoris, 3), np.round(std, 3)])
            result_sqi = np.vstack([result_sqi, sqis]) if result_sqi is not None else sqis
            title = 'skew: {} | kurt: {} | std: {}'.format(round(skewness, 3), round(kurtoris, 3), round(std, 3))
            print(title)
            plt.cla()
            plt.title(title)
            plt.plot(ppg_seg)
            plt.gcf().set_size_inches(10, 6)
            # plt.show()
            # path_save = '/mnt/ai_data/ABPM_FW/test_stream/pic_test_st_ref'
            path_save = '/mnt/ai_data/ABPM_FW/An/pic_sqi'
            fig_path = os.path.join(path_save,
                                    '{}.png'.format(num))
            plt.savefig(fig_path, dpi=100)

        df = pd.DataFrame(result_sqi)
        df.columns = ['num', 'skew', 'kurt', 'std']
        df.to_excel('/mnt/ai_data/ABPM_FW/test_stream/{}.xlsx'.format('compute_sqi'))
        a = 10

    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    @staticmethod
    def compute_fft(signal, fs=128):
        x = fft(signal)
        f_resolution = fs / len(signal)
        X_magnitude = np.abs(x)[:len(signal) // 2]
        _x = f_resolution * np.arange(len(x) // 2)
        return _x, X_magnitude

    def compute_ai_rptt(self, ppg, seg_accept, s_id):
        df = pd.read_excel(dfm.LABEL_PATH, index_col=None, header=1)
        df = df.fillna(method='ffill')
        key = df.columns.tolist()
        df.rename(columns={'Unnamed: 0': 'ID', 'Unnamed: 1': 'Name', 'Unnamed: 2': 'Gender', 'Unnamed: 3': 'Age',
                           'Unnamed: 4': 'STT', 'Unnamed: 5': 'Date', 'Unnamed: 6': 'Time'}, inplace=True)
        data = df[['ID', 'SBP', 'DBP', 'PUL', 'Gender', 'Age']]
        sub_label = data.query('ID == {}'.format(int(s_id)))
        SD_label = sub_label.iloc[:, 1:3].to_numpy()
        time_line_label = np.arange(3, 18, 3) * 60

        ppg_green, _ = wfdb.processing.resample_sig(ppg[:, 1], self.fs_raw, self.fs)
        time_seg = 8*128

        numoverlap = self.num_overlap
        segs = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_green) - time_seg,
                                                                       time_seg - numoverlap)[:, None]
        # ppg_ai = (ppg_green - np.min(ppg_green)) / (np.max(ppg_green) - np.min(ppg_green))
        ppg_ai = ppg_green  # * 10
        # ppg_ai = -np.flip(ppg_ai)

        ppg_ai = ppg_ai[segs[seg_accept[:-3]]]

        ppg_hr = ppg_ai.copy()
        hr = []
        for signal in ppg_hr:
            peaks = scipy.signal.find_peaks(signal, prominence=0.005, distance=0.1 * self.fs)[0]
            if len(peaks) < 3:
                continue
            sub_hr = 60*128 / np.mean(np.diff(peaks))
            hr.extend([sub_hr])
            a = 10
        hr = np.mean(np.asarray(hr), dtype=int)
        # ppg_ai = ppg_ai.reshape(-1, 256)
        # kur = np.mean(np.power(ppg_ai-np.mean(ppg_ai, axis=1)[:, None], 4), axis=1) / np.power(np.std(ppg_ai, axis=1), 4)
        # skew = np.mean(np.power(ppg_ai-np.mean(ppg_ai, axis=1)[:, None], 3), axis=1) / np.power(np.std(ppg_ai, axis=1), 3)
        # ppg_ai = (ppg_ai - np.min(ppg_ai, axis=1)[:, None] )/ (np.max(ppg_ai, axis=1)[:, None] - np.min(ppg_ai, axis=1)[:, None])
        ppg_ai = ppg_ai[:, :, np.newaxis, np.newaxis]

        sd_init = SD_label[0, :]
        sd_0 = np.concatenate([sd_init] * len(ppg_ai)).reshape(-1, len(sd_init))
        import tensorflow as tf
        predict = None
        ver = 4
        if ver == 3:
            from model.ann_model import cnn_4_0_5_5, cnn_test_3_25_1, cnn_4_0_8_0

            ai_model = cnn_test_3_25_1()
            ai_model.load_weights(tf.train.latest_checkpoint('/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.25_2_1.0.0.0/checkpoint')).expect_partial()
            predict = ai_model.predict([ppg_ai, sd_0])
        else:
            ai_model = tf.keras.models.load_model(os.path.join(dfm.WORKSPACE, 'save_model_fine_tune'))
            predict = ai_model.predict(ppg_ai)

            sd_init = np.mean(SD_label, axis=0, dtype=int)

            PP_0 = sd_init[0] - sd_init[1]
            MBP_0 = sd_init[0]/3 + 2*sd_init[1]/3

            rptt_init = predict  # [:len(predict) // 5]
            rptt_alpha = np.mean(rptt_init - np.mean(rptt_init)) / np.sum(np.abs(rptt_init - np.mean(rptt_init)))
            RPTT_0 = np.mean(rptt_init) * (1 - rptt_alpha)

            rptt = predict

            DBP_v2 = MBP_0 + 2/0.031*np.log(RPTT_0/rptt) - PP_0/3*np.power(RPTT_0/rptt, 2)
            SBP_v2 = DBP_v2 + PP_0*np.power(RPTT_0/rptt, 2)

            ind_true = np.ones_like(SBP_v2)
            s_fail = np.flatnonzero((SBP_v2 < 80) | (SBP_v2 > 180))
            d_fail = np.flatnonzero((DBP_v2 < 55) | (DBP_v2 > 100))
            sd_fail = np.flatnonzero((SBP_v2 - DBP_v2 < 35) | (SBP_v2 -DBP_v2 > 100))
            fail = np.unique(np.concatenate((s_fail, d_fail)))
            ind_true[fail] = 0
            # print(np.mean(predict))
            me = [round(np.mean(SBP_v2) - np.mean(SD_label[:, 0]), 3), round(np.mean(DBP_v2) - np.mean(SD_label[:, 1]), 2)]
            print(np.mean(SBP_v2) - np.mean(SD_label[:, 0]))
            print(np.mean(DBP_v2) - np.mean(SD_label[:, 1]))
            result = [np.mean(SBP_v2), np.mean(DBP_v2), np.mean(SD_label[:, 0]), np.mean(SD_label[:, 1]), hr, np.mean(sub_label.iloc[:, 3].to_numpy())]
            ind = np.flatnonzero(ind_true)
            plot = False

            if len(ind) > 30:
                # np.save('/mnt/ai_data/ABPM_FW/PPG_BP_8s_txt/{}.txt'.format(s_id), ppg_ai[ind, :, 0, 0])
                sd_label = []

                for i in range(0, 900, 180):
                    idx = np.flatnonzero(((seg_accept[ind] > round(i/8)) & (seg_accept[ind] <= round((i+180)/8))))
                    sd_label.extend([SD_label[i//180]]*len(idx))

                # sd_label = np.concatenate((sd_label), axis=0, dtype=int)
                sd_label = np.asarray(sd_label).astype(int)
                if len(ind) > len(sd_label):
                    sd_label = np.concatenate((sd_label, [sd_label[-1, :]]*(len(ind) - len(sd_label))), axis=0)
                elif len(ind) < len(sd_label):
                    sd_label = sd_label[:len(ind)]
                result_bp = np.hstack((sd_label, SBP_v2[ind], DBP_v2[ind], seg_accept[ind, None]))
                result_bp = np.round(result_bp).astype(int)
                np.save('/mnt/ai_data/ABPM_FW/PPG_npy/{}.npy'.format(s_id), result_bp)
                # np.savetxt('/mnt/ai_data/ABPM_FW/PPG_BP_8s_txt/{}.txt'.format(s_id), ppg_ai[ind, :, 0, 0].flatten(), delimiter='\r\n', fmt='%1.17e')
                # np.savetxt('/mnt/ai_data/ABPM_FW/Result_BP_2/{}.txt'.format(s_id), result_bp, fmt='%d')
                result = [int(s_id), PP_0, round(MBP_0), RPTT_0]
                # with open("/mnt/ai_data/ABPM_FW/ppg8s.txt", "ab") as f:
                #     f.write(b"\n")
                #     np.savetxt(f, np.asarray(result))
                print('******************* {}'.format(result))

            # plt.plot(SBP_v2, label='SBP_predict')
            # plt.plot(DBP_v2, label='DBP_predict')
            # plt.ylim(40, 150)
            # plt.legend(loc='upper right')
            # plt.title('{}'.format(s_id))
            # plt.show()
                plot = True


            if plot:
                # ind = np.flatnonzero(std_rptt < 0.08)

                time_line = np.arange(0, 8, 8)[None, :] + seg_accept[:-3, None]*8
                time_line = time_line.flatten()

                plt.clf()

                plt.plot(time_line[ind], SBP_v2[ind], '*--', label='SBP_predict')
                plt.plot(time_line_label, SD_label[:, 0], 'o-', label='SBP_actual')
                plt.plot(time_line[ind], DBP_v2[ind], '*--', label='DBP_predict')
                plt.plot(time_line_label, SD_label[:, 1], 'o-', label='DBP_actual')
                plt.ylim(50, 140)
                plt.xlabel('Time [s]')
                plt.ylabel('Blood Pressure [mmHg]')

                # plt.title('ID: {} {}\n{}\n{}'.format(s_id, len(SBP_v2), [np.mean(SBP_v2), np.mean(DBP_v2)], np.mean(SD_label, axis=0)))
                plt.title('ID: {} \n'.format(s_id))
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.gcf().set_size_inches(18, 8)

                # plt.show()

                fig_path = '/mnt/ai_data/ABPM_FW/fig_ai_ptt_3/' + '{}.png'.format(s_id)
                plt.savefig(fig_path, dpi=100)

            return result

    def compute_ai(self, ppg, seg_accept, s_id):
        df = pd.read_excel(dfm.LABEL_PATH, index_col=None, header=1)
        df = df.fillna(method='ffill')
        key = df.columns.tolist()
        df.rename(columns={'Unnamed: 0': 'ID', 'Unnamed: 1': 'Name', 'Unnamed: 2': 'Gender', 'Unnamed: 3': 'Age',
                           'Unnamed: 4': 'STT', 'Unnamed: 5': 'Date', 'Unnamed: 6': 'Time'}, inplace=True)
        data = df[['ID', 'SBP', 'DBP', 'PUL', 'Gender', 'Age']]
        sub_label = data.query('ID == {}'.format(int(s_id)))
        SD_label = sub_label.iloc[:, 1:3].to_numpy()
        time_line_label = np.arange(3, 18, 3) * 60

        ppg_green, _ = wfdb.processing.resample_sig(ppg[:, 1], self.fs_raw, self.fs)
        time_seg = self.len_seg

        numoverlap = self.num_overlap
        segs = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_green) - time_seg,
                                                              time_seg - numoverlap)[:, None]
        # ppg_ai = (ppg_green - np.min(ppg_green)) / (np.max(ppg_green) - np.min(ppg_green))
        ppg_ir, _ = wfdb.processing.resample_sig(ppg[:, 2], self.fs_raw, self.fs)
        ppg_ai = ppg_green
        # ppg_ai = np.column_stack((ppg_ai, ppg_ir))

        ppg_ai = ppg_ai[segs[seg_accept[:-3]]]
        # ppg_ai = ppg_green  # * 10
        # ppg_ai = -np.flip(ppg_ai)

        # ppg_ai = ppg_ai[segs[seg_accept[:-3]]]

        ppg_hr = ppg_green[segs[seg_accept[:-3]]]
        hr = []
        for signal in ppg_hr:
            peaks = scipy.signal.find_peaks(signal, prominence=0.005, distance=0.1 * self.fs)[0]
            if len(peaks) < 3:
                continue
            sub_hr = 60 * self.fs / np.mean(np.diff(peaks))
            hr.extend([sub_hr])

        hr = np.mean(np.asarray(hr), dtype=int)
        # ppg_ai = ppg_ai.reshape(-1, 256)
        # kur = np.mean(np.power(ppg_ai-np.mean(ppg_ai, axis=1)[:, None], 4), axis=1) / np.power(np.std(ppg_ai, axis=1), 4)
        # skew = np.mean(np.power(ppg_ai-np.mean(ppg_ai, axis=1)[:, None], 3), axis=1) / np.power(np.std(ppg_ai, axis=1), 3)
        # ppg_ai = (ppg_ai - np.min(ppg_ai, axis=1)[:, None] )/ (np.max(ppg_ai, axis=1)[:, None] - np.min(ppg_ai, axis=1)[:, None])

        ppg_ai = ppg_ai[:, :, np.newaxis, np.newaxis]
        # ppg_ai = ppg_ai[:, :, :, np.newaxis]

        sd_init = SD_label[0, :]
        sd_0 = np.concatenate([sd_init] * len(ppg_ai)).reshape(-1, len(sd_init))
        import tensorflow as tf
        predict = None
        ver = 4
        if ver == 3:
            from model.ann_model import cnn_4_0_5_5, cnn_test_3_25_1, cnn_4_0_8_0

            ai_model = cnn_test_3_25_1()
            ai_model.load_weights(tf.train.latest_checkpoint(
                '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.25_2_1.0.0.0/checkpoint')).expect_partial()
            predict = ai_model.predict([ppg_ai, sd_0])
        else:
            ai_model = tf.keras.models.load_model(os.path.join(dfm.WORKSPACE, 'save_model_fine_tune'))
            info = sub_label[['Gender', 'Age']].iloc[0, :].to_list()
            if info[0] == 'Male':
                info[0] = 0
            else:
                info[0] = 1
            info = np.concatenate([info] * len(ppg_ai)).reshape(-1, 2)
            info = info.astype(float)

            SD_INFO = np.mean(SD_label, axis=1)
            PP = SD_INFO[0] - SD_INFO[1]
            MAP = (SD_INFO[0] + 2 * SD_INFO[1]) / 3
            SD_info = [PP, MAP]
            SD_info = np.concatenate([SD_info] * len(ppg_ai)).reshape(-1, 2).astype(float)

            info = np.concatenate((info, SD_info), axis=1)

            predict = ai_model.predict((ppg_ai, SD_info))
            # predict = ai_model.predict(ppg_ai)

            DBP_v2 = predict[1]
            SBP_v2 = predict[0]

            # print(np.mean(predict))
            me = [round(np.mean(SBP_v2) - np.mean(SD_label[:, 0]), 3), round(np.mean(DBP_v2) - np.mean(SD_label[:, 1]), 2)]
            print(np.mean(SBP_v2) - np.mean(SD_label[:, 0]))
            print(np.mean(DBP_v2) - np.mean(SD_label[:, 1]))
            result = [np.mean(SBP_v2), np.mean(DBP_v2), np.mean(SD_label[:, 0]), np.mean(SD_label[:, 1]), hr, np.mean(sub_label.iloc[:, 3].to_numpy())]
            plot = True
            if plot:
                # ind = np.flatnonzero(std_rptt < 0.08)

                time_line = np.arange(0, 8, 8)[None, :] + seg_accept[:-3, None]*8
                time_line = time_line.flatten()

                plt.plot(time_line, SBP_v2, '*--', label='SBP_predict')
                plt.plot(time_line_label, SD_label[:, 0], 'o-', label='SBP_actual')
                plt.plot(time_line, DBP_v2, '*--', label='DBP_predict')
                plt.plot(time_line_label, SD_label[:, 1], 'o-', label='DBP_actual')
                plt.ylim(50, 150)
                plt.xlabel('Time [s]')
                plt.ylabel('Blood Pressure [mmHg]')

                # plt.title('ID: {} {}\n{}\n{}'.format(s_id, len(SBP_v2), [np.mean(SBP_v2), np.mean(DBP_v2)], np.mean(SD_label, axis=0)))
                plt.title('ID: {} \nME: {}'.format(s_id, me))
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.gcf().set_size_inches(18, 8)

                plt.show()
            return result

    def extract_data(self, file, plot_flag=False):
        ppg = np.load(file).reshape((-1, self.num_ch)) / PPG_GAIN

        ppg = -butter_bandpass_filter(ppg.T, 0.5, 10, self.fs_raw, 2).T
        ppg_green = ppg[:, 1]

        acceptable, percent_ppg = self.calculate_sqi_green(ppg_green)

        ppg_green, _ = wfdb.processing.resample_sig(ppg_green, self.fs_raw, self.fs)
        time_seg = 8 * self.fs

        numoverlap = self.num_overlap * self.fs // self.fs
        segs = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_green) - time_seg,
                                                                       time_seg - numoverlap)[:, None]
        ppg_ai = ppg_green  # / 0.06
        ppg_ir, _ = wfdb.processing.resample_sig(ppg[:, 2], self.fs_raw, self.fs)

        # ppg_ai = np.column_stack((ppg_ai, ppg_ir))

        ppg_ai = ppg_ai[segs[acceptable[3:-3]]]

        # ppg_ai = (ppg_ai - np.min(ppg_ai, axis=1)[:, None]) / (np.max(ppg_ai, axis=1)[:, None] - np.min(ppg_ai, axis=1)[:, None])

        return ppg_ai[:, :, np.newaxis, np.newaxis]
        # return ppg_ai[:, :, :, np.newaxis]

    def extract_data_rptt(self, file, plot_flag=False):
        ppg = np.load(file).reshape((-1, self.num_ch)) / PPG_GAIN

        ppg = -butter_bandpass_filter(ppg.T, 0.5, 10, self.fs_raw, 2).T
        ppg_green = ppg[:, 1]
        acceptable, percent_ppg = self.calculate_sqi_green(ppg_green)
        if percent_ppg < THR_TOTAL_SQI:
            return [], []

        ppg_green, _ = wfdb.processing.resample_sig(ppg_green, self.fs_raw, self.fs)
        time_seg = 8 * self.fs

        numoverlap = self.num_overlap * self.fs // self.fs
        segs = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_green) - time_seg,
                                                                       time_seg - numoverlap)[:, None]
        ppg_ai = ppg_green  # / 0.06

        ppg_ai = ppg_ai[segs[acceptable[:-3]]]
        total_rptt = []
        for sp in ppg_ai:
            rptt = HP.feature_ppg_fw_bw_continous(sp, self.fs, plot_flag=False)
            total_rptt.append(rptt)
        total_rptt = np.asarray(total_rptt)
        ind = np.flatnonzero(total_rptt != -1)
        mean = np.median(total_rptt[ind])
        ind_change = np.flatnonzero(total_rptt == -1)
        # total_rptt[ind_change] = mean
        total_rptt[ind] = mean

        # ppg_ai = (ppg_ai - np.min(ppg_ai, axis=1)[:, None]) / (np.max(ppg_ai, axis=1)[:, None] - np.min(ppg_ai, axis=1)[:, None])
        return ppg_ai[ind, :, np.newaxis, np.newaxis], total_rptt[ind, np.newaxis]


    def fft_red_ir(self, ppg, path, acceptable):
        ppg_fil = -butter_bandpass_filter(ppg.T, 0.5, 10, self.fs_raw, 2).T

        time_seg = 8 * self.fs_raw

        segs = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_fil) - time_seg,
                                                             time_seg - 0)[:, None]

        segs = segs[acceptable]
        s_id = os.path.basename(path).rstrip('.npy')


        y_lim = 2000




        for seg in segs:
            fig = plt.figure(figsize=(16, 9))
            fig.suptitle('FFT\nSubject ID: {}'.format(s_id))
            ax = []
            for i in range(6):
                ax.append(fig.add_subplot(3, 2, i + 1))
            for i_raw in range(0, 6, 2):
                ch = i_raw // 2

                signal = ppg[seg, ch]
                x = fft(signal)
                f_resolution = 256 / len(signal)
                X_magnitude = np.abs(x)[:len(signal) // 2]
                _x = f_resolution * np.arange(len(x) // 2)

                ax[i_raw].cla()
                ax[i_raw].plot(_x, X_magnitude, color=CHANNELS_COLOR[ch], label=CHANNELS_LABEL[ch])
                ax[i_raw].grid()
                ax[i_raw].set_ylim(-1, 10)

                ax[i_raw].set_xlim(-1, 20)

                ax[i_raw].set_ylabel('FFT Values')
                ax[i_raw].legend(loc='upper right')

                signal = ppg_fil[seg, ch]
                x = fft(signal, norm="ortho")
                f_resolution = 256 / len(signal)
                X_magnitude = np.abs(x)[:len(signal) // 2]
                _x = f_resolution * np.arange(len(x) // 2)

                i = i_raw + 1
                ax[i].cla()
                ax[i].plot(_x, X_magnitude, color=CHANNELS_COLOR[ch], label=CHANNELS_LABEL[ch])
                ax[i].grid()
                ax[i].set_xlim(-1, 20)
                ax[i].legend(loc='upper right')

            ax[0].set_title('RAW')
            ax[1].set_title('FILTER')
            ax[4].set_xlabel('Frequency [Hz]')
            ax[5].set_xlabel('Frequency [Hz]')
            dst = '/mnt/ai_data/ABPM_FW/collect_itr/pics'
            name_file = '{}.png'.format(s_id)
            path_save = os.path.join(dst, name_file)
            # plt.savefig(path_save, dpi=100)
            plt.show()

    def analyze(self, file, plot_flag=False):
        try:

            s_id = os.path.basename(file).rstrip('.npy')
            # s_id = '001'
            print(s_id)

            ppg = np.load(file).reshape((-1, self.num_ch)) / PPG_GAIN
            ppg_raw = ppg

            ppg = -butter_bandpass_filter(ppg.T, 0.5, 10, self.fs_raw, 2).T
            # ppg = -butter_bandpass_filter(ppg.T, 0.5, 3, self.fs_raw, 2).T
            ppg_green = ppg[:, 1]
            # ppg_green = smooth(ppg_green, window_len=25)

            acceptable, percent_ppg = self.calculate_sqi_green(ppg_green)
            print(percent_ppg)


            # if percent_ppg < THR_TOTAL_SQI:
            #     print('Fail SQI < {}%'.format(THR_TOTAL_SQI))
            #     return -1

            # info_path = os.path.join(os.path.dirname(file), 'calib_led_current.txt')
            # info = pd.read_csv(info_path, sep=" ", header=None).iloc[:, 2].to_numpy()
            # ratio_led = info[0]/info[2]
            # spo2 = self.compute_spo2(ppg_raw, acceptable, ratio_led=1)
            # bp = self.compute_ai(ppg, acceptable, s_id)
            # bp = self.compute_ai_rptt(ppg, acceptable, s_id)

            ppg_green, _ = wfdb.processing.resample_sig(ppg_green, self.fs_raw, self.fs)
            time_seg = self.len_seg

            seg = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_green) - time_seg,
                                                                 time_seg - self.num_overlap)[:, None]


            # print(int(percent_ppg/100*len(ppg_green)/self.fs))
            # print(int(len(ppg_green)/self.fs))
            title = 'SQI: {}%'.format(percent_ppg)

            # self.calculate_sqi_red_ir(ppg)
            # self.fft_red_ir(ppg_raw, file, acceptable)
            self.plot_good_signal(ppg_green, seg, acceptable, s_id)

            self.plot_good_signal_3CH(ppg, s_id=s_id)
            # self.plot_good_signal_3CH(ppg, s_id=s_id)
            df = pd.read_excel(dfm.LABEL_PATH, index_col=None, header=1)
            df = df.fillna(method='ffill')
            key = df.columns.tolist()
            df.rename(columns={'Unnamed: 0': 'ID', 'Unnamed: 1': 'Name', 'Unnamed: 2': 'STT', 'Unnamed: 3': 'Date',
                               'Unnamed: 4': 'Time'}, inplace=True)
            data = df[['ID', 'SPO2']]
            spo2_label = data.query('ID == {}'.format(int(s_id)))
            spo2_label = spo2_label.iloc[:, 1].to_numpy()
            spo2_label = int(np.mean(spo2_label))
             # bp = [0]
            return np.concatenate([[int(s_id)], np.round(bp).astype(int), [int(spo2), spo2_label]])
        except Exception as e:
            print(e)


    def analyze_fromfile(self, dir, plot_flag=False):
        ppg = None
        paths = glob.glob(dir + '/*.ppg')
        paths.sort()
        for path in paths:
            ppg_file = read_dat(path).reshape((-1, self.num_ch))
            ppg = np.vstack([ppg, ppg_file]) if ppg is not None else ppg_file

        ppg = ppg / PPG_GAIN

        ppg = -butter_bandpass_filter(ppg.T, 0.5, 10, self.fs_raw, 2).T

        ppg_green = ppg[:, 1]
        acceptable, percent_ppg = self.calculate_sqi_green(ppg_green)

        ppg_green, _ = wfdb.processing.resample_sig(ppg_green, self.fs_raw, self.fs)
        ppg_green = ppg_green
        time_seg = 8 * self.fs

        seg = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_green) - time_seg,
                                                             time_seg - self.num_overlap)[:, None]

        print(percent_ppg)
        title = 'SQI: {}%'.format(percent_ppg)
        # self.plot_ppg_3CH(ppg, num_ch=3, title=title)
        self.plot_good_signal(ppg_green, seg, acceptable)


    def plot_ppg_3CH(self, ppg, num_ch=NUM_CH, title=''):
        channels = ['RED', 'GREEN', 'IR', 'Ambient']
        colors = ['r', 'g', 'k', 'b']
        base_line = [0.5, 0, -0.5]

        time_steps = np.linspace(0, len(ppg[:, 0]) / FS_PROCESS, len(ppg[:, 0]))
        num_ch_process = num_ch

        for i in range(num_ch_process):
            plt.plot(time_steps, ppg[:, i] + base_line[i], color=colors[i], label='PPG_{}'.format(channels[i]))


        plt.ylabel('Amplitude [pA]')
        plt.legend(loc='upper right')
        plt.xlabel('Time [s]')
        plt.title(title)

        # plt.suptitle(title, fontsize=16)
        plt.gcf().set_size_inches(18, 8)
        plt.show()

    def plot_good_signal(self, ppg_filtered, seg, acceptable, s_id=''):
        ppg = ppg_filtered.copy()
        for ind in acceptable:
            good = plt.axvspan(seg[ind, 0], seg[ind, -1], color='yellow', alpha=0.5, label='good')
        # plt.ylim(-0.5, 0.5)
        plt.plot(ppg, color='g', label='Green')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        if len(acceptable):
            plt.legend(handles=[good], loc='upper right')
        plt.gcf().set_size_inches(15, 7)

        percent = round(100 * (len(acceptable) / len(seg)), 2)
        seconds = len(ppg_filtered) // self.fs
        mm_ss = str(datetime.timedelta(seconds=seconds))
        plt.title('Subject ID: {}\nDuration time: {}\nPass SQI: {}%'.format(s_id, mm_ss, percent))

        plt.show()


    def calculate_sqi_4s(self, ppg_filtered):
        time_seg = 4 * self.fs
        ppg = ppg_filtered.copy()
        ppg, _ = wfdb.processing.resample_sig(ppg, self.fs_raw, self.fs)
        seg = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg) - time_seg,
                                                              time_seg - self.num_overlap)[:, None]

        ppg = ppg[seg]
        result_sqi = None
        for ppg_seg in ppg:
            std = np.std(ppg_seg)
            skewness = np.mean(np.power((ppg_seg - np.mean(ppg_seg)), 3)) / np.power(std, 3)
            kurtosis = np.mean(np.power((ppg_seg - np.mean(ppg_seg)), 4)) / np.power(std, 4)
            sqis = np.asarray([round(skewness, 3), round(kurtosis, 3), np.round(std, 3)])
            result_sqi = np.vstack([result_sqi, sqis]) if result_sqi is not None else sqis

        acceptable = np.ones(len(seg))

        reject_kutosis = np.flatnonzero((result_sqi[:, 1] > THR_KURTOSIS) | (result_sqi[:, 1] < 1.75))  # 50
        acceptable[reject_kutosis] = 0
        ind_acceptable = np.flatnonzero(acceptable)
        percent_ppg = round(100 * len(ind_acceptable) / len(seg), 2)
        # percent_ppg = round(100 * len(ind_acceptable)*(time_seg - self.num_overlap) / len(ppg_filtered), 2)

        return ind_acceptable, percent_ppg


    def compute_spo2(self, ppg_raw, acceptable, ratio_led=1):
        ppg_lp = butter_lowpass_filter(ppg_raw.T, 10, self.fs_raw, 2).T
        ppg_fil = butter_bandpass_filter(ppg_raw.T, 0.5, 10, self.fs_raw, 2).T

        accept, _ = self.calculate_sqi_4s(ppg_fil[:, 0])
        accept_ir, _ = self.calculate_sqi_4s(ppg_fil[:, 2])
        accept = np.intersect1d(accept, accept_ir)

        time_seg = 4*256
        seg = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_fil[:, 0]) - time_seg,
                                                             time_seg)[:, None]
        seg = seg[accept]
        seg = seg[len(seg)//4:]
        ac_red = ppg_fil[:, 0][seg]
        dc_red = ppg_lp[:, 0][seg]
        ac_ir = ppg_fil[:, 2][seg]
        dc_ir = ppg_lp[:, 2][seg]

        # ac_red = np.sqrt(np.sum(np.power(ac_red, 2), axis=1)) / time_seg
        ac_red = np.sqrt(np.mean(np.power(ac_red, 2), axis=1))
        dc_red = np.mean(dc_red, axis=1)

        # ac_ir = np.sqrt(np.sum(np.power(ac_ir, 2), axis=1)) / time_seg
        ac_ir = np.sqrt(np.mean(np.power(ac_ir, 2), axis=1))
        dc_ir = np.mean(dc_ir, axis=1)

        ac_red = ac_red/ratio_led
        dc_red = dc_red/ratio_led

        r = np.log(ac_red/dc_red)/np.log(ac_ir/dc_ir)
        r_2 = (ac_red/dc_red)/(ac_ir/dc_ir)
        spo2 = 10.0002*r**3 + -52.887*r**2 + 26.871*r + 98.283
        spo2_2 = -16.666666*r_2*r_2 + 8.333333*r_2 + 100
        spo2_4 = 110 - 11*r_2

        ind = np.flatnonzero((103 > spo2_2) & (spo2_2 > 68))
        spo2_2 = spo2_2[ind]
        ind_100 = np.flatnonzero(spo2_2 > 100)
        spo2_2[ind_100] = 100

        return np.mean(spo2_2)



    def plot_good_signal_3CH(self, ppg_filtered, s_id=''):
        ppg = ppg_filtered.copy()
        timeline = np.arange(len(ppg[:, 0])) / self.fs_raw

        start, stop = 600, 620  # 470, 490
        ind_start, ind_stop = int(start*self.fs), int(stop*self.fs)
        ind = np.flatnonzero(((timeline>start) & (timeline<stop)))
        timeline = timeline[ind]
        ppg = ppg[ind]

        plt.clf()
        plt.subplot(311)
        plt.plot(timeline, ppg[:, 0] + 0.05, color=CHANNELS_COLOR[0], label=CHANNELS_LABEL[0])
        plt.xlim(start, stop)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.ylabel('Amplitude')

        plt.subplot(312)
        plt.plot(timeline, ppg[:, 1], color=CHANNELS_COLOR[1], label=CHANNELS_LABEL[1])
        plt.xlim(start, stop)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.ylabel('Amplitude')

        plt.subplot(313)
        plt.plot(timeline, ppg[:, 2] - 0.05, color=CHANNELS_COLOR[2], label=CHANNELS_LABEL[2])
        plt.xlim(start, stop)

        # plt.ylim(-0.1, 0.1)

        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')

        plt.gcf().set_size_inches(15, 7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        seconds = len(ppg_filtered) // self.fs_raw
        mm_ss = str(datetime.timedelta(seconds=seconds))
        # plt.suptitle('Subject ID: {}\nDuration time: {}'.format(s_id, mm_ss))
        plt.suptitle('Subject ID: {}'.format(s_id))
        fig_path = os.path.join('/mnt/ai_data/ABPM_FW/Pictures_PPG_ITR', '{}.png'.format(s_id))
        plt.savefig(fig_path, dpi=100)
        # plt.show()

    def calculate_sqi_real_time(self, ppg_filtered, seg, thr_sqi=THR_SQI):
        ppg = ppg_filtered.copy()
        ppg = ppg[seg]
        list_sqi = []
        for i in range(1, len(seg), 1):
            c = np.sum(ppg[i - 1] * ppg[i]) / np.sqrt(np.sum(np.power(ppg[i - 1], 2)) * np.sum(np.power(ppg[i], 2)))
            hat = (50 * (c + 1)) * 8 / 99
            sqi = np.exp(hat) / np.exp(8) * 100
            list_sqi.extend([sqi])

        list_sqi = np.asarray(list_sqi)
        ind_acceptable = np.flatnonzero(list_sqi > thr_sqi)  # 50
        return ind_acceptable


    def calculate_sqi_real_time_green(self, ppg_filtered, thr_sqi=THR_SQI):
        ppg = ppg_filtered.copy()
        ppg, _ = wfdb.processing.resample_sig(ppg, self.fs_raw, self.fs)
        seg = np.arange(0, self.len_seg, 1)[None, :] + np.arange(0, len(ppg) - self.len_seg,
                                                                 self.len_seg - self.num_overlap)[:, None]
        ppg = ppg[seg]
        list_sqi = []
        for i in range(1, len(seg), 1):
            c = np.sum(ppg[i - 1] * ppg[i]) / np.sqrt(np.sum(np.power(ppg[i - 1], 2)) * np.sum(np.power(ppg[i], 2)))
            hat = (50 * (c + 1)) * 8 / 99
            sqi = np.exp(hat) / np.exp(8) * 100
            list_sqi.extend([sqi])

        list_sqi = np.asarray(list_sqi)
        ind_acceptable = np.flatnonzero(list_sqi > thr_sqi)  # 50
        percent_ppg = round(100 * len(ind_acceptable) / len(seg), 2)
        return ind_acceptable, percent_ppg

    def calculate_sqi_green(self, ppg_filtered):
        time_seg = 8 * self.fs
        ppg = ppg_filtered.copy()
        ppg, _ = wfdb.processing.resample_sig(ppg, self.fs_raw, self.fs)
        seg = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg) - time_seg,
                                                              time_seg - self.num_overlap)[:, None]

        ppg = ppg[seg]
        result_sqi = None
        for ppg_seg in ppg:
            std = np.std(ppg_seg)
            skewness = np.mean(np.power((ppg_seg - np.mean(ppg_seg)), 3)) / np.power(std, 3)
            kurtosis = np.mean(np.power((ppg_seg - np.mean(ppg_seg)), 4)) / np.power(std, 4)
            sqis = np.asarray([round(skewness, 3), round(kurtosis, 3), np.round(std, 3)])
            result_sqi = np.vstack([result_sqi, sqis]) if result_sqi is not None else sqis

        acceptable = np.ones(len(seg))

        # reject_kutosis = np.flatnonzero((result_sqi[:, 1] > THR_KURTOSIS) | (result_sqi[:, 1] < 2))  # 50
        reject_kutosis = np.flatnonzero((result_sqi[:, 1] > THR_KURTOSIS) | (result_sqi[:, 1] < 1.8))  # 50
        reject_std = np.flatnonzero(result_sqi[:, 2] > THR_STD)  # 50
        acceptable[reject_kutosis] = 0
        acceptable[reject_std] = 0
        ind_acceptable = np.flatnonzero(acceptable)
        percent_ppg = round(100 * len(ind_acceptable) / len(seg), 2)
        # percent_ppg = round(100 * len(ind_acceptable)*(time_seg - self.num_overlap) / len(ppg_filtered), 2)

        return ind_acceptable, percent_ppg

    def calculate_sqi_red_ir(self, ppg_filtered):
        time_seg = 8 * self.fs
        ppg = ppg_filtered.copy()
        ppg_red, _ = wfdb.processing.resample_sig(ppg[:, 0], self.fs_raw, self.fs)
        ppg_ir, _ = wfdb.processing.resample_sig(ppg[:, 2], self.fs_raw, self.fs)
        seg = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_red) - time_seg,
                                                              time_seg - self.num_overlap)[:, None]

        # ppg_red = ppg_red[seg]
        ppg_red = ppg_ir[seg]
        result_sqi = None
        for ppg_seg in ppg_red:
            std = np.std(ppg_seg)
            skewness = np.mean(np.power((ppg_seg - np.mean(ppg_seg)), 3)) / np.power(std, 3)
            kurtosis = np.mean(np.power((ppg_seg - np.mean(ppg_seg)), 4)) / np.power(std, 4)
            sqis = np.asarray([round(skewness, 3), round(kurtosis, 3), np.round(std, 3)])
            # if kurtosis > THR_KURTOSIS:
            #     continue
            plt.plot(ppg_seg)
            plt.title(sqis)
            plt.show()

            result_sqi = np.vstack([result_sqi, sqis]) if result_sqi is not None else sqis

        acceptable = np.ones(len(seg))

        reject_kutosis = np.flatnonzero(result_sqi[:, 1] > THR_KURTOSIS)  # 50
        reject_std = np.flatnonzero(result_sqi[:, 2] > THR_STD)  # 50
        acceptable[reject_kutosis] = 0
        acceptable[reject_std] = 0
        ind_acceptable = np.flatnonzero(acceptable)
        percent_ppg = round(100 * len(ind_acceptable) / len(seg), 2)
        # percent_ppg = round(100 * len(ind_acceptable)*(time_seg - self.num_overlap) / len(ppg_filtered), 2)

        return ind_acceptable, percent_ppg