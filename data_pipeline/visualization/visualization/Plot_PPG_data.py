import glob
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os

# Define
PPG_GAIN = 2**19 / 32.25  #10 ** 9 / 62.5
FS = 256
THR = 80000


class ProcessPPG:
    def __init__(self, fs, start_byte, num_bytes):
        self.fs = fs
        self.start_byte = start_byte
        self.num_bytes = num_bytes

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def read_dat(self, filePath, signed=False):
        with open(filePath, "rb") as f:
            file = f.read()
            file = file[self.start_byte:]
            convert = lambda byte: np.int.from_bytes(byte, byteorder='big', signed=signed)
            signal = [convert(file[i:i + self.num_bytes]) for i in range(0, len(file), self.num_bytes)]

            # signal = np.asarray(signal) / PPG_GAIN
            signal = np.asarray(signal, dtype=int)
            # if len(signal) > THR:
            #     signal = signal.reshape(2, -1)
        return signal

    # @staticmethod
    def plot_ppg(self, ppg, title, num_fig=1):

        time_steps = np.linspace(0, len(ppg) / FS, len(ppg))
        plt.suptitle(title, fontsize=16)

        if num_fig == 1:
            plt.plot(time_steps, ppg, label='PPG')
            # plt.ylim(-3e-4, 3e-4)
            plt.legend(loc='upper right')
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [pA]')
        else:
            ppg_bp = self.butter_bandpass_filter(ppg, 0.5, 10, FS, 3)

            plt.subplot(211)
            plt.plot(time_steps, ppg, label='PPG')
            # plt.ylim(-3e-4, 3e-4)
            plt.legend(loc='upper right')
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [pA]')
            plt.title('Raw')
            plt.subplot(212)
            plt.plot(time_steps, ppg_bp, label='PPG_BP')
            # plt.ylim(-3e-4, 3e-4)
            plt.legend(loc='upper right')
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [pA]')
            plt.title('Bandpass')


        plt.gcf().set_size_inches(18, 8)
        plt.show()

    def plot_ppg_3CH(self, ppg, path, save_fig=False, filter='raw', num_ch=3):
        time_steps = np.linspace(0, len(ppg[:, 0]) / FS, len(ppg[:, 0]))
        title = '{}\n{}'.format(os.path.basename(path).rstrip('.PPG'), filter)
        # plt.vlines(1718, -1, 1, colors='red')

        # ppg[:, 1] -= 0.61
        # plt.plot(ppg)

        num_ch_process = num_ch

        fig, axis = plt.subplots(num_ch_process, 1, sharex=True, sharey=False, figsize=(19.2, 10.8))
        # channels = ['RED', 'GREEN', 'IR']
        # colors = ['r', 'g', 'k']

        channels = ['RED', 'GREEN', 'IR', 'Ambient']
        colors = ['r', 'g', 'k', 'b']

        # axis[3].plot(time_steps, ppg[:, 1], label='PPG')
        for i in range(num_ch_process):
            axis[i].plot(time_steps, ppg[:, i], color=colors[i], label='PPG_{}'.format(channels[i]))
            axis[i].set_title('{}'.format(channels[i]))
            # axis[i].set_ylim(-3e-4, 3e-4)
            axis[i].set_ylabel('Amplitude')
            axis[i].legend(loc='upper right')
        axis[num_ch_process-1].set_xlabel('Time [s]')

        fig.suptitle(title, fontsize=16)
        plt.gcf().set_size_inches(18, 8)
        if save_fig:
            dst = os.path.join('/mnt/ai_data/ABPM_FW/test_3led/fig', path.split('/')[-2])
            os.makedirs(dst, exist_ok=True)
            name_file = '{}_{}.png'.format(os.path.basename(path).rstrip('.ppg'), filter)
            path_save = os.path.join(dst, name_file)
            plt.savefig(path_save, dpi=100)
            plt.cla()
        else:
            plt.show()

    def plot_ppg_from_folder(self, dir, channels=1, save_fig=False, filter='raw'):
        paths = glob.glob(dir + '/*.ppg')
        paths.sort()
        for path in paths:
            ppg = self.read_dat(path, signed=False)
            title = os.path.basename(path).rstrip('.PPG')
            np.savetxt('float_result.txt', ppg, delimiter='\n', fmt='%1.4e')

            if channels == 1:
                # ppg = self.butter_bandpass_filter(ppg, 0.5, 10, FS, 5)
                ppg = -ppg
                self.plot_ppg(ppg, title)
            else:
                ppg = ppg[:len(ppg) - (len(ppg) % 3)]
                ppg = ppg.reshape(-1, 3)
                ppg = ppg[256 // 8:, :]
                ppg = -ppg
                ppg = self.butter_bandpass_filter(ppg.T, 0.5, 10, FS, 2).T
                # ppg = butter_lowpass_filter(ppg.T, 10, FS, 2).T
                self.plot_ppg_3CH(ppg, path, save_fig, filter)

    def plot_ppg_from_file(self, path, channels=1):
        ppg = self.read_dat(path, signed=False)
        title = os.path.basename(path).rstrip('.ppg')
        if channels == 1:
            # ppg = self.butter_bandpass_filter(ppg, 0.5, 10, FS, 5)
            ppg = -ppg
            self.plot_ppg(ppg, title, num_fig=2)
        else:
            ppg = ppg[:len(ppg) - (len(ppg) % channels)]
            ppg = ppg.reshape(-1, channels)


            ppg = ppg[256//8:-512, :]
            ppg = ppg
            ppg = -self.butter_bandpass_filter(ppg.T, 0.5, 10, FS, 2).T

            self.plot_ppg_3CH(ppg, title, num_ch=channels, filter='Bandpass')

    def plot_ppg_from_npy(self, path):
        ppg = np.load(path) / PPG_GAIN
        title = os.path.basename(path).rstrip('.npy')

        ppg = -self.butter_bandpass_filter(ppg.T, 0.5, 10, FS, 2).T

        self.plot_ppg_3CH(ppg, title, num_ch=3, filter='Bandpass')


if __name__ == '__main__':
    process = ProcessPPG(fs=FS, start_byte=512, num_bytes=3)

    path = '/mnt/ai_data/ABPM_FW/collect_itr/PPG_ITR/2/002.npy'
    # For .npy
    process.plot_ppg_from_npy(path)

    # For .ppg
    # process.plot_ppg_from_file(path, channels=3)

    # For folder
    # dir = '/mnt/ai_data/ABPM_FW/test_3led/8'
    # process.plot_ppg_from_folder(dir, channels=1, save_fig=False, filter='bandpass_05_10hz')

