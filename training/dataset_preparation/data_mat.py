import glob
import os.path
import time
from multiprocessing import Pool

import natsort
from tqdm import tqdm

from mat73 import loadmat
import numpy as np
import matplotlib.pyplot as plt
import wfdb as wf


def Build_Dataset(Path, FieldName='Subset'):
    Data = loadmat(Path)
    # Access 10-s segments of ECG, PPG and ABP signals
    Signals = Data[FieldName]['Signals']
    # Access SBP labels of each 10-s segment
    SBPLabels = Data[FieldName]['SBP']
    # Access Age of the subject corresponding to each of the 10-s segment
    Age = Data[FieldName]['Age']
    # Access Gender of the subject corresponding to each of the 10-s segment
    Gender = np.array(Data[FieldName]['Gender']).squeeze()
    # Convert Gender to numerical 0-1 labels
    Gender = (Gender == 'M').astype(float)
    # Access Height and Weight of the subject corresponding to each of the 10-s segment
    # If the subject is from the MIMIC-III matched subset, height and weight will be NaN
    # since they were only recorded in VitalDB
    Height = Data[FieldName]['Height']
    Weight = Data[FieldName]['Weight']
    # Concatenate the demographic information as one matrix
    Demographics = np.stack((Age, Gender, Height, Weight), axis=1)
    return Signals, SBPLabels, Demographics


def process_data(path, dir_save):
    file_name = os.path.basename(path).rstrip('.mat')
    # print("Processing file part: {}".format(file_name))
    data = loadmat(path)
    try:

        ppg = np.concatenate(data['Subj_Wins']['PPG_Raw']).flatten()
        abp = np.concatenate(data['Subj_Wins']['ABP_Raw']).flatten()
        ecg = np.concatenate(data['Subj_Wins']['ECG_Raw']).flatten()

        wf.wrsamp(record_name='{}'.format(file_name),
                  fs=125,
                  units=['mV', 'NU', 'mmHg'],
                  sig_name=['II', 'PLETH', 'ABP'],
                  p_signal=np.vstack((ecg, ppg, abp)).T,
                  # adc_gain=[128.0, 255.0, 4.8],
                  # baseline=[-65, -128, -325],
                  write_dir=dir_save)
    except Exception as error:
        print('{} - Error at: {}'.format(error, file_name))

def worker_function(args):
    path, dir_save = args
    return process_data(path, dir_save)


if __name__ == "__main__":
    paths = glob.glob('/mnt/ai_virtual/dataset/PulseDB/PulseDB_Vital/*.mat')
    dir_save = '/mnt/ai_virtual/dataset/PPG_AI'

    paths = natsort.natsorted(paths)

    args = []
    ar2 = []
    for path in paths:
        args.append((path, dir_save))
    progress_bar = tqdm(total=len(args), desc="Processing", position=0, leave=True)

    with Pool(os.cpu_count() - 4) as pool:
        for i in pool.imap_unordered(worker_function, args):
            progress_bar.update(1)
