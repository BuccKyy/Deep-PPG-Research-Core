import numpy as np

fs_raw = 256
def compute_spo2(ppg_raw):
    ppg_lp = butter_lowpass_filter(ppg_raw.T, 10, fs_raw, 2).T
    ppg_fil = butter_bandpass_filter(ppg_raw.T, 0.5, 10, fs_raw, 2).T
    time_seg = 8 * 256
    seg = np.arange(0, time_seg, 1)[None, :] + np.arange(0, len(ppg_fil[:, 0]) - time_seg,
                                                         time_seg)[:, None]
    ac_red = ppg_fil[:, 0][seg]
    dc_red = ppg_lp[:, 0][seg]
    ac_ir = ppg_fil[:, 2][seg]
    dc_ir = ppg_lp[:, 2][seg]

    ac_red = np.sqrt(np.mean(np.power(ac_red, 2), axis=1))
    dc_red = np.mean(dc_red, axis=1)

    ac_ir = np.sqrt(np.mean(np.power(ac_ir, 2), axis=1))
    dc_ir = np.mean(dc_ir, axis=1)

    r_2 = (ac_red / dc_red) / (ac_ir / dc_ir)
    spo2_4 = 108 - 8 * r_2

    ind = np.flatnonzero((103 > spo2_4) & (spo2_4 > 68))
    spo2_4 = spo2_4[ind]
    ind_100 = np.flatnonzero(spo2_4 > 100)
    spo2_4[ind_100] = 100

    return np.mean(spo2_4)