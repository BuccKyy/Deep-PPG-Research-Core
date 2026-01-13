import csv
import os.path
import pandas as pd
import matplotlib.lines as mlines
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tqdm
import wfdb
import wfdb.processing
from model.Sqeeze_Unet import dfm, tf, glob, json, AIData

import cross_correlation.preprocessing
from scipy import stats


def compute_error(path, tf_data):
    ds = tf_data.get_dataset_from_tfrecord_test_result(path, batch_size=1)
    ppg, RPTT = [], []
    SBP, DBP = [], []
    Ka, Kb = [], []
    HR = []
    calibrate = []
    for index, batch in enumerate(ds):
        list_tensors = [i for i in batch]
        ppg.extend([tensor.numpy() for tensor in list_tensors[0]])
        RPTT.extend([tensor.numpy() for tensor in list_tensors[1]])
        Ka.extend([tensor.numpy() for tensor in list_tensors[2]][0])
        Kb.extend([tensor.numpy() for tensor in list_tensors[2]][1])
        SBP.extend([tensor.numpy() for tensor in list_tensors[3]][0])
        DBP.extend([tensor.numpy() for tensor in list_tensors[3]][1])
        HR.extend([tensor.numpy() for tensor in list_tensors[4]])
        calibrate.extend([tensor.numpy() for tensor in list_tensors[2]])

    ppg = np.asarray(ppg)
    Ka = np.asarray(Ka).flatten()
    Kb = np.asarray(Kb).flatten()
    SBP = np.asarray(SBP).flatten()
    DBP = np.asarray(DBP).flatten()
    RPTT = np.asarray(RPTT).flatten()
    HR = np.asarray(HR).flatten()
    calibrate = np.asarray(calibrate)
    calibrate = np.reshape(calibrate, (-1, 2))

    Ka = np.mean(Ka)
    Kb = np.mean(Kb)

    def bp_algorithm():
        rptt_0 = np.mean(RPTT)
        SBP_0 = np.mean(SBP[:len(SBP) // 3], dtype=int)
        DBP_0 = np.mean(DBP[:len(DBP) // 3], dtype=int)
        PP_0 = SBP_0 - DBP_0
        MBP_0 = (SBP_0 + 2 * DBP_0) / 3

        # SBP_predict = MBP_0 + (2 / 3) * PP_0 * np.power((rptt_0 / RPTT), 2)
        # DBP_predict = MBP_0 - (1 / 3) * PP_0 * np.power((rptt_0 / RPTT), 2)

        DBP_predict = Kb - 2 / 0.031 * np.log(RPTT) - Ka / (3 * RPTT * RPTT)
        SBP_predict = DBP_predict + Ka / (RPTT * RPTT)

        return SBP_predict, DBP_predict

    def bp_ai():
        from lstm_model import lstm, lstm_8s
        from ann_model import cnn_test, cnn_test_3_25, cnn_test_3_25_1, cnn_test_3_26
        ai_model = cnn_test_3_25_1()

        ai_model.load_weights(tf.train.latest_checkpoint(os.path.join(dfm.WORKSPACE, 'checkpoint'))).expect_partial()

        # result_predict = ai_model.predict(ppg)
        result_predict = ai_model.predict((ppg, calibrate))
        result_predict = np.asarray(result_predict)

        SBP_predict = result_predict[0, :, 0]
        DBP_predict = result_predict[1, :, 0]

        return SBP_predict, DBP_predict

    SBP_predict, DBP_predict = bp_ai()
    # idx = np.flatnonzero(DBP_predict < 0)
    # SBP = np.delete(SBP, idx)
    # SBP_predict = np.delete(SBP_predict, idx)
    # DBP = np.delete(DBP, idx)
    # DBP_predict = np.delete(DBP_predict, idx)

    # return SBP, SBP_predict.astype(int), DBP, DBP_predict.astype(int)
    return SBP, SBP_predict, DBP, DBP_predict


def write_evaluate_tfrecord():
    os.makedirs(dfm.TRAIN_PATH, exist_ok=True)
    result_file_name = 'bp_ai'
    path_result = dfm.TRAIN_PATH + '/' + result_file_name + '_float_all_v' + dfm.VER_MODEL + '.csv'
    # path_result = dfm.TRAIN_PATH + '/' + result_file_name + '_float_test_v' + dfm.VER_MODEL + '.csv'
    # path_result = dfm.TRAIN_PATH + '/' + result_file_name + '_float_train_v' + dfm.VER_MODEL + '.csv'
    with open(path_result, 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['Record', 'SBP_actual', 'DBP_actual', 'SBP_predict', 'DBP_predict'])
        path_json = glob.glob(os.path.join(dfm.WORKSPACE, 'INFO_train_valid_test') + '/*.json')
        # with open(path_json[0], 'r') as openfile:
        #     rd = json.load(openfile)
        # tfrecord_paths = rd.get('train')
        tfrecord_paths = glob.glob(dfm.DIR_SAVE_TFRECORD + '/*.tfrecord')
        from natsort import natsorted
        natsorted(tfrecord_paths)
        tf_data = AIData(dfm.DIR_SAVE_TFRECORD, dfm.input_length)

        for path in tfrecord_paths:
            record_name = os.path.basename(path).rstrip('.tfrecord')
            print('\n{}'.format(record_name))
            # sbp_me, sbp_std, dbp_me, dbp_std, hr_me, hr_std = compute_error(path, tf_data, ai_model)
            SBP, SBP_predict, DBP, DBP_predict = compute_error(path, tf_data)
            for i in range(len(SBP)):
                writer.writerow([record_name, SBP[i], DBP[i], SBP_predict[i], DBP_predict[i]])
                # writer.writerow([record_name, sbp_me, sbp_std, dbp_me, dbp_std, hr_me, hr_std])


def loop_db(dir):
    # db = glob.glob(dir_tf + '*1_*.hea')
    db = glob.glob(dir + '*.hea')
    db.sort()
    for path in db:
        path = path.rstrip('.hea')
        sig = wfdb.rdsamp(path)
        ppg = sig[0][:, 6]
        ppg = wfdb.processing.resample_sig(ppg, 500, 125)[0]
        ppg = (ppg - 8192) / 16384 * 1000
        # abp = sig[0][:, 2]
        from cross_correlation.preprocessing import butter_bandpass_filter
        ppg = butter_bandpass_filter(ppg, 0.5, 8, 125, 4)

        len_seg = dfm.input_length  # 256
        num_overlap = 0
        seg = np.arange(0, len_seg, 1)[None, :] + np.arange(0, len(ppg) - len_seg, len_seg - num_overlap)[:, None]
        seg = seg.astype(int)
        ppg[seg] = (ppg[seg] - dfm.PPG_MIN) / (dfm.PPG_MAX - dfm.PPG_MIN)

        list_features = np.array([])
        cmt_info = sig[1]['comments'][0].split(' ')
        SBP = (int(cmt_info[cmt_info.index('<bp_sys_start>:') + 1]) + int(
            cmt_info[cmt_info.index('<bp_sys_end>:') + 1])) // 2
        DBP = (int(cmt_info[cmt_info.index('<bp_dia_start>:') + 1]) + int(
            cmt_info[cmt_info.index('<bp_dia_end>:') + 1])) // 2
        for i in seg:
            try:
                # second, peaks, troughs, p, tr = analyze(ppg[i])
                # if len(p) < 3:
                #     continue
                # plot_data(ppg[i])
                RPTT, K_a, K_b = AIData.extract_features(ppg[i], SBP, DBP)
                features = np.array([RPTT, K_a, K_b])
                if np.isnan(features).any():
                    raise Exception("features = nan")
                list_features = np.vstack([list_features, features]) if list_features.size > 0 else features
            except Exception as e:
                # print(e)
                continue

        if len(list_features.flatten()) < 4:
            continue

        list_Ka = list_features[:, 1][~np.isnan(list_features[:, 1])]
        list_Kb = list_features[:, 2][~np.isnan(list_features[:, 2])]

        if len(list_Ka) < 1 or len(list_Kb) < 1:
            continue

        feature_Ka = np.mean(list_Ka)
        feature_Kb = np.mean(list_Kb)
        list_RPTT = list_features[:, 0][~np.isnan(list_features[:, 0])]

        list_predict = np.array([])
        for r_ptt in list_RPTT:
            D = feature_Kb - 2 / 0.031 * np.log(r_ptt) - feature_Ka / (3 * r_ptt * r_ptt)
            S = D + feature_Ka / (r_ptt * r_ptt)
            result = np.array([S, D])
            list_predict = np.vstack([list_predict, result]) if list_predict.size > 0 else result

        print(os.path.basename(path))
        S = int(np.median(list_predict[:, 0]))
        D = int(np.median(list_predict[:, 1]))
        # print(list_features)
        print('Actual SBP, DBP: {}  {}'.format(SBP, DBP))
        print('Predict SBP, DBP: {}  {}'.format(S, D))
        print('-----------')


def write_evaluate_dat():
    os.makedirs(dfm.RESULT_PATH, exist_ok=True)
    result_file_name = 'ANN'
    path_result = dfm.RESULT_PATH + '/' + result_file_name + '_v' + dfm.VER_MODEL + '.csv'
    with open(path_result, 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['Record', 'SBP_ME', 'SBP_STD', 'DBP_ME', 'DBP_STD', 'HR_ME', 'HR_STD'])

        from ann_model import ann
        ai_model = ann()
        ai_model.load_weights(tf.train.latest_checkpoint(os.path.join(dfm.WORKSPACE, 'checkpoint'))).expect_partial()

        path_json = glob.glob(os.path.join(dfm.WORKSPACE, 'INFO_train_valid_test') + '/*.json')
        with open(path_json[0], 'r') as openfile:
            rd = json.load(openfile)

        tfrecord_test_dir = rd.get('test')
        # tfrecord_test_dir = rd.get('train')

        tf_data = AIData(dfm.DIR_SAVE_TFRECORD, dfm.input_length)
        for path in tfrecord_test_dir:
            record_name = os.path.basename(path).rstrip('.tfrecord')
            print('\n{}'.format(record_name))
            sbp_me, sbp_std, dbp_me, dbp_std, hr_me, hr_std = compute_error(path, tf_data, ai_model)
            writer.writerow([record_name, sbp_me, sbp_std, dbp_me, dbp_std, hr_me, hr_std])


def BHS_metric(err):
    """
    Computes the BHS Standard metric

    Arguments:
        err {array} -- array of absolute error

    Returns:
        tuple -- tuple of percentage of samples with <=5 mmHg, <=10 mmHg and <=15 mmHg error
    """

    leq5 = 0
    leq10 = 0
    leq15 = 0

    for i in range(len(err)):

        if (abs(err[i]) <= 5):
            leq5 += 1
            leq10 += 1
            leq15 += 1

        elif (abs(err[i]) <= 10):
            leq10 += 1
            leq15 += 1

        elif (abs(err[i]) <= 15):
            leq15 += 1

    return (leq5 * 100.0 / len(err), leq10 * 100.0 / len(err), leq15 * 100.0 / len(err))


def evaluate_BHS_Standard(sbps, dbps, maps, plot_flag=False):
    """
        Evaluates PPG2ABP based on
        BHS Standard Metric
    """

    def newline(p1, p2):
        """
        Draws a line between two points

        Arguments:
            p1 {list} -- coordinate of the first point
            p2 {list} -- coordinate of the second point

        Returns:
            mlines.Line2D -- the drawn line
        """
        ax = plt.gca()
        xmin, xmax = ax.get_xbound()

        if (p2[0] == p1[0]):
            xmin = xmax = p1[0]
            ymin, ymax = ax.get_ybound()
        else:
            ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
            ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

        l = mlines.Line2D([xmin, xmax], [ymin, ymax], linewidth=1, linestyle='--')
        ax.add_line(l)
        return l

    sbp_percent = BHS_metric(sbps)  # compute BHS metric for sbp
    dbp_percent = BHS_metric(dbps)  # compute BHS metric for dbp
    map_percent = BHS_metric(maps)  # compute BHS metric for map

    print('----------------------------')
    print('|        BHS-Metric        |')
    print('----------------------------')

    print('----------------------------------------')
    print('|     | <= 5mmHg | <=10mmHg | <=15mmHg |')
    print('----------------------------------------')
    print('| DBP |  {} %  |  {} %  |  {} %  |'.format(round(dbp_percent[0], 1), round(dbp_percent[1], 1),
                                                      round(dbp_percent[2], 1)))
    print('| MAP |  {} %  |  {} %  |  {} %  |'.format(round(map_percent[0], 1), round(map_percent[1], 1),
                                                      round(map_percent[2], 1)))
    print('| SBP |  {} %  |  {} %  |  {} %  |'.format(round(sbp_percent[0], 1), round(sbp_percent[1], 1),
                                                      round(sbp_percent[2], 1)))
    print('----------------------------------------')

    '''
        Plot figures
    '''
    if plot_flag:
        ## SBPS ##

        fig = plt.figure(figsize=(18, 4), dpi=120)
        ax1 = plt.subplot(1, 3, 1)
        ax2 = ax1.twinx()
        sns.distplot(sbps, bins=100, kde=False, rug=False, ax=ax1)
        sns.distplot(sbps, bins=100, kde=False, rug=False, ax=ax2)
        ax2.set_yticklabels(['0 \%', '3.67 \%', '7.34 \%',
                             '11.01 \%', '14.67 \%', '18.34 \%', '22.01 \%'])
        ax1.set_xlabel(r'$|$' + 'Error' + r'$|$' + ' (mmHg)', fontsize=11)
        ax1.set_ylabel('Number of Samples', fontsize=11)
        ax2.set_ylabel('Percentage of Samples', fontsize=11)
        plt.title('Absolute Error in SBP Prediction', fontsize=15)
        plt.xlim(xmax=60.0, xmin=0.0)
        plt.xticks(np.arange(0, 60 + 1, 5))
        p1 = [5, 0]
        p2 = [5, 10000]
        newline(p1, p2)
        p1 = [10, 0]
        p2 = [10, 10000]
        newline(p1, p2)
        p1 = [15, 0]
        p2 = [15, 10000]
        newline(p1, p2)
        plt.tight_layout()

        ## DBPS ##

        ax1 = plt.subplot(1, 3, 2)
        ax2 = ax1.twinx()
        sns.distplot(dbps, bins=100, kde=False, rug=False, ax=ax1)
        sns.distplot(dbps, bins=100, kde=False, rug=False, ax=ax2)
        ax2.set_yticklabels(['0 \%', '7.34 \%', '14.67 \%',
                             '22.01 \%', '29.35 \%', '36.68 \%', '44.02 \%'])
        ax1.set_xlabel(r'$|$' + 'Error' + r'$|$' + ' (mmHg)', fontsize=11)
        ax1.set_ylabel('Number of Samples', fontsize=11)
        ax2.set_ylabel('Percentage of Samples', fontsize=11)
        plt.title('Absolute Error in DBP Prediction', fontsize=15)
        plt.xlim(xmax=60.0, xmin=0.0)
        plt.xticks(np.arange(0, 60 + 1, 5))
        p1 = [5, 0]
        p2 = [5, 10000]
        newline(p1, p2)
        p1 = [10, 0]
        p2 = [10, 10000]
        newline(p1, p2)
        p1 = [15, 0]
        p2 = [15, 10000]
        newline(p1, p2)
        plt.tight_layout()

        ## MAPS ##

        ax1 = plt.subplot(1, 3, 3)
        ax2 = ax1.twinx()
        sns.distplot(maps, bins=100, kde=False, rug=False, ax=ax1)
        sns.distplot(maps, bins=100, kde=False, rug=False, ax=ax2)
        ax2.set_yticklabels(['0 \%', '7.34 \%', '14.67 \%', '22.01 \%',
                             '29.35 \%', '36.68 \%', '44.02 \%', '51.36 \%'])
        ax1.set_xlabel(r'$|$' + 'Error' + r'$|$' + ' (mmHg)', fontsize=11)
        ax1.set_ylabel('Number of Samples', fontsize=11)
        ax2.set_ylabel('Percentage of Samples', fontsize=11)
        plt.title('Absolute Error in MAP Prediction', fontsize=15)
        plt.xlim(xmax=60.0, xmin=0.0)
        plt.xticks(np.arange(0, 60 + 1, 5))
        p1 = [5, 0]
        p2 = [5, 10000]
        newline(p1, p2)
        p1 = [10, 0]
        p2 = [10, 10000]
        newline(p1, p2)
        p1 = [15, 0]
        p2 = [15, 10000]
        newline(p1, p2)
        plt.tight_layout()

        plt.show()


def bland_altman(data1, data2, x_axis=True):
    """
    Computes mean +- 1.96 sd

    Arguments:
        data1 {array} -- series 1
        data2 {array} -- series 2
    """

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, alpha=0.6, s=11)
    plt.axhline(md, color='black', linestyle='--', alpha=1)
    plt.axhline(md + 1.96 * sd, color='black', linestyle='--', alpha=1)
    plt.axhline(md - 1.96 * sd, color='black', linestyle='--', alpha=1)
    plt.text(x=np.min(mean), y=md + 3 * sd, s='{}'.format(round(md + 1.96 * sd, 2)), fontsize=10)
    plt.text(x=np.min(mean), y=md - 3 * sd, s='{}'.format(round(md - 1.96 * sd, 2)), fontsize=10)

    max = int(3 * (md + 1.96 * sd))
    min = int(3 * (md - 1.96 * sd))
    # plt.ylim(ymin=-30, ymax=30)
    # plt.ylim(ymin=-55, ymax=55)
    plt.ylim(ymin=-85, ymax=85)
    if x_axis:
        plt.xlabel('Avg. of Target and Estimated Value (mmHg)', fontsize=14)
    plt.ylabel('Error in Prediction (mmHg)', fontsize=14)
    print(md + 1.96 * sd, md - 1.96 * sd)
    ind = np.where(np.logical_and(md - 1.96 * sd < diff, diff < md + 1.96 * sd))
    print('Per point in LOAs: {}%'.format(round(len(ind[0]) / len(diff) * 100, 2)))


def evaluate_csv():
    import pandas as pd
    # path = '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.9_1.0.0.0/result/R-PTT_algorithm_v3.0.9.csv'
    # path = '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.9_1.0.0.0/result/R-PTT_algorithm_sit_v3.0.9.csv'
    path = '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.9_1.0.0.0/result/R-PTT_algorithm_sit_6_v3.0.9.csv'
    df = pd.read_csv(path)
    key = df.columns.tolist()
    SBP_actual = df[key[1]].to_numpy()
    DBP_actual = df[key[2]].to_numpy()

    list_sensor = np.arange(6, 7, 3)

    SBP_actual = np.repeat(SBP_actual, len(list_sensor))
    DBP_actual = np.repeat(DBP_actual, len(list_sensor))

    SBP_predict, DBP_predict = [], []

    for i in list_sensor:
        sensor = i * 2 + 1
        print(key[sensor])
        SBP_predict.extend(df[key[sensor]].tolist())
        DBP_predict.extend(df[key[sensor + 1]].tolist())

    SBP_predict = np.asarray(SBP_predict)
    DBP_predict = np.asarray(DBP_predict)
    me_sbp = SBP_predict - SBP_actual
    std_sbp = np.std(me_sbp)
    me_dbp = DBP_predict - DBP_actual
    std_dbp = np.std(me_dbp)

    # BHS_metric(me_sbp)

    print('ME SBP: {} - DBP: {}'.format(np.mean(me_sbp), np.mean(me_dbp)))
    print('STD SBP: {} - DBP: {}'.format(std_sbp, std_dbp))

    plt.subplot(1, 2, 1)
    # print('---------SBP---------')
    bland_altman(SBP_predict, SBP_actual)
    plt.title('SBP Prediction', fontsize=18)

    plt.subplot(1, 2, 2)
    # print('---------DBP---------')
    bland_altman(DBP_predict, DBP_actual)
    plt.title('DBP Prediction', fontsize=18)

    plt.suptitle('Bland-Altman Plot', fontsize=18)
    plt.gcf().set_size_inches(15, 8)
    plt.show()
    plt.close()


def bland_altman_plot(SBP_predict, SBP_actual, DBP_predict, DBP_actual, MAP_predict, MAP_actual):
    plt.subplot(1, 3, 1)
    bland_altman(SBP_predict, SBP_actual)
    plt.title('SBP', fontsize=15)
    # plt.suptitle('Bland-Altman Plot', fontsize=18)
    # plt.gcf().set_size_inches(8, 5)
    # plt.show()

    plt.subplot(1, 3, 2)
    bland_altman(DBP_predict, DBP_actual)
    plt.title('DBP', fontsize=15)
    # plt.suptitle('Bland-Altman Plot', fontsize=18)
    # plt.gcf().set_size_inches(8, 5)

    # plt.show()

    plt.subplot(1, 3, 3)
    bland_altman(MAP_predict, MAP_actual)
    plt.title('MAP', fontsize=15)

    plt.suptitle('Bland-Altman Plot', fontsize=18)
    plt.gcf().set_size_inches(15, 8)
    plt.show()
    plt.close()


def regression_plot(sbpTrues, sbpPreds, dbpTrues, dbpPreds, mapTrues, mapPreds):
    # plt.figure(figsize=(18, 6), dpi=120)
    plt.figure(figsize=(15, 8), dpi=120)

    plt.subplot(1, 3, 1)
    sns.regplot(x=sbpTrues, y=sbpPreds, scatter_kws={'alpha': 0.8, 's': 3}, line_kws={'color': 'r', 'lw': 3})
    plt.xlabel('Target Value (mmHg)', fontsize=14)
    plt.ylabel('Estimated Value (mmHg)', fontsize=14)
    plt.title('SBP', fontsize=18)
    # plt.show()

    # plt.figure(figsize=(8, 5), dpi=120)
    plt.subplot(1, 3, 2)
    sns.regplot(x=dbpTrues, y=dbpPreds, scatter_kws={'alpha': 0.8, 's': 3}, line_kws={'color': 'r', 'lw': 3})
    plt.xlabel('Target Value (mmHg)', fontsize=14)
    plt.ylabel('Estimated Value (mmHg)', fontsize=14)
    plt.title('DBP', fontsize=18)
    # plt.show()

    # plt.figure(figsize=(8, 5), dpi=120)
    plt.subplot(1, 3, 3)
    # sns.regplot(x=mapTrues, y=mapPreds, scatter_kws={'alpha': 0.5, 's': 3}, line_kws={'color': '#e0b0b4'})
    sns.regplot(x=mapTrues, y=mapPreds, scatter_kws={'alpha': 0.8, 's': 3}, line_kws={'color': 'r', 'lw': 3})
    plt.xlabel('Target Value (mmHg)', fontsize=14)
    plt.ylabel('Estimated Value (mmHg)', fontsize=14)
    plt.title('MAP', fontsize=18)

    plt.suptitle('Regression Plot', fontsize=18)
    # plt.figure(figsize=(8, 5), dpi=120)

    plt.show()

    '''
        Printing statistical analysis values like r and p value
    '''
    print('DBP')
    print(scipy.stats.linregress(dbpTrues, dbpPreds))
    print('MAP')
    print(scipy.stats.linregress(mapTrues, mapPreds))
    print('SBP')
    print(scipy.stats.linregress(sbpTrues, sbpPreds))


def evaluate_model_csv():
    # path = '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.9_1.0.0.0/result/R-PTT_algorithm_v3.0.9.csv'
    # path = '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.9_1.0.0.0/result/R-PTT_algorithm_sit_v3.0.9.csv'
    path = '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.10_1.0.0.0/training-v0/ANN_v3.0.10.csv'
    df = pd.read_csv(path)
    key = df.columns.tolist()
    SBP_actual = df[key[1]].to_numpy()
    DBP_actual = df[key[3]].to_numpy()
    HR_actual = df[key[5]].to_numpy()

    SBP_predict = df[key[2]].to_numpy()
    DBP_predict = df[key[4]].to_numpy()
    HR_predict = df[key[6]].to_numpy()

    MAP_actual = SBP_actual / 3 + DBP_actual * 2 / 3
    MAP_predict = SBP_predict / 3 + DBP_predict * 2 / 3

    me_sbp = SBP_predict - SBP_actual
    std_sbp = np.std(me_sbp)
    me_dbp = DBP_predict - DBP_actual
    std_dbp = np.std(me_dbp)
    me_map = MAP_predict - MAP_actual
    std_map = np.std(me_map)

    # print('ME SBP: {} - DBP: {} - MAP: {}'.format(np.mean(me_sbp), np.mean(me_dbp), np.mean(me_map)))
    # print('STD SBP: {} - DBP: {} - MAP: {}'.format(std_sbp, std_dbp, std_map))

    print('---------------------')
    print('|   AAMI Standard   |')
    print('---------------------')

    print('-----------------------')
    print('|     |  ME   |  STD  |')
    print('-----------------------')
    print('| DBP | {} | {} |'.format(np.round(np.mean(me_dbp), 3), np.round(np.std(me_dbp), 3)))
    print('| MAP | {} | {} |'.format(np.round(np.mean(me_map), 3), np.round(np.std(me_map), 3)))
    print('| SBP | {} | {} |'.format(np.round(np.mean(me_sbp), 3), np.round(np.std(me_sbp), 3)))
    print('-----------------------')

    bland_altman_plot(SBP_predict, SBP_actual, DBP_predict, DBP_actual, MAP_predict, MAP_actual)
    # evaluate_BHS_Standard(me_sbp, me_dbp, me_map)
    # regression_plot(SBP_predict, SBP_actual, DBP_predict, DBP_actual, MAP_predict, MAP_actual)


def evaluate_algorithm_csv():
    # path = '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.14_1.0.0.0/result/R-PTT_fb_cal_v2_v3.0.14.csv'
    # path = '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.14_1.0.0.0/training-v0/rptt_v1_v3.0.14.csv'
    # path = '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.14_1.0.0.0/result/R-PTT_fb_cal_vRPTT0_v3.0.14.csv'  # !!!
    # path = '/mnt/ai_data/PPG2ABP_data_third/model_unet_v3.0.14_1.0.0.0/result/R-PTT_fb_cal_vK_ab_v3.0.14.csv'
    # path = os.path.join(dfm.TRAIN_PATH, 'bp_ai_v{}.csv'.format(dfm.VER_MODEL))
    path = os.path.join(dfm.TRAIN_PATH, 'bp_ai_float_all_v{}.csv'.format(dfm.VER_MODEL))
    # path = os.path.join(dfm.TRAIN_PATH, 'bp_ai_float_test_v{}.csv'.format(dfm.VER_MODEL))
    # path = os.path.join(dfm.TRAIN_PATH, 'bp_ai_float_train_v{}.csv'.format(dfm.VER_MODEL))
    df = pd.read_csv(path)
    key = df.columns.tolist()
    SBP_actual = df[key[1]].to_numpy()
    DBP_actual = df[key[2]].to_numpy()
    print(len(SBP_actual))
    plt.hist(np.vstack((SBP_actual, DBP_actual)).T, bins=38, label=['SBP', 'DBP'])
    plt.xlabel('Blood Pressure (mmHg)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    # plt.grid()
    plt.gcf().set_size_inches(10, 5)
    plt.show()
    # plt.close()

    SBP_predict = df[key[3]].to_numpy()
    DBP_predict = df[key[4]].to_numpy()

    MAP_actual = SBP_actual / 3 + DBP_actual * 2 / 3
    MAP_predict = SBP_predict / 3 + DBP_predict * 2 / 3

    me_sbp = SBP_predict - SBP_actual
    std_sbp = np.std(me_sbp)
    me_dbp = DBP_predict - DBP_actual
    std_dbp = np.std(me_dbp)
    me_map = MAP_predict - MAP_actual
    std_map = np.std(me_map)

    print('-----------------------')
    print('|     |  MAE  |  STD  |')
    print('-----------------------')
    print('| DBP | {} | {} |'.format(np.round(np.mean(abs(me_dbp)), 3), np.round(np.std(abs(me_dbp)), 3)))
    print('| MAP | {} | {} |'.format(np.round(np.mean(abs(me_map)), 3), np.round(np.std(abs(me_map)), 3)))
    print('| SBP | {} | {} |'.format(np.round(np.mean(abs(me_sbp)), 3), np.round(np.std(abs(me_sbp)), 3)))

    # print('STD SBP: {} - DBP: {} - MAP: {}'.format(std_sbp, std_dbp, std_map))

    print('---------------------')
    print('|   AAMI Standard   |')
    print('---------------------')

    print('-----------------------')
    print('|     |  ME   |  STD  |')
    print('-----------------------')
    print('| DBP | {} | {} |'.format(np.round(np.mean(me_dbp), 3), np.round(np.std(me_dbp), 3)))
    print('| MAP | {} | {} |'.format(np.round(np.mean(me_map), 3), np.round(np.std(me_map), 3)))
    print('| SBP | {} | {} |'.format(np.round(np.mean(me_sbp), 3), np.round(np.std(me_sbp), 3)))
    print('-----------------------')

    evaluate_BHS_Standard(me_sbp, me_dbp, me_map, plot_flag=False)
    bland_altman_plot(SBP_predict, SBP_actual, DBP_predict, DBP_actual, MAP_predict, MAP_actual)
    regression_plot(SBP_predict, SBP_actual, DBP_predict, DBP_actual, MAP_predict, MAP_actual)


if __name__ == "__main__":
    # write_evaluate_tfrecord()
    # evaluate_model_csv()
    evaluate_algorithm_csv()
    # evaluate_csv()
