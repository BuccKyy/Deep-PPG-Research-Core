import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import wfdb

import cross_correlation.preprocessing
from keras.utils import plot_model

from model.Sqeeze_Unet import *
from scipy import stats


def save_h5():
    arg = (dfm.input_length, dfm.num_classes, dfm.deconv_ksize, dfm.model_width, dfm.dropout, dfm.Conv_activation, dfm.activation)
    unet_model = SqeezeUnet1D(*arg).SqueezeUNet()
    unet_model.load_weights(tf.train.latest_checkpoint('/mnt/ai_data/PPG2ABP_data_second/model_unet_3/checkpoint')).expect_partial()
    unet_model.save('/mnt/ai_data/PPG2ABP_data_second/model_unet/unet_v{}.h5'.format(1))


def train_model():
    # arg = (dfm.input_length, dfm.num_classes, dfm.deconv_ksize, dfm.model_width, dfm.dropout, dfm.Conv_activation, dfm.activation)
    # arg = (dfm.input_length, dfm.num_classes, dfm.deconv_ksize, dfm.model_width, dfm.dropout, dfm.Conv_activation, 'tanh')
    # unet_model = SqeezeUnet1D(*arg).SqueezeUNet()

    # handle = AIProcess(unet_model, work_dir=dfm.WORKSPACE)
    handle = AIProcess(work_dir=dfm.WORKSPACE)
    # handle.compile_model()
    handle.train(epochs=dfm.EPOCH, batch_size=dfm.BATCH_SIZE, dir_tfrecord=dfm.DIR_SAVE_TFRECORD, lists_exist=False)
    # handle.train_dat(epochs=50, batch_size=1)

    # _arg = (dfm.input_length, 2, dfm.Conv_activation, dfm.activation, dfm.dropout)
    # cnn_model = CNN(*_arg).cnn_1d()

    # handle = AIProcess(cnn_model, work_dir=dfm.WORKSPACE)
    # # handle.compile_model()
    # handle.train(epochs=50, batch_size=16, dir_tfrecord=dfm.DIR_SAVE_TFRECORD)
    # handle.train_dat(epochs=50, batch_size=1)


def result_from_wfdb():
    arg = (dfm.input_length, dfm.num_classes, dfm.deconv_ksize, dfm.model_width, dfm.dropout, dfm.Conv_activation, dfm.activation)
    unet_model = SqeezeUnet1D(*arg).SqueezeUNet()
    dir_wfdb = '/mnt/ai_data/PPG2ABP_data_second/Cuff-Less_Blood_Pressure_Estimation'
    list_db = list(map(lambda path: path.rstrip('.hea'), glob.glob(dir_wfdb + '/*.hea')))
    list_train, list_test = train_test_split(list_db, test_size=0.2, shuffle=True, random_state=11)
    list_train, list_valid = train_test_split(list_train, test_size=0.25, shuffle=True, random_state=11)

    unet_model.load_weights(tf.train.latest_checkpoint('/mnt/ai_data/PPG2ABP_data_second/model_unet_6/checkpoint')).expect_partial()

    tf_data = AIData(dir_wfdb, dfm.input_length)
    ppg, abp = tf_data.process_data(list_train[10])
    seg = np.arange(0, 256, 1)[None, :] + np.arange(0, len(ppg) - 256, 256 - 0)[:, None]
    seg = seg.astype(int)

    ppg = ppg[seg]
    abp = abp[seg]

    abp_predict = unet_model.predict(ppg)

    plt.subplot(211)
    ppg = ppg * 5.3 - 2.4
    plt.plot(ppg.flatten(), color='b', label='ppg')
    plt.legend(loc='upper right')
    plt.title('PPG')
    plt.grid()

    plt.subplot(212)
    # predict = abp_predict * 141.91 + 50.33
    # actual = np.asarray(abp).flatten() * 141.91 + 50.33
    predict = abp_predict
    actual = np.asarray(abp).flatten()
    plt.plot(predict.flatten(), color='r', label='predict')
    plt.plot(actual, color='k', linestyle='--', label='actual')
    plt.legend(loc='upper right')
    plt.title('\n\nABP')

    plt.grid()

    plt.gcf().set_size_inches(18, 8)
    plt.show()
    plt.close()


def result_predict():
    arg = (dfm.input_length, dfm.num_classes, dfm.deconv_ksize, dfm.model_width, dfm.dropout, dfm.Conv_activation, dfm.activation)
    unet_model = SqeezeUnet1D(*arg).SqueezeUNet()

    # from New_Model import build_new_model
    # unet_model = build_new_model()

    unet_model.load_weights(tf.train.latest_checkpoint(os.path.join(dfm.WORKSPACE, 'checkpoint'))).expect_partial()

    path_json = glob.glob(os.path.join(dfm.WORKSPACE, 'INFO_train_valid_test') + '/*.json')
    with open(path_json[0], 'r') as openfile:
        rd = json.load(openfile)

    # tfrecord_test_dir = rd.get('train')
    tfrecord_test_dir = rd.get('test')

    tf_data = AIData(dfm.DIR_SAVE_TFRECORD, dfm.input_length)
    # tf_data = AIData(dfm.DIR_SAVE_TFRECORD, 125 * 7)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_12/AI_data_part_1_721.tfrecord', batch_size=10)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_7/AI_data_part_1_1676.tfrecord', batch_size=10)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_7/AI_data_part_1_2422.tfrecord', batch_size=10)

    for path in tfrecord_test_dir:
        # if not 'AI_data_part_1_1258' or 'AI_data_part_1_2801' in path:
        # if 'AI_data_part_1_1258' not in path:
        # # if 'AI_data_part_1_2649' not in path:
        # if '1_1676' not in path:
        # if '1_1124' not in path:
        #     continue
        print('\n{}'.format(os.path.basename(path).rstrip('.tfrecord')))
        # ds = tf_data.get_dataset_from_tfrecord(path, batch_size=1)
        # ppg, abp = [], []
        # for index, batch in enumerate(ds):
        #     list_tensors = [i for i in batch]
        #     ppg.extend([tensor.numpy() for tensor in list_tensors[0]])
        #     abp.extend([tensor.numpy() for tensor in list_tensors[1]])
        # _ppg = np.asarray(ppg)

        ppg, abp = tf_data.get_numpy_from_tfrecord_2(path, dfm.BATCH_SIZE)
        # _ppg = ppg[:, :, np.newaxis]
        _ppg = ppg
        abp_predict = unet_model.predict(_ppg)


        x_0 = 20000  # 7000
        x_1 = 22500  # 10500
        plt.subplot(311)
        ppg_0 = _ppg[:, :, 0]  # * 5.3 - 2.4
        plt.plot(ppg_0.flatten(), color='b', label='ppg')
        plt.legend(loc='upper right')
        # plt.xlim((x_0, x_1))
        # plt.ylim(0, 1)
        plt.title('PPG')
        plt.grid()

        plt.subplot(312)
        predict = abp_predict.flatten()  #* (dfm.ABP_MAX - dfm.ABP_MIN) + dfm.ABP_MIN
        actual = np.asarray(abp).flatten()   #* (dfm.ABP_MAX - dfm.ABP_MIN) + dfm.ABP_MIN

        # predict = np.asarray(abp_predict).flatten()
        # actual = np.asarray(abp).flatten()
        # plt.plot(_ppg.flatten(), color='b', label='ppg')
        plt.plot(predict, color='r', label='predict')
        plt.plot(actual, color='k', linestyle='--', alpha=0.6, label='actual')
        plt.legend(loc='upper right')
        # plt.xlim((x_0, x_1))
        # plt.ylim(0, 1)
        plt.grid()
        plt.title('\n\nABP')

        plt.subplot(313)
        predict_smooth = cross_correlation.preprocessing.smooth(predict)
        # plt.plot(_ppg.flatten(), color='b', label='ppg')
        plt.plot(predict_smooth, color='r', label='predict')
        plt.plot(actual, color='k', linestyle='--', alpha=0.6, label='actual')
        plt.legend(loc='upper right')
        # plt.xlim((x_0, x_1))
        # plt.ylim(0, 1)
        plt.grid()
        plt.title('\n\nABP_smooth')

        ME = np.mean(predict - actual)
        STD = np.std(predict - actual)
        plt.suptitle('{} | ME: {} - STD: {}'.format(os.path.basename(path).rstrip('.tfrecord'), np.round(ME, 3), np.round(STD, 3)), fontsize=20)
        plt.gcf().set_size_inches(18, 8)
        plt.show()
        # plt.close()


def result_predict_test():
    arg = (dfm.input_length, dfm.num_classes, dfm.deconv_ksize, dfm.model_width, dfm.dropout, dfm.Conv_activation, dfm.activation)
    unet_model = SqeezeUnet1D(*arg).SqueezeUNet_test()

    # from New_Model import build_new_model
    # unet_model = build_new_model()

    unet_model.load_weights(tf.train.latest_checkpoint(os.path.join(dfm.WORKSPACE, 'checkpoint'))).expect_partial()

    path_json = glob.glob(os.path.join(dfm.WORKSPACE, 'INFO_train_valid_test') + '/*.json')
    with open(path_json[0], 'r') as openfile:
        rd = json.load(openfile)

    tfrecord_test_dir = rd.get('test')
    # tfrecord_test_dir = rd.get('train')

    tf_data = AIData(dfm.DIR_SAVE_TFRECORD, dfm.input_length)
    # tf_data = AIData(dfm.DIR_SAVE_TFRECORD, 125 * 7)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_12/AI_data_part_1_721.tfrecord', batch_size=10)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_7/AI_data_part_1_1676.tfrecord', batch_size=10)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_7/AI_data_part_1_2422.tfrecord', batch_size=10)

    for path in tfrecord_test_dir:
        # if not 'AI_data_part_1_1258' or 'AI_data_part_1_2801' in path:
        # if 'AI_data_part_1_1258' not in path:
        # # if 'AI_data_part_1_2649' not in path:
        # if '1_1676' not in path:
        #     continue
        print('\n{}'.format(os.path.basename(path).rstrip('.tfrecord')))
        # ds = tf_data.get_dataset_from_tfrecord(path, batch_size=1)
        # ppg, abp = [], []
        # for index, batch in enumerate(ds):
        #     list_tensors = [i for i in batch]
        #     ppg.extend([tensor.numpy() for tensor in list_tensors[0]])
        #     abp.extend([tensor.numpy() for tensor in list_tensors[1]])
        # _ppg = np.asarray(ppg)

        ppg, abp = tf_data.get_numpy_from_tfrecord_2(path, dfm.BATCH_SIZE)
        # _ppg = ppg[:, :, np.newaxis]
        _ppg = ppg
        abp_predict = unet_model.predict(_ppg)

        x_0 = 20000  # 7000
        x_1 = 22500  # 10500
        plt.subplot(311)
        ppg_0 = _ppg[:, :, 0]  # * 5.3 - 2.4
        plt.plot(ppg_0.flatten(), color='b', label='ppg')
        plt.legend(loc='upper right')
        # plt.xlim((x_0, x_1))
        # plt.ylim(0, 1)
        plt.title('PPG')
        plt.grid()

        plt.subplot(312)
        predict = abp_predict.flatten() #* (dfm.ABP_MAX - dfm.ABP_MIN) + dfm.ABP_MIN
        actual = np.asarray(abp).flatten() #* (dfm.ABP_MAX - dfm.ABP_MIN) + dfm.ABP_MIN

        # predict = np.asarray(abp_predict).flatten()
        # actual = np.asarray(abp).flatten()
        # plt.plot(_ppg.flatten(), color='b', label='ppg')
        plt.plot(predict, color='r', label='predict')
        plt.plot(actual, color='k', linestyle='--', alpha=0.8, label='actual')
        plt.legend(loc='upper right')
        # plt.xlim((x_0, x_1))
        # plt.ylim(0, 1)
        plt.grid()
        plt.title('\n\nABP')

        plt.subplot(313)
        predict_smooth = cross_correlation.preprocessing.smooth(predict)
        # plt.plot(_ppg.flatten(), color='b', label='ppg')
        plt.plot(predict_smooth, color='r', label='predict')
        plt.plot(actual, color='k', linestyle='--', alpha=0.8, label='actual')
        plt.legend(loc='upper right')
        # plt.xlim((x_0, x_1))
        # plt.ylim(0, 1)
        plt.grid()
        plt.title('\n\nABP_smooth')

        ME = np.mean(np.abs(predict - actual))
        STD = np.std(predict - actual)
        plt.suptitle('{} | ME: {} - STD: {}'.format(os.path.basename(path).rstrip('.tfrecord'), np.round(ME, 3), np.round(STD, 3)), fontsize=20)
        plt.gcf().set_size_inches(18, 8)
        plt.show()
        plt.close()


def round_5(x, base=5):
    return int(base * round(x / base))


def evalute_result(SBP_predict, DBP_predict, SBP_actual, DBP_actual):
    SBP_predict = np.asarray(SBP_predict)
    DBP_predict = np.asarray(DBP_predict)
    SBP_actual = np.asarray(SBP_actual)
    DBP_actual = np.asarray(DBP_actual)

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
    from model.test_evaluate import evaluate_BHS_Standard, bland_altman_plot, regression_plot
    evaluate_BHS_Standard(me_sbp, me_dbp, me_map, plot_flag=False)

    bland_altman_plot(SBP_predict, SBP_actual, DBP_predict, DBP_actual, MAP_predict, MAP_actual)
    regression_plot(SBP_predict, SBP_actual, DBP_predict, DBP_actual, MAP_predict, MAP_actual)


def result_predict_1(ai_model):
    # arg = (dfm.input_length, dfm.num_classes, dfm.deconv_ksize, dfm.model_width, dfm.dropout, dfm.Conv_activation, dfm.activation)
    # unet_model = SqeezeUnet1D(*arg).SqueezeUNet()
    # unet_model = SqeezeUnet1D(*arg).SqueezeUNet_test()
    # _arg = (dfm.input_length, 2, dfm.Conv_activation, dfm.activation, dfm.dropout)
    # unet_model = CNN(*_arg).ANN_new()


    # from ResNet_1DCNN import build_resnet
    # ai_model = build_resnet()

    ai_model.load_weights(tf.train.latest_checkpoint(os.path.join(dfm.WORKSPACE, 'checkpoint'))).expect_partial()

    path_json = glob.glob(os.path.join(dfm.WORKSPACE, 'INFO_train_valid_test') + '/*.json')
    with open(path_json[0], 'r') as openfile:
        rd = json.load(openfile)

    tfrecord_test_dir = rd.get('train')
    # tfrecord_test_dir = rd.get('valid')
    # random.shuffle(tfrecord_test_dir)
    # tfrecord_test_dir = glob.glob(dfm.DIR_SAVE_TFRECORD + '/*.tfrecord')

    tf_data = AIData(dfm.DIR_SAVE_TFRECORD, dfm.input_length)

    S_ac, D_ac = [], []
    S_p, D_p = [], []

    for path in tfrecord_test_dir:
        # if '1_1676' not in path:
        # if '1_1419' in path:
        #     continue
        print('\n{}'.format(os.path.basename(path).rstrip('.tfrecord')))
        ds = tf_data.get_dataset_from_tfrecord_test_result(path, batch_size=dfm.BATCH_SIZE)
        ppg, ppg_raw, mag = [], [], []
        SBP, DBP = [], []
        HR = []
        for index, batch in enumerate(ds):
            list_tensors = [i for i in batch]
            ppg.extend([tensor.numpy() for tensor in list_tensors[0]])
            # ppg_raw.extend([tensor.numpy() for tensor in list_tensors[1]])

            SBP.extend([tensor.numpy() for tensor in list_tensors[1]][0])
            DBP.extend([tensor.numpy() for tensor in list_tensors[1]][1])
            HR.extend([tensor.numpy() for tensor in list_tensors[2]])
            mag.extend([tensor.numpy() for tensor in list_tensors[3]])


        _ppg = np.asarray(ppg)
        _ppg_raw = np.asarray(ppg_raw)
        mag = np.asarray(mag)
        SBP = np.asarray(SBP).flatten()
        DBP = np.asarray(DBP).flatten()
        HR = np.asarray(HR).flatten()

        # in_model = np.concatenate((_ppg, _ppg_calib), axis=2)
        in_model = mag / 100
        in_model = np.expand_dims(in_model, 3)
        # in_model = (_ppg, HR)

        result_predict = ai_model.predict(in_model)
        result_predict = np.asarray(result_predict)

        SBP_predict = result_predict[0, :, 0]
        DBP_predict = result_predict[1, :, 0]

        SBP_predict += np.mean(SBP[:5]) - np.mean(SBP_predict)
        DBP_predict += np.mean(DBP[:5]) - np.mean(DBP_predict)

        S_ac.extend(SBP)
        D_ac.extend(DBP)
        S_p.extend(SBP_predict)
        D_p.extend(DBP_predict)

        is_plot = True
        if is_plot:
            plt.subplot(311)
            plt.plot(_ppg.flatten(), color='b', label='ppg')
            # plt.plot(_ppg_calib.flatten()+3, color='r', label='ppg_calib')
            plt.legend(loc='upper right')
            # plt.xlim((x_0, x_1))
            # plt.ylim(0, 1)
            plt.title('PPG')
            plt.grid()

            plt.subplot(312)
            plt.plot(SBP_predict, color='r', label='SBP_predict')
            plt.plot(SBP, color='k', linestyle='--', label='SBP_actual')
            plt.legend(loc='upper right')
            e = SBP_predict - SBP
            plt.title('\n\nSBP | ME: {}  STD: {}'.format(np.mean(e), np.std(e)))
            plt.grid()

            plt.subplot(313)
            plt.plot(DBP_predict, color='r', label='DBP_predict')
            plt.plot(DBP, color='k', linestyle='--', label='DBP_actual')
            plt.legend(loc='upper right')
            e = DBP_predict - DBP
            plt.title('\n\nDBP | ME: {}  STD: {}'.format(np.mean(e), np.std(e)))
            plt.grid()

            plt.suptitle(os.path.basename(path).rstrip('.tfrecord'), fontsize=20)
            plt.gcf().set_size_inches(16, 9)
            plt.show()
            plt.close()

    # plot_histogram_SD_actual(np.asarray([S_ac, D_ac]).T)
    evalute_result(S_p, D_p, S_ac, D_ac)


def plot_histogram_SD_actual(SD):
    # plt.hist(SD.flatten(), bins=100, label='PPG')
    plt.hist(SD, bins=38, label=['SBP', 'DBP'])
    plt.xlabel('Blood Pressure (mmHg)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    # plt.grid()
    plt.gcf().set_size_inches(10, 5)
    plt.show()
    # plt.close()


def _result_predict_1(ai_model):

    # from ann_model import ann_rptt2bp, ann_rptt2bp_calibrated
    # ai_model = ann_rptt2bp_calibrated()

    ai_model.load_weights(tf.train.latest_checkpoint(os.path.join(dfm.WORKSPACE, 'checkpoint'))).expect_partial()
    # ai_model = tf.keras.models.load_model('/mnt/ai_data/PPG2ABP_data_fourth/model_5.0.0.4/save_model')

    path_json = glob.glob(os.path.join(dfm.WORKSPACE, 'INFO_train_valid_test') + '/*.json')
    with open(path_json[0], 'r') as openfile:
        rd = json.load(openfile)

    tfrecord_test_dir = rd.get('test')
    # tfrecord_test_dir = rd.get('train')

    tf_data = AIData(dfm.DIR_SAVE_TFRECORD, dfm.input_length)

    for path in tfrecord_test_dir:
        # if '1_1676' not in path:
        # if '1_1419' in path:
        #     continue
        print('\n{}'.format(os.path.basename(path).rstrip('.tfrecord')))
        ds = tf_data.get_dataset_from_tfrecord_test_result_4(path, batch_size=dfm.BATCH_SIZE*4)
        ppg, RPTT = [], []
        SBP, DBP = [], []
        Ka, Kb = [], []
        HR = []
        abp = []
        calib = []
        ppg_calib = []
        for index, batch in enumerate(ds):
            list_tensors = [i for i in batch]
            ppg.extend([tensor.numpy() for tensor in list_tensors[0]])
            # RPTT.extend([tensor.numpy() for tensor in list_tensors[1]])
            # Ka.extend([tensor.numpy() for tensor in list_tensors[2]][0])
            # Kb.extend([tensor.numpy() for tensor in list_tensors[2]][1])
            SBP.extend([tensor.numpy() for tensor in list_tensors[1]][0])
            DBP.extend([tensor.numpy() for tensor in list_tensors[1]][1])
            abp.extend([tensor.numpy() for tensor in list_tensors[2]])
            calib.extend([tensor.numpy() for tensor in list_tensors[3]])
            ppg_calib.extend([tensor.numpy() for tensor in list_tensors[4]])
            HR.extend([tensor.numpy() for tensor in list_tensors[5]])


            # HR.extend([tensor.numpy() for tensor in list_tensors[4]])

        _ppg = np.asarray(ppg)
        abp = np.asarray(abp)
        calib = np.concatenate([calib])
        Ka = np.asarray(Ka).flatten()
        Kb = np.asarray(Kb).flatten()
        SBP = np.asarray(SBP).flatten()
        DBP = np.asarray(DBP).flatten()
        RPTT = np.asarray(RPTT).flatten()
        HR = np.asarray(HR).flatten()

        Ka[:] = np.median(Ka)
        Kb[:] = np.median(Kb)

        # input = np.vstack((RPTT, HR, Ka, Kb)).T
        # input_model = (_ppg, calib)
        input_model = ppg_calib

        # custom
        input_model = np.concatenate([input_model])

        result_predict = ai_model.predict(input_model)
        result_predict = np.asarray(result_predict)
        result_predict = np.squeeze(result_predict)
        Ratio_predict = result_predict

        # SBP_predict = result_predict.copy()
        # SBP_predict = result_predict[0, :]
        # DBP_predict = result_predict[1, :]


        plt.subplot(411)
        plt.plot(_ppg.flatten(), label='PPG')
        plt.legend(loc='upper right')
        # plt.xlim((x_0, x_1))
        # plt.ylim(0, 1)
        plt.title('PPG')
        plt.grid()

        plt.subplot(412)
        plt.plot(abp.flatten(), color='r', label='SBP_predict')
        plt.legend(loc='upper right')
        plt.grid()

        # plt.subplot(413)
        # plt.plot(SBP_predict, color='r', label='SBP_predict')
        # plt.plot(SBP, color='k', linestyle='--', label='SBP_actual')
        # # plt.ylim(np.min(SBP)-10, np.max(SBP)+10)
        # plt.legend(loc='upper right')
        # e = SBP_predict - SBP
        # plt.title('\n\nSBP | ME: {}  STD: {}'.format(np.mean(e), np.std(e)))
        # plt.grid()

        plt.subplot(414)
        # plt.plot(DBP_predict, color='r', label='DBP_predict')
        # plt.plot(DBP, color='k', linestyle='--', label='DBP_actual')
        plt.plot(Ratio_predict, color='r', label='predict')
        plt.plot(HR, color='k', linestyle='--')
        # plt.ylim(np.min(DBP)-10, np.max(DBP)+10)
        plt.legend(loc='upper right')
        # e = DBP_predict - DBP
        # plt.title('\n\nDBP | ME: {}  STD: {}'.format(np.mean(e), np.std(e)))
        plt.grid()

        plt.suptitle(os.path.basename(path).rstrip('.tfrecord'), fontsize=20)
        plt.gcf().set_size_inches(16, 9)
        plt.show()
        plt.close()


def result_predict_2():
    from ann_model import ann
    ai_model = ann()

    # from ResNet_1DCNN import build_resnet
    # ai_model = build_resnet()

    ai_model.load_weights(tf.train.latest_checkpoint(os.path.join(dfm.WORKSPACE, 'checkpoint'))).expect_partial()

    path_json = glob.glob(os.path.join(dfm.WORKSPACE, 'INFO_train_valid_test') + '/*.json')
    with open(path_json[0], 'r') as openfile:
        rd = json.load(openfile)

    # tfrecord_test_dir = rd.get('test')
    tfrecord_test_dir = rd.get('train')

    tf_data = AIData(dfm.DIR_SAVE_TFRECORD, dfm.input_length)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_12/AI_data_part_1_721.tfrecord', batch_size=10)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_7/AI_data_part_1_1676.tfrecord', batch_size=10)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_7/AI_data_part_1_2422.tfrecord', batch_size=10)

    for path in tfrecord_test_dir:
        # if '1_1676' not in path:
        # if '1_1419' in path:
        #     continue
        print('\n{}'.format(os.path.basename(path).rstrip('.tfrecord')))
        ds = tf_data.get_dataset_from_tfrecord_test_result(path, batch_size=dfm.BATCH_SIZE)
        ppg, RPTT = [], []
        SBP, DBP = [], []
        Ka, Kb = [], []
        HR = []
        for index, batch in enumerate(ds):
            list_tensors = [i for i in batch]
            ppg.extend([tensor.numpy() for tensor in list_tensors[0]])
            RPTT.extend([tensor.numpy() for tensor in list_tensors[1]])
            Ka.extend([tensor.numpy() for tensor in list_tensors[2]][0])
            Kb.extend([tensor.numpy() for tensor in list_tensors[2]][1])
            SBP.extend([tensor.numpy() for tensor in list_tensors[3]][0])
            DBP.extend([tensor.numpy() for tensor in list_tensors[3]][1])
            HR.extend([tensor.numpy() for tensor in list_tensors[4]])

        _ppg = np.asarray(ppg)

        Ka = np.asarray(Ka).flatten()
        Kb = np.asarray(Kb).flatten()
        SBP = np.asarray(SBP).flatten()
        DBP = np.asarray(DBP).flatten()
        RPTT = np.asarray(RPTT).flatten()
        HR = np.asarray(HR).flatten()
        # SBP = abp[ :, 0]
        # DBP = abp[1, :, 0]
        # hr = abp[2, :, 0]

        # abp = np.vstack((SBP, DBP)).T

        # abp_predict = ai_model.predict(_ppg)
        result_predict = ai_model.predict(_ppg)
        result_predict = np.asarray(result_predict)
        RPTT_predict = result_predict[0, :, 0] * dfm.FS
        HR_predict = result_predict[1, :, 0]

        DBP_predict = Kb - 2 / 0.031 * np.log(RPTT_predict) - Ka / (3 * RPTT_predict * RPTT_predict)
        SBP_predict = DBP_predict + Ka / (RPTT_predict * RPTT_predict)

        # from cross_correlation.preprocessing import smooth
        # DBP_predict = smooth(DBP_predict)
        # SBP_predict = smooth(SBP_predict)

        x_0 = 11500  # 2800
        x_1 = 14000  # 19800


        plt.subplot(321)
        plt.plot(_ppg.flatten(), color='b', label='ppg')
        plt.legend(loc='upper right')
        # plt.xlim((x_0, x_1))
        # plt.ylim(0, 1)
        plt.title('PPG')
        plt.grid()

        plt.subplot(323)
        plt.plot(RPTT_predict, color='r', label='predict')
        plt.plot(RPTT, color='k', linestyle='--', label='actual')
        plt.legend(loc='upper right')
        e = RPTT_predict - RPTT
        ME = str(np.round(np.mean(e), 2))
        STD = str(np.round(np.std(e), 2))
        print(ME)

        plt.title('\n\nRPTT | ME: {}  STD: {}'.format(ME, STD))
        plt.grid()

        plt.subplot(324)
        plt.plot(HR_predict, color='r', label='HR_predict')
        plt.plot(HR, color='k', linestyle='--', label='HR_actual')
        plt.legend(loc='upper right')
        e = HR_predict - HR
        plt.title('\n\nHR | ME: {}  STD: {}'.format(np.mean(e), np.std(e)))
        plt.grid()

        plt.subplot(325)
        plt.plot(SBP_predict, color='r', label='SBP_predict')
        plt.plot(SBP, color='k', linestyle='--', label='SBP_actual')
        plt.legend(loc='upper right')
        e = SBP_predict - SBP
        plt.title('\n\nSBP | ME: {}  STD: {}'.format(np.mean(e), np.std(e)))
        plt.grid()

        plt.subplot(326)
        plt.plot(DBP_predict, color='r', label='DBP_predict')
        plt.plot(DBP, color='k', linestyle='--', label='DBP_actual')
        plt.legend(loc='upper right')
        e = DBP_predict - DBP
        plt.title('\n\nDBP | ME: {}  STD: {}'.format(np.mean(e), np.std(e)))
        plt.grid()

        plt.suptitle(os.path.basename(path).rstrip('.tfrecord'), fontsize=20)
        plt.gcf().set_size_inches(16, 9)
        plt.show()
        plt.close()

def analyze_data():
    from tqdm import tqdm
    from cross_correlation.preprocessing import lowpass_remez, highpass_remez

    DIR_DB = '/mnt/ai_data/PPG2ABP_data_second/Cuff-Less_Blood_Pressure_Estimation'
    db = list(map(lambda path: path.rstrip('.hea'), glob.glob(DIR_DB + '/*Part_1*.hea')))
    # db = list(map(lambda path: path.rstrip('.hea'), glob.glob(DIR_DB + '/*.hea')))

    x_max = np.empty(2)
    x_min = [100, 100]

    def plot_ppg_abp(ppg, abp, ppg_dc=None, abp_dc=None):
        ppg = ppg.flatten()  # * 5.3 - 2.4
        abp = np.asarray(abp).flatten()
        op = ((np.max(abp) - np.min(abp)) / 2 + np.min(abp))

        ppg, y_first, y_second, abp = AIData.derivative(ppg, abp)
        print('first: {} - {}'.format(np.max(y_first), np.min(y_first)))
        print('second: {} - {}'.format(np.max(y_second), np.min(y_second)))
        # y_first = (y_first + 2.8)/8
        # y_second = (y_second + 80) / (95+80)

        y = np.stack((ppg, y_first, y_second), axis=1)

        peaks, _ = scipy.signal.find_peaks(abp, height=op, distance=30)
        troughs, _ = scipy.signal.find_peaks(-abp, distance=50)

        # if np.std(abp[peaks]) > 3 or np.std(abp[troughs]) > 3:
        #     raise Exception('bad')

        plt.subplot(311)
        plt.plot(ppg, color='b', label='ppg')
        plt.plot(y_first, '--', label='1st')
        plt.plot(y_second, '--', label='2nd')
        # plt.plot(ppg_dc)
        if ppg_dc is not None:
            plt.plot(ppg_dc, color='red', label='abp_dc')

        # kte_i = compute_KTE(ppg*10)
        # plt.plot(kte_i, color='r', linestyle='--', label='KTE')
        plt.legend(loc='upper right')
        # plt.xlim((x_0, x_1))
        # plt.ylim(0, 1)
        plt.title('PPG')
        # plt.title('ABP - raw')
        plt.grid()

        plt.subplot(312)
        # abp = np.asarray(abp).flatten() * 141.91 + 50.33
        plt.plot(abp, color='r', label='abp')
        if abp_dc is not None:
            plt.plot(abp_dc, linestyle='--', label='abp_dc')
            plt.axhline(y=np.mean(abp_dc), label='mean_abp_dc', color='green')
        # plt.plot(ppg.flatten(), color='b', linestyle='--', alpha=0.4, label='abp_raw')

        # plt.axhline(y=op, color='green')

        # plt.plot(peaks, abp[peaks], 'x', label='peaks')
        # plt.plot(troughs, abp[troughs], 'x', label='troughs')
        plt.legend(loc='upper right')
        # plt.xlim((x_0, x_1))
        # plt.ylim(0, 1)
        # plt.title('\n\nABP | std peaks: {} | std peaks: {}'.format(np.std(abp[peaks]), np.std(abp[troughs])))
        plt.title('\n\nABP')
        plt.grid()

        plt.subplot(313)
        plt.plot(ppg, color='r', label='ppg')
        plt.plot(abp, color='k', linestyle='--', alpha=0.6, label='abp')
        plt.legend(loc='upper right')

        plt.title('\n\nPPG-ABP')
        plt.grid()

    def norm(sig, s_type='ppg'):
        x_min = 0
        x_max = 1
        if s_type == 'ppg':
            x_min = -2.4
            x_max = 2.9
        if s_type == 'abp':
            x_min = 50.33
            x_max = 192.24
        sig = (sig - x_min) / (x_max - x_min)
        return sig

    save_dir = '/mnt/ai_data/PPG2ABP_data_second/pic_analyze_data/'
    for i in tqdm(range(len(db))):
        # if '1_1676' not in db[i]:
        #     continue
        info = wfdb.rdsamp(db[i])
        # if os.path.exists(save_dir + os.path.basename(db[i]) + '.png'):
        #     continue
        try:
            # sig_ppg = lowpass_remez(info[0][:, 1])
            sig_ppg = info[0][:, 1]
            sig_abp = info[0][:, 2]
            # ppg, abp = sig_ppg, sig_abp
            # ppg, abp, ppg_dc, abp_dc = AIData.corr_process_2(sig_ppg, sig_abp)
            ppg, abp = AIData.corr_process_2(sig_ppg, sig_abp)

            # if len(ppg) < 30000:
            #     raise Exception('{} - len ppg after process <30000'.format(os.path.basename(db[i])))
            ppg = norm(ppg, 'ppg')
            abp = norm(abp, 'abp')
            r = stats.pearsonr(ppg, abp)[0]

            plt.suptitle('{} r={}'.format(os.path.basename(db[i]), r), fontsize=16)

            plot_ppg_abp(ppg, abp)
            plt.gcf().set_size_inches(11, 8)
            plt.show()
            plt.close()
            # plt.savefig(save_dir + os.path.basename(db[i]) + '.png')
            # plt.clf()

        except Exception as e:
            print(e)
            # continue
    # print('PPG: {} | {}'.format(x_min[0], x_max[0]))
    # print('ABP: {} | {}'.format(x_min[1], x_max[1]))
    # x_max = np.empty(2)
    # x_min = np.empty(2)
    # ppg, abp = [], []
    # tf_data = AIData(dfm.DIR_SAVE_TFRECORD, 256)
    # ds = tf_data.get_dataset_from_tfrecord('/mnt/ai_data/PPG2ABP_data_second/data_tfrecord_2/AI_data_11.tfrecord', batch_size=10)
    # for index, batch in tqdm(enumerate(ds)):
    #     list_tensors = [i for i in batch]
    #     ppg.extend([tensor.numpy() for tensor in list_tensors[0]])
    #     abp.extend([tensor.numpy() for tensor in list_tensors[1]])
    # x_max[0] = np.max(ppg)
    # x_max[1] = np.max(abp)
    # x_min[0] = np.min(ppg)
    # x_min[1] = np.min(abp)
    # print(x_min)
    # print(x_max)


if __name__ == "__main__":
    # from lstm_model import lstm, lstm_8s, lstm_old
    from ann_model import ann, ann_v308, ann_old, cnn_test, cnn_test_3_26, cnn_test_3_25_1, cnn_4_0_1, cnn_4_0_3, cnn_4_0_4, cnn_4_0_5, cnn_4_0_5_3, cnn_4_0_5_4, cnn_4_0_8_2, cnn_4_0_8_0, cnn_4_0_8_1, cnn_4_0_8_4
    # ai_model = cnn_4_0_8_2()
    from ResNet_1DCNN import build_resnet
    ai_model = build_resnet()
    # ai_model.summary()
    # path_pic = 'architecture.png'
    # plot_model(ai_model, to_file=path_pic, show_shapes=True)

    # train_model()
    # result_predict()

    _result_predict_1(ai_model)
    # result_predict_1(ai_model)  # run this
    # result_predict_2()
    # result_from_wfdb()
    # analyze_data()
