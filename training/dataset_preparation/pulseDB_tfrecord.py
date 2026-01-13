import os.path
import sys
from multiprocessing import Pool

import numpy as np
import json
from mat73 import loadmat
import tensorflow as tf
import model.define_model as dfm

from tfrecord.generate_tfrecord import AIData


def save_tf_record(writer, ppg_total, sbp_total, dbp_total, ppg_calibrate, bp_calibrate):
    for i in range(len(sbp_total)):
        try:
            ppg = ppg_total[i]
            sbp = sbp_total[i]
            dbp = dbp_total[i]
            bp = [sbp, dbp]

            feature = {
                'ppg_target': AIData._float_feature(ppg.tolist()),
                'bp_target': AIData._float_feature(bp),
                'ppg_calibrate': AIData._float_feature(ppg_calibrate.tolist()),
                'bp_calibrate': AIData._float_feature(bp_calibrate),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


def process_work(info, db_dir, tf_record_dir):
    try:
        rd_name = info['subject']
        path = os.path.join(db_dir, '{}.mat'.format(rd_name))
        data = loadmat(path)
        ind = info['ind_seg']
        ppg_total = np.concatenate(data['Subj_Wins']['PPG_F'])
        sbp_total = np.concatenate(data['Subj_Wins']['SegSBP'])
        dbp_total = np.concatenate(data['Subj_Wins']['SegDBP'])

        ppg_calibrate = ppg_total[info['ind_calibrate']][0]
        sbp_calibrate = sbp_total[info['ind_calibrate']][0]
        dbp_calibrate = dbp_total[info['ind_calibrate']][0]
        bp_calibrate = [sbp_calibrate, dbp_calibrate]

        ppg_total = ppg_total[ind]
        sbp_total = sbp_total[ind]
        dbp_total = dbp_total[ind]

        file_name = '{}.tfrecord'.format(rd_name)
        tfrecord_path = os.path.join(tf_record_dir, file_name)
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            save_tf_record(writer, ppg_total, sbp_total, dbp_total, ppg_calibrate, bp_calibrate)
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == "__main__":
    db_dir = '/mnt/ai_data/PulseDB/PulseDB-MIMIC'
    path_json = '/mnt/ai_data/PPG2BP/json/data_1.json'
    tf_record_dir = dfm.DIR_SAVE_TFRECORD
    os.makedirs(tf_record_dir, exist_ok=True)

    with open(path_json, 'r') as openfile:
        info_json = json.load(openfile)
    info_json = info_json['data']
    LEN_INFO_JSON = len(info_json)
    args = []
    for info in info_json:
        args.append((info, db_dir, tf_record_dir))

    with Pool(os.cpu_count() - 3) as pool:
        pool.starmap(process_work, args)
