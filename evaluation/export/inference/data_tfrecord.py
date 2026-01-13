import glob
import os
import json

import numpy as np

import wfdb as wf
import tensorflow as tf

from multiprocessing import Pool
from os.path import join, basename, dirname

class AIData:
    """
    data_strategy: a dictionary with keys are mode ['train', 'valid', 'test'] and value are list of file path
    process_function: function for process file data with arguments are (filename, *process_function_config)
    """

    def __init__(self, save_data_path):

        self.tf_record_path = save_data_path
        self.sample_shape = None
        self.label_shape = None

    @staticmethod
    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def save_tf_record(writer, data, label):
        for i in range(data.shape[0]):
            feature = {
                'sample': AIData.float_feature(data[i].flatten().tolist()),
                'label': AIData.int64_feature(label[i].flatten().tolist())
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    def __parse_batch(self, record_batch):
        feature_description = {
            'sample': tf.io.FixedLenFeature([*self.sample_shape], tf.float32),
            'label': tf.io.FixedLenFeature([*self.label_shape], tf.int64)
        }

        example = tf.io.parse_example(record_batch, feature_description)
        data = example['sample']
        data = tf.reshape(data, (125, 3, 1))
        return data, example['label']

    def load_tf_record(self, mode, batch_size=128):
        if self.sample_shape is None:
            data_info_file = join(self.tf_record_path, 'data_info.json')
            if not os.path.isfile(data_info_file):
                print('Data is unavailable')
                return None
            with open(data_info_file, 'r') as read_file:
                data_info = json.load(read_file)
            self.sample_shape = data_info['summary']['sample_shape']
            self.label_shape = data_info['summary'].get('label_shape')

        files = glob.glob(join(self.tf_record_path, mode, '*.tfrecord'))
        if len(files) < 1:
            return None

        ds = tf.data.TFRecordDataset(files, num_parallel_reads=os.cpu_count())
        ds = ds.map(self.__parse_batch, num_parallel_calls=os.cpu_count())

        if mode == 'train':
            ds = ds.shuffle(batch_size * 2, reshuffle_each_iteration=True)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size * 2)

        return ds
