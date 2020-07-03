"""FIC dataset"""

import pickle
import csv
import os
import datetime as dt
import logging
import glob
import tensorflow as tf
import numpy as np

FREQUENCY = 100
DEFAULT_LABEL = "Idle"
FLIP_ACC = [-1., 1., 1.]
FLIP_GYRO = [1., -1., -1.]
TIME_FACTOR = 1

TRAIN_IDS = ['1_1','1_2','2_1','2_2','2_3','3_1','3_2','3_3','6_1','6_2','6_3',
  '9_1','10_1','11_1']
VALID_IDS = ['4_1','4_2','4_3','7_1']
TEST_IDS = ['5_1','8_1','12_1']

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class Dataset():

  def __init__(self, src_dir, exp_dir, dom_hand_spec, label_spec,
    label_spec_inherit, exp_uniform, exp_format):
    self.src_dir = src_dir
    self.exp_dir = exp_dir
    self.exp_uniform = exp_uniform
    self.exp_format = exp_format
    # Read data
    pickle_path = os.path.join(src_dir, "fic_pickle.pkl")
    if not os.path.isfile(pickle_path):
      raise RunimeError('Pickle file not found')
    with open(pickle_path, 'rb') as f:
      self._data = pickle.load(f)

  def ids(self):
    return [(self._data['subject_id'][i], self._data['session_id'][i]) \
      for i in range(0, len(self._data['subject_id']))]

  def check(self, id):
    return True

  def data(self, i, id):
    logging.info("Reading processed data from pickle")
    timestamps = self._data['signals'][i][:,0]
    acc = self._data['signals'][i][:,1:4]
    gyro = self._data['signals'][i][:,4:7]
    return timestamps, {"hand": (acc, gyro)}

  def labels(self, i, id, timestamps):
    num = len(timestamps)
    # Read annotations
    annotations = self._data['bite_gt'][i]
    # Read labels
    labels_1 = np.empty(num, dtype='U25')
    labels_1.fill(DEFAULT_LABEL)
    for start_time, end_time in annotations:
      start_frame = np.argmax(np.array(timestamps) >= start_time)
      end_frame = np.argmax(np.array(timestamps) > end_time)
      labels_1[start_frame:end_frame] = "Intake"
    return list(labels_1)

  def dominant(self, id):
    return "NA"

  def write(self, path, id, timestamps, data, _, label_1):
    frame_ids = list(range(0, len(timestamps)))
    id_s = '_'.join([str(x) for x in id])
    def _format_time(t):
      return (dt.datetime.min + dt.timedelta(seconds=t)).time().strftime('%H:%M:%S.%f')
    timestamps = [_format_time(t) for t in timestamps]
    acc = np.asarray(data["hand"][0])
    gyro = np.asarray(data["hand"][1])
    assert len(timestamps) == len(acc), \
      "Number timestamps and acc readings must be equal"
    assert len(timestamps) == len(gyro), \
      "Number timestamps and acc readings must be equal"
    if self.exp_format == 'csv':
      with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "frame_id", "timestamp", "acc_x", "acc_y",
          "acc_z", "gyro_x", "gyro_y", "gyro_z", "label_1"])
        for i in range(0, len(timestamps)):
          writer.writerow([id_s, frame_ids[i], timestamps[i],
            acc[i][0], acc[i][1], acc[i][2], gyro[i][0], gyro[i][1],
            gyro[i][2], label_1[i]])
    elif self.exp_format == 'tfrecord':
      with tf.io.TFRecordWriter(path) as tfrecord_writer:
        for i in range(0, len(timestamps)):
          example = tf.train.Example(features=tf.train.Features(feature={
            'example/subject_id': _bytes_feature(id_s.encode()),
            'example/frame_id': _int64_feature(frame_ids[i]),
            'example/timestamp': _bytes_feature(timestamps[i].encode()),
            'example/acc': _floats_feature(acc[i].ravel()),
            'example/gyro': _floats_feature(gyro[i].ravel()),
            'example/label_1': _bytes_feature(label_1[i].encode())
          }))
          tfrecord_writer.write(example.SerializeToString())

  def done(self):
    logging.info("Done")

  def get_flip_signs(self):
    return FLIP_ACC, FLIP_GYRO

  def get_frequency(self):
    return FREQUENCY

  def get_time_factor(self):
    return TIME_FACTOR

  def get_train_ids(self):
    return TRAIN_IDS

  def get_valid_ids(self):
    return VALID_IDS

  def get_test_ids(self):
    return TEST_IDS
