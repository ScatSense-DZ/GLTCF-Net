

import math
import os
from random import shuffle
import tensorflow as tf
import numpy as np
from scipy.io import loadmat

# from utils.utils import cvtColor, preprocess_input, preprocess_dsm


class ClutterDataset(tf.keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, train, dataset_path):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        self.input_shape        = input_shape
        self.batch_size         = batch_size
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        images = []
        targets = []
        # print('index is {}', index)
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i = i % self.length
            name = self.annotation_lines[i].split()[0]

            #   从文件中读取数据
            dem = loadmat(os.path.join(os.path.join(self.dataset_path, "dem"), name + ".mat"))
            graz = loadmat(os.path.join(os.path.join(self.dataset_path, "graz"), name + ".mat"))
            sha = loadmat(os.path.join(os.path.join(self.dataset_path, "shadow"), name + ".mat"))
            sig = loadmat(os.path.join(os.path.join(self.dataset_path, "sig"), name + ".mat"))

            #   数据增强

            dem = np.reshape((int(self.input_shape[0]), int(self.input_shape[1]), 1))
            graz = np.reshape((int(self.input_shape[0]), int(self.input_shape[1]), 1))
            sha = np.reshape((int(self.input_shape[0]), int(self.input_shape[1]), 1))
            sig = np.reshape((int(self.input_shape[0]), int(self.input_shape[1]), 1))

            #   数据融合
            temp = np.concatenate([dem, graz, sha], axis=-1)

            images.append(temp)
            targets.append(sig)

        images = np.array(images)
        targets = np.array(targets)
        return images, targets

    def on_epoch_end(self):
        shuffle(self.annotation_lines)
        
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
