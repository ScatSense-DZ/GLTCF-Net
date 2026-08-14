

import math
import os
import tensorflow as tf
import numpy as np
from scipy.io import loadmat


def ClutterDataset(annotationLines, inputShape, dataset_path):
    in_a = []
    in_b = []
    targets = []
    length = len(annotationLines)

    for i in range(0, length):
        i = i % length
        # print(i)
        name = annotationLines[i].split()[0]
        # print(name)

        #   从文件中读取数据
        dem = loadmat(os.path.join(os.path.join(dataset_path, 'dem'), name + ".mat"))
        dem = dem['data']
        graz = loadmat(os.path.join(os.path.join(dataset_path, 'graz'), name + ".mat"))
        graz = graz['data']
        sha = loadmat(os.path.join(os.path.join(dataset_path, 'shadow'), name + ".mat"))
        sha = sha['data']
        sig = loadmat(os.path.join(os.path.join(dataset_path, 'sig'), name + ".mat"))
        sig = sig['data']

        # 数据reshape
        dem = np.reshape(dem, (int(inputShape[0]), int(inputShape[1]), 1))
        graz = np.reshape(graz, (int(inputShape[0]), int(inputShape[1]), 1))
        sha = np.reshape(sha, (int(inputShape[0]), int(inputShape[1]), 1))
        sig = np.reshape(sig, (int(inputShape[0]), int(inputShape[1]), 1))

        dem = dem / np.max(dem)
        graz = graz / np.max(graz)
        sig = sig / (-70)

        #  数据融合
        x = np.concatenate([graz, sha], axis=-1)

        in_a.append(dem)
        in_b.append(x)
        targets.append(sig)

    in_a = np.array(in_a)
    in_b = np.array(in_b)

    # print(images.shape)
    targets = np.array(targets)
    # print(targets.shape)

    return in_a, in_b, targets


def ClutterDatasetV2(annotationLines, inputShape, dataset_path):
    in_a = []
    in_b = []
    targets = []
    length = len(annotationLines)

    for i in range(0, length):
        i = i % length
        # print(i)
        name = annotationLines[i].split()[0]
        # print(name)

        #   从文件中读取数据
        dem = loadmat(os.path.join(os.path.join(dataset_path, 'dem'), name + ".mat"))
        dem = dem['data']
        graz = loadmat(os.path.join(os.path.join(dataset_path, 'graz'), name + ".mat"))
        graz = graz['data']
        sha = loadmat(os.path.join(os.path.join(dataset_path, 'shadow'), name + ".mat"))
        sha = sha['data']
        sig = loadmat(os.path.join(os.path.join(dataset_path, 'sig'), name + ".mat"))
        sig = sig['data']

        # 数据reshape
        dem = np.reshape(dem, (int(inputShape[0]), int(inputShape[1]), 1))
        graz = np.reshape(graz, (int(inputShape[0]), int(inputShape[1]), 1))
        sha = np.reshape(sha, (int(inputShape[0]), int(inputShape[1]), 1))
        sig = np.reshape(sig, (int(inputShape[0]), int(inputShape[1]), 1))

        dem = dem / np.max(dem)
        graz = graz / np.max(graz)
        sig = sig / (-70)

        #  数据融合
        x = np.concatenate([dem, graz, sha], axis=-1)

        in_b.append(x)
        targets.append(sig)

    in_b = np.array(in_b)

    # print(images.shape)
    targets = np.array(targets)
    # print(targets.shape)

    return in_b, targets