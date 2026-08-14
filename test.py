

import os
import scipy.io as sci
import tensorflow as tf
from utils.ClutterDataset import ClutterDataset
from tensorflow.keras.models import load_model
from network.AFIM import SwinTransformerBlock


inputShape = (128, 128, 1)
dataPath = 'dataset'
with open(os.path.join(dataPath, "test.txt"), "r") as f:
    valLines = f.readlines()

x_t_test, x_c_test, y_test = ClutterDataset(valLines, inputShape, dataPath)

model = load_model('logs/OverallNetwork/20260810/best_epoch_weights_20260810.hdf5',
                   custom_objects={'SwinTransformerBlock': SwinTransformerBlock,
                                   'tf': tf})

pre1 = model.predict([x_t_test, x_c_test])

sci.savemat("result/test/pre1.mat", {'pre1': pre1})
sci.savemat("result/test/sig.mat", {'y_val': y_test})

