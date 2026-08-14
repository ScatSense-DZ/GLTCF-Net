

"""
    Train the proposed model
    @author: Zhu Dong
    @date: 2026.08.05
"""


import os
import datetime
import pandas as pd
from network.AFIM import AFE_module
from network.OverallNetwork import OverallNetwork
from utils.CallBack import LossHistory
from utils.ClutterDataset import ClutterDataset
from utils.LearningRate import get_lr_scheduler
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainGpu = [0, ]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in trainGpu)
    gpusPerNode = len(trainGpu)
    print('Number of devices: {}'.format(gpusPerNode))

    t_inputShape = (128, 128, 1)

    c_inputShape = (128, 128, 2)

    outputShape = (128, 128, 1)

    date = '20260814'


    batch_size = 16

    epochs = 300

    Init_lr = 0.001
    Min_lr = Init_lr * 0.1

    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 1e-4

    # ------------------------------------------------------------------#
    #   lr_decay_type   可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'


    save_period = 5


    save_dir = os.path.join('logs/OverallNetwork', str(date))
    os.makedirs(save_dir, exist_ok=True)

    eval_flag = True
    eval_period = 5

    dataPath = 'dataset'

    with open(os.path.join(dataPath, "train.txt"), "r") as f:
        trainLines = f.readlines()
    with open(os.path.join(dataPath, "val.txt"), "r") as f:
        valLines = f.readlines()

    if True:
        nbs = 8
        lr_limit_max = 0.01 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 0.001 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': Adam(lr=Init_lr_fit, beta_1=momentum),
            'sgd': SGD(lr=Init_lr_fit, momentum=momentum, nesterov=True)
        }['adam']

        x_t_train, x_c_train, y_train = ClutterDataset(trainLines, t_inputShape, dataPath)
        x_t_val, x_c_val, y_val = ClutterDataset(valLines, t_inputShape, dataPath)

        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        Logging = TensorBoard(log_dir)
        # loss_history    = LossHistory(log_dir)

        CheckpointBest = ModelCheckpoint(os.path.join(save_dir, 'best_epoch_weights_' + str(date) + '.hdf5'),
                                         monitor='val_loss', save_best_only=True, period=1)

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1)
        
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epochs)

        LrScheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)

        model = OverallNetwork(t_inputShape, c_inputShape, kernel_size=3)

        model.compile(optimizer=optimizer, loss='mse', run_eagerly=True, metrics=['accuracy'])

        # Loss_history = LossHistory(model)

        callbacks = [Logging, CheckpointBest, LrScheduler] #,Loss_history

        print('**********************************************************')
        print('******************  Start training  *********************')
        print('**********************************************************')

        history = model.fit([x_t_train, x_c_train], y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=([x_t_val, x_c_val], y_val), callbacks=callbacks)

        print('**********************************************************')
        print('******************  Finish training  *********************')
        print('**********************************************************')

        # pd.DataFrame(history.history).to_csv('result/' + 'loss.csv', index=False)
