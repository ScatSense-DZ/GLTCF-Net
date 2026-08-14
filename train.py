

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

    # 设置用到的显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainGpu = [0, ]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in trainGpu)
    gpusPerNode = len(trainGpu)
    print('Number of devices: {}'.format(gpusPerNode))

    # 输入图片的大小
    t_inputShape = (128, 128, 1)

    c_inputShape = (128, 128, 2)

    outputShape = (128, 128, 1)

    date = '20260814'

    # 训练batch_size
    batch_size = 16

    # 模型总共训练的epoch
    epochs = 300

    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=7e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 0.001
    Min_lr = Init_lr * 0.1

    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=7e-3
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 1e-4

    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'

    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 5

    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = os.path.join('logs/OverallNetwork', str(date))
    os.makedirs(save_dir, exist_ok=True)
    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 5

    #   Potsdam_path   数据集路径
    dataPath = 'dataset'

    #   读取数据集对应的txt
    with open(os.path.join(dataPath, "train.txt"), "r") as f:
        trainLines = f.readlines()
    with open(os.path.join(dataPath, "val.txt"), "r") as f:
        valLines = f.readlines()

    if True:
        #   判断当前batch_size，自适应调整学习率
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

        # 获得学习率下降的公式
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
