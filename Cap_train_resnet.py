import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from resnet2 import ResNet
import os
import numpy as np
from six.moves import cPickle
from resnet import resnet18, resnet10, resnet34, resnet48, resnet101

gpu = tf.config.experimental.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)



# 加载数据
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255. -1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def main():                                                           # 1. ！！！注意改变训练  这里要改
    SaveWightPath = r"E:\AOItest\AOIpyOtherMethods\Cap_ResNet\\weights_resnet101_epo15_Data1\resnet18"
    Train_batchsize = 30   # res18 - 80   res48- 50   res101 - 30
    val_batchsize = 30
    (x, y), (x_test, y_test) = Cap_load_data()
    y = tf.squeeze(y, axis=1)  # [n, 1] => [n]
    y_test = tf.squeeze(y_test, axis=1)  # [n, 1] => [n]
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(1000).map(preprocess).batch(Train_batchsize).repeat()
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.shuffle(500).map(preprocess).batch(val_batchsize).repeat()

    network = resnet101()                                            #  2. ！！！注意改变训练  这里要改

    network.build(input_shape=(None, 64, 64, 3))
    network.summary()

    # 用 keras 的高层API直接训练
    network.compile(optimizer='rmsprop',
                  loss=tf.losses.binary_crossentropy,
                  metrics=['accuracy'])

                                                                     #  3. ！！！注意 ，改变训练 检查epochs 是否要改
    network.fit(train_db, epochs=15, verbose=2, steps_per_epoch=x.shape[0] // Train_batchsize,
                validation_steps=x_test.shape[0] // val_batchsize, validation_data=test_db, validation_freq=1)
    network.save_weights(SaveWightPath)
    # network.save('./Cap_model_resnet.h5')

def Cap_load_data():
    # path = r'E:\Deep_learn\Cap_detection\DataSet'
    path = r'E:\Deep_learn\Cap_detection\DataSet1'  # Data1 在Data基础上加了更多的小电容
    fpath = os.path.join(path, 'train.pkl')
    x_train, y_train = load_batch(fpath)
    fpath = os.path.join(path, 'test.pkl')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))  #标签
    y_test = np.reshape(y_test, (len(y_test), 1))

    # if K.image_data_format() == 'channels_last':
    #     x_train = x_train.transpose(0, 2, 3, 1)
    #     x_test = x_test.transpose(0, 2, 3, 1)
    return (x_train, y_train), (x_test, y_test)
def load_batch(fpath):  # 使用cpick 读取文件
    with open(fpath, 'rb') as f:
        d = cPickle.load(f, encoding='bytes')
    data = d['data']
    labels = d['labels']
    data = data.reshape(data.shape[0], 64, 64, 3)
    return data, labels
if __name__ == "__main__":
    main()