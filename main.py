# -*- coding:utf-8 -*-
import numpy as np
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split


def build_deep_cnn(num_classes=2):
    model = Sequential()

    model.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def load_dataset(data_dir):
    train_data = np.load('{0}/train_data.npy'.format(data_dir))
    train_label = np.load('{0}/train_label.npy'.format(data_dir))

    return train_data, train_label


def calc_accuracy(predicted_arr, answer_arr):
    num_data = len(predicted_arr)
    return sum([1.0 for i in range(num_data) if predicted_arr[i] == answer_arr[i]])/len(num_data)

if __name__ == '__main__':
    X, y = load_dataset("./resources/data")

    # シャッフル
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    # X内で並べ替え(cv=>keras)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    train_X = np.array([x.transpose(2, 0, 1) for x in train_X])
    test_X = np.array([x.transpose(2, 0, 1) for x in test_X])

    # modelの準備
    model = build_deep_cnn()
    model.summary()
    init_learning_rate = 1e-2
    opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.85, nesterov=False)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["acc"])

    # callbackの設定
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

    # trainデータでmodelを学習
    hist = model.fit(train_X, train_y,
                     batch_size=100,
                     nb_epoch=50,
                     validation_split=0.1,
                     verbose=1,
                     callbacks=[early_stopping])

    # testデータで検証
    predict_y = model.predict_classes(test_X)
    print(calc_accuracy(predict_y, test_y))
