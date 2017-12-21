# Since the structure of the neural network is almost identical to that of the 10-classification NN, from the result we can see
# that the accuracy of the binary classifier both on training data and test data are far higher than that of the 10-classification NN.
# So, the binary classification task is easier than classification into 10 categories because there should be more training epoches for
# the 10-classification NN to get such high accuracy as that of the binary classification.

import numpy as np
import tensorflow as tf
import json
import h5py
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Input, Flatten, Conv2D, MaxPooling1D, Dropout, Conv1D, MaxPooling2D,Dense
from keras.models import Model
from keras import optimizers


def load_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test
    ytrain_1hot = np.zeros((ytrain.size, 10))
    for i in range(ytrain.size):
        ytrain_1hot[i][int(ytrain[i])] = 1
    ytest_1hot = np.zeros((ytest.size, 10))
    for i in range(ytest.size):
        ytest_1hot[i][int(ytest[i])] = 1
    xtrain = xtrain / 255
    xtest = xtest / 255
    return xtrain, ytrain_1hot, xtest, ytest_1hot


def build_multilayer_nn():
    nn = Sequential()
    nn.add(Flatten(input_shape=(32, 32, 3)))
    nn.add(Dense(units=100, activation="relu"))
    nn.add(Dense(units=10, activation="softmax"))
    nn.summary()
    return nn


def train_multilayer_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=20, batch_size=32)


def build_convolution_nn():
    a=Input(shape=(10, 10))
    b=Conv1D(10, (2), activation='relu', padding="same")(a)
    c=Conv1D(10, (2), activation='relu', padding="same")(b)
    d=MaxPooling1D(pool_size=(2))(c)
    e=Dropout(0.25)(d)
    f=Conv1D(10, (2), activation='relu', padding="same")(e)
    g=Conv1D(10, (2), activation='relu', padding="same")(f)
    h=MaxPooling1D(pool_size=(2))(g)
    i=Dropout(0.4)(h)
    j=Flatten()(i)
    k=Dense(units=50, activation="relu")(j)
    l=Dense(units=25, activation="relu")(k)
    output = Dense(units=1, activation="sigmoid")(l)
    model=Model(a,output)
    return model


def train_convolution_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.05)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=3, batch_size=12)


def get_binary_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test
    ytrain_1hot = np.zeros((ytrain.size, 1))
    for i in range(ytrain.size):
        if (int(ytrain[i]) in [2, 3, 4, 5, 6, 7]):
            ytrain_1hot[i] = 1
    ytest_1hot = np.zeros((ytest.size, 1))
    for i in range(ytest.size):
        if (int(ytest[i]) in [2, 3, 4, 5, 6, 7]):
            ytest_1hot[i] = 1
    xtrain = xtrain / 255
    xtest = xtest / 255
    return xtrain, ytrain_1hot, xtest, ytest_1hot


def build_binary_classifier():
    # The accuracy for training data and test data is shown below
    # 50000/50000 [==============================] - 173s 3ms/step - loss: 0.1016 - acc: 0.9604
    # [0.13250257595181464, 0.95150000000000001]
    bnn = Sequential()
    bnn.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)))
    bnn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    bnn.add(MaxPooling2D(pool_size=(2, 2)))
    bnn.add(Dropout(0.25))
    bnn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    bnn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    bnn.add(MaxPooling2D(pool_size=(2, 2)))
    bnn.add(Dropout(0.4))
    bnn.add(Flatten())
    bnn.add(Dense(units=250, activation="relu"))
    bnn.add(Dense(units=100, activation="relu"))
    bnn.add(Dense(units=1, activation="sigmoid"))
    return bnn


def train_binary_classifier(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.05)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=32)


if __name__=='__main__':
    alpha = 0.3
    f = open('player.json', 'r')
    player = json.load(f)
    f.close()
    f = open('tagDATA2.json', 'r')
    game = json.load(f)
    f.close()

    xtrain = np.zeros((4410, 10, 10))
    ytrain = np.zeros((4410, 10))

    j = 0
    for i in player:
        category = player[i]['category']
        gameplay = player[i]['gameplay']
        for m in range(category+1):
            xtrain[j][m] = gameplay
        for k in range(len(gameplay)):
            if gameplay[k] > alpha:
                ytrain[j][k] = 1
        j += 1
    ytrain=ytrain.T
    y=np.zeros((10,4410,1))
    for i in range(10):
        for j in range(4410):
            y[i][j]=[ytrain[i][j]]
    ytrain=y
    print(y.shape)
    cnn0 = build_convolution_nn()
    train_convolution_nn(cnn0, xtrain, ytrain[0])
    cnn0.save('model0.h5')

    cnn1 = build_convolution_nn()
    train_convolution_nn(cnn1, xtrain, ytrain[1])
    cnn1.save('model1.h5')

    cnn2 = build_convolution_nn()
    train_convolution_nn(cnn2, xtrain, ytrain[2])
    cnn2.save('model2.h5')

    cnn3 = build_convolution_nn()
    train_convolution_nn(cnn3, xtrain, ytrain[3])
    cnn3.save('model3.h5')

    cnn4 = build_convolution_nn()
    train_convolution_nn(cnn4, xtrain, ytrain[4])
    cnn4.save('model4.h5')

    cnn5 = build_convolution_nn()
    train_convolution_nn(cnn5, xtrain, ytrain[5])
    cnn5.save('model5.h5')

    cnn6 = build_convolution_nn()
    train_convolution_nn(cnn6, xtrain, ytrain[6])
    cnn6.save('model6.h5')

    cnn7 = build_convolution_nn()
    train_convolution_nn(cnn7, xtrain, ytrain[7])
    cnn7.save('model7.h5')

    cnn8 = build_convolution_nn()
    train_convolution_nn(cnn8, xtrain, ytrain[8])
    cnn8.save('model8.h5')

    cnn9 = build_convolution_nn()
    train_convolution_nn(cnn9, xtrain, ytrain[9])
    cnn9.save('model9.h5')

    xtest_single=np.array([xtrain[1]])
    if type(xtest_single)!=type(0):
        prediction=[]
        prediction.append(cnn0.predict(xtest_single, batch_size=2))
        prediction.append(cnn1.predict(xtest_single, batch_size=2))
        prediction.append(cnn2.predict(xtest_single, batch_size=2))
        prediction.append(cnn3.predict(xtest_single, batch_size=2))
        prediction.append(cnn4.predict(xtest_single, batch_size=2))
        prediction.append(cnn5.predict(xtest_single, batch_size=2))
        prediction.append(cnn6.predict(xtest_single, batch_size=2))
        prediction.append(cnn7.predict(xtest_single, batch_size=2))
        prediction.append(cnn8.predict(xtest_single, batch_size=2))
        prediction.append(cnn9.predict(xtest_single, batch_size=2))
        print(prediction)
        y2=ytrain.T
        print(y2[0][1])
# Write any code for testing and evaluation in this main section.
