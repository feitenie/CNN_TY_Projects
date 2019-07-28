#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:06:55 2019
CNN_Typhoon_tracking_prediction
"""
import numpy as np
import pandas as pd
import glob
import cv2
import random

import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras import backend
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
a = [[240, 190, 55]]

def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])

def load_data():
    df = pd.read_excel("./1958＿ty_list.xlsx")
    #df['filter'] = pd.Series(np.ones(df.shape[0]))
    #df['侵臺路徑分類'].unique() : array([2, '---', 3, '特殊', 1, 5, 6, 9, 4, 8, 7]
    df['侵臺路徑分類'][df['侵臺路徑分類'] == "特殊"] = 0
    #df["filter"][df['侵臺路徑分類'] == "---"] = 0
    df = df[df["侵臺路徑分類"] != "---"]
    df = df.assign(figname = lambda x: x.年份.astype(str) + x.英文名稱)
    train = dict(df[['figname', '侵臺路徑分類']].values)
    
    files = glob.glob("./Ty_track_fig/*.png")
    fig_names = [file.split("/")[-1][9:-4] for file in files]

    chosen = list(set(train.keys()).intersection(fig_names))
    usd_img = [ "./Ty_track_fig/OBS_traj_"+name+".png" for name in chosen]
    return usd_img, chosen, train
            

def Preprocessing(usd_img, chosen, train):     
    x = np.array([cv2.imread(file) for file in usd_img])
    X = np.zeros((len(usd_img),1024,1024))
    X = rgb2gray(x)
    X = X[:,200:860:2,200:900:2] # pooling
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X = X/255

    Y = [train[name] for name in chosen]
    Y = keras.utils.to_categorical(Y, 10)
    
    indices = np.arange(X.shape[0])
    random.seed(66)
    random.shuffle(indices)    
    num02 = int(0.2 * X.shape[0])

    # X 
    X = X[indices]
    X_train = X[num02:]
    X_test = X[:num02]
    # Y
    Y = Y[indices]
    Y_train = Y[num02:]
    Y_test = Y[:num02]
    return X_train, X_test, Y_train, Y_test

def CNN(img_rows,img_cols):
    model = Sequential()
    model.add(Convolution2D(20, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu', input_shape=(img_rows,img_cols,1)))
    model.add(Convolution2D(20, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu' ))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(Convolution2D(40, (3,3), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(40, (5,5), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(Convolution2D(40, (5,5), use_bias=True, padding='SAME', strides=1, activation='selu'))
    model.add(MaxPooling2D(pool_size=(5,5)))

    model.add(Flatten())
    model.add(Dense(128, activation='selu'))
    model.add(Dense(128, activation='selu'))
    model.add(Dense(10, activation='softmax'))
    return model

def plot_confusion_matrix(y_true, y_pred, classes, normalize = False, title = None, cmap = plt.cm.Blues):

    if not title:
        if normalize:
            title = "Normalized Confusion Matrix"
        else:
            title = "Confusion Matrix (Numbers)"
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion matrix (Numbers)')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im)
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
if __name__ == "__main__":
    
    usd_img, chosen, train = load_data()
    X_train, X_test, Y_train, Y_test = Preprocessing(usd_img, chosen, train)
    
    # hyperparameter
    BATCH_SIZE = 15
    EPOCHS = 10
    learning_rate = 0.0005
    #
    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    #model
    model = CNN(img_rows, img_cols)
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = optimizers.Adam(lr = learning_rate), metrics=['accuracy'])
    H = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True)
    print("TEST accuracy = " + str( model.evaluate(X_test, Y_test)[1]))
    print("TRAIN accuracy = " + str(model.evaluate(X_train, Y_train)[1]))
    classes = np.arange(10).astype("str")
    plot_confusion_matrix(Y_test.argmax(axis = 1), model.predict(X_test).argmax(axis = 1), classes)