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
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D

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
    indices = random.shuffle(indices)    
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

def CNN():
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

if __name__ == "__main__":
    
    usd_img, chosen, train = load_data()
    X_train, X_test, Y_train, Y_test = Preprocessing(usd_img, chosen, train)
    
    # hyperparameter
    BATCH_SIZE = 10
    EPOCHS = 60
    learning_rate = 0.001
    #
    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    #model
    model = CNN(img_rows, img_cols)
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(lr = learning_rate), metrics=['accuracy'])
    model.fit(X_train, Y_train, valudation_data = (X_test, Y_test), batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True)
    print(model.evaluate(X_test, Y_test))