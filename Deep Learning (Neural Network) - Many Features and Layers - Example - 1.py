#!/usr/bin/env python
# coding: utf-8

import pandas as pd

dataset = pd.read_csv('pima-indians-diabetes.data.csv', header=None)
dataset.info()
dataset
dataset.head(2)

y = dataset[8]
dataset.columns

X = dataset[[0,1,2,3,4,5,6,7]]

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()

model.add(Dense(units=10, input_dim=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100)

model.save("dia_model.h5")



