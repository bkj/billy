#!/usr/bin/env python

"""
    eda.py
"""

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC

import bcolz

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, MaxPooling2D

from keras import backend as K
if K._BACKEND == 'tensorflow':
    def limit_mem():
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        cfg.gpu_options.visible_device_list="0"
        K.set_session(K.tf.Session(config=cfg))
    
    limit_mem()

# --
# IO

train_test_split = pd.read_csv('./data/cub/train_test_split.txt', sep=' ', header=None)
train_sel = np.array(train_test_split[1].astype('bool'))

image_ids = pd.read_csv('./data/cub/images.txt', sep=' ', header=None).set_index(1) - 1

df = pd.read_csv('./data/feats-crow-448', sep='\t', header=None)
df[0] = df[0].apply(lambda x: '/'.join(x.split('/')[2:]))

lookup = pd.DataFrame(np.arange(df.shape[0]), index=df[0])
df = df.loc[np.array(lookup.loc[image_ids.index]).squeeze()]

labs = np.array(df[0].apply(lambda x: x.split('/')[0]))
feats = np.array(df[range(1, df.shape[1])])
nfeats = normalize(feats)

train_labs, test_labs = labs[train_sel], labs[~train_sel]
train_feats, test_feats = nfeats[train_sel], nfeats[~train_sel]

# --
# Linear classifier

svc = LinearSVC().fit(train_feats, train_labs)
(svc.predict(test_feats) == test_labs).mean()

# fc2 => 0.579
# crow => 0.660

# --
# Softmax classifier w/ bilinear features

bili = bcolz.open('./bilinear.bc')[:]

df = pd.read_csv('./convpaths', header=None)
df[0] = df[0].apply(lambda x: '/'.join(x.split('/')[-2:]))
lookup = pd.DataFrame(np.arange(df.shape[0]), index=df[0])
ord_ = np.array(lookup.loc[image_ids.index]).squeeze()

labs = np.array(df[0].apply(lambda x: x.split('/')[0]))[ord_]
n_classes = np.unique(labs).shape[0]
train_labs, test_labs = labs[train_sel], labs[~train_sel]

train_bili = bili[ord_[train_sel]]
test_bili = bili[ord_[~train_sel]]

y_train = np.array(pd.get_dummies(train_labs)).argmax(1)
y_test = np.array(pd.get_dummies(test_labs)).argmax(1)

model = Sequential()
model.add(Dense(n_classes, input_shape=(train_bili.shape[1], )))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

fitist = model.fit(
    train_bili, y_train, 
    verbose=True, 
    batch_size=32,
    validation_data=(test_bili, y_test),
    epochs=50
)

# dropout 0.25 = 0.7477
# dropout 0.50 = ~0.75 @ e11 (got impatient)
# dropout 0.75 = 0.7684

# --
# Raw features (no outer product)

conv = bcolz.open('./conv.bc')[:]
train_conv = conv[ord_[train_sel]]
test_conv = conv[ord_[~train_sel]]

# Normalize
fmodel = Sequential()
fmodel.add(Flatten(input_shape=train_conv.shape[1:]))
fmodel.add(Lambda(lambda x: K.l2_normalize(x, 1)))
train_conv_2 = fmodel.predict(train_conv, batch_size=64, verbose=True)
test_conv_2 = fmodel.predict(test_conv, batch_size=64, verbose=True)

model = Sequential()
model.add(Dense(n_classes, input_shape=(train_conv_2.shape[1], )))
model.add(Dropout(0.75))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

y_train = np.array(pd.get_dummies(train_labs)).argmax(1)
y_test = np.array(pd.get_dummies(test_labs)).argmax(1)

fitist = model.fit(
    train_conv_2, y_train, 
    verbose=True, 
    batch_size=32,
    validation_data=(test_conv_2, y_test),
    epochs=50
)

# dropout=0.9 + maxpooling => 0.5656
