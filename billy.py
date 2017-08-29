#!/usr/bin/env python

"""
    billy.py
    
    Compare models using CROW features w/ those using bilinear features
"""

import os
import bcolz
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

import keras
from keras.layers import *
from keras.models import Sequential

from keras import backend as K
if K._BACKEND == 'tensorflow':
    def limit_mem():
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        cfg.gpu_options.visible_device_list="0"
        K.set_session(K.tf.Session(config=cfg))
    
    limit_mem()

# --
# Prep

train_test_split = pd.read_csv('./data/cub/train_test_split.txt', sep=' ', header=None)
train_sel = np.array(train_test_split[1].astype('bool'))

image_ids = pd.read_csv('./data/cub/images.txt', sep=' ', header=None).set_index(1) - 1

# --
# Linear classifier

# Metadata IO
df = pd.read_csv('./data/feats-crow-448', sep='\t', header=None)
df[0] = df[0].apply(lambda x: '/'.join(x.split('/')[2:]))

lookup = pd.DataFrame(np.arange(df.shape[0]), index=df[0])
df = df.loc[np.array(lookup.loc[image_ids.index]).squeeze()]

labs = np.array(df[0].apply(lambda x: x.split('/')[0]))
feats = np.array(df[range(1, df.shape[1])])
nfeats = normalize(feats)

train_labs, test_labs = labs[train_sel], labs[~train_sel]
train_feats, test_feats = nfeats[train_sel], nfeats[~train_sel]

# Train classifier
svc = LinearSVC().fit(train_feats, train_labs)
(svc.predict(test_feats) == test_labs).mean()

# fc2 => 0.579
# crow => 0.660

# --
# Load bilinear features

# Metadata IO
df = pd.read_csv('./data/convpaths', header=None)
df[0] = df[0].apply(lambda x: '/'.join(x.split('/')[-2:]))
lookup = pd.DataFrame(np.arange(df.shape[0]), index=df[0])
ord_ = np.array(lookup.loc[image_ids.index]).squeeze()

labs = np.array(df[0].apply(lambda x: x.split('/')[0]))[ord_]
n_classes = np.unique(labs).shape[0]
train_labs, test_labs = labs[train_sel], labs[~train_sel]

y_train = np.array(pd.get_dummies(train_labs)).argmax(1)
y_test = np.array(pd.get_dummies(test_labs)).argmax(1)

# Data IO
bili = bcolz.open('./data/bilinear.bc')[:]
train_bili = bili[ord_[train_sel]]
test_bili = bili[ord_[~train_sel]]

# --
# Train model w/ bilinear features

model = Sequential()
model.add(Dense(n_classes, input_shape=(train_bili.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

fitist = model.fit(
    train_bili, y_train, 
    verbose=True, 
    batch_size=32,
    validation_data=(test_bili, y_test),
    epochs=50,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ReduceLROnPlateau(patience=5)
    ]
)

# all features
# dropout 0.25 = 0.7477
# dropout 0.50 = ~0.75 @ e11 (got impatient)
# dropout 0.75 = 0.7684

# top 10K features by std
# dropout 0.50 = ~0.75 @ e38 (got impatient)

# --
# Train model on subset of bilinear features

# Option 1
# Choose some dimensions w/ the highest standard deviation
stds = train_bili.std(axis=0)
k = 16
sel = np.argsort(-stds)[:512 * k]
train_sbili, test_sbili = train_bili[:,sel], test_bili[:,sel]

# Option 2 (better)
# Use PCA to reduce dimensionality
from sklearn.decomposition import PCA
pca = PCA(n_components=512).fit(train_bili[:1000])
pca_train_bili = pca.transform(train_bili)
pca_test_bili = pca.transform(test_bili)

# !! Even w/ LinearSVC -- this works better than normal features
svc = LinearSVC().fit(pca_train_bili, train_labs)
(svc.predict(pca_test_bili) == test_labs).mean()
# 0.741

model = Sequential()
model.add(Dense(n_classes, input_shape=(pca_train_bili.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

fitist = model.fit(
    pca_train_bili, y_train, 
    verbose=True, 
    batch_size=32,
    validation_data=(pca_test_bili, y_test),
    epochs=50,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ReduceLROnPlateau(patience=5)
    ]
)

# k=16, dropout = 0.50 -> 0.751 @ e50
# pca=0.9, dropout=0.50 -> 0.757 @ e50

