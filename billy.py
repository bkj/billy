#!/usr/bin/env python

"""
    billy.py
    
    Compare models using CROW features w/ those using bilinear features
    
    !! Need to clean up the ordering
"""

import os
import sys
import bcolz
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
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
# Meta

meta = pd.read_csv('./data/cub/meta.tsv', header=None, sep='\t')
meta.columns = ('id', 'fname', 'train')
meta['lab'] = meta.fname.apply(lambda x: x.split('/')[0])

train_sel = np.array(meta.train)
train_labs, test_labs = np.array(meta.lab[meta.train]), np.array(meta.lab[~meta.train])

# --
# Linear classifier

# Metadata IO
crow = pd.read_csv('./data/feats-crow-448', sep='\t', header=None)
crow = np.array(crow[crow.columns[1:]])
ncrow = normalize(crow)

train_ncrow, test_ncrow = ncrow[train_sel], ncrow[~train_sel]

# Train classifier
svc = LinearSVC().fit(train_ncrow, train_labs)
(svc.predict(test_ncrow) == test_labs).mean() # 0.660

# --
# Load bilinear features

# Data IO
bili = bcolz.open('./data/bilinear.bc')[:]

train_bili, test_bili = bili[np.array(meta.train)], bili[np.array(~meta.train)]

# --
# Use PCA to reduce dimensionality, then train SVM
# Problem is that PCA on this matrix seems to be very expensive (~232K columns)
# so we were just computing on subset of rows

rsel = np.random.choice(train_bili.shape[0], 1000, replace=False) 
pca = PCA(n_components=512).fit(train_bili[rsel]) # untested

npca_train_bili = normalize(pca.transform(train_bili))
npca_test_bili = normalize(pca.transform(test_bili))

svc = LinearSVC().fit(npca_train_bili, train_labs)
(svc.predict(npca_test_bili) == test_labs).mean()

# 0.757 (normalized, unwhiten)
# ^^ Basically as good as using full bilinear features

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



