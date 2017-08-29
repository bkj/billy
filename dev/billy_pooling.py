#!/usr/bin/env python

"""
    billy_pooling.py
    
    Compute bilinear features on the fly, w/ a 1x1 conv. layer to reduce dimensionality
    and avoid the quadratic explosion
"""

import bcolz
import pandas as pd

import keras
from keras.layers import *
from keras.models import Sequential, Model

from keras import backend as K
if K._BACKEND == 'tensorflow':
    def limit_mem():
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        cfg.gpu_options.visible_device_list="0"
        K.set_session(K.tf.Session(config=cfg))
    
    limit_mem()

def bili_pooling(x):
    shp = K.shape(x)
    bsz, w, h, c = shp[0], shp[1], shp[2], shp[3]
    
    # Bilinear pooling
    b1 = K.reshape(x, (bsz * w * h, c, 1))
    b2 = K.reshape(x, (bsz * w * h, 1, c))
    d  = K.reshape(K.batch_dot(b1, b2), (bsz, w * h, c * c))
    d = K.sum(d, 1)
    
    # Normalize
    d = K.sqrt(K.relu(d))
    d = K.l2_normalize(d, -1)
    return d


def bili_pooling_asym(data):
    x, y = data
    shp_x = K.shape(x)
    bsz, w, h, c_x = shp_x[0], shp_x[1], shp_x[2], shp_x[3]
    
    shp_y = K.shape(y)
    c_y = shp_y[3]
    
    # Bilinear pooling
    bx = K.reshape(x, (bsz * w * h, c_x, 1))
    by = K.reshape(y, (bsz * w * h, 1, c_y))
    
    # Normalize
    d = K.reshape(K.batch_dot(bx, by), (bsz, w * h, c_x * c_y))
    d = K.sum(d, 1)
    d = K.sqrt(K.relu(d))
    d = K.l2_normalize(d, axis=-1)
    return d

# --
# Prep

train_test_split = pd.read_csv('./data/cub/train_test_split.txt', sep=' ', header=None)
train_sel = np.array(train_test_split[1].astype('bool'))

image_ids = pd.read_csv('./data/cub/images.txt', sep=' ', header=None).set_index(1) - 1

# --
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
conv = bcolz.open('./conv.bc')[:]
train_conv = conv[ord_[train_sel]]
test_conv = conv[ord_[~train_sel]]

train_conv = train_conv.reshape((train_conv.shape[0], 28, 28, 512))
test_conv = test_conv.reshape((test_conv.shape[0], 28, 28, 512))


N = 32

inp_x = Input(shape=train_conv.shape[1:])
inp_y = Input(shape=train_conv.shape[1:])

conv_x = Conv2D(N, (1, 1), data_format="channels_last")(inp_x)
bili = Lambda(bili_pooling_asym, output_shape=(N * train_conv.shape[-1],))([inp_x, inp_y])

x = Dense(512)(bili)
x = Dropout(0.5)(x)
x = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=[inp_x, inp_y], outputs=x)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

conv_fitist = model.fit(
    [train_conv] * 2, y_train, 
    verbose=True, 
    batch_size=32,
    validation_data=([test_conv] * 2, y_test),
    epochs=50,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ReduceLROnPlateau(patience=5)
    ]
)
