import os
import bcolz
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize

from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping

from keras import backend as K
if K._BACKEND == 'tensorflow':
    def limit_mem():
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        cfg.gpu_options.visible_device_list="0"
        K.set_session(K.tf.Session(config=cfg))
    
    limit_mem()

# --
# Load metadata

image_ids = pd.read_csv('./data/cub/images.txt', sep=' ', header=None).set_index(1) - 1
tmp = pd.read_csv('./convpaths', header=None)[0]
tmp = tmp.apply(lambda x: '/'.join(x.split('/')[-2:]))
lookup = pd.DataFrame(np.arange(tmp.shape[0]), index=tmp)

ord_ = np.array(lookup.loc[image_ids.index]).squeeze()

labs = np.array(tmp.apply(lambda x: x.split('/')[0]))[ord_]
n_classes = np.unique(labs).shape[0]

# --
# Train/test split

train_test_split = pd.read_csv('./data/cub/train_test_split.txt', sep=' ', header=None)
train_sel = np.array(train_test_split[1].astype('bool'))

conv = bcolz.open('./conv.bc')[:]
bili = bcolz.open('./bilinear.bc')[:]

train_conv, test_conv = conv[ord_[train_sel]], conv[ord_[~train_sel]]
train_bili, test_bili = bili[ord_[train_sel]], bili[ord_[~train_sel]]
train_labs, test_labs = labs[train_sel], labs[~train_sel]

y_train = np.array(pd.get_dummies(train_labs)).argmax(1)
y_test = np.array(pd.get_dummies(test_labs)).argmax(1)

del conv
del bili

np.save("./data/tmp/train_conv", train_conv)
np.save("./data/tmp/test_conv", test_conv)
np.save("./data/tmp/train_bili", train_bili)
np.save("./data/tmp/test_bili", test_bili)
np.save("./data/tmp/y_train", y_train)
np.save("./data/tmp/y_test", y_test)

train_conv = np.load('./data/tmp/train_conv.npy')
test_conv = np.load('./data/tmp/test_conv.npy')
train_bili = np.load('./data/tmp/train_bili.npy')
test_bili = np.load('./data/tmp/test_bili.npy')
y_train = np.load('./data/tmp/y_train.npy')
y_test = np.load('./data/tmp/y_test.npy')

# --
# Model 1: Train on pooled conv features

def conv2crow(x):
    return normalize(x.max(axis=(1, 2)))

X_train = conv2crow(train_conv)
X_test  = conv2crow(test_conv)

model = Sequential()
model.add(Dense(n_classes, input_shape=(X_train.shape[1], )))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=5)
crow_fitist = model.fit(
    X_train, y_train, 
    verbose=True, 
    batch_size=16,
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[es]
)

max(crow_fitist.history['val_acc'])

"""
    mean pooling, no normalization: 0.6407 @ e25
    mean pooling, normalization: 0.6507 @ e30
    max pooling, no normalization: ...
    max pooling, normalization: 0.6790 @ e48 (**)
    
    max pooling, normalization, dropout=0.5: 0.6900 @ e50 (**)
    max + sum pooling, normalization, dropout=0.5: 0.7114 @46
    max + sum + length pooling, normalization, dropout=0.5: 0.7134 @40
"""

# --
# Model 2: Train on bili features 
# However, dimensionality is super high, so pick high variance dimensoins

stds = train_bili.std(axis=0)

X_train = train_bili
X_test  = test_bili
sel = np.argsort(-stds)[:50000]
X_train, X_test = normalize(X_train[:,sel]), normalize(X_test[:,sel])

model = Sequential()
model.add(Dense(n_classes, input_shape=(X_train.shape[1], )))
model.add(Dropout(0.75))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=5)
bili_fitist = model.fit(
    X_train, y_train, 
    verbose=True, 
    batch_size=64,
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[es]
)

# ~ 0.75 w/ top 10K by stddev


# --
# Model 3: Train on _unpooled_ conv features.
# Eg, the same root information as bili, but w/o bilinear pooling

def conv2flat(x):
    fmodel= Sequential()
    fmodel.add(Flatten(input_shape=x.shape[1:]))
    fmodel.add(Lambda(lambda x: K.l2_normalize(x, 1)))
    return fmodel.predict(x, verbose=True, batch_size=32)

X_train = conv2flat(train_conv)
X_test  = conv2flat(test_conv)

model = Sequential()
model.add(Dense(n_classes, input_shape=(X_train.shape[1], )))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=5)
conv_fitist = model.fit(
    X_train, y_train, 
    verbose=True, 
    batch_size=32,
    validation_data=(X_test, y_test),
    epochs=6,
    callbacks=[es]
)

"""

    dropout=0, 4 epochs
    loss: 0.1548 - acc: 0.9992 - val_loss: 2.2141 - val_acc: 0.4570
    
"""


# --
# Model 4: Learning the bilinear pooling (w/ one branch)

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

# def batch_dot(x, y):
#     return K.T.batched_tensordot(x, y, axes=[2, 1])

# b = 16
# def block_bili_pooling(x, b=b):
#     shp = K.shape(x)
#     bsz, w, h, c = shp[0], shp[1], shp[2], shp[3]
    
#     # (Blocked) Bilinear pooling
#     b1 = K.reshape(x, (bsz * w * h * b, c / b, 1))
#     b2 = K.reshape(x, (bsz * w * h * b, 1, c / b))
#     bd = batch_dot(b1, b2)
    
#     # Aggregate
#     d = K.sum(K.reshape(bd, (bsz, w * h, c ** 2 / b)), 1)
    
#     # Normalize
#     d = K.l2_normalize(K.epsilon() + K.sign(d) * K.sqrt(K.abs(d)), -1)
    
#     return d


n_classes = 200
X_train = train_conv
X_test = test_conv

N = 512

model = Sequential()
model.add(Conv2D(512, (1, 1), data_format="channels_last", input_shape=X_train.shape[1:]))
model.add(Lambda(bili_pooling, output_shape=(N ** 2 / b,)))
model.add(Dense(n_classes))
model.add(Dropout(0.9))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=5)
conv_fitist = model.fit(
    X_train[:X_train.shape[0] // 16 * 16], y_train[:X_train.shape[0] // 16 * 16], 
    verbose=True, 
    batch_size=16,
    validation_data=(X_test[:X_test.shape[0] // 16 * 16], y_test[:X_test.shape[0] // 16 * 16]),
    epochs=50,
    callbacks=[es]
)

# 0.7474 @ 43e

# # --
# # Model 5: Attempting self-attention

# def conv2att(x):
#     fmodel= Sequential()
#     fmodel.add(MaxPooling2D((2, 2), input_shape=x.shape[1:]))
#     return fmodel.predict(x, verbose=True, batch_size=32)


# # X_train = conv2att(train_conv)
# # X_test  = conv2att(test_conv)

# inp    = Input(shape=X_train.shape[1:])

# pool   = GlobalAveragePooling2D()(inp)
# pool   = Lambda(lambda x: K.l2_normalize(x, -1))(pool)

# d1     = Dense(n_classes)(pool)
# d2     = Dense(n_classes, activation='softmax')(d1)
# model  = Model(inputs=inp, outputs=d2)
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(monitor='val_acc', patience=5)
# conv_fitist = model.fit(
#     X_train, y_train, 
#     verbose=True, 
#     batch_size=32,
#     validation_data=(X_test, y_test),
#     epochs=50,
#     callbacks=[es]
# )


# # --

# def conv_softmax(x):
#     shp = K.shape(x)
    
#     bsz = shp[0]
#     h = shp[1]
#     w = shp[2]
    
#     x = K.reshape(x, (bsz, h * w))
#     x = K.softmax(x)
#     x = K.reshape(x, (bsz, h, w, 1))
#     return x
    

# inp = Input(shape=X_train.shape[1:])

# att = Conv2D(1024, (1, 1), activation='relu', use_bias=True, name='att1')(inp)
# att = Conv2D(1024 * 2, (1, 1), activation='relu', use_bias=True, name='att2')(att)
# att = Conv2D(1024 * 4, (1, 1), use_bias=True, name='att3')(att)
# # att = Lambda(conv_softmax)(att)

# # pool = multiply([inp, att])
# pool = GlobalAveragePooling2D()(att)
# pool = Lambda(lambda x: K.l2_normalize(x, -1))(pool)

# d1     = Dense(n_classes)(pool)
# d2     = Dense(n_classes, activation='softmax')(d1)
# model  = Model(inputs=inp, outputs=d2)
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# # model.get_layer(name='att1').set_weights([np.ones((1, 1, 512, 128))])
# # model.get_layer(name='att2').set_weights([np.ones((1, 1, 128, 1))])
# model.get_layer(name='att1').trainable = True
# model.get_layer(name='att2').trainable = True

# es = EarlyStopping(monitor='val_acc', patience=5)
# conv_fitist = model.fit(
#     X_train, y_train, 
#     verbose=True, 
#     batch_size=32,
#     validation_data=(X_test, y_test),
#     epochs=50,
#     callbacks=[es]
# )

