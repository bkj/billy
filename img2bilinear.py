#!/usr/bin/env python

"""
    get_conv.py
"""

import sys
import bcolz
import argparse
import numpy as np

from keras import backend as K
if K._BACKEND == 'tensorflow':
    def limit_mem():
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        cfg.gpu_options.visible_device_list="0"
        K.set_session(K.tf.Session(config=cfg))
    
    limit_mem()

import torch
from torch.nn import functional as F

from tdesc.workers import VGG16Worker
from keras.models import Model

# --

def compute_bilinear_torch(x):
    x = torch.bmm(x.unsqueeze(2), x.unsqueeze(1)).sum(0)
    x = F.relu(x).sqrt().view(1, -1).squeeze()
    x = F.normalize(x, dim=0)
    x = x.cpu().data.numpy()
    return x

class ConvWorker(VGG16Worker):
    def __init__(self, target_dim):
        super(ConvWorker, self).__init__(crow=True, target_dim=target_dim)
        self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)
    
    def featurize(self, path, img, return_feat=False):
        feat = self.model.predict(img).squeeze()
        print feat.shape
        tfeat = feat.reshape(feat.shape[0] * feat.shape[1], feat.shape[2])
        print tfeat.shape
        tfeat = torch.FloatTensor(tfeat).cuda()
        print tfeat.size()
        bili = compute_bilinear_torch(tfeat)
        return path, bili


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, default='./bilinear.bc')
    parser.add_argument('--target-dim', type=int, default=448)
    return parser.parse_args()

SIZE = 28
DIM = 512

if __name__ == "__main__":
    args = parse_args()
    
    # !! The dimensions of this will vary depending on args.target_dim
    conv = bcolz.carray(np.empty((0, DIM ** 2), 'float32'), 
        chunklen=16, mode='w', rootdir=args.outpath)
    
    worker = ConvWorker(target_dim=args.target_dim).run(io_threads=3, timeout=10)
    for p, w in worker:
        conv.append(w)
        conv.flush()
        
        print p
        sys.stdout.flush()
