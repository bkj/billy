#!/usr/bin/env python

"""
    bilinear.py
    
    Compute bilinear features (sum of outer products of tiles)
    
    GPU accelerated -- doing it in numpy is _super_ slow
"""

import bcolz
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

# def compute_bilinear(x):
#     mm = np.sum(np.array([np.outer(xx, xx) for xx in x]), axis=0)
#     mm = np.sqrt(np.maximum(0, mm)).flatten()
#     mm /= np.sqrt((mm ** 2).sum())
#     return mm

def compute_bilinear_torch(x):
    x = torch.bmm(x.unsqueeze(2), x.unsqueeze(1)).sum(0)
    x = F.relu(x).sqrt().view(1, -1).squeeze()
    x = F.normalize(x, dim=0)
    x = x.cpu().data.numpy()
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='./conv.bc')
    parser.add_argument('--outpath', type=str, default='./bilinear.bc')
    parser.add_argument('--n-chunks', type=int, default=100)
    return parser.parse_args()

SIZE = 28
DIM = 512

if __name__ == "__main__":
    args = parse_args()
    
    conv = bcolz.open(args.inpath)
    # conv = conv.reshape((conv.shape[0], SIZE * SIZE, DIM))
    
    bili = bcolz.carray(np.empty((0, DIM * DIM), 'float32'), 
        chunklen=16, mode='w', rootdir=args.outpath)

    inds = np.array_split(range(len(conv)), args.n_chunks)
    for ind in tqdm(inds):
        for c in torch.FloatTensor(conv[ind]).cuda():
            bili.append(compute_bilinear_torch(c))
        
        bili.flush()