#!/usr/bin/env python

"""
    prep-cub.py
    
    Process the CUB metadata into something saner
"""

import pandas as pd
import numpy as np

train_sel = pd.read_csv('./data/cub/train_test_split.txt', sep=' ', header=None)
train_sel = np.array(train_sel[1].astype('bool'))

meta = pd.read_csv('./data/cub/images.txt', sep=' ', header=None)
meta.columns = ('id', 'fname')
meta['train'] = train_sel

meta.to_csv('./data/cub/meta.tsv', sep='\t', header=False, index=False)
