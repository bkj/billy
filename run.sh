#!/bin/bash

# run.sh
# 

# Featurize
INPATH=./data/cub/images/
find $INPATH -type f | python -m tdesc --model vgg16 > ./data/feats-fc
find $INPATH -type f | python -m tdesc --model vgg16 --crow --target-dim 448 > ./data/feats-crow-448

# Get last convolutional features
find $INPATH -type f | ./get_conv.py --target-dim 448 --outpath ./conv.bc > ./convpaths

# Expand to bilinear features
./bilinear.py --inpath ./conv.bc --outpath ./bilinear.bc