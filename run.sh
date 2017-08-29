#!/bin/bash

# run.sh
# 

INPATH=./data/cub/images/

# Regular features
find $INPATH -type f | python -m tdesc --model vgg16 --crow --target-dim 448 > ./data/feats-crow-448

# Get last convolutional features, then expand to bilinear features
find $INPATH -type f | ./get_conv.py --target-dim 448 --outpath ./conv.bc > ./data/convpaths
./bilinear.py --inpath ./conv.bc --outpath ./data/bilinear.bc

# run ./billy.py