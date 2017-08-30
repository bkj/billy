#!/bin/bash

# run.sh
# 

INPATH=./data/cub/images/
python prep-cub.py # creates './data/cub/meta.tsv'

# Regular features
find $INPATH -type f | python -m tdesc --model vgg16 --crow --target-dim 448 > ./data/feats-crow-448

# Bilinear features
cat ./data/cub/meta.tsv | cut -d$'\t' -f2 | sed 's@^@./data/cub/images/@' | ./img2bilinear.py --outpath ./data/bilinear.bc

# run ./billy.py
