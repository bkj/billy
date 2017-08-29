### billy

Looking using bilinear features for classification.

This code (approximately) reproduces the results of 

    http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf

Need to download the `CUB_200_2011` dataset and rename `./data/cub`

### Prereqs

- tdesc -- https://github.com/bkj/tdesc
- keras
- sklearn
- bcolz

### Usage

```
INPATH=./data/cub/images/

# Regular features
find $INPATH -type f | python -m tdesc --model vgg16 --crow --target-dim 448 > ./data/feats-crow-448

# Get last convolutional features, then expand to bilinear features
find $INPATH -type f | ./get_conv.py --target-dim 448 --outpath ./conv.bc > ./data/convpaths
./bilinear.py --inpath ./conv.bc --outpath ./data/bilinear.bc

# run ./billy.py
```
