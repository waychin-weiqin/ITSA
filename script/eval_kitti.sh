#!/bin/bash

python -u eval.py --cuda 0 \
                  --verbose \
                  --model GwcNet \
                  --loadmodel /home/wei/data2/domain/stereo/baseline/checkpoints/GwcNet.tar \
                  --kitti15


exit 1