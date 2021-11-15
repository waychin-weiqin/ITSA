#!/bin/bash

python -u eval.py --cuda 0 \
                  --verbose \
                  --model PSMNet \
                  --loadmodel checkpoints/PSMNet/best.tar \
                  --kitti15


exit 1