#!/bin/bash

python -u main.py  --epochs 20 \
                   --batch 12 \
                   --itsa \
                   --model PSMNet \
                   --lambd 0.1 \
                   --eps 0.5 \
                   --lr 0.001 \
                   --maxdisp 192 \
                   --savemodel ./checkpoints/PSMNet/ \
                   --verbose 
                   
exit 1