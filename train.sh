#!/bin/sh

# train

# baseline
nohup python -u main.py --output_dir ./logs/base >out0

# cutout with 1 hole and length 16
nohup python -u main.py --output_dir ./logs/cutout-hole1-length16 --cutout >out1

# mixup with alpha 1.0
nohup python -u main.py --output_dir ./logs/mixup-alpha1 --mixup >out2

# cutmix with alpha 1.0 and prob 0.5
nohup python -u main.py --output_dir ./logs/cutmix-alpha1-prob0.5 --cutmix_prob 0.5 >out3
