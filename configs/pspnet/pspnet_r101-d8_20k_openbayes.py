_base_ = './pspnet_r50-d8_20k_openbayes.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))