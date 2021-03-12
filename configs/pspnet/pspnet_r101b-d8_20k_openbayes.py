_base_ = './pspnet_r50-d8_20k_openbayes.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101))