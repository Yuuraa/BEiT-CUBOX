_base_ = './isanet_r50-d8_512x512_80k_cubox_none.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))