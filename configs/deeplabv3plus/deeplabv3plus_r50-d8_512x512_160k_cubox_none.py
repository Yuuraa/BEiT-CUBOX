_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/cubox_none.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    decode_head=dict(num_classes=201), auxiliary_head=dict(num_classes=201))