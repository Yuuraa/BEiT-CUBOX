_base_ = [
    '../_base_/models/isanet_r50-d8.py', '../_base_/datasets/cubox_none.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=201), auxiliary_head=dict(num_classes=201))