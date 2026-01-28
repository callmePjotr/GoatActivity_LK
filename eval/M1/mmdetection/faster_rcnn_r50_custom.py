_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'custom_animal_coco.py',
    'mmdet::_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
        )
    )
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=150,
    val_interval=1
)

val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=1e-4
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[80, 120],
        gamma=0.1
    )
]

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=10,
        min_delta=0.001,
        rule='greater'
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/bbox_mAP',
        rule='greater',
        interval=0,
        max_keep_ckpts=1,
        save_last=False
    )
)


work_dir = './work_dirs/fasterrcnn_animal'
