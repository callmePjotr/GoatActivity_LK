_base_ = [
    'mmdet::_base_/models/retinanet_r50_fpn.py',
    'custom_animal_coco.py',
    'mmdet::_base_/default_runtime.py'
]

# =========================
# MODEL
# =========================
model = dict(
    bbox_head=dict(
        num_classes=1  # ('animal',)
    )
)

# =========================
# TRAINING
# =========================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=1
)

val_cfg = dict()
test_cfg = dict()

# =========================
# OPTIMIZER (STABIL!)
# =========================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=1e-4
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# =========================
# LR SCHEDULER
# =========================
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
        milestones=[35, 45],
        gamma=0.1
    )
]

# =========================
# EARLY STOPPING
# =========================
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='bbox_mAP',
        patience=7,
        min_delta=0.001,
        rule='greater'
    )
]

# =========================
# CHECKPOINTS
# =========================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/bbox_mAP',
        rule='greater'
    )
)


# =========================
# WORK DIR
# =========================
work_dir = './work_dirs/retinanet_animal'
