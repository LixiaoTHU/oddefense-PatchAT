_base_ = [
    '../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/schedules/schedule_1x.py', '../configs/_base_/default_runtime.py'
]

checkpoint_at = "/home/ckpt/resnet50_linf_eps4_pure.pth"
model = dict(
    backbone=dict(frozen_stages=1, init_cfg=dict(type='Pretrained', checkpoint=checkpoint_at)),
    train_cfg=dict(rcnn=dict(clip=6)),
)

dataset_type = 'CocoDataset'
data_root = '/home/share/datasets/mscoco2017/'
work_dir = "/home/workdir/frcnn/"

# adversarial trainging and eval config
free_m = 4
times = 4


# full version

adv_cfg = dict(
    adv_flag=True,
    adv_type="all", # assert in ["all", "mtd", "cwa", "ours"]
    free_m=free_m,
    linf_config = dict(
        linf_flag=True,
        epsilon=4,
    ),
    patch_config = dict(
        patch_flag=False,
        scale_factor = 1/10/1.414*2,
        step_size=8,
        epsilon=64,
        without_gradient_info = False,
    ),
)


test_adv_cfg = dict(
    adv_flag=False,
    bbox_patch_config = dict(
        patch_flag = True,
        adv_type="cls", # assert in ["cls", "reg", "cwa", "dag", "ours"]
        patch_type="square", # assert in ["square", "circle"]
        step_size=2,
        epsilon=255,
        num_steps=200,
        scale_factor=0.15,
    ),
    universal_patch_config = dict(
        patch_flag = False,
    )
)

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=int(500/free_m),
    warmup_ratio=0.001,
    step=[times*10//free_m])
runner = dict(type='AdvEpochBasedRunner', max_epochs=times*12//free_m)
optimizer_config = dict(_delete_=True,
                    type='AdvOptimizerHook',
                    grad_clip=dict(max_norm=100, norm_type=2)) # ignore previous setting
optimizer = dict(
    _delete_ = True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    paramwise_cfg=dict(norm_decay_mult=0.,
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)},
        bypass_duplicate=True
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

auto_scale_lr = dict(enable=False, base_batch_size=16)