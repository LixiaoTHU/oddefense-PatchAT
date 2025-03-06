#  adapted from https://github.com/LYMDLUT/DN-DETR-mmdetection
_base_ = [
    '../configs/_base_/datasets/coco_detection.py', '../configs/_base_/default_runtime.py'
]
checkpoint_at = "/home/ckpt/resnet50_linf_eps4_pure.pth"
model = dict(
    type='DABDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_at)),
    bbox_head=dict(
        type='DNDETRHead',
        num_query=300,
        query_dim=4,
        random_refpoints_xy=False,
        bbox_embed_diff_each_layer=False,
        num_classes=80,
        in_channels=2048,
        transformer=dict(
            type='DNTransformer',
            d_model=256,
            num_patterns=0,
            num_queries=300,
            encoder=dict(
                type='DABDetrTransformerEncoder',
                num_layers=3,
                d_model=256,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU'),
                    ),
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DABDetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                d_model=256,
                query_dim=4,
                iter_update=True,
                keep_query_pos=False,
                query_scale_type='cond_elewise',
                modulate_hw_attn=True,
                bbox_embed_diff_each_layer=False,
                transformerlayers=dict(
                    type='DABDetrTransformerDecoderLayer',
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU'),
                    ),
                    attn_cfgs=[
                        dict(
                        type='DFMultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0),
                        dict(
                        type='DPMultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0)],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        positional_encoding=dict(
            type='SinePositionalEncodingHW', temperatureH=10, temperatureW=10, num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=300))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.

dataset_type = 'CocoDataset'
data_root = '/home/share/datasets/mscoco2017/'
work_dir = "/home/workdir/dndetr/"

free_m = 8
times = 4

adv_cfg = dict(
    adv_flag=True,
    adv_type="all", # assert in ["all", "mtd", "cwa", "ours"]
    free_m=free_m,
    linf_config = dict(
        linf_flag=True,
        epsilon=4,
    ),
    patch_config = dict(
        patch_flag=True,
        scale_factor = 1/10/1.414,
        step_size=4,
        epsilon=32,
        without_gradient_info = False,
    ),
)
test_adv_cfg = dict(
    adv_flag=True,
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

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

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

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.1,
    paramwise_cfg=dict(norm_decay_mult=0.,
        custom_keys={'backbone': dict(lr_mult=0.01, decay_mult=1.0)},
        bypass_duplicate=True
    )
)

optimizer_config = dict(type='AdvOptimizerHook',
                    grad_clip=dict(max_norm=0.1, norm_type=2)) # ignore previous setting
# learning policy
lr_config = dict(policy='step', step=[times*10//free_m])
runner = dict(type='AdvEpochBasedRunner', max_epochs=times*12//free_m)

log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

auto_scale_lr = dict(enable=False, base_batch_size=16)