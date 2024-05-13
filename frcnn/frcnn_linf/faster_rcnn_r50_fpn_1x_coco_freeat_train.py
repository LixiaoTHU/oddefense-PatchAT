_base_ = [
    '../faster_rcnn_r50_fpn_1x_coco_freeat_base.py'
]

work_dir = "/home/workdir/frcnn/linf/"

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

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=int(500/free_m),
    warmup_ratio=0.001,
    step=[times*10//free_m])
runner = dict(type='AdvEpochBasedRunner', max_epochs=times*12//free_m)