_base_ = [
    '../dn-detr.py'
]

test_adv_cfg = dict(
    adv_flag=True,
    bbox_patch_config = dict(
        patch_flag = True,
        adv_type="cls", # assert in ["cls", "reg", "cwa", "dag", "ours"]
        patch_type="square", # assert in ["square", "circle"]
        step_size=2,
        epsilon=255,
        num_steps=200,
        scale_factor=1.0/10.0/1.414,
    ),
    universal_patch_config = dict(
        patch_flag = False,
    )
)