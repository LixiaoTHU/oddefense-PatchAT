# PBCAT

This is the code for submission entitled PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection.


First, you need to change the variables **checkpoint_at**, **data_root**, and **work_dir** to your own path in the following files:

- `frcnn/faster_rcnn_r50_fpn_1x_coco_freeat_base.py`
- `fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_base.py`
- `dn-detr/dn-detr.py`

---

### Environment
```bash
conda create -n patch python=3.10
conda activate patch

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -U openmim
mim install mmcv-full~=1.7.0
pip install mmdet~=2.28.0

pip install -r requirements.txt

pip install mmcv==1.7.0
pip install scikit-image
pip install yapf==0.40.1
```

---

### Faster-rcnn:

1. **Training:**

    ```bash
    bash tools/dist_train.sh frcnn/frcnn_gradient/faster_rcnn_r50_fpn_1x_coco_freeat_train.py [num_gpus]
    ```

2. **Evaluation:**

    ```bash
    bash tools/dist_test.sh frcnn/test/square_patch.py [weight path] [num_gpus] --eval bbox
    ```


---

### FCOS:

1. **Training:**

    ```bash
    bash tools/dist_train.sh fcos/fcos_gradient/fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_all.py [num_gpus]
    ```

2. **Evaluation:**

    ```bash
    bash tools/dist_test.sh fcos/test/square_patch.py [weight path] [num_gpus] --eval bbox
    ```


---

### DN-DETR:

1. **Training:**

    ```bash
    bash tools/dist_train.sh dn-detr/dn-detr.py [num_gpus]
    ```

2. **Evaluation:**

    ```bash
    bash tools/dist_test.sh dndetr/test/square_patch.py [weight path] [num_gpus] --eval bbox
    ```

### CKPTS:
https://drive.google.com/drive/folders/15OtbR-x8kuyBaK0Z_EdbGw_YIO_Rw_b6?usp=sharing
