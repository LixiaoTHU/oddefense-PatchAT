<div align="center">
  <h3>PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection</h3>
  <a href="https://arxiv.org/abs/2506.23581">
        <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2506.23581-b31b1b.svg">
  </a>
</div>

<h2 id="quick-start">Quick Start</h2>
This is the official implementation for ''PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection'', ICCV 2025.  
  
This work is based on our prior work "**On the Importance of Backbone to the Adversarial Robustness of Object Detectors**"(IEEE TIFS) to defend against pixel-based adversarial attacks.  
🔗 Project page: [https://github.com/thu-ml/oddefense](https://github.com/thu-ml/oddefense)

<h3>Preparation</h3>

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

<h3>Train and Evaluate</h3>

1. **Modify Config Files**  
   Modify the variables **checkpoint_at**, **data_root**, and **work_dir** to your own path in the following files:
    - `frcnn/faster_rcnn_r50_fpn_1x_coco_freeat_base.py`
    - `fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_base.py`
    - `dn-detr/dn-detr.py`

2. **Training**  
   Run the following command to start training:
    ```bash
    bash tools/dist_train.sh [config_file] [num_gpus]
    ```

3. **Evaluation**  
  Run the following command to evaluate your model:
    ```bash
    bash tools/dist_test.sh [config_file] [ckpt_path] [num_gpus] --eval bbox
    ```

<h2 id="models">Models</h2>

| **Model**       | **Config File**                                                                                     | **Checkpoint**                          |
|------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------|
| Faster-RCNN  | [`faster_rcnn_r50_fpn_1x_coco_freeat_train.py`](frcnn/frcnn_gradient/faster_rcnn_r50_fpn_1x_coco_freeat_train.py)            | <a href='https://drive.google.com/file/d/1CN_ne8CUnwzzvQ2gHDNvwDQKHfkAXS2a/view?usp=drive_link'> click to download </a> |
| FCOS            | [`fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_train.py`](fcos/fcos_gradient/fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_train.py)                                       | <a href='https://drive.google.com/file/d/1SE1jsbBjc7-lo9UcsNCblU-aShmI9QH8/view?usp=drive_link'> click to download </a>            |
| DN-DETR         | [`dn-detr.py`](dn-detr/dn-detr.py)                                   | <a href='https://drive.google.com/file/d/1CN_ne8CUnwzzvQ2gHDNvwDQKHfkAXS2a/view?usp=drive_link'> click to download </a>         |

<h3>
Acknowledgement
</h3>

If you find that our work is helpful to you, please star this project and consider cite:
```
@inproceedings{li2025pbcat,
  title={PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection},
  author={Li, Xiao and Zhu, Yiming and Huang, Yifan and Zhang, Wei and He, Yingzhe and Shi, Jie and Hu, Xiaolin},
  booktitle={IEEE InternationalConference on Computer Vision},
  year={2025}
}
@article{li2025importance,
  title={On the importance of backbone to the adversarial robustness of object detectors},
  author={Li, Xiao and Chen, Hang and Hu, Xiaolin},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  publisher={IEEE}
}
```
