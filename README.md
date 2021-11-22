# 🧩 BEiT Semantic Segmentation
> ***Semantic segmantation with CUBOX dataset***
>
>
> Original Work 
> https://github.com/microsoft/unilm/tree/master/beit/semantic_segmentation
>
> Related Paper
> https://arxiv.org/abs/2106.08254


<br/>

#### **🐋 Docker Image**  
```yoorachoi/beit:dist```

<br/>

#### 🙋‍♀️ Maintainer**  
[Yura Choi](https://github.com/Yuuraa)

<br/>

---
### 📌 **Table Of Contents**

- Experiment Environment
    - Hardware
    - Setup
    - Structure
- Scripts
    - Train Script
    - Test Script
- Acknowledgement

<br/>

---
<br/>

### 💻 **Experiment Environment**
<br/>


#### **⚙️ Hardware**
Both
    - nvidia-docker pre-installed 

Trained on:
    - **GPU** - 4 x GeForce RTX 2080 Ti (11GB)
    - **RAM** - 256GB
    - vision lab server 112

Tested on:
    - **GPU** - 2 x GeForce RTX 2080 Ti (11GB)
    - **SSD** - Samsung MZ7LH3T8 (3.5 TB)


#### **⛳ Setup**
1. Download Docker Image 
    ```bash
    docker pull yoorachoi/beit:dist
    ```
2. Download Dataset to \[PATH_TO_DATASET]
3. Execute Docker Container
    ```bash
    docker run -it --gpus all --ipc host \
    --mount type="bind",source=[PATH_TO_DATASET],target="/dataset" \
    yoorachoi/beit:dist /bin/bash
    ```
<br/>

####  **📁  Structure**
```
/unilm
├── backbone
|   └── beit.py
├── configs
    ├── _base_
    |    ├── datasets
    |    |    ├── cubox.py
    |    |    ├── cubox_test.py
    |    |    ├── cubox_test_none.py
    |    |    └── ...
    |    ├── models
    |    |    └── upernet_beit_cubox.py
    |    └── schedules
    |    |    ├── schedule_160k.py
    |    |    └── schedule_320k.py
    └── beit/upernet
         ├── upernet_beit_base_12_256_slide_160k_ade20k_pt2ft.py
         └── upernet_beit_base_12_256_slide_160k_ade20k_pt2ft_test.py



/dataset
├── images
|   ├── train
|   |   ├── none
|   |   |   ├── class1
|   |   |   |   ├── image1
|   |   |   |   └── image2
|   |   |   |   └── ...
|   |   |   └── class2
|   |   |   └── class3
|   |   |   └── ...
|   |   └── semitransparent
|   |   |   ├── class1
|   |   |   └── class2
|   |   |   └── class3
|   |   |   └── ...
|   |   └── wiredense
|   |   |   ├── class1
|   |   |   └── class2
|   |   |   └── class3
|   |   |   └── ...
|   |   └── wireloose
|   |   |   ├── class1
|   |   |   └── class2
|   |   |   └── class3
|   |   |   └── ...
|   |   └── wiremedium
|   |   |   ├── class1
|   |   |   └── class2
|   |   |   └── class3
|   |   |   └── ...
|   └── validation # same structure as train
|   └── test        # same structure as train
|
└──  seg_map          # same structure as images

```



<br/>

## 📜 Scripts (inside docker container)
<br/>

### 1️⃣ Train Script
Available Config files
- configs/beit/upernet/upernet_beit_base_12_256_slide_160k_ade20k_pt2ft.py


Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS>  --work-dir <SAVE_PATH> --seed 0  --deterministic --options model.pretrained=<IMAGENET_CHECKPOINT_PATH/URL>
```

Train with total train set & validate with total test set:
```bash
bash tools/dist_train.sh \
    configs/beit/upernet/upernet_beit_base_12_256_slide_160k_ade20k_pt2ft.py 4 \
    --work-dir /path/to/save --seed 0  --deterministic \
    --options model.pretrained=https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth
```

More config files can be found at [`configs/beit/upernet`](configs/beit/upernet) in original repository.

<br/>

### 2️⃣ Evaluation Script

Command format:
```
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, evaluate a BEiT-base backbone with UperNet:
```bash
bash tools/dist_test.sh configs/beit/upernet/upernet_beit_base_12_640_slide_160k_ade20k_pt2ft.py \ 
    https://unilm.blob.core.windows.net/beit/beit_base_patch16_640_pt22k_ft22ktoade20k.pth  4 --eval mIoU
```

Expected results:
```
+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 53.61 | 64.82 | 84.62 |
+--------+-------+-------+-------+
```


---
<br/>
<br/>

## Acknowledgment 

This code is built using the [beit repository](https://github.com/microsoft/unilm/tree/master/beit/semantic_segmentation), which is based on  [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library, [Timm](https://github.com/rwightman/pytorch-image-models) library, the [Swin](https://github.com/microsoft/Swin-Transformer) repository, [XCiT](https://github.com/facebookresearch/xcit) and the [SETR](https://github.com/fudan-zvg/SETR) repository.
