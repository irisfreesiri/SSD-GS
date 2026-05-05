<h1 align="center"> SSD-GS: Scatter and Shadow Decomposition for Realistic Relighting in 3D Gaussian Splatting </h1>

<p align="center">
    <a href="https://www.linkedin.com/in/iris-zheng-620295264/" target="_blank">Iris Zheng</a>, 
    <a href="https://www.linkedin.com/in/guojun-tang-a857b9199/" target="_blank">Guojun Tang</a>, 
    <a href="https://scholar.google.com/citations?user=0x45kYQAAAAJ&hl=en" target="_blank">Alexander Doronin</a>, 
    <a href="https://scholar.google.com/citations?user=wS0ittUAAAAJ&hl=en" target="_blank">Paul Teal</a>, 
    <a href="https://fanglue.github.io/" target="_blank">Fang-Lue Zhang</a>
</p>
<p align="center"> Victoria University of Wellington </p>

[Paper](https://arxiv.org/pdf/2604.13333) | 
[ICLR 2026](https://iclr.cc/virtual/2026/poster/10011264) | 
[Video](https://www.youtube.com/watch?v=mbNmavFGOs8) |
[Poster](https://iclr.cc/media/PosterPDFs/ICLR%202026/10011264.png?t=1776286281.582543) | 
[Dataset](https://myvuwac-my.sharepoint.com/:f:/g/personal/zhengjun1_myvuw_ac_nz/IgADS6vPIEEkTr6Wr64NsT3vAS1dIytv1eDrsh2PW78S-3o?e=F3me3f)

https://github.com/user-attachments/assets/bc3c91e5-96cd-4d16-8d88-124ac93e16cd

## 🔧 Installation

Clone the repository:
```bash
git clone https://github.com/irisfreesiri/SSD-GS.git
cd SSD-GS
```

Create a new conda environment:
```bash
conda create --name ssd-gs python=3.10 pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.4 cuda-toolkit=12.4 cuda-cudart=12.4 -c pytorch -c "nvidia/label/cuda-12.4.0"
conda activate ssd-gs
pip install ninja  # speedup torch cuda extensions compilation
pip install -r requirements.txt
```

## 📁 Dataset Preparation

This project uses three datasets in the [original paper](https://arxiv.org/pdf/2604.13333). Please download the [datasets](https://myvuwac-my.sharepoint.com/:f:/g/personal/zhengjun1_myvuw_ac_nz/IgADS6vPIEEkTr6Wr64NsT3vAS1dIytv1eDrsh2PW78S-3o?e=F3me3f), unzip them, and organize them following the directory structure below:

```text
SSD-GS/
├── data/
│   ├── Real_NRHints/
│   │   ├── Cat/
│   │   ├── CatSmall/
│   │   └── ...
│   ├── Synthetic_GS3/
│   │   ├── AnisoMetal/
│   │   ├── Drums/
│   │   └── ...
│   └── Synthetic_SSS-GS/
│       ├── bunny_small/
│       ├── candle_small/
│       └── ...
```

## 🚀 Training, Rendering, and Evaluation

We provide both **batch scripts** for full experiments and **standalone scripts** for flexible usage.

The full pipeline includes:

- **Training**: optimize Gaussian parameters and decomposition modules  
- **Rendering**: generate novel view images  
- **Evaluation**: compute quantitative metrics  

---

### 📦 Batch Experiments

Each batch script runs **training + rendering + evaluation** automatically.

⚠️Note: before execution, make sure to set the **correct CUDA architecture** according to your GPU, e.g.:
```python
env["TORCH_CUDA_ARCH_LIST"] = "8.6"   # RTX 3090 = 8.6
```
(You can find the correct compute capability for NVIDIA GPUs here:
https://developer.nvidia.com/cuda-gpus)
#### Real NRHints Dataset

```bash
python batch_script_real_nrhints.py
```

#### Synthetic SSS-GS Dataset

```bash
python batch_script_synthetic_gs3.py
```

#### Synthetic SSS-GS Dataset

```bash
python batch_script_synthetic_sssgs.py
```

---

### 🏋️ Training via `train.py`

To train a single scene, run:

```bash
python train.py -s <path_to_scene> -m <output_model_path> --use_nerual_phasefunc --eval
```

#### Example:
```bash
python train.py \
  -s data/Real_NRHints/Cat \
  -m output/real_nrhints_Cat_20260430 \
  --data_device cpu \
  --view_num 2000 \
  --iterations 100000 \
  --use_nerual_phasefunc \
  --cam_opt \
  --pl_opt \
  --eval
```
The following arguments are commonly used for training:

| Argument                      |      Type | Description                                                     |
| ----------------------------- | --------: | --------------------------------------------------------------- |
| `-s`                          |       str | Path to the input scene dataset                                 |
| `-m`                          |       str | Path to save the trained model                                  |
| `--iterations`                |       int | Total number of training iterations                             |
| `--use_nerual_phasefunc`      |      flag | Enable the neural phase-function module                         |
| `--cam_opt`                   |      flag | Enable camera-pose optimization, mainly for real-world scenes   |
| `--pl_opt`                    |      flag | Enable point-light optimization, mainly for real-world scenes   |
| `--eval`                      |      flag | Enable evaluation split during training                         |

### 🎥 Rendering via `render.py`
After training, render novel views from a trained model:
```bash
python render.py -m <output_model_path> --iteration <iteration> --use_nerual_phasefunc --opt_pose
```

#### Example:
```bash
python render.py \
  -m output/real_nrhints_Cat_20260430 \
  --iteration 100000 \
  --opt_pose \
  --use_nerual_phasefunc
```

The following arguments are commonly used for rendering:

| Argument                 | Type | Description                                                               |
| ------------------------ | ---: | ------------------------------------------------------------------------- |
| `-m`                     |  str | Path to the trained model directory                                       |
| `--iteration`            |  int | Model iteration used for rendering                                        |
| `--opt_pose`             | flag | Use optimized camera poses during rendering, mainly for real-world scenes |
| `--use_nerual_phasefunc` | flag | Enable the neural phase-function module during rendering                  |

### 📊 Evaluation via `metrics.py`
After rendering, compute quantitative metrics:
```bash
python metrics.py -m <output_model_path>
```
#### Example:
```bash
python metrics.py -m output/real_nrhints_Cat_20260430
```
The evaluation script computes image-quality metrics for the rendered views, such as PSNR, SSIM, and LPIPS, depending on the available outputs.

## 📌 Release Roadmap

The repository is under active development.  
Upcoming updates include:

- ~~Code Release~~ ✅
- ~~Dataset Release~~ ✅
- ~~Training & Rendering Instructions~~ ✅
- ~~Teaser / Demo Video~~ ✅

## 📚 Acknowledgments

We have intensively borrow codes from [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting), [gs^3](https://github.com/gsrelight/gs-relight), and [gsplat](https://github.com/nerfstudio-project/gsplat). We also use [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for it's efficient MLP implementation. Many thanks to the authors for sharing their codes.
