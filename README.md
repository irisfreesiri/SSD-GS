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


## 📌 Release Roadmap

The repository is under active development.  
Upcoming updates include:

- ~~Code Release~~ ✅
- ~~Dataset Release~~ ✅
- Training & Rendering Instructions
- Pretrained Models
- ~~Teaser / Demo Video~~ ✅

## 📚 Acknowledgments

We have intensively borrow codes from [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting), [gs^3](https://github.com/gsrelight/gs-relight), and [gsplat](https://github.com/nerfstudio-project/gsplat). We also use [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for it's efficient MLP implementation. Many thanks to the authors for sharing their codes.
