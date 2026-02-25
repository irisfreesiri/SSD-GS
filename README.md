# SSD-GS
🎉 Official code release of "SSD-GS: Scattering and Shadow Decomposition for Relightable 3D Gaussian Splatting" (ICLR 2026).


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
- Dataset Release
- Training & Rendering Instructions
- Pretrained Models
- Teaser / Demo Video

## 📚 Acknowledgments

We have intensively borrow codes from [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting), [gs^3](https://github.com/gsrelight/gs-relight), and [gsplat](https://github.com/nerfstudio-project/gsplat). We also use [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for it's efficient MLP implementation. Many thanks to the authors for sharing their codes.
