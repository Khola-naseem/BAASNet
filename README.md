# BAASNet: Boundary-Aware Deep Learning for Accurate Polyp Segmentation



- **Authors**: Khola Naseem, Nabeel Khalid, Andreas Dengel, Sheraz Ahmed  
- **Repo (example)**: `https://github.com/Khola-naseem/BAASNet`

---

## Installation

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.12 (CUDA recommended)
- torchvision ≥ 0.13
- OpenCV, Albumentations, NumPy,timm, tqdm, scikit-image, scikit-learn,  matplotlib

```bash
pip install -r requirements.txt
```

### Example conda environment setup
```bash
conda create -n baasnet python=3.10 -y
conda activate baasnet

# Install a CUDA build of PyTorch (adjust to match your CUDA toolkit)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Common deps
pip install timm
pip install scikit-learn
pip install scikit-image
pip install opencv-python-headless==4.5.5.64
pip install albumentations==1.3.0
```

---


### Data preparation

Organize datasets like this (adjust paths to your setup):

```
datasets/
  Kvasir-SEG/               # images/, masks/
  PolyDB/                   # images/, masks/ (BLI/FICE/LCI/NBI/WLI)
  EndoScene/                # images/, masks/
  PraNet/                   # CVC-ColonDB, ETIS, CVC-300, Kvasir, CVC-ClinicDB
  EndoTect/                 # images/, masks/
```

> For downloading links and official splits, follow the **Data Availability** section of the paper and/or the dataset homepages.

### Configuration

- Default image size: **256×256** (resize during training/inference).  
- Typical augmentations: horizontal/vertical flip, rotation, Gaussian blur, shift–scale transforms.

---

## Training

```

python train.py --data_root /path/data

```

---

## Inference

```
python inference.py --data_path /Data/ --model_path best_model.pth --output_dir output


```

---

## Evaluation

```
python test_new.py --data_path /data/  --model_path best_model.pth
```

## Acknowledgements

We build on standard PyTorch tooling and public datasets (Kvasir-SEG, PolyDB, CVC-EndoSceneStill, PraNet test sets, EndoTect).  
Please follow the original dataset licenses and cite appropriately.

---

