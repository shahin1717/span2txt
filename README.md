# 🤟 Spanish Sign Language Recognition (LSE)
> Real-time hand sign detection using OpenCV + PyTorch CNN

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)
![Accuracy](https://img.shields.io/badge/Accuracy-99.35%25-brightgreen)

---

## 📋 Overview

A deep learning pipeline that recognizes **19 letters of the Spanish Sign Language (LSE) alphabet** from images and live webcam feed.

- **Preprocessing** — OpenCV skin mask to isolate and crop the hand region
- **Model** — MobileNetV2 (transfer learning, fine-tuned on LSE dataset)
- **Accuracy** — 99.35% on validation set
- **Inference** — Real-time webcam prediction with bounding box overlay

---

## 🔤 Supported Letters

`A B C D E F G I K L M N O P Q R S T U`

---

## 📁 Project Structure

```
span2txt/
├── data/                  # raw images (not included)
│   ├── A/
│   ├── B/
│   └── ...
├── data_cleaned/          # preprocessed crops (generated)
├── src/
│   ├── preprocess.py      # OpenCV skin mask + hand crop
│   ├── dataset.py         # PyTorch Dataset & transforms
│   ├── model.py           # MobileNetV2 architecture
│   ├── train.py           # training loop
│   ├── evaluate.py        # metrics + confusion matrix
│   └── webcam.py          # real-time inference
├── sign_model.pth         # trained model weights (generated)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/span2txt.git
cd span2txt
```

### 2. Create environment
```bash
conda create -n signlang python=3.10 -y
conda activate signlang
```

### 3. Install dependencies
```bash
# Install PyTorch with CUDA (adjust cu128 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install opencv-python scikit-learn matplotlib seaborn tqdm
```

### 4. Prepare your dataset
Place your images in this structure:
```
data/
├── A/  image1.jpg  image2.jpg ...
├── B/  ...
└── ...
```

### 5. Preprocess
```bash
cd src
python preprocess.py
```

### 6. Train
```bash
python train.py
```

### 7. Evaluate
```bash
python evaluate.py
```

### 8. Webcam demo
```bash
python webcam.py
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 99.35% |
| Macro F1-Score | 0.994 |
| Training Epochs | 40 |
| Image Size | 128×128 |

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Per-class Performance

| Letter | Precision | Recall | F1 |
|--------|-----------|--------|----|
| A | 1.000 | 1.000 | 1.000 |
| B | 1.000 | 1.000 | 1.000 |
| C | 1.000 | 1.000 | 1.000 |
| F | 1.000 | 0.943 | 0.971 |
| M | 0.982 | 0.982 | 0.982 |
| N | 0.982 | 0.982 | 0.982 |
| S | 0.952 | 1.000 | 0.976 |

---

## 🧠 How It Works

### Step 1 — OpenCV Preprocessing
Each image goes through a skin detection pipeline before training and inference:

1. Convert BGR → HSV colour space
2. Create binary mask for skin-coloured pixels
3. Flood fill to close holes inside the hand
4. Find largest contour = the hand
5. Crop bounding box with 15% padding

### Step 2 — MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet (1.2M images)
- Last 3 convolutional blocks unfrozen and fine-tuned
- Custom classifier head: `1280 → 512 → 19 classes`
- Dropout layers to prevent overfitting

### Step 3 — Training Strategy
- **Weighted CrossEntropy loss** to handle class imbalance
- **Adam optimizer** with weight decay
- **ReduceLROnPlateau** scheduler
- Data augmentation: random flip, rotation, colour jitter, affine

---

## 🛠️ Requirements

- Python 3.10
- PyTorch 2.x + CUDA
- OpenCV 4.x
- torchvision
- scikit-learn
- matplotlib
- seaborn
- tqdm

---

## 📷 Dataset

- **19 classes** (LSE alphabet letters)
- **~100 images per class** (~1998 total)
- Plain/white background
- High resolution JPG photos

Dataset not included in this repo. To use your own dataset follow the folder structure above.

---

## 📄 License

MIT License — feel free to use and modify.