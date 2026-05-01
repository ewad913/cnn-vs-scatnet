# CNN vs ScatNet — Binary Image Classification

> Project for the **Visual Intelligence** course, University of Verona (2025–2026)  
> Author: **Ewad JAMIN**

## Overview

This project implements and compares two architectures for binary image classification (Cats vs Dogs):

- **CNN** : a standard 3-block convolutional neural network with learned filters
- **ScatNet** : a scattering network using fixed wavelet filters (via [Kymatio](https://www.kymat.io/))

The comparison covers three dimensions:
1. **Predictive performance** — 5-fold cross-validation, accuracy, F1-score
2. **Filter analysis** — learned CNN filters vs fixed ScatNet wavelets
3. **Interpretability** — 6 XAI attribution methods (Saliency, Input×Gradient, Guided Backpropagation, Deconvolution, Integrated Gradients, Occlusion)

The Saliency method is implemented from scratch and validated against [Captum](https://captum.ai/)'s implementation.

## Results

| Model   | Accuracy | F1-score |
|---------|----------|----------|
| CNN     | 0.7581   | 0.7575   |
| ScatNet | 0.7793   | 0.7785   |

ScatNet achieves slightly higher accuracy with more stable cross-validation performance. CNN produces more spatially localized attribution maps.

## Repository Structure

```
.
├── notebook/
│   └── Project_VI_Ewad_JAMIN.ipynb   # Main notebook (training, evaluation, XAI)
├── models/                           # Saved model
│   ├── cnn_best.pth
│   └── scatnet_best.pth
├── requirements.txt
└── .gitignore
```

## Dataset

The [Cats vs Dogs dataset](https://www.kaggle.com/datasets/alifrahman/dataset-for-wbc-classification) from Kaggle contains 2802 RGB images (1401 per class).

- **Train:** 1000 images/class
- **Test:** 401 images/class

The notebook expects the dataset to be available via Google Drive (mounted in Colab). Update the paths in the first cells if running locally.

## Setup

### Running on Google Colab (recommended)

The notebook is designed to run on Google Colab with GPU acceleration. Mount your Google Drive and update the dataset path in the configuration cell.

### Running locally

```bash
pip install -r requirements.txt
jupyter notebook notebook/Project_VI_Ewad_JAMIN.ipynb
```

> **Note:** The `kymatio` and `captum` libraries require PyTorch to be installed first.

## Pretrained Models

Saved model weights (`.pth`) are stored in the `models/` folder if provided.  
If not present, re-run the notebook — training takes approximately:
- **CNN:** ~30 epochs × 5 folds
- **ScatNet:** ~10 epochs × 5 folds (converges faster)

## Dependencies

See `requirements.txt` for the full list. Key libraries:

| Library | Purpose |
|---------|---------|
| `torch` / `torchvision` | Model training |
| `kymatio` | Scattering transform (ScatNet) |
| `captum` | XAI attribution methods |
| `scikit-learn` | Cross-validation, metrics |
| `Pillow`, `numpy`, `matplotlib` | Data processing & visualization |