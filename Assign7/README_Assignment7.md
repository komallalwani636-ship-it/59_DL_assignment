# Practical 7 – Transfer Learning with AlexNet on STL-10

**Subject:** Deep Learning

## Problem Statement
Fine-tune a pretrained **AlexNet** (ImageNet weights) on the **STL-10** image classification dataset using transfer learning, and visualize the dataset, training progress, and results.

## Dataset
**STL-10** — 10 classes (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck), 96×96 native resolution images. Auto-downloaded via `torchvision.datasets.STL10` (predefined train split of 5,000 images and test split of 8,000 images).

## What This Notebook Does
The notebook is structured block-by-block:

1. **Setup & Imports** – PyTorch, torchvision, sklearn, matplotlib, seaborn.
2. **Configuration & Reproducibility** – sets random seed, selects device (CUDA/MPS/CPU), and defines hyperparameters (image size 224, batch size 32, 10 epochs, learning rate 1e-3, fine-tune mode).
3. **Load Dataset & Visualize Raw Samples** – downloads STL-10, displays a grid of 18 sample images with labels, and plots the class distribution.
4. **Data Augmentation & DataLoaders** – applies random resized crop, horizontal flip, and color jitter for training; standard resize + normalization for validation/test. Carves a 15% validation split out of the training set. Builds train/val/test `DataLoader`s.
5. **Load Pretrained AlexNet & Modify Classifier Head** – loads AlexNet with ImageNet weights, replaces the final classification layer for 10 classes. Supports two modes:
   - `feature_extract`: freeze all convolutional layers, train only the new head
   - `finetune` (default): also unfreezes the last convolutional block for adaptation
6. **Loss, Optimizer, Scheduler** – `CrossEntropyLoss`, Adam optimizer (only on trainable params), `StepLR` scheduler.
7. **Training Loop** – trains for the configured number of epochs, tracking train/val loss and accuracy each epoch, and keeps the best model checkpoint based on validation accuracy.
8. **Visualize Training Curves** – plots loss and accuracy over epochs for train vs validation.
9. **Evaluate on Test Set & Confusion Matrix** – computes test accuracy, a full classification report, and a confusion matrix heatmap.
10. **Visualize Predictions** – displays grids of correctly and incorrectly classified test images with true/predicted labels.
11. **Per-Class Accuracy Chart** – bar chart of accuracy per class against the overall test accuracy, plus a final summary (best val accuracy, test accuracy, training time, trainable vs total parameters).

All generated plots are saved as PNGs to `./outputs/`:
`01_sample_images.png`, `02_class_distribution.png`, `03_training_curves.png`, `04_confusion_matrix.png`, `05_correct_predictions.png`, `06_incorrect_predictions.png`, `07_per_class_accuracy.png`.

## How to Run
1. Open `DL_Assignment_7.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. If running in Colab or a fresh environment, uncomment the pip install line in Block 1:
   ```
   !pip install torch torchvision matplotlib scikit-learn seaborn -q
   ```
3. Run all cells sequentially.
4. **GPU strongly recommended** (CUDA or Apple Silicon MPS) — the notebook auto-detects and uses the best available device. AlexNet fine-tuning on CPU will be slow but will still run.
5. STL-10 (~2.5 GB) will be auto-downloaded to `./data` on first run.

## Requirements
```
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
```

## Configuration (editable in Block 2)
| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 224 | Input size expected by AlexNet |
| `BATCH_SIZE` | 32 | DataLoader batch size |
| `EPOCHS` | 10 | Number of training epochs |
| `LR` | 1e-3 | Adam learning rate |
| `MODE` | `"finetune"` | `"feature_extract"` or `"finetune"` |

## Expected Output
- Sample image grid and class distribution bar chart for STL-10
- Train/validation loss and accuracy curves over 10 epochs
- Test accuracy, classification report, and confusion matrix
- Correct vs incorrect prediction galleries
- Per-class accuracy bar chart
- Final summary: best validation accuracy, test accuracy, training time, and trainable/total parameter counts

## File Structure
```
├── DL_Assignment_7.ipynb   # Main notebook
├── data/                    # Auto-downloaded STL-10 dataset (created on run)
├── outputs/                 # Saved visualization PNGs (created on run)
└── README.md                # This file
```
