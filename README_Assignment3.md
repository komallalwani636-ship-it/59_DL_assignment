# Practical 3 – Forward Propagation & Backpropagation (Learning Rate / Epoch Analysis)

**Subject:** Deep Learning
**Department:** CSE-AI | **Semester:** 5 | **Academic Year:** 2026-27
**Name:** Netra Lalwani | **Roll No:** 59 | **PRN:** 12413745

## Problem Statement
Implement forward propagation and backpropagation using TensorFlow/Keras. Analyze the effect of different learning rates and the number of epochs on model performance.

## Dataset
MNIST handwritten digits dataset, downloaded via `kagglehub` (`hojjatk/mnist-dataset`) and parsed manually from IDX binary files.

## What This Notebook Does
1. Downloads and parses the raw MNIST IDX files (train/test images and labels).
2. Normalizes pixel values to `[0, 1]` and flattens images to 784-length vectors.
3. Visualizes sample digits and the training class distribution.
4. Defines a Sequential MLP (`784 → 128 → 64 → 10`, ReLU + Softmax).
5. Implements an **explicit manual training loop** using `tf.GradientTape` to demonstrate forward propagation and backpropagation step-by-step, tracking loss, accuracy, and gradient norms per epoch.
6. Runs a full experiment sweep over **4 learning rates** (`0.1, 0.01, 0.001, 0.0001`) × **3 epoch settings** (`5, 15, 30`) = 12 configurations.
7. Visualizes results as heatmaps (accuracy & loss vs LR/epochs) and line plots (validation accuracy/loss per epoch per LR).
8. Demonstrates an unstable (too-high LR) vs stable training comparison.
9. Compares a fixed learning rate against an exponentially decaying learning rate schedule.
10. Selects the best-performing configuration, retrains it, and evaluates with a confusion matrix and classification report.
11. Visualizes misclassified digits and the learned first-layer weight patterns.
12. Prints a final ranked summary table of all 12 configurations.

## How to Run
1. Open `DL_Assignment_3.ipynb` in Google Colab or Jupyter Notebook.
2. Run all cells sequentially.
3. GPU recommended (Runtime → Change runtime type → GPU) since 12 models are trained; CPU also works but is slower.

## Requirements
```
kagglehub
numpy
pandas
tensorflow
matplotlib
seaborn
scikit-learn
```

## Expected Output
- Manual training loop reaches **~97% accuracy** in 3 epochs
- Best configuration: **lr=0.001, epochs=30 → ~97.86% test accuracy**
- Learning rate `0.1` shows unstable/noisy validation curves; `0.001` and `0.0001` are most stable
- Confusion matrix, classification report, and misclassified examples for the best model
- Comparison plots for LR stability and fixed vs decaying LR schedules

## File Structure
```
├── DL_Assignment_3.ipynb   # Main notebook
└── README.md                # This file
```
