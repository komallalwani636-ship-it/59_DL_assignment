# Practical 1 – Data Preprocessing, Normalization, Train-Test Split & Visualization

**Subject:** Deep Learning
**Department:** CSE-AI | **Semester:** 5 | **Academic Year:** 2026-27
**Name:** Netra Lalwani | **Roll No:** 59 | **PRN:** 12413745

## Problem Statement
Install and configure TensorFlow/Keras in Google Colab. Perform data preprocessing, normalization, train-test split, and visualization on a sample dataset.

## Dataset
MNIST handwritten digits dataset, downloaded via `kagglehub` (`hojjatk/mnist-dataset`).

## What This Notebook Does
1. Installs `kagglehub` and `mlxtend`.
2. Downloads and loads the MNIST training images/labels using `mlxtend.data.loadlocal_mnist`.
3. Inspects dataset shape and dimensions.
4. Normalizes pixel values from `[0, 255]` to `[0, 1]`.
5. Splits the data into training (80%) and testing (20%) sets using `train_test_split` with stratification.
6. Visualizes a single sample digit and a grid of 10 sample digits.
7. Plots the class distribution of digits (0–9) using a histogram.
8. Prints final shapes of training/testing images and labels, and the pixel value range.

## How to Run
1. Open `DL_Assignment_1.ipynb` in Google Colab or Jupyter Notebook.
2. Run all cells sequentially (Runtime → Run all).
3. No GPU required.

## Requirements
```
kagglehub
mlxtend
numpy
matplotlib
scikit-learn
```

## Expected Output
- Dataset shape: Images `(60000, 784)`, Labels `(60000,)`
- Pixel value range after normalization: `0.0` to `1.0`
- Train/Test split: `(48000, 784)` / `(12000, 784)`
- Visualizations: single digit sample, 10-digit grid, class distribution histogram

## File Structure
```
├── DL_Assignment_1.ipynb   # Main notebook
└── README.md                # This file
```
