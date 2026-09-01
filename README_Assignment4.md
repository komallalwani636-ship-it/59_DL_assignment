# Practical 4 – Convolutional Neural Network (CNN) for Image Classification

**Subject:** Deep Learning
**Department:** CSE-AI | **Semester:** 5 | **Academic Year:** 2026-27
**Name:** Netra Lalwani | **Roll No:** 59 | **PRN:** 12413745

## Problem Statement
Design and implement a Convolutional Neural Network (CNN) for image classification using the Tomato or Soybean disease dataset.

## Dataset
Fruit and Vegetable Disease (Healthy vs Rotten) dataset, downloaded via `kagglehub` (`muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten`) — 28 classes, ~29,277 images across fruits/vegetables labeled Healthy or Rotten/Diseased.

## What This Notebook Does
1. Downloads the dataset and auto-detects the directory containing class-labeled image folders.
2. Explores the dataset:
   - Counts images per class and overall class imbalance ratio
   - Healthy vs Rotten/Diseased pie chart
   - Sample image grid across random classes
   - Image dimension distribution scatter plot
3. Loads images with `image_dataset_from_directory`, splitting into train/validation/test sets.
4. Applies data augmentation (random flip, rotation, zoom, contrast) and rescaling, and visualizes augmentation examples.
5. Computes class weights to handle class imbalance.
6. Defines a deeper CNN architecture (4 conv blocks with BatchNorm, 128×128×3 input) for reference.
7. Builds and trains a **lighter/faster CNN** (3 conv blocks, 64×64×3 input) for a quicker run.
8. Compiles with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
9. Trains for 5 epochs and plots training/validation accuracy and loss curves.
10. Evaluates on the test set: classification report (precision/recall/F1 per class), confusion matrix.
11. Visualizes misclassified examples for error analysis.

## How to Run
1. Open `DL_Assignment_4.ipynb` in Google Colab or Jupyter Notebook.
2. **Enable GPU** (Runtime → Change runtime type → GPU) — training on CPU will be very slow given the dataset size.
3. Run all cells sequentially. The dataset download (~4.77 GB) may take a few minutes.

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
- Total classes: **28**, Total images: **~29,277**
- Class imbalance ratio noted between largest and smallest class
- Training/validation accuracy and loss curves over 5 epochs
- Classification report and confusion matrix on the test set
- Misclassified example gallery for error analysis

> Note: With only 5 epochs on the lightweight architecture (as run in the reference output), accuracy remains modest (~30%). Increasing epochs, using the deeper architecture, enabling data augmentation, or fine-tuning a pretrained backbone (e.g., MobileNetV2, already imported) will substantially improve results.

## File Structure
```
├── DL_Assignment_4.ipynb   # Main notebook
└── README.md                # This file
```
