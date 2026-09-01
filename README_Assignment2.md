# Practical 2 – Multilayer Perceptron (MLP) Classification

**Subject:** Deep Learning
**Department:** CSE-AI | **Semester:** 5 | **Academic Year:** 2026-27
**Name:** Netra Lalwani | **Roll No:** 59 | **PRN:** 12413745

## Problem Statement
Design and implement a Multilayer Perceptron (MLP) for classification of the Iris or Wine dataset, and evaluate its performance using accuracy and a confusion matrix.

## Dataset
Wine dataset, loaded via `sklearn.datasets.load_wine()` (13 features, 3 classes).

## What This Notebook Does
1. Loads the Wine dataset and splits it into training (75%) and testing (25%) sets, stratified by class.
2. Standardizes features using `StandardScaler`.
3. Builds and trains an `MLPClassifier` with hidden layers `(64, 32, 16)`, ReLU activation, Adam optimizer, and early stopping.
4. Evaluates the model: accuracy, confusion matrix, classification report.
5. Reduces features to 2D using PCA and trains a second MLP for decision-boundary visualization.
6. Computes multiclass ROC curves and AUC scores.
7. Extracts hidden-layer representations and visualizes them with t-SNE.
8. Renders an 8-panel dashboard:
   - Confusion matrix
   - Training loss curve
   - Validation accuracy per epoch
   - PCA decision boundary
   - Multiclass ROC curves
   - t-SNE of hidden representations
   - Top input feature weights (layer 1)
   - Per-class accuracy

## How to Run
1. Open `DL_Assignment_2.ipynb` in Google Colab or Jupyter Notebook.
2. Run all cells sequentially.
3. No GPU required; runs in under a minute on CPU.

## Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Expected Output
- Test Accuracy: **~97.78%**
- Precision/Recall/F1 per class (see classification report)
- Confusion matrix with near-perfect diagonal
- Clear class separation visible in PCA decision boundary and t-SNE plots

## File Structure
```
├── DL_Assignment_2.ipynb   # Main notebook
└── README.md                # This file
```
