# Practical 5 – RNN, LSTM & GRU for Sequence Classification

**Subject:** Deep Learning  
**Department:** CSE-AI | **Semester:** 5 | **Academic Year:** 2026-27  
**Name:** Netra Lalwani | **Roll No:** 59 | **PRN:** 12413745

## Problem Statement

Implement and compare **RNN, LSTM, and GRU** models for sequence classification, and analyze their performance using appropriate evaluation metrics.

## Dataset

**IMDB Movie Review Dataset** — a binary sentiment classification dataset containing movie reviews labeled as either positive or negative.

The reviews are converted into integer sequences using a fixed vocabulary size and padded to a common sequence length before being provided to the recurrent models.

## What This Notebook Does

1. Loads the IMDB movie review dataset using TensorFlow/Keras.
2. Preprocesses the text data by converting reviews into integer sequences.
3. Pads the sequences to a fixed length so that they can be processed in batches.
4. Splits the dataset into training and testing sets.
5. Defines a common embedding layer and classification architecture for fair comparison.
6. Implements a **Simple RNN** model for sequence classification.
7. Implements an **LSTM** model for sequence classification.
8. Implements a **GRU** model for sequence classification.
9. Trains all three models using the same dataset, optimizer, loss function, batch size, and number of epochs.
10. Evaluates each model using accuracy, precision, recall, and F1-score.
11. Generates confusion matrices for the three models.
12. Plots training and validation accuracy for comparison.
13. Plots training and validation loss for comparison.
14. Creates a final comparison table and visualization of the model performance.
15. Analyzes the differences between RNN, LSTM, and GRU in terms of sequence learning and performance.

## Model Architectures

### Simple RNN

```text
Input Sequence
      ↓
Embedding
      ↓
SimpleRNN
      ↓
Dense
      ↓
Sigmoid
      ↓
Sentiment Prediction
```

### LSTM

```text
Input Sequence
      ↓
Embedding
      ↓
LSTM
      ↓
Dense
      ↓
Sigmoid
      ↓
Sentiment Prediction
```

### GRU

```text
Input Sequence
      ↓
Embedding
      ↓
GRU
      ↓
Dense
      ↓
Sigmoid
      ↓
Sentiment Prediction
```

All three models use the same preprocessing and classification setup so that their performance can be compared fairly.

## Evaluation Metrics

The following metrics are used to evaluate the models:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Training and validation curves are also used to analyze model convergence and generalization.

## Model Comparison

The final performance is summarized using a comparison table:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Simple RNN | — | — | — | — |
| LSTM | — | — | — | — |
| GRU | — | — | — | — |

The actual values are obtained after running the notebook.

## How to Run

1. Open `DL_Assignment_5.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required libraries if they are not already available.
3. Run all cells sequentially.
4. GPU is recommended for faster training, but the notebook can also run on CPU.
5. Review the training curves, confusion matrices, classification reports, and final model comparison.

## Requirements

```text
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Expected Output

- Preprocessed and padded IMDB sequences
- RNN training and validation curves
- LSTM training and validation curves
- GRU training and validation curves
- Accuracy, precision, recall, and F1-score for each model
- Confusion matrix for each model
- Model performance comparison table
- Model performance comparison chart
- Analysis of RNN, LSTM, and GRU performance

## Conclusion

The notebook provides a comparative analysis of Simple RNN, LSTM, and GRU architectures for sequence classification. The results demonstrate how different recurrent architectures process sequential information and how their architectural differences affect classification performance, training behavior, and generalization.

## File Structure

```text
├── DL_Assignment_5.ipynb   # Main notebook
└── README.md               # This file
```
