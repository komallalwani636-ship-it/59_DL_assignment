# FinSentBERT — Financial Sentiment Classification with BERT

FinSentBERT is a deep learning project that fine-tunes a pre-trained **BERT (`bert-base-uncased`)** model to classify financial tweets and headlines into three sentiment classes:

- **Bearish**
- **Bullish**
- **Neutral**

The project demonstrates how transfer learning can be applied to financial text, where sentiment often depends heavily on context. For example, words such as *cut*, *lower*, or *rise* can have different meanings depending on what is being discussed.

## Overview

Financial sentiment analysis is more challenging than ordinary text classification because financial tweets contain:

- Company tickers such as `$TSLA`
- Short and noisy text
- Domain-specific terminology
- Context-dependent sentiment
- A significant class imbalance, with Neutral being the dominant class

The project therefore compares a traditional machine learning baseline with a fine-tuned BERT model and uses **macro-F1** as the main evaluation metric.

## Dataset

The project uses the Hugging Face dataset:

`zeroshot/twitter-financial-news-sentiment`

The dataset contains approximately **11,900 English financial tweets** annotated with three labels:

| Label | Sentiment |
|---|---|
| 0 | Bearish |
| 1 | Bullish |
| 2 | Neutral |

Dataset: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment

## Project Workflow

The notebook follows this pipeline:

```text
Financial Tweets
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
TF-IDF + Logistic Regression
       ↓
BERT WordPiece Tokenization
       ↓
Pre-trained BERT Representation
       ↓
BERT Fine-tuning
       ↓
Evaluation
       ↓
Error Analysis
       ↓
Attention-based Explainability
       ↓
Gradio Web Demo
```

## Methodology

### 1. Data Preprocessing

The financial tweets are loaded from the Hugging Face dataset. URLs are removed and whitespace is normalized before the text is passed to the models.

The dataset is divided into training, validation, and test data. The notebook also performs exploratory analysis to examine class distribution and distinctive words associated with each sentiment.

### 2. Classical Baseline

Before fine-tuning BERT, the project establishes a traditional NLP baseline using:

- **TF-IDF**
- **Unigrams and bigrams**
- **Logistic Regression**
- **Balanced class weights**

This provides a reference point for measuring the benefit of contextual BERT representations.

### 3. BERT Tokenization

The project uses the WordPiece tokenizer associated with:

`bert-base-uncased`

Financial text is converted into BERT-compatible tokens and IDs. A maximum sequence length is used to control computational cost.

### 4. Pre-trained BERT Analysis

Before fine-tuning, the final-layer **[CLS] embeddings** are extracted from BERT for a subset of test tweets.

These embeddings are projected into two dimensions using **t-SNE** to visualize whether the three sentiment classes are naturally separated by the pre-trained model.

### 5. BERT Fine-tuning

The model architecture consists of:

```text
bert-base-uncased
        ↓
12 Transformer Layers
        ↓
768-dimensional [CLS] representation
        ↓
Linear Classification Head
        ↓
3 Sentiment Classes
```

The model contains approximately **110 million parameters**.

To address the imbalanced dataset, the project uses a **class-weighted cross-entropy loss**, giving greater importance to the minority Bearish and Bullish classes.

The notebook trains the model for 4 epochs using a learning rate of `2e-5`, with a linear warm-up schedule.

## Evaluation

The project evaluates the models using:

- Accuracy
- Macro-F1
- Per-class precision, recall, and F1-score
- Confusion matrix

**Macro-F1 is treated as the primary metric** because the Neutral class makes up roughly two-thirds of the dataset. Accuracy alone could therefore hide poor performance on the minority classes.

The notebook compares:

1. TF-IDF + Logistic Regression
2. Fine-tuned BERT

The exact metric values are generated when the notebook is executed and are displayed in the evaluation section.

## Representation Analysis

The project visualizes BERT embeddings before and after fine-tuning.

Before fine-tuning, the sentiment representations are relatively mixed. After fine-tuning, the embeddings form more distinct sentiment-specific clusters.

This provides a visual demonstration of how transfer learning adapts a general-purpose language representation to the financial sentiment classification task.

## Error Analysis

The notebook identifies BERT's most confident incorrect predictions.

This analysis shows that some errors are difficult even for humans because financial sentiment can be ambiguous. For example, a lowered price target may still be above the current market price, making its overall sentiment less obvious.

The project therefore treats model errors as an opportunity to understand limitations in both the model and the underlying labels.

## Explainability

The project also examines BERT's attention patterns.

The final-layer attention heads are averaged and the attention received by the **[CLS] token** is inspected. This provides an indication of which tokens the model is focusing on when forming its classification representation.

The notebook demonstrates that attention can be concentrated around financially meaningful words such as:

- `tumble`
- `slashes`
- `beats`
- `jumps`

This makes the predictions easier to inspect rather than treating the model as a completely opaque classifier.

## Inference

After training, the fine-tuned model can classify new financial headlines.

Example inputs used in the notebook include financial-market statements such as:

```text
Nifty hits record high as FII inflows surge
```

The model returns probabilities for:

```text
Bearish
Bullish
Neutral
```

## Interactive Gradio Demo

The project includes a **Gradio web interface**.

Users can enter a financial headline and receive a live sentiment prediction with probabilities for all three classes.

Example:

```text
TCS shares fall after weak deal wins in Q2
```

The interface provides the model's predicted sentiment and confidence distribution.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- BERT
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- t-SNE
- Gradio

## Installation

Install the required libraries:

```bash
pip install -U transformers datasets accelerate gradio
```

The notebook also uses standard Python machine learning libraries such as PyTorch, NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

## Running the Project

The project is implemented as a Jupyter/Google Colab notebook.

1. Open `DL_Assignment_8.ipynb`.
2. Install the required dependencies.
3. Select a GPU runtime, preferably a **T4 GPU** in Google Colab.
4. Run the notebook cells sequentially.
5. Wait for dataset loading, model fine-tuning, evaluation, and visualization steps to complete.
6. Launch the Gradio section to interact with the trained model.

A complete GPU run is expected to take approximately **10 minutes**, depending on the runtime environment.

## Project Structure

```text
.
├── DL_Assignment_8.ipynb
└── README.md
```

The notebook contains the complete implementation, including preprocessing, baseline modeling, BERT fine-tuning, evaluation, visualization, error analysis, explainability, inference, and the Gradio demo.

## Key Learning Outcomes

This project demonstrates:

- Financial text preprocessing
- NLP exploratory data analysis
- TF-IDF-based text classification
- Logistic Regression as an NLP baseline
- BERT WordPiece tokenization
- Transfer learning with BERT
- Fine-tuning a Transformer for classification
- Handling class imbalance using weighted loss
- Macro-F1 based evaluation
- Confusion-matrix and error analysis
- t-SNE visualization of embeddings
- Attention-based model explainability
- Deployment of an NLP model through a Gradio interface

## Future Work

Possible extensions mentioned in the project include:

- Fine-tuning a finance-specific model such as **FinBERT / `ProsusAI/finbert`**
- Experimenting with larger Transformer architectures such as RoBERTa or DeBERTa-v3
- Applying data augmentation to minority classes
- Adding Indian-market financial data from sources such as Moneycontrol or Economic Times
- Exploring Hinglish financial text

## References

1. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019.
2. Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
3. Hugging Face Dataset: `zeroshot/twitter-financial-news-sentiment`

## Author

Deep Learning Assignment — Assignment 8
