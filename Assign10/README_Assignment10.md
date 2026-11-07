# Practical 10 – Audio Classification with Speech-to-Text (STT) and Text-to-Speech (TTS)

**Subject:** Deep Learning
**Department:** CSE-AI | **Semester:** 5 | **Academic Year:** 2026-27
**Name:** Netra Lalwani | **Roll No:** 59 | **PRN:** 12413745

## Problem Statement
Design and implement a deep learning application for audio classification that incorporates Speech-to-Text (STT) and Text-to-Speech (TTS) using an appropriate deep learning architecture, and evaluate its performance using suitable metrics.

## Application
Identify the spoken Indian language (Hindi, Marathi, Tamil, Telugu), transcribe the speech, and speak the text back.

```
speech ──► CRNN language classifier ──► language
                   │
                   ▼
      Whisper-small (STT) ──► transcript
                   │
                   ▼
      MMS-TTS / VITS (TTS) ──► synthetic speech
```

## Dataset
Google **FLEURS** (validation split), downloaded from the Hugging Face Hub (`google/fleurs`): real people reading sentences in Hindi, Marathi, Tamil and Telugu, with native-script transcripts. 200 utterances are used per language (800 in total).

## Models and Metrics
| Part | Architecture | Metrics |
|---|---|---|
| Audio classification | **CRNN**: 64-band log-mel spectrogram → 3 CNN blocks → BiGRU → linear (0.98 M parameters) | Accuracy, Precision, Recall, F1, confusion matrix |
| Speech-to-text | **Whisper-small** (Transformer encoder–decoder) | WER, CER |
| Text-to-speech | **MMS-TTS** (VITS) | Round-trip CER (Whisper on synthetic speech), duration |

## What This Notebook Does
1. Installs `jiwer` and `uroman`, imports all libraries and selects the GPU.
2. Downloads the FLEURS Parquet files, resamples audio to 16 kHz, trims silence and normalizes the volume.
3. Splits the data into train / validation / test (526 / 118 / 156) using `StratifiedGroupKFold`, grouped by sentence id so no sentence appears in two sets.
4. Performs exploratory analysis: utterances per language, duration boxplot, waveforms and log-mel spectrograms, and audio playback.
5. Builds and trains the CRNN for 25 epochs (AdamW, cosine learning-rate schedule, label smoothing, SpecAugment), keeping the best validation macro-F1.
6. Evaluates the classifier on the test set: accuracy, precision, recall, F1, normalized confusion matrix, accuracy versus clip length, and a t-SNE plot of the learned embeddings.
7. Transcribes test clips with Whisper-small (using the CRNN-predicted language) and computes WER and CER per language.
8. Synthesizes speech with MMS-TTS, then evaluates it with round-trip CER and compares human and synthetic spectrograms and durations.
9. Runs the complete pipeline demo (identify → transcribe → speak back) and prints a results summary.

## How to Run
1. Open `DL_Assignment_10.ipynb` in Google Colab.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Run all cells sequentially (Runtime → Run all). It takes about 10–12 minutes.
4. An internet connection is required (FLEURS data and the Whisper / MMS-TTS models are downloaded from the Hugging Face Hub).

## Requirements
```
torch
torchaudio
transformers
huggingface_hub
jiwer
uroman
librosa
soundfile
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Expected Output
Results from the recorded run (values can vary slightly between runs because of random cropping and training):

- Split sizes: Train `526` | Val `118` | Test `156`
- CRNN parameters: `0.98 M`
- **Classifier (test set):** Accuracy `0.9231`, Macro-F1 `0.9215`

| Language | Precision | Recall | F1 |
|---|---|---|---|
| Hindi | 0.9118 | 0.7949 | 0.8493 |
| Marathi | 1.0000 | 0.9756 | 0.9877 |
| Tamil | 0.9167 | 0.9429 | 0.9296 |
| Telugu | 0.8696 | 0.9756 | 0.9195 |

- **Speech-to-text (Whisper-small, 10 test clips per language):**

| Language | WER | CER |
|---|---|---|
| Hindi | 1.019 | 1.220 |
| Marathi | 1.011 | 0.391 |
| Tamil | 0.741 | 0.275 |
| Telugu | 1.989 | 1.830 |
| **Average** | **1.190** | **0.929** |

- **Text-to-speech (MMS-TTS), round-trip CER:** human `0.385` vs synthetic `0.568`

| Language | CER (human) | CER (synthetic) |
|---|---|---|
| Hindi | 0.179 | 0.172 |
| Marathi | 0.301 | 0.398 |
| Tamil | 0.202 | 0.256 |
| Telugu | 0.859 | 1.446 |

- Pipeline demo: a Tamil test clip was predicted as Tamil (89.4%), transcribed and spoken back.
- Visualizations: language counts and durations, waveforms and log-mel spectrograms, training curves, confusion matrix, accuracy vs clip length, t-SNE of embeddings, WER/CER bar chart, human vs synthetic spectrograms, round-trip CER bar chart.

## Observations
- The CRNN identifies the language reliably from a few seconds of audio (92.3% accuracy); Marathi is recognized best and Hindi has the lowest recall.
- WER can exceed 1.0 because insertions (e.g. repeated tokens produced by Whisper-small) count as errors. CER is the fairer metric for Indian languages, whose words are long.
- Whisper-small works best on Tamil and Marathi, while Hindi and especially Telugu give repetitive or wrong-script output. This also limits the Telugu round-trip CER for the TTS evaluation.
- MMS-TTS speech is about as intelligible as the human recording for Hindi, and degrades for Marathi, Tamil and Telugu.

## File Structure
```
├── DL_Assignment_10.ipynb          # Main notebook
├── 59_DL_Assignment_10.pdf         # Practical implementation sheet
└── README.md                       # This file
```
