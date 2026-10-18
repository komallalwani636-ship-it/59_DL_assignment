# Practical 9 – Vision Transformer vs CNN: Indian Food Classification

**Subject:** Deep Learning
**Department:** CSE-AI | **Semester:** 5 | **Academic Year:** 2026-27
**Name:** Netra Lalwani | **Roll No:** 59 | **PRN:** 12413745

## Problem Statement

Fine-tune a pre-trained **Vision Transformer (ViT-B/16)** for image classification on an Indian food dataset (20 dishes), and compare it with **CNN** models — a custom CNN trained from scratch and a pre-trained **ResNet-50** — in terms of accuracy, macro-F1, model size, speed, robustness and explainability.

## Dataset

**`rajistics/indian_food_images`** (Hugging Face Hub) — about 6,300 real-world food photos across **20 classes**:

burger, butter naan, chai, chapati, chole bhature, dal makhani, dhokla, fried rice, idli, jalebi, kaathi rolls, kadai paneer, kulfi, masala dosa, momos, paani puri, pakode, pav bhaji, pizza, samosa.

| Split | Images |
|---|---:|
| Train | 4,795 |
| Validation (10% stratified from train) | 533 |
| Test (official test split) | 941 |

This is a **fine-grained** task: several dishes look very similar (chapati vs butter naan, dal makhani vs kadai paneer), and the photos vary widely in size, lighting and plating.

## Models Compared

| Model | Type | Pre-training | Parameters |
|---|---|---|---:|
| Custom CNN | 4 conv blocks + global pooling, trained from scratch | None | 1.18 M |
| ResNet-50 | CNN with residual connections | ImageNet-1k | 23.55 M |
| ViT-B/16 (`google/vit-base-patch16-224`) | Vision Transformer | ImageNet-21k → 1k | 85.81 M |

**How ViT works:** the 224×224 image is split into 14×14 = **196 patches** of 16×16 pixels. Each patch is projected to a 768-d token, a learnable **[CLS]** token and position embeddings are added, and 12 Transformer encoder layers apply **multi-head self-attention**, so every patch can attend to every other patch from the first layer. The classifier reads the final [CLS] token.

**How a CNN differs:** convolutions only see a local neighbourhood that grows with depth. This built-in bias (locality and translation equivariance) makes CNNs data-efficient, while ViTs depend on large-scale pre-training.

## What This Notebook Does

1. Installs and imports PyTorch, Torchvision, Hugging Face Transformers/Datasets and Scikit-learn; sets seeds and selects the GPU.
2. Loads the Indian food dataset, resizes every image once (shorter side 240 px) and caches it in memory for faster epochs.
3. Creates a stratified train/validation split and keeps the official test split for final evaluation.
4. Performs EDA: images per class, original image sizes and one sample image per class.
5. Builds a shared data pipeline (224×224 images, random resized crop, flip and colour jitter for training; center crop for test), so all models see the same data.
6. Wraps each model so that input normalization happens inside the model (ImageNet mean/std for the CNNs, 0.5/0.5 for ViT).
7. Trains all three models with the same training loop:
   - AdamW optimizer, cosine learning-rate schedule with 10% warm-up
   - Label smoothing 0.1 and mixed-precision (fp16) training
   - Best checkpoint chosen by validation macro-F1
   - Custom CNN: 15 epochs, lr 1e-3 · ResNet-50: 5 epochs, lr 3e-4 · ViT: 5 epochs, lr 3e-5
8. Plots training/validation loss, validation accuracy and validation macro-F1.
9. Compares the models on the test set: accuracy, macro-F1, top-3 accuracy, parameters, model size, training time and inference speed.
10. Plots per-class F1 scores and normalized confusion matrices.
11. Shows test images where ViT and ResNet-50 disagree.
12. Explains predictions with **Grad-CAM** (ResNet-50) and **Attention Rollout** (ViT).
13. Runs a **robustness stress test** with Gaussian noise, Gaussian blur and occlusion at increasing severity.
14. Lets you upload your own food photo and see the top-5 predictions of each model.
15. Prints a final summary of the results.

## Results (this run, Tesla T4 GPU)

| Model | Accuracy | Macro-F1 | Top-3 Acc | Params (M) | Size (MB) | Train time (min) | Inference (img/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN (scratch) | 0.708 | 0.679 | 0.883 | 1.2 | 4.7 | 6.5 | 1350 |
| ResNet-50 (CNN) | 0.906 | 0.907 | 0.968 | 23.5 | 94.2 | 3.7 | 787 |
| **ViT-B/16 (Transformer)** | **0.934** | **0.935** | **0.970** | 85.8 | 343.3 | 4.0 | 382 |

**Best model on test macro-F1:** ViT-B/16.

- **Transfer learning gain** (ResNet-50 vs scratch CNN): **+19.9** accuracy points
- **ViT vs ResNet-50** on the clean test set: **+2.8** accuracy points
- ViT was right and ResNet wrong on **40** test images; ResNet was right and ViT wrong on **14**
- Classes where ViT gains most over ResNet: Kulfi (+0.10), Kaathi Rolls (+0.08), Pav Bhaji (+0.08), Pakode (+0.06)
- Classes where ResNet beats ViT: Momos (−0.03), Paani Puri (−0.02), Butter Naan (−0.01), Jalebi (−0.01)

### Robustness — accuracy retained at maximum severity (% of clean accuracy)

| Corruption | Custom CNN | ResNet-50 | ViT-B/16 |
|---|---:|---:|---:|
| Gaussian noise (σ = 0.3) | 11.0 % | 6.7 % | **63.6 %** |
| Gaussian blur (σ = 4) | 9.3 % | 58.7 % | **87.1 %** |
| Occlusion (patch = 0.6 × image side) | 66.8 % | **93.6 %** | 93.1 % |

Under the heaviest occlusion, ViT scored 0.869 accuracy vs 0.848 for ResNet-50.

## Key Observations

1. **Pre-training matters most.** Both pre-trained models beat the scratch CNN by a wide margin, reaching high accuracy within one or two epochs, while the scratch CNN climbed slowly over 15 epochs.
2. **ViT vs CNN.** ViT-B/16 had the best accuracy and macro-F1, with its biggest gains on dishes where global context (the whole plate and side dishes) helps.
3. **Cost.** ViT has about 3.6× more parameters than ResNet-50 and runs at about half the inference speed; ResNet-50 remains a strong accuracy-per-compute choice.
4. **Robustness.** ViT held up far better under noise and blur. Under occlusion, ViT and ResNet-50 retained a similar share of their clean accuracy.
5. **Explainability.** Grad-CAM gives compact, local heat-maps for ResNet-50, while ViT's attention rollout is spread across the whole plate, showing more global reasoning.

## How to Run

1. Open `DL_Assignment_9.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order (the first cell installs `transformers` and `datasets`).
4. The dataset and pre-trained weights download automatically on the first run.
5. The full run takes about **20–25 minutes** on a T4 GPU.
6. In Section 12, upload any food photo to test the three models.

## Requirements

```text
torch
torchvision
transformers
datasets
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
```

## File Structure

```text
├── DL_Assignment_9.ipynb   # Main notebook
└── README.md               # This file
```

## References

1. Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
2. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
3. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
4. Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020.
5. Naseer et al., *Intriguing Properties of Vision Transformers*, NeurIPS 2021.
6. Dataset: `rajistics/indian_food_images`, Hugging Face Hub.
