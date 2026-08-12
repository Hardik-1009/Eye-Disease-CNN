# Glaucoma Detection from Retinal Fundus Images

A deep learning based binary image classification project for detecting **glaucoma from retinal fundus images**. The project compares a baseline CNN with class weighting, data augmentation, and transfer learning using EfficientNetB0, followed by fine-tuning and decision-threshold optimization.

---

## 📌 Project Overview

Glaucoma is a progressive eye disease that can cause irreversible vision loss. Early detection is therefore important for timely treatment and management.

In this project, retinal fundus images are classified into:

- **Glaucoma**
- **Normal**

The main objective is not only to achieve high classification accuracy but also to improve **glaucoma recall**, since failing to detect a glaucoma case is more concerning than incorrectly flagging a normal image.

The project follows the progression:

```text
Baseline CNN
      ↓
Class-Weighted CNN
      ↓
CNN + Data Augmentation
      ↓
EfficientNetB0 Transfer Learning
      ↓
Fine-Tuned EfficientNetB0
      ↓
Decision Threshold Optimization
```

---

## 📂 Dataset

The project uses the **Fondo del Ojo Normal Glaucoma Retinopatia** dataset from Kaggle.

**Dataset:**  
https://www.kaggle.com/datasets/jeremypoveda/fondo-del-ojo-normal-glaucoma-retinopatia

### Classes

| Label | Class |
|------:|-------|
| 0 | Glaucoma |
| 1 | Normal |

### Dataset Distribution

| Dataset | Glaucoma | Normal | Total |
|---|---:|---:|---:|
| Training | 6,950 | 10,880 | 17,830 |
| Validation | 1,351 | 2,215 | 3,566 |
| Test | 241 | 533 | 774 |

The original training data were divided into training and validation sets using an **80:20 stratified split**.

The test set was kept separate and was used only for final evaluation.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Kaggle Notebook

---

## ⚙️ Preprocessing

All retinal fundus images were resized to:

```text
224 × 224 × 3
```

RGB images were used as model input.

Class encoding:

```text
0 → Glaucoma
1 → Normal
```

For the EfficientNetB0 experiments, the model's built-in input preprocessing was used.

---

# 🧠 Model Development

## 1. Baseline CNN

A custom CNN was first trained from scratch to establish a baseline.

The architecture consisted of:

```text
Input
  ↓
Conv2D
  ↓
MaxPooling
  ↓
Conv2D
  ↓
MaxPooling
  ↓
Conv2D
  ↓
MaxPooling
  ↓
Flatten
  ↓
Dense
  ↓
Dropout
  ↓
Sigmoid
```

### Baseline Test Results

| Metric | Result |
|---|---:|
| Accuracy | 73.39% |
| Glaucoma Precision | 72.73% |
| Glaucoma Recall | 23.24% |
| Glaucoma F1 | 35.22% |
| Macro F1 | 59.24% |
| ROC-AUC | 68.64% |

The baseline model showed a strong tendency to classify images as normal.

---

## 2. Class-Weighted CNN

Since the training dataset contained more normal images than glaucoma images, class weighting was introduced.

Approximate class weights:

```text
Glaucoma → 1.27
Normal   → 0.82
```

The CNN architecture was kept unchanged so that the effect of class weighting could be evaluated independently.

### Results

| Metric | Baseline CNN | Class-Weighted CNN |
|---|---:|---:|
| Accuracy | 73.39% | **76.74%** |
| Glaucoma Precision | 72.73% | **80.20%** |
| Glaucoma Recall | 23.24% | **33.61%** |
| Glaucoma F1 | 35.22% | **47.37%** |
| Macro F1 | 59.24% | **66.22%** |
| ROC-AUC | 68.64% | **74.68%** |

Class weighting improved glaucoma recall by more than 10 percentage points.

---

## 3. CNN + Data Augmentation

Data augmentation was investigated to improve generalization.

The following transformations were used:

- Horizontal flipping
- Small rotations
- Random zoom
- Small translations
- Mild contrast variation

Class weighting was retained.

However, this experiment did not improve the test performance.

### Results

| Metric | Class-Weighted CNN | CNN + Augmentation |
|---|---:|---:|
| Accuracy | 76.74% | 71.19% |
| Glaucoma Precision | 80.20% | 59.38% |
| Glaucoma Recall | 33.61% | 23.65% |
| Glaucoma F1 | 47.37% | 33.83% |
| Macro F1 | 66.22% | 57.71% |
| ROC-AUC | 74.68% | 65.75% |

Therefore, the investigated augmentation strategy was not retained in the final model.

---

# 🚀 Transfer Learning with EfficientNetB0

The next experiment used **EfficientNetB0 pretrained on ImageNet**.

The pretrained convolutional base was initially frozen and a custom classification head was added:

```text
EfficientNetB0
      ↓
Global Average Pooling
      ↓
Dense(128)
      ↓
Dropout
      ↓
Sigmoid
```

Class-weighted training was retained.

### EfficientNetB0 Results

| Metric | Result |
|---|---:|
| Accuracy | **82.04%** |
| Glaucoma Precision | **81.48%** |
| Glaucoma Recall | 54.77% |
| Glaucoma F1 | 65.51% |
| Macro F1 | 76.68% |
| ROC-AUC | **86.32%** |

EfficientNetB0 produced a substantial improvement over the custom CNN.

The improvement in glaucoma recall was:

```text
23.24% → 54.77%
```

while ROC-AUC improved from:

```text
68.64% → 86.32%
```

---

# 🔧 Fine-Tuning EfficientNetB0

The upper layers of EfficientNetB0 were subsequently unfrozen to allow the pretrained features to adapt to retinal fundus images.

Approximately the final 30 layers were fine-tuned while earlier layers remained frozen.

Batch normalization layers were kept frozen for training stability.

A small learning rate of:

```text
1 × 10⁻⁵
```

was used.

### Fine-Tuned Results at Threshold 0.50

| Metric | Result |
|---|---:|
| Accuracy | 81.40% |
| Glaucoma Precision | 72.15% |
| Glaucoma Recall | **65.56%** |
| Glaucoma F1 | **68.70%** |
| Macro F1 | **77.73%** |
| ROC-AUC | 85.38% |

Fine-tuning increased glaucoma recall:

```text
54.77% → 65.56%
```

---

# 🎚️ Decision Threshold Optimization

The default classification threshold of `0.50` was not assumed to be optimal.

Since the sigmoid output represents the probability of:

```text
1 → Normal
```

increasing the threshold makes the model more likely to classify an image as:

```text
0 → Glaucoma
```

Thresholds from `0.20` to `0.80` were evaluated using the **validation set**.

### Threshold Analysis

| Threshold | Glaucoma Precision | Glaucoma Recall | Glaucoma F1 |
|---:|---:|---:|---:|
| 0.20 | 94.81% | 65.68% | 77.60% |
| 0.25 | 93.07% | 69.57% | 79.62% |
| 0.30 | 91.99% | 72.66% | 81.19% |
| 0.35 | 90.38% | 75.04% | 82.00% |
| 0.40 | 89.26% | 77.12% | 82.75% |
| 0.45 | 87.78% | 79.57% | 83.47% |
| 0.50 | 86.42% | 82.37% | 84.35% |
| **0.55** | **83.94%** | **84.96%** | **84.45%** |
| 0.60 | 81.27% | 87.41% | 84.23% |
| 0.65 | 78.37% | 89.14% | 83.41% |
| 0.70 | 75.00% | 91.29% | 82.35% |
| 0.75 | 72.15% | 93.38% | 81.40% |
| 0.80 | 67.84% | 95.61% | 79.37% |

The threshold **0.55** produced the highest validation glaucoma F1-score and was therefore selected.

The test set was not used during threshold selection.

---

# 🏆 Final Model

The final selected model is:

> **Fine-Tuned EfficientNetB0 + Class Weighting + Decision Threshold = 0.55**

### Final Test Performance

| Metric | Result |
|---|---:|
| **Accuracy** | **80.88%** |
| Glaucoma Precision | 69.62% |
| **Glaucoma Recall** | **68.46%** |
| **Glaucoma F1** | **69.04%** |
| Macro Precision | 77.73% |
| Macro Recall | 77.48% |
| **Macro F1** | **77.60%** |
| **ROC-AUC** | **85.38%** |

---

## 📊 Final Confusion Matrix

```text
                    Predicted
                 Glaucoma  Normal
Actual Glaucoma     165      76
       Normal        72     461
```

The final model correctly detected:

```text
165 / 241
```

glaucoma cases.

Therefore:

```text
Glaucoma Recall = 68.46%
```

---

# 📈 Overall Model Comparison

| Model | Accuracy | G. Precision | G. Recall | G. F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 73.39% | 72.73% | 23.24% | 35.22% | 59.24% | 68.64% |
| Class-Weighted CNN | 76.74% | 80.20% | 33.61% | 47.37% | 66.22% | 74.68% |
| CNN + Augmentation | 71.19% | 59.38% | 23.65% | 33.83% | 57.71% | 65.75% |
| EfficientNetB0 | **82.04%** | **81.48%** | 54.77% | 65.51% | 76.68% | **86.32%** |
| Fine-Tuned EfficientNetB0 | 81.40% | 72.15% | 65.56% | 68.70% | **77.73%** | 85.38% |
| **Fine-Tuned EfficientNetB0 + Threshold 0.55** | **80.88%** | 69.62% | **68.46%** | **69.04%** | 77.60% | 85.38% |

---

# 🔍 Key Findings

### Class Weighting

Class weighting improved glaucoma recall:

```text
23.24% → 33.61%
```

This helped reduce the model's tendency to favor the majority normal class.

### Data Augmentation

The investigated augmentation strategy did not improve performance.

ROC-AUC decreased from:

```text
74.68% → 65.75%
```

### Transfer Learning

EfficientNetB0 produced the largest improvement in the project.

ROC-AUC increased from:

```text
68.64% → 86.32%
```

and glaucoma recall increased from:

```text
23.24% → 54.77%
```

### Fine-Tuning

Fine-tuning further improved glaucoma recall:

```text
54.77% → 65.56%
```

### Threshold Optimization

Selecting a threshold of `0.55` using the validation set increased test glaucoma recall to:

```text
68.46%
```

while maintaining:

```text
Glaucoma Precision = 69.62%
```

---

# 📌 Key Takeaways

The project demonstrates the importance of evaluating multiple approaches rather than relying solely on accuracy.

The overall improvement in glaucoma recall was:

```text
Baseline CNN
     ↓
23.24%
     ↓
Class-Weighted CNN
     ↓
33.61%
     ↓
CNN + Data Augmentation
     ↓
23.65%
     ↓
EfficientNetB0
     ↓
54.77%
     ↓
Fine-Tuned EfficientNetB0
     ↓
65.56%
     ↓
Threshold Optimization
     ↓
68.46%
```

The largest improvement came from **transfer learning with EfficientNetB0**, while fine-tuning and threshold optimization further improved glaucoma detection.

---

# 📁 Repository Structure

```text
glaucoma-detection/
│
├── README.md
│
├── notebooks/
│   └── glaucoma_detection.ipynb
│
├── results/
│   ├── baseline_confusion_matrix.png
│   ├── efficientnet_confusion_matrix.png
│   ├── final_confusion_matrix.png
│   ├── loss_curves.png
│   ├── accuracy_curves.png
│   └── threshold_analysis.png
│
├── report/
│   └── glaucoma_detection_report.pdf
│
└── requirements.txt
```

---

# 🚀 Future Work

Possible extensions include:

- Testing ResNet, DenseNet, ConvNeXt, or other pretrained architectures
- Cross-validation for more robust performance estimates
- External validation using an independent glaucoma dataset
- Grad-CAM for model interpretability
- Optimization for a specific sensitivity/specificity target
- Ensemble models
- Retinal image quality assessment
- Robustness testing under different imaging conditions

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

The model is not intended to replace professional medical diagnosis. Clinical deployment would require extensive external validation, clinical testing, regulatory approval, and evaluation by qualified healthcare professionals.

---

# 👨‍💻 Author

**Hardik Singhal**

M.Sc. Statistics  
Indian Institute of Technology Kanpur

---

# 📚 References

- Tan, M. & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. International Conference on Machine Learning (ICML).
- TensorFlow / Keras documentation
- Scikit-learn documentation
- Kaggle — Fondo del Ojo Normal Glaucoma Retinopatia Dataset
