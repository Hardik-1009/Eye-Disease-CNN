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
