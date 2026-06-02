# YOLOv8n Fine-Tuning for Fine-Grained Card Symbol Detection
This repository accompanies a research project that investigates **fine-tuning strategies for YOLOv8** on a challenging **fine-grained, small-object detection task**: recognizing playing card symbols (rank and suit) in real-world images.
The work focuses on how different levels of model adaptation affect performance when objects are **very small, visually similar, and densely packed**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c3e57c00-3c09-4e66-8d2e-7480afc3d826" alt="CardDetection" width="600"/>
  </p>

## Tech Stack
**Core:** Python · PyTorch · Ultralytics YOLOv8

**Data & Experiment Tracking:** Roboflow · Weights & Biases (W&B)

**Visualization & Utilities:** OpenCV · NumPy · Pandas · Matplotlib · Seaborn

**Topics:** Object Detection · Fine-Grained Small-Object Detection · 
Transfer Learning · Model Fine-Tuning · Layer Freezing

## Research Goal
We specifically examine whether increasing the number of trainable 
layers improves detection performance, and whether full fine-tuning 
risks **catastrophic forgetting** when adapting a COCO-pretrained 
model to a narrow, specialized domain.

---

## Hypotheses
- **H1:** Increasing the number of trainable layers during fine-tuning improves detection performance **without causing catastrophic forgetting**, as pretrained features can be refined rather than overwritten.
- **H2:** Most classification errors occur between **visually similar card symbols**, reflecting the inherent difficulty of fine-grained discrimination at very small scales.

---

## Methodology
We fine-tune **YOLOv8n** (pretrained on COCO) on a dataset of ~2,200 
real-world images containing ~60,000 annotated bounding boxes across 
**53 classes** (rank and suit combinations plus Joker). All models 
are trained for 100 epochs with identical hyperparameters, only 
varying which parts of the network are updated.

Three freezing strategies of increasing capacity are compared:
  - **Head Only:** Detection head trained, backbone + neck frozen (0.76M parameters)
  - **Neck + Head:** Backbone frozen, neck and head trained (1.75M parameters)
  - **Entire Model:** Full end-to-end fine-tuning (3.02M parameters)

---

## Key Findings
- **Full model fine-tuning performed best**, achieving:
  - Precision: **0.82**
  - Recall: **0.77**
  - mAP@50: **0.85** - mean Average Precision at IoU threshold 0.5, meaning a 
    detection counts as correct if it overlaps the ground truth by at least 50%. 
    Strong overall detection performance at this standard threshold.
  - mAP@50–95: **0.52** - same metric averaged across stricter IoU thresholds 
    (0.5 to 0.95), penalizing imprecise bounding boxes more heavily. A solid 
    result given the difficulty of tightly bounding very small objects.

- Training only the head led to **severe performance degradation**, showing that mid- and low-level features must adapt for fine-grained tasks.
- No evidence of **catastrophic forgetting** was observed, even when fine-tuning all layers.
- **Misclassifications predominantly occurred between visually similar symbols**, such as:
  - Rotationally ambiguous ranks (e.g. 6 vs. 9)
  - Same-rank cards across different suits
- Localization was generally accurate; **classification remained the main bottleneck**.

---

## Conclusions
Fine-grained small-object detection benefits significantly from **maximizing trainable model capacity** when domain shift is modest. While YOLOv8n can reliably localize tiny card symbols, **discriminating subtle visual differences** remains challenging and dominates error behavior.

This work demonstrates that end-to-end fine-tuning is both **safe and effective** for specialized detection tasks involving small, visually similar objects.
