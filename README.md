# YOLOv8n Fine-Tuning for Fine-Grained Card Symbol Detection
This repository accompanies a research project that investigates **fine-tuning strategies for YOLOv8** on a challenging **fine-grained, small-object detection task**: recognizing playing card symbols (rank and suit) in real-world images.
The work focuses on how different levels of model adaptation affect performance when objects are **very small, visually similar, and densely packed**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c3e57c00-3c09-4e66-8d2e-7480afc3d826" alt="CardDetection" width="600"/>
  </p>

## Research Goal
To evaluate how varying the number of trainable layers in a pretrained **YOLOv8n** model impacts detection accuracy, training stability, and error patterns when applied to fine-grained playing card recognition.

---

## Hypotheses
- **H1:** Increasing the number of trainable layers during fine-tuning improves detection performance **without causing catastrophic forgetting**, as pretrained features can be refined rather than overwritten.
- **H2:** Most classification errors occur between **visually similar card symbols**, reflecting the inherent difficulty of fine-grained discrimination at very small scales.

---

## Experimental Setup
- **Dataset:** ~2,200 images with ~60,000 annotated objects across **53 classes**
- **Model:** YOLOv8n (pretrained on COCO)
- **Fine-tuning strategies:**
  - **Head Only:** Detection head trained, backbone + neck frozen
  - **Neck + Head:** Backbone frozen, neck and head trained
  - **Entire Model:** Full end-to-end fine-tuning

---

## Key Findings
- **Full model fine-tuning performed best**, achieving:
  - Precision: **0.82**
  - Recall: **0.77**
  - mAP@50: **0.85**
  - mAP@50–95: **0.52**
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
