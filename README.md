**HMGF-Net: Hierarchical Memory-Guided Graph Fusion Network for Traffic Flow Prediction**

## Overview

HMGF-Net is a novel traffic forecasting framework that integrates:

* Transformer-based spatiotemporal encoding
* Memory-guided traffic pattern refinement
* Hierarchical graph fusion
* Multi-scale static diffusion priors

to achieve accurate and robust traffic flow prediction.

---

## Requirements

* Python >= 3.9
* PyTorch >= 2.0
* CUDA >= 11.0 (optional)

## Dataset

The **PEMS03**, **PEMS04**, **PEMS07**, and **PEMS08** datasets used in this study are publicly available at:

https://drive.google.com/file/d/1_8GD3bnN5n1A0zn6e9oGlymna-pqppl9/view?usp=drive_link

After downloading, please organize the datasets as follows:

```text
datasets/
├── PEMS03/
├── PEMS04/
├── PEMS07/
└── PEMS08/
```

---

