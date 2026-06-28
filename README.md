            # HMGF-Net: Hierarchical Memory-Guided Graph Fusion Network for Traffic Flow Prediction

Official implementation of the paper:

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

The processed **PEMS03**, **PEMS04**, **PEMS07**, and **PEMS08** datasets used in this study are publicly available at:

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

## Training

Train HMGF-Net using:

```bash
python train.py
```

or specify the configuration file:

```bash
python train.py --config configs/PEMS08.conf
```

Example:

```bash
python train.py --config configs/PEMS04.conf
```

---

## Evaluation

Evaluate a trained model by running:

```bash
python test.py --config configs/PEMS08.conf
```

---

## Project Structure

```text
HMGF-Net/
├── configs/
├── datasets/
├── model/
├── engine.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

---

## Citation

If you find this repository useful for your research, please cite:

```bibtex
@article{guo2026hmgfnet,
  title={HMGF-Net: Hierarchical Memory-Guided Graph Fusion Network for Traffic Flow Prediction},
  author={Guo, Guoxing and others},
  journal={},
  year={2026}
}
```

---

## Contact

For any questions, please feel free to open an issue on GitHub.

---

## License

This project is released under the MIT License.
