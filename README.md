# OFETNow-ML

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data%20%26%20Models-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Pre-trained machine-learning models and inference workflow for predicting organic field-effect transistor (OFET) performance from polymer structure and device parameters. For the database, please visit https://www.ofetnow.top/.

> **Machine-Learning Prediction of Carrier Mobility and Polarity Type of Polymer Semiconductors: from High-Fidelity Data to Experimental Realization**

---

## Overview

OFETNow-ML provides two classification models:

- **Mobility Classifier** — predicts high vs. low charge-carrier mobility (threshold: 0.5 cm² V⁻¹ s⁻¹).
- **Carrier-Type Classifier** — predicts the dominant transport type (p-type / n-type / ambipolar).

Both models use device parameters, Morgan fingerprints (radius 8, 2048 bits), and RDKit molecular descriptors as input. This repository releases the training matrices used for model fitting, pre-trained models, the inference pipeline, and a 30-entry future-blind benchmark dataset.

---

## Installation

> [!IMPORTANT]
> Reproducing the released cross-validation results and model files requires the pinned environment below.

| Software | Version |
|---|---:|
| Python | 3.9 |
| NumPy | 2.0.2 |
| pandas | 2.3.1 |
| scikit-learn | 1.6.1 |
| XGBoost | 2.1.4 |
| joblib | 1.5.3 |
| RDKit | 2024.9.6 |

```bash
git clone https://github.com/wonderingbiggy/OFETNow.git
cd OFETNow
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Run on the included example (5 entries)
python examples/run_prediction.py

# Run on your own data
python examples/run_prediction.py --input data/your_input.json

# CSV input is also supported
python examples/run_prediction.py --input data/future_blind.csv
```

Results are saved to the `outputs/` directory. See `data/input_template.json` for the expected input format and `data/README.md` for detailed field descriptions.

Interactive Jupyter notebooks are available for generic prediction (`notebooks/OFETNow_generic_prediction.ipynb`) and for repeated cross-validation and model training (`notebooks/model_training_and_cv.ipynb`).

---

## Repository Structure

```
OFETNow-ML/
├── README.md
├── LICENSE
├── DATA_MODEL_LICENSE.md
├── requirements.txt
├── models/                 # Pre-trained models, ablation models, scalers, and feature lists of pre-trained models and ablation models
├── data/                   # Training matrices, input templates, raw data, and benchmark data
├── notebooks/              # Prediction and model-training notebooks
└── examples/               # Command-line prediction script
```

See `models/model_metadata.json` for model configuration details and `data/README.md` for data documentation.

---

## License

Code is released under the [MIT License](LICENSE). Data and model artifacts are released under [CC BY 4.0](DATA_MODEL_LICENSE.md).

---

## Citation

```bibtex
@article{Li2026OFETnow,
  title   = {Machine-Learning Prediction of Carrier Mobility and Polarity Type of Polymer Semiconductors: from High-Fidelity Data to Experimental Realization},
  author  = {Li, Yuan-Kai and Li, Si-Lu and Liu, Yi and Wu, Hao-Tian and Zhang, Tian-Yu and Wang, Jie-Yu and Yao, Ze-Fan and Pei, Jian},
  journal = {Journal of the American Chemical Society},
  year    = {2026},
  note    = {Under revision}
}
```
