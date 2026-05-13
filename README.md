# OFETNow-ML

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data%20%26%20Models-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Pre-trained machine-learning models and inference workflow for predicting organic field-effect transistor (OFET) performance from polymer structure and device parameters, for the database, please visit http://ofetnow.top.

> **Machine-Learning Accelerated Discovery of Polymer Semiconductors using a Closed-Loop Strategy from High-Fidelity Data to Experimental Realization**

---

## Overview

OFETNow-ML provides two classification models:

- **Mobility Classifier** — predicts high vs. low charge-carrier mobility (threshold: 0.5 cm² V⁻¹ s⁻¹).
- **Carrier-Type Classifier** — predicts the dominant transport type (p-type / n-type / ambipolar).

Both models use device parameters, Morgan fingerprints (radius 8, 2048 bits), and RDKit molecular descriptors as input. The full training dataset is not included; this repository releases pre-trained models, the inference pipeline, and a 30-entry future-blind benchmark dataset.

---

## Installation

```bash
git clone https://github.com/wonderingbiggy/OFETNow.git
cd OFETNow-ML
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

An interactive Jupyter notebook is also available at `notebooks/OFETNow_generic_prediction.ipynb`.

---

## Repository Structure

```
OFETNow-ML/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── models/                 # Pre-trained models, scalers, and feature lists
├── data/                   # Input templates, example data, and benchmark dataset
├── notebooks/              # Interactive prediction notebook
└── examples/               # Command-line prediction script
```

See `models/model_metadata.json` for model configuration details and `data/README.md` for data documentation.

---

## License

Code is released under the [MIT License](LICENSE). Models and benchmark data are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Citation

```bibtex
@article{OFETNow-ML,
  title   = {Machine-Learning Accelerated Discovery of Polymer Semiconductors using a Closed-Loop Strategy from High-Fidelity Data to Experimental Realization},
  author  = {Yuan-Kai Li, Si-Lu Li, Hao-Tian Wu, Ze-Fan Yao, Jie-Yu Wang, Jian Pei},
  Under review
```
