# Supply-Chain-Fraud-Detection

A Python project that detects fraud in supply-chain data using graph neural networks (GNNs). The repository contains preprocessing, model training, evaluation code, and a small app interface for inference/visualization.

> Note: This README was written from the repository layout (README.md, `app.py`, `gnn model.py`, `src/`, `data/`). If you want me to incorporate exact command-line flags, dataset filenames, or function names from the code, allow me to read the files and I’ll update the README with precise usage and examples.

## Table of contents
- [About](#about)
- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Quick start](#quick-start)
  - [Run the web app / API (app.py)](#run-the-web-app--api-apppy)
  - [Train the GNN model (gnn model.py)](#train-the-gnn-model-gnn-modelpy)
  - [Run inference](#run-inference)
- [Model overview](#model-overview)
- [Evaluation](#evaluation)
- [Configuration & hyperparameters](#configuration--hyperparameters)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About
This project aims to identify fraudulent actors or transactions in supply-chain datasets by representing entities and interactions as a graph and training a graph neural network (GNN) to classify suspicious nodes/edges. Typical use-cases:
- Flagging potentially fraudulent suppliers or purchase orders
- Prioritizing investigations
- Visualizing suspicious subgraphs for analyst review

## Features
- Graph construction from transactional / entity data
- GNN model implementation for node/edge classification
- Training and evaluation scripts
- Lightweight web app for inference and visualization (app.py)

## Repository structure
- README.md — this file
- app.py — web app / API for inference (Flask/FastAPI/streamlit style; please confirm)
- gnn model.py — training and model code for GNN
- src/ — helper modules, preprocessing, utilities
- data/ — datasets (not included in repo; add data here or set path)
- models/ — (optional) saved trained model files (create if needed)
- notebooks/ — (optional) exploratory notebooks (create if needed)

Adjust the structure above to match exact filenames/paths in your repo.

## Requirements
- Python 3.8+
- Recommended packages (examples):
  - numpy, pandas
  - scikit-learn
  - torch (PyTorch) and torch-geometric (or DGL) — if using PyG or DGL
  - networkx
  - joblib
  - Flask or FastAPI (or streamlit) — depending on app implementation
  - matplotlib / seaborn for visualizations

Install with pip:
```bash
python -m pip install -r requirements.txt
```
(If `requirements.txt` is not present, create one or install packages manually.)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/0hruv/Supply-Chain-Fraud-Detection.-.git
cd Supply-Chain-Fraud-Detection.-
```
2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
Place your dataset files inside the `data/` directory. Typical required files:
- entities.csv — list of suppliers/customers/entities and attributes
- transactions.csv — transactional edges (from, to, amount, timestamp, etc.)
- labels.csv — labels for supervised training (node or edge-level fraud labels)

If your code expects different filenames or a single consolidated file, update the `data/` path or the code accordingly.

## Quick start

### Run the web app / API (app.py)
The repository contains `app.py` which likely serves a web UI or REST endpoint for inference. Example usage (adjust if the app uses a different framework):

```bash
# If Flask
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000

# Or run directly
python app.py
```

Open http://localhost:5000 in your browser (or check the app logs for the actual address/port).

If the app expects a model path or data path, run:
```bash
python app.py --model-path models/best_model.pt --data-path data/
```
(Replace flags with the actual arguments used by app.py — I can extract them if you want me to read the file.)

### Train the GNN model (gnn model.py)
Example training command (fill in actual args used by the script):
```bash
python "gnn model.py" --data data/ --save-dir models/ --epochs 100 --batch-size 32 --lr 0.001
```
If the script uses a `main()` function, you can call it directly or run the file.

Saved model checkpoint(s) will be written to `models/`.

### Run inference
Either use the web app endpoint or execute an inference script (example):
```bash
python src/infer.py --model models/best_model.pt --input data/new_transactions.csv --output results/predictions.csv
```

## Model overview
The project uses a Graph Neural Network (GNN) to learn from graph-structured supply chain data:
- Graph nodes represent entities (suppliers, manufacturers, customers, etc.)
- Graph edges represent interactions (orders, shipments, payments)
- Node and/or edge features can include amounts, counts, time deltas, categorical encodings
- The GNN aggregates neighborhood information to predict fraud likelihood

Common GNN architectures used for such tasks:
- GCN (Graph Convolutional Network)
- GraphSAGE
- GAT (Graph Attention Network)
- Custom architectures combining temporal features

If your repository uses PyTorch Geometric or DGL, the training script will show the exact layers and loss functions.

## Evaluation
Suggested evaluation metrics for fraud detection:
- Precision, Recall, F1-score (particularly for the positive/fraud class)
- ROC-AUC and PR-AUC (precision-recall curve)
- Confusion matrix for threshold selection
- Precision@k for prioritizing top-k suspicious entities

Include cross-validation (time-split if data is temporal) and careful sampling because fraud is usually rare.

## Configuration & hyperparameters
Keep hyperparameters in a config file or use CLI args:
- learning rate, weight decay
- number of GNN layers
- hidden dimension size
- dropout
- batch size, epochs
- class weights / sampling strategy for imbalanced labels

Document the defaults in the training script or a YAML/JSON config.

## Tips for production / improvements
- Use temporal graph models when timestamps matter (e.g., TGN, temporal GNNs)
- Incorporate graph explainability (GNNExplainer, integrated gradients) to help analysts interpret results
- Maintain feature drift monitoring and model retraining pipelines
- Use domain knowledge to derive robust features (e.g., velocity features, supplier risk scores)

## Contributing
Contributions welcome. Suggested workflow:
1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests where applicable
4. Open a pull request with a clear description



## Contact
Maintainer: 0hruv  
If you have questions or want to contribute, open an issue or pull request on the repository.

---

If you want I can:
- Update this README with exact commands and examples extracted from `app.py` and `gnn model.py` (I need access to the source files in the repo).
- Commit this README.md directly to the repository for you (I’ll need permission to push).
- Create a `requirements.txt` or a minimal `models/` README that documents saved checkpoint names and expected input formats.


## Deploy on Streamlit Community Cloud ✅
Follow these steps to deploy the Streamlit app for free using Streamlit Community Cloud.

1. Create a minimal `requirements.txt` in the repo root (example includes `streamlit`, `pandas`, and visualization libs).
2. Push your branch to GitHub and open a PR when ready.
3. Sign in to https://share.streamlit.io with GitHub and click **New app** → choose the repo, branch, and `app.py` as the app file → click **Deploy**.
4. If your app requires secrets (API keys), add them in the Cloud UI under **Settings → Secrets** and access them in code via `st.secrets`.

Notes:
- The Community Cloud will install `requirements.txt` and run your app automatically; logs and build errors are visible in the app dashboard.
- Avoid committing secrets or large generated files (use `.gitignore`).

# Supply-Chain-Fraud-Detection.-
>>>>>>> feat/deploy/streamlit
