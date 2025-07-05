# CPC Dataset Modeling

This repository contains datasets for the Choice Prediction Competition (CPC) project and a simple training script.

## Files

- `Competition background2025.06.10 (1).pdf` – background description of the competition.
- `Readme CPC training variables names 2025.04.22 (2).docx` – description of variables in the dataset.
- `Training 2025.04.22DM (2).xlsx` – spreadsheet of training data.
- `gpt_cpc_1000 oneshot DM (1).csv` – CSV version with textual descriptions used by the training script.
- `train_model.py` – Python script that trains a regression model predicting the A-rate from task parameters.

## Requirements

- Python 3.11+
- `pandas`
- `scikit-learn`

Install the Python dependencies with:

```bash
pip install pandas scikit-learn
```

## Usage

Run the training script:

```bash
python train_model.py
```

The script reads the CSV file, splits it into training and test sets, builds a pipeline with TF‑IDF encoding for text fields and a random forest regressor, then prints the mean absolute error on the test set.
