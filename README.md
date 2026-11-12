# Titanic ML Training Pipeline

Containerized machine learning training environment for predicting passenger survival on the Titanic using a Random Forest classifier.

## Project Structure
- `train_model.py`: Data cleaning, feature engineering, model training, and artifact export.
- `Dockerfile`: JupyterLab and Python 3.11 environment setup.
- `data/`: Input dataset (`titanic.csv`).
- `models/`: Generated model artifacts (`.joblib`), metrics (`model_metadata.json`), and evaluation plots.

## Usage
```bash
docker build -t titanic-training .
docker run --rm -v "$(pwd):/home/jovyan/work" titanic-training python train_model.py
