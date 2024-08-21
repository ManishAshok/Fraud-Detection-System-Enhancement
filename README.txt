
# Fraud Detection System Enhancement Project

## Overview
This project enhances fraud detection capabilities by analyzing transaction data to identify suspicious activities. It uses Python and R for data analysis, with the machine learning model deployed on a Hadoop framework. SQL is used for data querying and preprocessing.

## Files Included

### Data Files
- `transaction_data.csv`: Synthetic transaction data including transaction amount, location, and fraud status.
- `X_train.csv`, `y_train.csv`: Training set for model training.
- `X_test.csv`, `y_test.csv`: Test set for model evaluation.

### Python Scripts
- `data_preprocessing.py`: Preprocesses the transaction data, normalizes features, and splits the data into training and testing sets.
- `model_training.py`: Trains a Random Forest classifier on the preprocessed data and evaluates its performance.
- `eda.py`: Performs exploratory data analysis (EDA) on the transaction data.

### How to Run the Project

1. **Data Preprocessing**:
   Run the `data_preprocessing.py` script to preprocess the transaction data.
   ```bash
   python data_preprocessing.py
   ```

2. **Exploratory Data Analysis**:
   Run the `eda.py` script to explore the transaction data.
   ```bash
   python eda.py
   ```

3. **Model Training and Evaluation**:
   Run the `model_training.py` script to train a Random Forest model and evaluate its performance.
   ```bash
   python model_training.py
   ```

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, matplotlib, seaborn

Install the required packages using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
