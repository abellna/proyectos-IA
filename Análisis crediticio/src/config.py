from pathlib import Path

PROCESSED_PATH = Path("data/processed/dataset_clean.csv")
TARGET_COLUMN = 'loan_status'
SELECTED_COLUMNS = ['income', 'int_rate', 'percent_income', 'ownership_MORTGAGE', 'ownership_RENT', 'previous_loans_Yes']