import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
	RAW_DATA_PATH, 
	PROCESSED_DATA_DIR, 
	TARGET_COLUMN, 
	RANDOM_STATE,
	TEST_SIZE, 
	VALID_SIZE
)
from utils import ensure_directories

def load_data(): 
	df = pd.read_csv(RAW_DATA_PATH)
	return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame: 
	df = df.copy()
	
	# Remove extra spaces from column names
	df.columns = df.columns.str.strip()
 
	# Drop columns that leak the answer
	leakage_columns = ["Churn Value", "Churn Score", "Churn Reason"]
 
	df = df.drop(columns = [c for c in leakage_columns if c in df.columns])
 
	# Drop columns that are not useful for first version
	drop_columns = [
		"CustomerID", "Count", "Country", "State", "City", "Zip Code", 
		"Lat Long", "Latitude", "Longitude"] 
 
	df = df.drop(columns = [c for c in drop_columns if c in df.columns])
 
	# Handle missing values
	for col in df.columns: 
		if df[col].dtype == "object": 
			df[col] = df[col].fillna("Unknown")
		else: 
			df[col] = df[col].fillna(df[col].median())
	# Encode target
	df[TARGET_COLUMN] = (
         df[TARGET_COLUMN]
	 .astype(str)
	 .str.strip()
	 .map({"Yes": 1, "No": 0})
	)
 
	return df

def split_and_save(df: pd.DataFrame):
	ensure_directories([PROCESSED_DATA_DIR])
	
	X = df.drop(columns=[TARGET_COLUMN])
	y = df[TARGET_COLUMN]
       
	X_train_full, X_test, y_train_full, y_test = train_test_split(
	       X, y, 
	       test_size = TEST_SIZE,
	       random_state = RANDOM_STATE,
	       stratify = y
        )
       
	X_train, X_valid, y_train, y_valid = train_test_split(
	       X_train_full, y_train_full,
	       test_size = VALID_SIZE,
	       random_state = RANDOM_STATE,
	       stratify = y_train_full
        )
       
	train_df = X_train.copy()
	train_df[TARGET_COLUMN] = y_train
       
	valid_df = X_valid.copy()
	valid_df[TARGET_COLUMN] = y_valid
       
	test_df = X_test.copy()
	test_df[TARGET_COLUMN] = y_test
       
	train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
	valid_df.to_csv(PROCESSED_DATA_DIR / "valid.csv", index=False)
	test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
       
	print("Saved train, valid, and test datasets.")
       
if __name__ == "__main__":
	df = load_data()
	df = clean_data(df)
	split_and_save(df)	
