import joblib 
import pandas as pd

from config import MODELS_DIR

def predict_single(input_df: pd.DataFrame, model_name = "xgboost"):
	model = joblib.load(MODELS_DIR / f"{model_name}.pkl") 
	pred = model.predict(input_df)[0]
	prob = model.predict_proba(input_df)[0][1]
	return pred, prob