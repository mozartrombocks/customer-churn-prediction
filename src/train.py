import joblib 
import pandas as pd

from xgboost import XGBClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
 
from config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COLUMN
from features import build_preprocessor
from utils import ensure_directories
 
def load_split(filename): 
        df = pd.read_csv(PROCESSED_DATA_DIR / filename)
        X = df.drop(columns = [TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        return X, y

def train_models(): 
        ensure_directories([MODELS_DIR])
         
        X_train, y_train = load_split("train.csv")
        X_valid, y_valid = load_split("valid.csv")
         
        preprocessor, _, _ = build_preprocessor(X_train)
         
        logistic_pipeline = Pipeline(steps = [
		("preprocessor", preprocessor), 
		("model", LogisticRegression(max_iter = 1000, random_state = 42))
	])
         
        rf_pipeline = Pipeline(steps = [
		("preprocessor", preprocessor),
		("model", RandomForestClassifier(n_estimators = 100, 
                                    		max_depth = 8,
                                        	min_samples_split = 10, 
                                           	random_state = 42))
	])
        
        xg_pipeline = Pipeline(steps = [
                ("preprocessor", preprocessor), 
                ("model", XGBClassifier(n_estimators = 100, 
                                        max_depth = 6, 
                                        learning_rate = 0.05,
                                        subsample = 0.8, 
                                        colsample_bytree = 0.8,  
                                        random_state = 42,
                                        eval_metric = "logloss"  
                                        ))
        ])
         
        models = {
		"logistic_regression": logistic_pipeline,
		"random_forest": rf_pipeline,
                "xgboost": xg_pipeline  
	}
         
        for model_name, pipeline in models.items(): 
                pipeline.fit(X_train, y_train)
                y_pred_proba = pipeline.predict_proba(X_valid)[:, 1]
                auc = roc_auc_score(y_valid, y_pred_proba)
                 
                print(f"{model_name} Validation ROC-AUC: {auc:.4f}")
                 
                y_pred = pipeline.predict(X_valid)
                print(classification_report(y_valid, y_pred))
                 
                joblib.dump(pipeline, MODELS_DIR / f"{model_name}.pkl")
                
        print("Models saved successfully.")
        
if __name__ == "__main__":
	train_models()
                  
