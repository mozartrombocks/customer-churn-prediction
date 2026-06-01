import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, TARGET_COLUMN
from utils import ensure_directories

def plot_feature_importance():
        ensure_directories([FIGURES_DIR])
        
        model_pipeline = joblib.load(MODELS_DIR / "random_forest.pkl")
        df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
        X = df.drop(columns = [TARGET_COLUMN])
        
        preprocessor = model_pipeline.named_steps["preprocessor"]
        model = model_pipeline.named_steps["model"]
        	
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_
        
        feature_names = [
                name.replace("cat__", "")
                .replace("num__", "")
                for name in feature_names
        ]


           
        importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
        }).sort_values(by="importance", ascending=False)

        # Keep only the top 15
        top_features = importance_df.head(15)

        plt.figure(figsize=(12, 8))
        plt.barh(
                top_features["feature"][::-1],
                top_features["importance"][::-1]
        )

        plt.xlabel("Feature Importance")
        plt.title("Top 15 Random Forest Features")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png")
        plt.close()

        print(top_features)
        
        top_features.to_csv(
                FIGURES_DIR / "top_features.csv",
                index=False
        )
        
        importance_df.to_csv(
                FIGURES_DIR / "all_feature_importance.csv",
                index=False
)
        
        
if __name__ == "__main__":
	plot_feature_importance()