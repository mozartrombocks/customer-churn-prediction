import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
	roc_auc_score, 
	roc_curve, 
	confusion_matrix, 
	ConfusionMatrixDisplay, 
	classification_report
)

from config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR,  TARGET_COLUMN
from utils import ensure_directories


def load_split(filename): 
        df = pd.read_csv(PROCESSED_DATA_DIR / filename)
        X = df.drop(columns = [TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        return X, y

def evaluate_model(model_name = "xgboost"):
        ensure_directories([FIGURES_DIR])
        
        model = joblib.load(MODELS_DIR / f"{model_name}.pkl") 
        
        print(f"Evaluating {model_name} on test set...")
        
        X_test, y_test = load_split("test.csv")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Test ROC_AUC: {auc:.4f}") 
        print(classification_report(y_test, y_pred))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)		
        
        plt.figure(figsize = (8, 6))
        plt.plot(fpr, tpr, label=f"{model_name} AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle = "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "roc_curve.png")
        plt.close()
 
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "confusion_matrix.png")
        plt.close()
        

if __name__ == "__main__":
    for model_name in [
        "logistic_regression",
        "random_forest",
        "xgboost"
    ]:
        evaluate_model(model_name)
        print("-" * 60)