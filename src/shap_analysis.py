import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COLUMN
from utils import ensure_directories


def run_shap_analysis(model_name="random_forest", sample_size=500):
    """
    Run SHAP explainability analysis for the trained XGBoost model.

    Outputs:
    - shap_summary_plot.png
    - shap_bar_plot.png
    """

    ensure_directories([FIGURES_DIR])

    # Load trained pipeline
    model_pipeline = joblib.load(MODELS_DIR / f"{model_name}.pkl")

    # Load test data
    df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Extract fitted preprocessor and trained model
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    # Transform X into the format the model actually sees
    X_transformed = preprocessor.transform(X)

    # Convert sparse matrix to dense if necessary
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    feature_names = [
        name.replace("cat__", "").replace("num__", "")
        for name in feature_names
    ]

    # Use a sample for faster SHAP computation
    if X_transformed.shape[0] > sample_size:
        X_sample = X_transformed[:sample_size]
    else:
        X_sample = X_transformed

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Handle binary classification output
    if isinstance(shap_values, list):
        shap_values_class_1 = shap_values[1]
    else:
        shap_values_class_1 = shap_values

    # SHAP summary plot
    shap.summary_plot(
        shap_values_class_1,
        X_sample,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_summary_plot.png",
        bbox_inches="tight",
        dpi=300
    )
    plt.close()

    # SHAP bar plot
    shap.summary_plot(
        shap_values_class_1,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_bar_plot.png",
        bbox_inches="tight",
        dpi=300
    )
    plt.close()

    print("SHAP analysis complete.")
    print(f"Saved: {FIGURES_DIR / 'shap_summary_plot.png'}")
    print(f"Saved: {FIGURES_DIR / 'shap_bar_plot.png'}")


if __name__ == "__main__":
    run_shap_analysis()
        
        
        
        
        