import pandas as pd 
import matplotlib.pyplot as plt

from config import RAW_DATA_PATH, FIGURES_DIR
from utils import ensure_directories

def run_eda():
    ensure_directories([FIGURES_DIR])

    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = df.columns.str.strip()

    # 1. Churn distribution
    churn_counts = df["Churn Label"].value_counts()

    plt.figure(figsize=(7, 5))
    plt.bar(churn_counts.index, churn_counts.values)
    plt.xlabel("Churn Label")
    plt.ylabel("Number of Customers")
    plt.title("Churn Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_churn_distribution.png")
    plt.close()

    # 2. Contract type vs churn rate
    contract_churn = (
        df.groupby("Contract")["Churn Label"]
        .apply(lambda x: (x == "Yes").mean())
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(contract_churn.index, contract_churn.values)
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Rate")
    plt.title("Churn Rate by Contract Type")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_contract_churn.png")
    plt.close()

    # 3. Tenure vs churn
    df["Tenure Group"] = pd.cut(
        df["Tenure Months"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        include_lowest=True
    )

    tenure_churn = (
        df.groupby("Tenure Group", observed=True)["Churn Label"]
        .apply(lambda x: (x == "Yes").mean())
    )

    plt.figure(figsize=(9, 5))
    plt.bar(tenure_churn.index.astype(str), tenure_churn.values)
    plt.xlabel("Tenure Group")
    plt.ylabel("Churn Rate")
    plt.title("Churn Rate by Tenure Group")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_tenure_churn.png")
    plt.close()

    # 4. Monthly charges vs churn
    plt.figure(figsize=(8, 5))

    churn_yes = df[df["Churn Label"] == "Yes"]["Monthly Charges"]
    churn_no = df[df["Churn Label"] == "No"]["Monthly Charges"]

    plt.hist(churn_no, bins=30, alpha=0.6, label="No Churn")
    plt.hist(churn_yes, bins=30, alpha=0.6, label="Churn")

    plt.xlabel("Monthly Charges")
    plt.ylabel("Number of Customers")
    plt.title("Monthly Charges Distribution by Churn Status")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_monthly_charges_churn.png")
    plt.close()

    print("EDA figures saved successfully.")


if __name__ == "__main__":
    run_eda()
 