import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(df: pd.DataFrame): 
    # Identify numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', "bool"]).columns.tolist()

    # Define transformers for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='median')),
	('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='most_frequent')),
	('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
	transformers=[
	    ('num', numeric_transformer, numeric_features),
	    ('cat', categorical_transformer, categorical_features)
	]
    )

    return preprocessor, numeric_features, categorical_features
