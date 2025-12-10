import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder


def preprocess_features(df: pd.DataFrame, numerical_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Preprocess the features in the DataFrame."""
    # Normalize numerical features
    if numerical_cols:
        for col in numerical_cols: 
            normalize_features(df, col)

    if categorical_cols:
        for col in categorical_cols:
            encode_categorical_features(df, col)
    return df

def normalize_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize numerical features in the dataset with StandardScaler."""
    scaler = StandardScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df

def encode_categorical_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Encode categorical features in the dataset."""
    encoder = OrdinalEncoder()
    df[col] = encoder.fit_transform(df[[col]])
    return df