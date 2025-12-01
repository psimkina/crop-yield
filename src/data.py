import pandas as pd 
from sklearn.model_selection import train_test_split

def load_data(file_path) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path, index_col=0)

def clean_dataframe(df: pd.DataFrame): 
    """Clean the DataFrame by handling missing values and duplicates."""

    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    df = df.drop_duplicates()
    df = df.dropna()

    return df

def split_data(df: pd.DataFrame, target_column: str, test_size: float=0.2, random_state: int=42):
    """Split the DataFrame into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test