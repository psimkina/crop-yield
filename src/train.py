from pathlib import Path

from data import load_data, clean_dataframe, split_data
from features import preprocess_features
from model import create_catboost_model, train_catboost_model

NUMERICAL_COLS = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
CATEGORICAL_COLS = ["area", "item"]
TARGET_COLUMN = "hg/ha_yield"


def main():

    # resolve paths
    root = Path(__file__).resolve().parents[1]  
    data_path = root / "data" / "raw" / "yield_df.csv"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.cbm"

    # prepare data for training
    df = load_data(data_path)
    df = clean_dataframe(df)

    # preprocess features
    df = preprocess_features(df, NUMERICAL_COLS, None) # encoding of categorical features is skipped for CatBoost

    # split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df, target_column=TARGET_COLUMN)

    print("Data prepared for training.")

    # create and train CatBoost model
    model = create_catboost_model()
    model = train_catboost_model(model, X_train, y_train, cat_features=CATEGORICAL_COLS)

    print("Model trained.")

    model.save_model(model_path)

if __name__ == "__main__":
    main()