import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import setup, compare_models, predict_model, pull, save_model
import joblib


def split_data(data: pd.DataFrame, test_size: float = 0.2):
    X = data.drop("class", axis=1)
    y = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


def train_model(split_output: dict):
    X_train = split_output["X_train"]
    y_train = split_output["y_train"]
    X_test = split_output["X_test"]
    y_test = split_output["y_test"]

    train = pd.concat([X_train, y_train], axis=1)
    train.columns = list(X_train.columns) + ['class']

    setup(
        data=train,
        target='class',
        session_id=123,
        fix_imbalance=True,
        verbose=False,
    )

    best_model = compare_models()

    save_model(best_model, "data/06_models/best_model")
    joblib.dump(best_model, "data/06_models/best_model.pkl")

    test_df = X_test.copy()
    test_df['class'] = y_test.values
    predict_model(best_model, data=test_df)
    metrics = pull()
    metrics.to_csv("data/08_reporting/model_metrics.csv", index=False)

    return best_model
