import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()

    # Zakładamy, że 'class' to target (jadalny/trujący)
    X = data_copy.drop(columns=["class"])
    y = data_copy["class"]

    categorical_cols = X.columns.tolist()
    
    
    preprocessor_mushrooms = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols),
        ]
    )

    transformed = preprocessor_mushrooms.fit_transform(X)
    feature_names = preprocessor_mushrooms.named_transformers_["cat"].get_feature_names_out(categorical_cols)

    df_out = pd.DataFrame(transformed, columns=feature_names, index=data_copy.index)
    df_out["class"] = y.values

    joblib.dump(preprocessor_mushrooms, "data/06_models/preprocessor.pkl")

    return df_out
