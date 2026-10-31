import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_save_model():
    df = pd.read_csv("dataset.csv")
    feature_cols = [
        "age",
        "sex",
        "chest pain type",
        "resting bp s",
        "cholesterol",
        "fasting blood sugar",
        "resting ecg",
        "max heart rate",
        "exercise angina",
        "oldpeak",
        "ST slope",
    ]
    X = df[feature_cols]
    y = df["target"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "model.joblib")
    print("Model saved as model.joblib")


if __name__ == "__main__":
    train_and_save_model()
