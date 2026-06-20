import joblib

from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import preprocess_data


def train_model():

    X_train, X_test, y_train, y_test = preprocess_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/heart_model.pkl")

    return model, X_test, y_test