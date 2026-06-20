from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from src.preprocessing import preprocess_data


def compare_models():

    X_train, X_test, y_train, y_test = preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=1000,
            random_state=42
        )
    }

    accuracies = []

    print("\nMODEL COMPARISON\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        accuracies.append(accuracy * 100)

        print(f"{name} Accuracy: {round(accuracy * 100, 2)}%")

    plt.figure(figsize=(8, 5))

    plt.bar(models.keys(), accuracies)

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")

    plt.xticks(rotation=15)

    plt.show()