"""
Trains a classifier on the Iris dataset and logs the experiment with MLflow.
Demonstrates core MLOps concepts: experiment tracking, model versioning, reproducibility.
"""
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "model/model.joblib"


def train_model(n_estimators=100, max_depth=5, random_state=42):
    # Load data
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Start an MLflow run — this tracks parameters, metrics, and the model itself
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        # Log parameters and metrics to MLflow — this is what makes it "MLOps"
        # rather than just "a script that trains a model"
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Also save locally so the serving API can load it without needing
        # an MLflow tracking server running
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        return model, accuracy, f1


if __name__ == "__main__":
    train_model()
