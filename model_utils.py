from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)


# Dataset location
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "student-mat.csv"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "320/student+performance.zip"
)


# Input features
FEATURES = [
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2"
]

# Target variable
TARGET = "G3"


# Download dataset
from pathlib import Path

DATA_FILE = Path(__file__).parent / "student-mat.csv"

def ensure_dataset():

    if DATA_FILE.exists():
        return DATA_FILE

    raise FileNotFoundError(
        "student-mat.csv is missing. "
        "Please upload it beside app.py."
    )

# Load dataset
def load_dataset():

    path = ensure_dataset()

    data = pd.read_csv(
        path,
        sep=";"
    )

    data = data.dropna(
        subset=FEATURES + [TARGET]
    )

    return data


# Train model
def train_model(data):

    X = data[FEATURES]

    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions


# Evaluate model
def evaluate_model(y_test, predictions):

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "mae": mae,
        "r2": r2
    }


# Predict marks
def predict_marks(model, values):

    student = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    prediction = model.predict(student)[0]

    # Restrict displayed marks to 0-20
    prediction = max(
        0,
        min(20, prediction)
    )

    return prediction
