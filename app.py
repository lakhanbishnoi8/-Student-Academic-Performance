import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model_utils import (
    load_dataset,
    train_model,
    evaluate_model,
    predict_marks,
    FEATURES,
    TARGET
)

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Application heading
st.title("🎓 Student Academic Performance Prediction")

st.subheader("Machine Learning Based Academic Analysis")

st.write(
    "This application predicts student final marks "
    "using the Linear Regression algorithm."
)

# Load dataset
@st.cache_data
def get_data():
    return load_dataset()

# Train model
@st.cache_resource
def get_model():
    data = get_data()
    return train_model(data)

# Load model and dataset
try:
    data = get_data()
    model, X_test, y_test, y_pred = get_model()

    metrics = evaluate_model(y_test, y_pred)

except Exception as e:
    st.error(f"Error loading project: {e}")
    st.stop()


# Sidebar
st.sidebar.title("Project Information")

st.sidebar.write("Domain: Education")
st.sidebar.write("Algorithm: Linear Regression")
st.sidebar.write("Learning: Supervised Learning")
st.sidebar.write("Language: Python")


# Navigation tabs
tab1, tab2, tab3 = st.tabs([
    "Predict Marks",
    "Model Performance",
    "Dataset Information"
])


# TAB 1: Prediction
with tab1:

    st.header("Enter Student Information")

    studytime = st.selectbox(
        "Study Time",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Less than 2 hours",
            2: "2 to 5 hours",
            3: "5 to 10 hours",
            4: "More than 10 hours"
        }[x]
    )

    failures = st.number_input(
        "Past Class Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences = st.number_input(
        "School Absences",
        min_value=0,
        max_value=93,
        value=4
    )

    g1 = st.number_input(
        "First Period Marks",
        min_value=0.0,
        max_value=20.0,
        value=12.0
    )

    g2 = st.number_input(
        "Second Period Marks",
        min_value=0.0,
        max_value=20.0,
        value=14.0
    )

    if st.button("Predict Final Marks"):

        student_data = [
            studytime,
            failures,
            absences,
            g1,
            g2
        ]

        result = predict_marks(model, student_data)

        st.success(
            f"Predicted Final Marks: {result:.2f} / 20"
        )

        st.progress(result / 20)

        st.info(
            "This is an estimated result generated "
            "by the machine learning model."
        )


# TAB 2: Model Performance
with tab2:

    st.header("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean Absolute Error",
        f"{metrics['mae']:.2f}"
    )

    col2.metric(
        "R2 Score",
        f"{metrics['r2']:.3f}"
    )

    col3.metric(
        "Testing Records",
        len(y_test)
    )

    st.subheader("Actual vs Predicted Marks")

    fig, ax = plt.subplots()

    ax.scatter(y_test, y_pred)

    ax.set_xlabel("Actual Marks")
    ax.set_ylabel("Predicted Marks")

    ax.set_title("Student Performance Prediction")

    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    plt.close(fig)


# TAB 3: Dataset Information
with tab3:

    st.header("Dataset Information")

    st.write("Dataset: UCI Student Performance")

    st.write("Total Records:", len(data))

    st.write("Total Columns:", len(data.columns))

    st.subheader("Input Features")

    st.write("""
    1. Study Time
    2. Past Failures
    3. School Absences
    4. First Period Marks (G1)
    5. Second Period Marks (G2)
    """)

    st.write("Target Variable: G3")

    st.write(
        "G3 represents the student's final mathematics grade."
    )

    st.dataframe(data[FEATURES + [TARGET]].head(10))

st.divider()

st.caption(
    "Student Academic Performance Prediction | "
    "Python | Machine Learning | Linear Regression"
)