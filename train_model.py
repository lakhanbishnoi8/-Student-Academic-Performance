from model_utils import (
    load_dataset,
    train_model,
    evaluate_model
)


# Load dataset
data = load_dataset()

# Train model
model, X_test, y_test, predictions = train_model(data)

# Evaluate model
result = evaluate_model(
    y_test,
    predictions
)

print("-----------------------------------")
print("STUDENT PERFORMANCE PREDICTION")
print("-----------------------------------")

print("Dataset Records:", len(data))

print("Training Records:", len(data) - len(y_test))

print("Testing Records:", len(y_test))

print("Mean Absolute Error:", result["mae"])

print("R2 Score:", result["r2"])

print("-----------------------------------")

print("Model Training Completed Successfully!")