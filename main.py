import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Step 2: Select input features
features = ["studytime", "failures", "absences", "G1", "G2"]

X = data[features]
y = data["G3"]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("STUDENT ACADEMIC PERFORMANCE PREDICTION")
print("-" * 45)
print("Model trained successfully!")

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Step 6: Take student information
print("\nEnter Student Information")

studytime = int(input("Study time category (1-4): "))
failures = int(input("Past class failures (0-4): "))
absences = int(input("Number of absences (0-93): "))
g1 = float(input("First-period marks (0-20): "))
g2 = float(input("Second-period marks (0-20): "))

# Step 7: Validate input
if not 1 <= studytime <= 4:
    raise ValueError("Study time must be between 1 and 4.")

if not 0 <= failures <= 4:
    raise ValueError("Failures must be between 0 and 4.")

if not 0 <= absences <= 93:
    raise ValueError("Absences must be between 0 and 93.")

if not 0 <= g1 <= 20 or not 0 <= g2 <= 20:
    raise ValueError("Previous marks must be between 0 and 20.")

# Step 8: Predict marks
student = pd.DataFrame(
    [[studytime, failures, absences, g1, g2]],
    columns=features
)

predicted_marks = model.predict(student)[0]

# Keep the displayed estimate within the dataset's grade range
predicted_marks = max(0, min(20, predicted_marks))

print("\nPREDICTION RESULT")
print("-" * 30)
print(f"Predicted Final Marks: {predicted_marks:.2f} / 20")

# Step 9: Display a graph
plt.scatter(y_test, predictions)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Student Marks")

plt.show()

