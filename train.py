import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# Load dataset
df = pd.read_csv("car_data.csv")

# Features and target
X = df[["year", "km_driven", "mileage", "engine"]]
y = df["price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

# Save model and score
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(r2, open("r2_score.pkl", "wb"))

print("Model trained and saved successfully!")