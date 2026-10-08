import os
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load master dataset
df = pd.read_csv(os.path.join(BASE_DIR, "master_dataset.csv"))

print("Dataset loaded.")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())


# Remove rows with missing values
df = df.dropna(
    subset=[
        "state",
        "district",
        "city",
        "year",
        "crime_type",
        "crime_count"
    ]
).copy()


# Create Label Encoders
le_state = LabelEncoder()
le_district = LabelEncoder()
le_city = LabelEncoder()
le_crime = LabelEncoder()


# Convert categorical columns into numbers
df["state_enc"] = le_state.fit_transform(
    df["state"].astype(str)
)

df["district_enc"] = le_district.fit_transform(
    df["district"].astype(str)
)

df["city_enc"] = le_city.fit_transform(
    df["city"].astype(str)
)

df["crime_type_enc"] = le_crime.fit_transform(
    df["crime_type"].astype(str)
)


# Features
X = df[
    [
        "state_enc",
        "district_enc",
        "city_enc",
        "crime_type_enc",
        "year"
    ]
]


# Target
y = df["crime_count"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


print("Training model...")

model.fit(X_train, y_train)

print("Model training completed.")


# Test model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("\nModel Performance")
print("-----------------")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


# Save model
joblib.dump(
    model,
    os.path.join(BASE_DIR, "crime_model_dr1.pkl")
)


# Save encoders
joblib.dump(
    le_state,
    os.path.join(BASE_DIR, "le_state.pkl")
)

joblib.dump(
    le_district,
    os.path.join(BASE_DIR, "le_district.pkl")
)

joblib.dump(
    le_city,
    os.path.join(BASE_DIR, "le_city.pkl")
)

joblib.dump(
    le_crime,
    os.path.join(BASE_DIR, "le_crime.pkl")
)


print("\nAll model files have been created successfully!")

print("crime_model_dr1.pkl")
print("le_state.pkl")
print("le_district.pkl")
print("le_city.pkl")
print("le_crime.pkl")