import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


print("========================================")
print("AI DISEASE PREDICTION - MODEL TRAINING")
print("========================================")

# Get project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path
dataset_path = os.path.join(
    BASE_DIR,
    "dataset",
    "Training.csv"
)

print("\nLoading dataset:")
print(dataset_path)

# Check dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"Training.csv not found at:\n{dataset_path}"
    )

# Load dataset
data = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")
print("Rows:", len(data))
print("Columns:", len(data.columns))


# Remove unnamed columns
data = data.loc[
    :,
    ~data.columns.str.contains("^Unnamed")
]


# Remove missing rows
data = data.dropna()

print("After cleaning:", data.shape)


# Check target column
if "prognosis" not in data.columns:
    raise ValueError(
        "The dataset does not contain the 'prognosis' column."
    )


# Separate features and target
X = data.drop("prognosis", axis=1)
y = data["prognosis"]


print("\nNumber of symptoms:", len(X.columns))
print("Number of diseases:", y.nunique())


# Encode disease names
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Train model
print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")


# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========================================")
print("MODEL RESULTS")
print("========================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_,
        zero_division=0
    )
)


# ========================================
# SAVE MODEL FILES
# ========================================

print("\n========================================")
print("SAVING MODEL FILES")
print("========================================")


model_path = os.path.join(
    BASE_DIR,
    "disease_model.pkl"
)

encoder_path = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

features_path = os.path.join(
    BASE_DIR,
    "features.pkl"
)


# Save model
joblib.dump(model, model_path)


# Save encoder
joblib.dump(encoder, encoder_path)


# Save feature names
joblib.dump(
    list(X.columns),
    features_path
)


# Check files
print("\nChecking saved files...")

files = [
    model_path,
    encoder_path,
    features_path
]

for file_path in files:

    if os.path.exists(file_path):

        size = os.path.getsize(file_path)

        print(
            os.path.basename(file_path),
            "->",
            size,
            "bytes"
        )

        if size == 0:
            raise RuntimeError(
                f"{file_path} was created but is EMPTY."
            )

    else:

        raise FileNotFoundError(
            f"{file_path} was not created."
        )


print("\n========================================")
print("SUCCESS!")
print("========================================")

print("All model files were created correctly.")

print("\nYou can now run:")
print("python app.py")