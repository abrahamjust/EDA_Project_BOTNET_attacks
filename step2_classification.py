import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Load the combined dataset
df = pd.read_csv("combined_data.csv")

# Split into features and target variable
X = df.drop(columns=["attack_type"])
y = df["attack_type"]

# Encode the labels (attack types) into numeric values
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Create a pipeline to scale features, apply PCA, and train the RandomForest model
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Scale features
    ("pca", PCA(n_components=10)),  # Apply PCA
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))  # Train Random Forest with parallel processing
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score (Weighted): {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Map numeric predictions back to label names
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# Create a DataFrame comparing actual vs predicted
comparison_df = pd.DataFrame({
    "Actual": y_test_labels,
    "Predicted": y_pred_labels
})

# Display a sample of 20 predictions
print("\nSample Predictions:")
print(comparison_df.sample(20))

# Feature importance (Random Forest)
importances = pipeline.named_steps["rf"].feature_importances_
print("\nFeature Importances:\n", importances)

import joblib
joblib.dump(pipeline, "botnet_random_forest_model.pkl")

# to load the model
#loaded_model = joblib.load("botnet_random_forest_model.pkl")
#predictions = loaded_model.predict(X_test)
