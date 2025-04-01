import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("music_features.csv")

# Separate features and target
X = df.drop(columns=["genre"])  # Features (MFCCs)
y = df["genre"]  # Target (genre labels)

# Encode genre labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(model, "models/music_genre_model.pkl")
joblib.dump(encoder, "models/music_genre_encoder.pkl")

print(f"✅ Model trained successfully with {X_train.shape[1]} features and saved!")