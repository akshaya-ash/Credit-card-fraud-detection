# save_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('creditcard.csv')  # Ensure this file is in your directory

# Prepare features and labels
X = df.drop(columns=['Class', 'Time'])
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, 'random_forest_model.pkl')
print("✅ Model saved as random_forest_model.pkl")
