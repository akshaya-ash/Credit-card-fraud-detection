import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ------------------------
# Step 1: Load the Dataset
# ------------------------
df = pd.read_csv('creditcard.csv')
print("Dataset shape:", df.shape)
print(df.head())

# Show fraud vs. non-fraud distribution
print("\nFraudulent vs Non-Fraudulent transactions:")
print(df['Class'].value_counts())

# --------------------------------
# Step 2: Preprocessing the Data
# --------------------------------
# Drop 'Time' column
df.drop(['Time'], axis=1, inplace=True)

# Scale the 'Amount' column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# -------------------------------
# Step 3: Balance the Dataset
# -------------------------------
fraud_count = df['Class'].value_counts()[1]
non_fraud_indices = df[df['Class'] == 0].index
fraud_indices = df[df['Class'] == 1].index

# Undersample non-fraud to match fraud count
non_fraud_sample = non_fraud_indices[:fraud_count]
balanced_indices = non_fraud_sample.union(fraud_indices)
df_balanced = df.loc[balanced_indices]

# Split features and target
X_balanced = df_balanced.drop('Class', axis=1)
y_balanced = df_balanced['Class']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

print("\nBalanced dataset shape:", df_balanced.shape)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ----------------------------
# Step 4: Train the Random Forest Classifier
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Step 5: Evaluate the Classifier
# ------------------------------
y_pred = model.predict(X_test)

print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------------
# Step 6: Save the Trained Model and Scaler
# -----------------------------------------
# OLD
model = joblib.load('model_full.pkl')
scaler = joblib.load('scaler_full.pkl')


print("\nModel and Scaler saved successfully!")


