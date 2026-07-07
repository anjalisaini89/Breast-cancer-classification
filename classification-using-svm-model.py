import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# Download and Load Dataset
# ==========================================

path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
print("Dataset Path:", path)

df = pd.read_csv(path + "/data.csv")

# ==========================================
# Exploratory Data Analysis (EDA)
# ==========================================

print("\nFirst 5 Records")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# Remove unnecessary columns
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

print("\nColumns")
print(df.columns)

# ==========================================
# Data Visualization
# ==========================================

sns.pairplot(
    df[
        [
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "diagnosis",
        ]
    ],
    hue="diagnosis",
)

plt.show()

plt.figure(figsize=(5,4))
sns.countplot(x="diagnosis", data=df)
plt.title("Diagnosis Count")
plt.show()

# ==========================================
# Prepare Data
# ==========================================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Train SVM Model
# ==========================================

model = SVC()

model.fit(X_train, y_train)

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

print("\nPredictions")
print(y_pred)

# ==========================================
# Model Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

# ==========================================
# Plot Confusion Matrix
# ==========================================

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()