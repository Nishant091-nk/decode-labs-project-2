# ============================================
# Project 2: Data Classification Using AI
# Dataset: Iris Dataset
# Algorithm: K-Nearest Neighbors (KNN)
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)

# ============================================
# Step 1: Load Dataset
# ============================================

iris = load_iris()

X = iris.data
y = iris.target

# ============================================
# Step 2: Dataset Information
# ============================================

print("=" * 50)
print("IRIS DATASET INFORMATION")
print("=" * 50)

print("\nFeature Names:")
print(iris.feature_names)

print("\nTarget Names:")
print(iris.target_names)

print("\nTotal Samples:", X.shape[0])
print("Total Features:", X.shape[1])

# ============================================
# Step 3: Convert into DataFrame
# ============================================

df = pd.DataFrame(X, columns=iris.feature_names)
df["Species"] = y

print("\nFirst 5 Rows:")
print(df.head())

# ============================================
# Step 4: Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================================
# Step 5: Feature Scaling
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================
# Step 6: Train KNN Model
# ============================================

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

# ============================================
# Step 7: Prediction
# ============================================

y_pred = model.predict(X_test)

# ============================================
# Step 8: Accuracy
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL ACCURACY")
print("=" * 50)

print(f"Accuracy : {accuracy * 100:.2f}%")

# ============================================
# Step 9: Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ============================================
# Step 10: Classification Report
# ============================================

print("\nClassification Report")

print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# ============================================
# Step 11: Plot Confusion Matrix
# ============================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# ============================================
# Step 12: Predict New Flower
# ============================================

print("\nPredicting a New Flower")

sample = [[5.1, 3.5, 1.4, 0.2]]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("Predicted Flower:", iris.target_names[prediction[0]])

print("\nProject Completed Successfully!")