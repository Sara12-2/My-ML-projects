# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Load the dataset from a CSV file
column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Glass_Type']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data',
                   names=column_names)

# Drop the 'Id' column
data = data.drop(columns='Id')

# Split the features and target
X = data.drop(columns='Glass_Type')
y = data['Glass_Type']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create the SVM model with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='auto')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Plot Heatmap for Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="YlGnBu")
plt.title('Heatmap of Confusion Matrix')
plt.ylabel('Actual class')
plt.xlabel('Predicted class')
plt.show()
# Step 10: Correlation Heatmap of the Features
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Feature Correlations")
plt.show()

# Step 11: Predict new data in a loop
while True:
    user_input = input("\nWould you like to predict a new record? (y/n): \n")
    if user_input.lower() == 'n':
        print("\nExiting the prediction loop.\n")
        break
    elif user_input.lower() == 'y':
        try:
            # Input feature values from the user
            new_features = []
            for feature in X.columns:  # Using all features for simplicity
                while True:
                    try:
                        # Input validation to ensure numeric values
                        value = float(input(f"\nEnter {feature}: \n"))
                        new_features.append(value)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a numeric value.")
        except Exception as e:
            print("An error occurred during input:", str(e))
