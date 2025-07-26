from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Step 1: Load the Breast Cancer dataset from scikit-learn
cancer = datasets.load_breast_cancer()
X = cancer.data[:, :4]  # Use only the first 4 features
y = cancer.target
feature_names = cancer.feature_names[:4]  # Use the names of the first 4 features
target_names = cancer.target_names

# Step 2: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets (70% training, 30% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 4: Build the Decision Tree Classifier
model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
model.fit(X_train, y_train)
#step 5predict the testing data 
y_pred=model.predict(X_test)
#step 6 evalauate the model performance 
print("classification report:\n",classification_report(y_test,y_pred,zero_division=1))
print("confusion matrix:\n",confusion_matrix(y_test,y_pred))
print("accuracy score:\n",accuracy_score(y_test,y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 7: Plot the Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(model, feature_names=feature_names, class_names=target_names, filled=True)
plt.title("Decision Tree for Breast Cancer Dataset (First 4 Features)")
plt.show()

# Step 8: 3D Visualization of Decision Boundaries (first 3 features)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using first 3 features
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, cmap='coolwarm', s=60)
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_zlabel(feature_names[2])
plt.title("3D Decision Boundaries on Breast Cancer Test Data")
plt.show()

# Step 9: Predict new data in a loop
while True:
    user_input = input("Would you like to predict a new record? (y/n): ")
    if user_input.lower() == 'n':
        print("Exiting the prediction loop.")
        break
    elif user_input.lower() == 'y':
        try:
            # Input feature values from the user
            new_features = []  # Using the same 4 features for simplicity
            for feature in feature_names:
                while True:
                    # Input validation to ensure numeric values
                    try:
                        value = float(input(f"Enter {feature}: "))
                        new_features.append(value)
                        break  # Exit the loop after valid input
                    except ValueError:
                        print(f"Invalid input for {feature}. Please enter a numeric value.")

            # Normalize the user input
            new_data = scaler.transform([new_features])  # Transform must receive 2D array
            prediction = model.predict(new_data)

            print(f"Predicted Class: {target_names[prediction[0]]} (Benign or Malignant)")
        except Exception as e:
            print(f"An error occurred: {e}")
        else:
            print("invalid input .please type y for yes or n")