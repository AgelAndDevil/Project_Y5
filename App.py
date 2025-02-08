# Importing required libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st

# Step 1: Exploratory Data Analysis (EDA)
data = sns.load_dataset('iris')

# Displaying basic dataset information
print("Dataset Information:")
print(data.info())

print("\nDataset Description:")
print(data.describe())

# Checking for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualizing the dataset with pairplot
sns.pairplot(data, hue="species", palette="Set2")
plt.show()

# Visualizing distributions
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for feature in features:
    sns.histplot(data=data,x=feature, kde=True, bins=20, hue=data['species'], palette='Set2')
    plt.title(f"Distribution of {feature}")
    plt.show()

# plt.figure(figsize=(8,6))
# for i,column in enumerate(nums,1):
#     plt.subplot(2,2,i)
#     # sns.histplot(data=data,x=column,kde=True,hue='species')
#     sns.histplot(data=data,x=column,kde=True,color='darkblue')
#     plt.title(column)
#     plt.xlabel(column)
#     plt.ylabel('Distribution')
# plt.tight_layout()
# plt.show()

# Step 2: Binary Logistic Regression with scikit-learn
# Adding binary target column
data['is_setosa'] = (data['species'] == 'setosa').astype(int)

# Splitting data
X = data[features]
y = data['is_setosa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model with scikit-learn
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = clf.predict(X_test)
print("\nScikit-learn Logistic Regression:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 2: Binary Logistic Regression with PyTorch
# Preparing data for PyTorch
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Train-test split
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Defining the model
class BinaryLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(BinaryLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = BinaryLogisticRegression(input_dim=X.shape[1])

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
y_pred_tensor = model(X_test_tensor)
y_pred_binary = (y_pred_tensor > 0.5).float()
accuracy = (y_pred_binary.eq(y_test_tensor).sum().item() / y_test_tensor.shape[0]) * 100
print(f"PyTorch Logistic Regression Accuracy: {accuracy:.2f}%")

# Step 3: Streamlit Web Application
st.title("Binary Logistic Regression for Iris Dataset")
model_choice = st.radio("Select the model for prediction:", ("Scikit-learn Logistic Regression", "PyTorch Binary Logistic Regression"))

# Accuracy
sklearn_accuracy = accuracy_score(y_test, clf.predict(X_test))
pytorch_accuracy = (y_pred_binary.eq(y_test_tensor).sum().item() / y_test_tensor.shape[0]) * 100

st.sidebar.title("Model Accuracy")
st.sidebar.write(f"Scikit-learn Accuracy: {sklearn_accuracy * 100:.2f}%")
st.sidebar.write(f"PyTorch Accuracy: {pytorch_accuracy:.2f}%")

# Group input fields and button in a form
with st.form(key="prediction_form"):
    # Input features
    sepal_length = st.number_input("Sepal Length", min_value=4.0, max_value=8.0, step=0.1)
    sepal_width = st.number_input("Sepal Width", min_value=2.0, max_value=4.5, step=0.1)
    petal_length = st.number_input("Petal Length", min_value=1.0, max_value=7.0, step=0.1)
    petal_width = st.number_input("Petal Width", min_value=0.1, max_value=3.0, step=0.1)
    
    # Submit button
    submit_button = st.form_submit_button(label="Predict")


if submit_button:
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]

    if model_choice == "Scikit-learn Logistic Regression":
            # Scikit-learn model prediction
        sklearn_prediction = clf.predict(input_features)[0]
        prediction_binary = "Setosa" if sklearn_prediction == 1 else "Not Setosa"
        st.write(f"Prediction (Scikit-learn): {prediction_binary}")
        
    elif model_choice == "PyTorch Binary Logistic Regression":
        # PyTorch model prediction
        input_tensor = torch.tensor(input_features, dtype=torch.float32)
        pytorch_prediction = model(input_tensor)
        pytorch_binary = "Setosa" if pytorch_prediction.item() > 0.5 else "Not Setosa"
        st.write(f"Prediction (PyTorch): {pytorch_binary}")


# if st.button("Predict"):
#     input_features = torch.tensor([[sepal_length, sepal_width, petal_length, petal_width]], dtype=torch.float32)
#     prediction = model(input_features)
#     prediction_binary = "Setosa" if prediction.item() > 0.5 else "Not Setosa"
#     st.write(f"Prediction: {prediction_binary}")
