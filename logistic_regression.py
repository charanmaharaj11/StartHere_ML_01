import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def load_and_preprocess_data():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # All four features
    y = iris.target  # Numeric labels (0, 1, 2)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train):
    # Fit the logistic regression model using all 4 features
    log_reg = LogisticRegression(multi_class='ovr', solver='liblinear')
    log_reg.fit(X_train, y_train)
    
    return log_reg

def plot_decision_boundaries(X_train_scaled, y_train, log_reg):
    # Create a meshgrid to plot decision boundaries using Sepal Length and Sepal Width
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Set the other features (petal length and petal width) to their mean values
    mean_petal_length = X_train_scaled[:, 2].mean()  # Mean of petal length
    mean_petal_width = X_train_scaled[:, 3].mean()   # Mean of petal width

    # Create the feature set for predictions (keep sepal features from meshgrid, and use mean for others)
    X_mesh = np.c_[xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), mean_petal_length), np.full_like(yy.ravel(), mean_petal_width)]

    # Predict the class labels for each point in the meshgrid
    Z = log_reg.predict(X_mesh)
    Z = Z.reshape(xx.shape)  # Reshape to match the meshgrid shape

    # Plot decision boundaries
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    # Plot the training data points
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', marker='o', s=100, alpha=0.7)

    plt.title("Logistic Regression Decision Boundary (Using Sepal Length & Sepal Width)")
    plt.xlabel("Sepal Length (Standardized)")
    plt.ylabel("Sepal Width (Standardized)")
    plt.colorbar()  # Show color bar indicating class labels
    plt.show()

if __name__ == '__main__':
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data()
    log_reg = train_model(X_train_scaled, y_train)
    plot_decision_boundaries(X_train_scaled, y_train, log_reg)
