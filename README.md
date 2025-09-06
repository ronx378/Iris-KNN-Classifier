[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronx378/Iris-KNN-Classifier/blob/main/Iris_KNN_Classifier.ipynb)

# Iris Classification with KNN, Pipeline & GridSearchCV
Iris Classification with scikit-learn by building an automated, robust workflow to classify iris species using the Iris dataset.

# Project Highlights
- Pipeline: Standardizes features (with StandardScaler) before fitting the K-Nearest Neighbors (KNN) classifier for fair distance-based learning.

- Hyperparameter Tuning: Uses GridSearchCV for exhaustive, cross-validated search for the optimal number of neighbors, ensuring generalizable performance.

- Cross-Validation: Applies 5-fold cross-validation to avoid overfitting and improve model robustness.

- Evaluation: Calculates test set accuracy, confusion matrix, and a detailed classification report.

# Code Overview
1. Load & Preprocess Data:
    Loads the iris dataset, splits into stratified train/test sets.

2. Pipeline Construction:
    Combines feature scaling and KNN classifier into a single pipeline for cleaner, less error-prone code.

3. Hyperparameter Tuning:
    Runs a grid search over different K values (n_neighbors=[1,3,5,7,9,11]) to find the best model parameters using 5-fold CV.

4. Model Evaluation:
    Assesses performance with accuracy, precision, recall, F1, and confusion matrix on the held-out test set.

# How to Run
- Install dependencies:
    text
    pip install scikit-learn
    Save the notebook or script and run it.

# Example Output
text
Best k: {'knn__n_neighbors': 11}
Accuracy: 1.0

Classification Report:
              precision    recall  f1-score   support

     setosa       1.00      1.00      1.00        10
 versicolor       1.00      1.00      1.00        10
  virginica       1.00      1.00      1.00        10

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

Confusion Matrix:
 [[10  0  0]
  [ 0 10  0]
  [ 0  0 10]]
Your model's accuracy and classification performance may vary slightly depending on the train-test split.