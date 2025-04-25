# Project 2: Boosting Trees

## Group Members

**Samarth Rajput**  
A20586237

**Jenil Panchal**  
A20598955

**Yashashree Reddy Karri**  
A20546825

**Krishna Reddy**  
A20563553

## Table of Contents  
1. [Introduction](#introduction)  
2. [Environment Setup](#environment-setup)  
3. [Project Overview](#project-overview)  
4. [Running the Code](#running-the-code)  
5. [Testing the Model](#testing-the-model)  
6. [Visualizations](#visualizations)  
7. [Tunable Parameters](#tunable-parameters)  
8. [Contributors](#contributors)

## Introduction  
This project presents a custom implementation of a **Gradient Boosting Classifier** using decision trees as base learners. Unlike typical libraries like scikit-learn, this implementation was developed from scratch to understand the underlying mechanisms and explore how gradient boosting can handle complex, nonlinear classification problems. The project supports both binary and multi-class classification and includes features like early stopping, residual calculation based on logistic loss, and visualization of model decision boundaries and performance metrics.

## Project Overview
The primary objective of this project is to develop and test a Gradient Boosting Tree Classifier that can handle non‑linear decision boundaries and complex classification problems. Gradient Boosting is particularly valuable when the relationship between features and target variables is non‑linear and complex, as it can capture intricate patterns through the sequential addition of weak learners.

The `run_gradient_boosting.py` script provides an end‑to‑end workflow that:

1. Loads a dataset (generated or user‑supplied CSV).
2. Shuffles and normalizes the data.
3. Splits the data into training and test sets.
4. Trains the Gradient Boosting model.
5. Evaluates performance on training and test sets.
6. Produces visualizations to assess model fit.

Typical use‑cases include spam detection, customer‑churn prediction, or binary medical diagnosis.

---
## Model Description
The Gradient Boosting Tree Classifier is an ensemble of **decision stumps**. Each stump makes a simple decision based on a single feature and threshold; the ensemble combines these decisions to create a powerful classifier.

### Loss and Gradient

> **Heads-up:** If the equations below appear as plain text in VS Code, install a Markdown extension with MathJax support (e.g. *Markdown Preview Enhanced* or *Markdown+Math*).

For binary labels \( y \in \{-1,1\} \) we minimise the **logistic loss**

$$
\mathcal{L}(y, F(x)) \;=\; \log\Bigl(1 + e^{-y\,F(x)}\Bigr)
$$

The corresponding negative gradient (our *residuals*) is

$$
\tilde r \;=\; -\frac{\partial \mathcal{L}}{\partial F(x)}
           \;=\; \frac{y}{1 + e^{y\,F(x)}}.
$$

### Algorithm

1. **Initialisation:** \( F_0(x) = 0 \).
2. **For** \( m = 1, \dots, M \):
   1. Compute residuals \( \tilde r^{(m)} \).
   2. Fit a decision stump \( h_m(x) \) to the residuals.
   3. Update the ensemble  
      \( F_m(x) \;=\; F_{m-1}(x) + \eta\,h_m(x) \)  
      where \( \eta \) is the learning-rate.
   4. If accuracy hasn’t improved for 20 rounds, trigger **early stopping**.

The `DecisionStump` class enumerates candidate splits for every feature-threshold pair, chooses the split that minimises squared error of residuals, and predicts a constant value on each side of the split.


---
## Environment Setup  
To run this project smoothly, follow these steps:
```bash
### 1. Clone the Repository
git clone https://github.com/Samarth677/Project2.git
cd Project2

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
(The list is short: `numpy`, `matplotlib`, `pytest`.)

### 3. Generate a synthetic dataset *(optional)*
```bash
python generate_binary_data.py --samples 600 --noise 0.25 --visualize
```
This saves `hard_test.csv` under `data/` and **Figure_5.png** under `Output_Images/`.

### 4. Train & evaluate the model
```bash
python run_gradient_boosting.py
```
You’ll see console metrics, and the five figures in `Output_Images/` will refresh.

### 5. Run the visualtizaion
```bash
python visualize.py
```
All five tests should pass in approximately 0.4 seconds.

##Project Overview
This project includes:

A GradientBoostingClassifier written from scratch in models/gradient_boosting.py

Synthetic data generation in data/synthetic_data.py

Unit tests in tests/test_gradient_boosting.py

Visualization script visualize.py for graphical analysis of model performance


Testing the Model
We wrote 5 unit tests in tests/test_gradient_boosting.py to ensure reliability:
pytest -v

### Test Cases & Sample Output:

#### Running binary classification test...
✅ Binary Accuracy: 0.85  
✅ Binary classification test passed!

#### Running multiclass classification test...
✅ Multi-class Accuracy: 0.90  
✅ Multiclass classification test passed!

#### Running overfit small dataset test...
✅ Overfitting test accuracy: 1.00  
✅ Overfitting test passed!

#### Running early stopping test...
✅ Trees built (early stopping): 200 / 200  
- Class 0: 100 trees  
- Class 1: 100 trees  
ℹ️  Early stopping did not trigger, but the model still completed correctly.  
✅ Early stopping test complete!

#### Running high-dimensional data test...
✅ Accuracy on high-dimensional data: 1.00  
✅ High-dimensional test passed!


## Visualizations

The `visualize.py` script generates the following visual outputs to help evaluate model behavior:

- **Decision Boundary**  
  Shows how the classifier separates different classes in the feature space.

- **PCA Projection (True vs Predicted)**  
  Projects high-dimensional data into 2D using PCA and overlays true vs predicted labels to inspect clustering and accuracy visually.

- **Probability Heatmap**  
  A heatmap displaying the model’s confidence levels across the feature space, highlighting areas of high and low certainty.

- **Classification Report Heatmap**  
  Displays key classification metrics (precision, recall, F1-score) in a matrix form, offering a snapshot of performance per class.

- **Predicted Probability Distribution**  
  A histogram of predicted class probabilities, useful for assessing the confidence distribution of the classifier’s predictions.



## Tunable Parameters

| Parameter        | Description                              | Default |
|------------------|------------------------------------------|---------|
| `n_estimators`   | Number of trees to boost                 | 100     |
| `learning_rate`  | Contribution of each tree                | 0.1     |
| `max_depth`      | Depth of each individual tree            | 1       |
| `early_stopping` | Enable early stopping                    | False   |
| `patience`       | Rounds to wait without improvement       | 5       |
| `val_fraction`   | Fraction of data reserved for validation | 0.1     |

## Q&A

### 1) What does the model you have implemented do and when should it be used?

The implemented model is a **Gradient Boosting Classifier** that builds an ensemble of decision trees to perform classification tasks. It works by training trees sequentially, each focusing on correcting the errors of its predecessor using gradient descent on a loss function. The classifier supports both **binary and multi-class classification**, with optional **early stopping** to avoid overfitting.

**Use cases:**

- When high accuracy is desired on structured/tabular data  
- When interpretability and iterative refinement are needed  
- Suitable for datasets with a mix of feature types (numerical, or categorical if preprocessed)

---

### 2) How did you test your model to determine if it is working reasonably correctly?

You implemented a comprehensive test suite in `tests/test_gradient_boosting.py`, covering:

- Binary classification test for basic correctness  
- Multi-class classification support validation  
- Overfitting test on a small dataset to verify learning capacity  
- Early stopping mechanism to ensure training halts with no improvement  
- High-dimensional data handling to confirm robustness in complex feature spaces

---

### 3) What parameters have you exposed to users of your implementation in order to tune performance?

The `GradientBoostingClassifier` exposes the following tunable parameters:

- `n_estimators`: Number of boosting rounds (trees)  
- `learning_rate`: Shrinks the contribution of each tree  
- `max_depth`: Maximum depth of each individual tree  
- `early_stopping`: Boolean flag to enable early stopping  
- `patience`: Number of rounds without improvement before early stopping  
- `val_fraction`: Fraction of training data used for validation during early stopping

---

### 4) Are there specific inputs that your implementation has trouble with? Can they be worked around?

**Noisy Data**  
The model may struggle to generalize when the dataset contains significant noise. This can lead to overfitting, especially when too many trees are added.  
**Workaround:** Use a lower `learning_rate` or limit the number of trees via `n_estimators`.

**High-Dimensional Data**  
With a large number of features, the model may become computationally expensive and harder to interpret, potentially capturing spurious relationships.  
**Workaround:** Apply dimensionality reduction (e.g., PCA) or manual feature selection.

**Categorical Features**  
This implementation expects only numeric input and does not natively support categorical data.  
**Workaround:** Preprocess data using one-hot encoding or label encoding before training.


