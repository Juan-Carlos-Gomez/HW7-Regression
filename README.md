# HW 7: logistic regression

In this assignment, we implemented a classifier using logistic regression, optimized with gradient descent.


* **Logistic Regression** = Linear model + Sigmoid function + Cross-entropy loss  

* **Sigmoid** transforms any number into a probability  

* **Loss** tells us how wrong we are

* **Gradient** tells us how to improve

* **Training** repeatedly calculates loss and gradient, then updates weights  

* **Batch gradient descent** is stochastic and helps avoid local minima  


## Logistic regression


A machine learning algorithm used for binary classification (predicting one of two outcomes: Yes/No, 1/0, True/False).

1. A linear model (weighted sum of features)
2. A sigmoid function (converts any number to a probability between 0 and 1)
3. Gradient descent (learns the best weights by minimizing prediction error)


Formula: prediction = sigmoid(features × weights)


## Dataset 

The dataset (`data/nsclc.csv`) contains medical records from lung cancer patients. Your goal is to **predict whether a patient has Non-Small Cell Lung Cancer (NSCLC) or Small Cell Lung Cancer** based on their medical history.

#### Class Labels
- **1** = Non-Small Cell Lung Cancer (NSCLC)
- **0** = Small Cell Lung Cancer

#### Pre-selected Features (6 total)
1. **Penicillin V Potassium 500 MG** - Antibiotic use (binary: 0 or 1)
2. **Computed tomography of chest and abdomen** - CT scan performed (binary: 0 or 1)
3. **Plain chest X-ray (procedure)** - X-ray performed (binary: 0 or 1)
4. **Low Density Lipoprotein Cholesterol** - LDL level (continuous)
5. **Creatinine** - Kidney function indicator (continuous)
6. **AGE_DIAGNOSIS** - Age at diagnosis (continuous)

> **Note**: These features are pre-selected for testing (see `main.py`). Explore other features in your unit tests! Complete feature list available in `logreg/utils.py`.

#### Data Splits
- **X_train, y_train**: 80% of data (~160 samples, 6 features)
- **X_val, y_val**: 20% of data (~40 samples, 6 features)

#### Preprocessing (StandardScaler)
Features are **standardized** before training:
- Mean = 0, Standard Deviation = 1
- Improves convergence speed
- Example: Age range 40-80 → normalized to approximately -2 to +2


### Why Logistic Regression?

**The Problem**: Linear regression predicts continuous values (like house prices), but classification requires probabilities between 0 and 1. You can't directly use linear regression for Yes/No decisions.

**The Solution**: **Logistic regression** combines a linear model with the **sigmoid function**, which "squeezes" any number into the range [0, 1]. This transforms your linear output into a valid probability.

**Why it works for this problem:**
- **Binary classification**: We have exactly two outcomes (NSCLC or Small Cell)
- **Probabilistic output**: We get confidence scores (e.g., 73% chance of NSCLC)
- **Efficient learning**: Gradient descent easily optimizes the weights
- **Interpretable results**: Weights show which medical features matter most

**Mathematical insight**: The sigmoid function creates an S-shaped curve that naturally models how probability increases with evidence. Strong evidence pushes predictions toward 0 or 1, while weak evidence stays near 0.5.


## Algorithm Implementation

The logistic regression classifier is implemented in `regression/logreg.py` with three core methods that work together:

#### 1. `make_prediction(X)` - Convert Features to Probabilities
```python
z = X @ W  # Linear combination of features and weights
y_pred = 1 / (1 + np.exp(-z))  # Apply sigmoid function
```
- **Input**: Feature matrix (n_samples × n_features)
- **Process**: Computes linear score, applies sigmoid to convert to probability [0, 1]
- **Output**: Predicted probability for each sample

#### 2. `loss_function(y_true, y_pred)` - Measure Prediction Error
```python
loss = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
```
- **Input**: True labels (0 or 1) and predicted probabilities
- **Process**: Calculates binary cross-entropy loss
- **Output**: Single scalar loss value (lower = better predictions)
- **Why this works**: Heavily penalizes confident wrong predictions, encouraging accurate probabilities

#### 3. `calculate_gradient(y_true, X)` - Determine How to Improve Weights
```python
y_pred = sigmoid(X @ W)
gradient = (1/m) * X.T @ (y_pred - y_true)
```
- **Input**: True labels and feature matrix
- **Process**: Computes prediction error, multiplies by features to get direction and magnitude
- **Output**: Gradient vector (one value per weight)
- **How it's used**: Weights updated via `W_new = W_old - learning_rate * gradient`

#### Training Loop
The `train_model()` method orchestrates the training process:
1. Add bias column to features
2. For each epoch:
   - Shuffle training data randomly
   - Split into batches
   - For each batch:
     - Make predictions
     - Calculate loss and gradient
     - Update weights using gradient descent
   - Evaluate on validation set
3. Stop when weights stop changing significantly or max iterations reached

This iterative approach allows the model to learn which medical features are most predictive of lung cancer type.


## Unit Tests

Comprehensive unit tests are provided in `test/test_logreg.py` to validate the implementation. Tests are organized into four categories:

#### TestPrediction (4 tests)
Validates that `make_prediction()` correctly converts features to probabilities:
- **test_prediction_shape**: Output dimensions match input samples
- **test_prediction_range**: All predictions fall in valid range [0, 1]
- **test_prediction_extreme_values**: Extreme weights produce correct probability behavior
- **test_prediction_zero_weights**: Zero weights produce neutral 0.5 predictions

#### TestLossFunction (5 tests)
Ensures `loss_function()` accurately measures prediction error:
- **test_loss_perfect_prediction**: Low loss for accurate predictions
- **test_loss_worst_prediction**: High loss for completely wrong predictions
- **test_loss_random_prediction**: Loss is positive and finite
- **test_loss_symmetry**: Loss function properties are correct
- **test_loss_single_sample**: Manual calculation verification

#### TestGradient (4 tests)
Verifies `calculate_gradient()` correctly computes weight updates:
- **test_gradient_shape**: Output dimensions match number of weights
- **test_gradient_perfect_fit**: Reasonable gradient values for well-fitting data
- **test_gradient_magnitude**: Gradient values are not unreasonably large
- **test_gradient_numerically**: **Gold standard** - Analytical gradient matches numerical approximation via finite differences

#### TestTraining (4 tests)
Validates the complete training workflow:
- **test_training_updates_weights**: Weights change during training
- **test_training_loss_decreases**: Loss history is populated and tracked
- **test_training_with_real_data**: Model trains successfully on the NSCLC dataset
- **test_training_convergence**: Model converges on linearly separable data

#### Running Tests
```bash
# Run all tests
pytest test/test_logreg.py -v

# Run specific test class
pytest test/test_logreg.py::TestPrediction -v

# Run specific test
pytest test/test_logreg.py::TestPrediction::test_prediction_range -v
```



## Tasks + Grading
* Algorithm Implementation (6):
  * Complete the `make_prediction` method (2)
  * Complete the `loss_function` method (2)
  * Complete the `calculate_gradient` method (2)
* Unit tests (3):
  * Unit test for `test_prediction`
  * Unit test for `test_loss_function`
  * Unit test for `test_gradient`
  * Unit test for `test_training`

* Code Readability (1)
* Extra credit (0.5)
  * Github actions and workflows (up to 0.5)

## Getting started

Fork this repository to your own GitHub account. Work on the codebase locally and commit changes to your forked repository. 

You will need following packages:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pytest](https://docs.pytest.org/en/7.2.x/)

## Additional notes

Try tuning the hyperparameters if you find that your model doesn't converge. Too high of a learning rate or too large of a batch size can sometimes cause the model to be unstable (e.g. loss function goes to infinity). If you're interested, scikit-learn also has some built-in [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) that you can use for testing.

We're applying a pretty simple model to a relatively complex problem here, so you should expect your classifier to perform decently but not amazingly. It's also possible for a given optimization run to get stuck in a local minima depending on the initialization. With that said, if your implementation is correct and you found reasonable hyperparameters, you should almost always at least do better than chance.
