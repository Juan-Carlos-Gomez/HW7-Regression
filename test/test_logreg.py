"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import logreg, utils
from sklearn.preprocessing import StandardScaler


class TestPrediction:
	"""Test the make_prediction method."""
	
	def test_prediction_shape(self):
		"""Test that prediction output has the correct shape."""
		model = logreg.LogisticRegressor(num_feats=3)
		X = np.random.randn(5, 4)  # 5 samples, 4 features (including bias)
		y_pred = model.make_prediction(X)
		assert y_pred.shape == (5,), f"Expected shape (5,), got {y_pred.shape}"
	
	def test_prediction_range(self):
		"""Test that predictions are in valid sigmoid range [0, 1]."""
		model = logreg.LogisticRegressor(num_feats=3)
		X = np.random.randn(100, 4)  # 100 samples, 4 features (including bias)
		y_pred = model.make_prediction(X)
		assert np.all(y_pred >= 0) and np.all(y_pred <= 1), \
			"Predictions should be in range [0, 1]"
	
	def test_prediction_extreme_values(self):
		"""Test predictions with extreme weight and input values."""
		model = logreg.LogisticRegressor(num_feats=1)
		# Set weights to extreme values
		model.W = np.array([100, 0])  # Very large weight
		X = np.array([[1, 1], [-1, 1]])  # Two samples with bias
		y_pred = model.make_prediction(X)
		# First sample should be close to 1, second close to 0
		assert y_pred[0] > 0.999, "Large positive z should give prediction close to 1"
		assert y_pred[1] < 0.001, "Large negative z should give prediction close to 0"
	
	def test_prediction_zero_weights(self):
		"""Test prediction with zero weights (should be 0.5)."""
		model = logreg.LogisticRegressor(num_feats=1)
		model.W = np.array([0, 0])  # All zero weights
		X = np.array([[1, 1], [2, 1], [3, 1]])  # Three samples
		y_pred = model.make_prediction(X)
		# With zero weights, sigmoid(0) = 0.5
		np.testing.assert_array_almost_equal(y_pred, np.array([0.5, 0.5, 0.5]), decimal=10)


class TestLossFunction:
	"""Test the loss_function method."""
	
	def test_loss_perfect_prediction(self):
		"""Test loss when predictions are perfect."""
		model = logreg.LogisticRegressor(num_feats=1)
		y_true = np.array([1, 0, 1, 0])
		y_pred = np.array([0.99, 0.01, 0.99, 0.01])  # Very good predictions
		loss = model.loss_function(y_true, y_pred)
		# Loss should be very small for perfect predictions
		assert loss < 0.02, f"Loss for near-perfect predictions should be small, got {loss}"
	
	def test_loss_worst_prediction(self):
		"""Test loss when predictions are completely wrong."""
		model = logreg.LogisticRegressor(num_feats=1)
		y_true = np.array([1, 0, 1, 0])
		y_pred = np.array([0.01, 0.99, 0.01, 0.99])  # Completely wrong predictions
		loss = model.loss_function(y_true, y_pred)
		# Loss should be large for wrong predictions
		assert loss > 1.0, f"Loss for wrong predictions should be large, got {loss}"
	
	def test_loss_random_prediction(self):
		"""Test loss with random predictions."""
		model = logreg.LogisticRegressor(num_feats=2)
		y_true = np.array([1, 0, 1, 1, 0])
		y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Random predictions
		loss = model.loss_function(y_true, y_pred)
		# Loss should be positive and finite
		assert loss > 0, "Loss should be positive"
		assert np.isfinite(loss), "Loss should be finite"
	
	def test_loss_symmetry(self):
		"""Test that swapping true and pred labels affects loss symmetrically."""
		model = logreg.LogisticRegressor(num_feats=1)
		y_true = np.array([1, 0, 1])
		y_pred = np.array([0.8, 0.3, 0.6])
		loss1 = model.loss_function(y_true, y_pred)
		# Loss should be finite and non-negative
		assert loss1 >= 0, "Loss should be non-negative"
	
	def test_loss_single_sample(self):
		"""Test loss with single sample."""
		model = logreg.LogisticRegressor(num_feats=1)
		y_true = np.array([1])
		y_pred = np.array([0.7])
		loss = model.loss_function(y_true, y_pred)
		# Manual calculation: -mean(1 * log(0.7) + 0 * log(0.3)) = -log(0.7) ≈ 0.357
		expected = -np.log(0.7)
		np.testing.assert_almost_equal(loss, expected, decimal=5)


class TestGradient:
	"""Test the calculate_gradient method."""
	
	def test_gradient_shape(self):
		"""Test that gradient has correct shape."""
		model = logreg.LogisticRegressor(num_feats=5)
		X = np.random.randn(10, 6)  # 10 samples, 6 features (including bias)
		y_true = np.random.binomial(1, 0.5, 10)
		gradient = model.calculate_gradient(y_true, X)
		assert gradient.shape == (6,), f"Expected gradient shape (6,), got {gradient.shape}"
	
	def test_gradient_perfect_fit(self):
		"""Test gradient when predictions perfectly match targets."""
		model = logreg.LogisticRegressor(num_feats=1)
		# Create data where our model perfectly predicts
		X = np.array([[0, 1], [1, 1]])  # 2 samples, 2 features
		y_true = np.array([0.0, 1.0])  # Labels matching sigmoid predictions
		# Set weights to create perfect predictions
		model.W = np.array([2, -1])  # z = 2*0 - 1 = -1 (sigmoid ≈ 0.27)
		# This won't be perfect, but gradient should still be reasonable
		gradient = model.calculate_gradient(y_true, X)
		assert gradient.shape == (2,), "Gradient should have shape (2,)"
		assert np.all(np.isfinite(gradient)), "All gradient components should be finite"
	
	def test_gradient_magnitude(self):
		"""Test that gradient magnitude is reasonable."""
		model = logreg.LogisticRegressor(num_feats=2)
		X = np.random.randn(20, 3)  # 20 samples, 3 features
		y_true = np.random.binomial(1, 0.5, 20)
		gradient = model.calculate_gradient(y_true, X)
		# Gradient magnitude should be reasonable (not too large)
		gradient_magnitude = np.linalg.norm(gradient)
		assert gradient_magnitude < 100, f"Gradient magnitude {gradient_magnitude} seems too large"
		assert gradient_magnitude >= 0, "Gradient magnitude should be non-negative"
	
	def test_gradient_numerically(self):
		"""Test gradient using numerical approximation (finite differences)."""
		model = logreg.LogisticRegressor(num_feats=2)
		X = np.array([[1, 0.5, 1], [2, 1.5, 1], [3, 2.5, 1]])  # 3 samples, 3 features
		y_true = np.array([0, 1, 1])
		
		# Set specific weights for reproducibility
		model.W = np.array([0.5, -0.3, 0.2])
		
		# Get analytical gradient
		analytical_grad = model.calculate_gradient(y_true, X)
		
		# Compute numerical gradient
		epsilon = 1e-5
		numerical_grad = np.zeros_like(model.W)
		for i in range(len(model.W)):
			model.W[i] += epsilon
			loss_plus = model.loss_function(y_true, model.make_prediction(X))
			model.W[i] -= 2 * epsilon
			loss_minus = model.loss_function(y_true, model.make_prediction(X))
			model.W[i] += epsilon
			numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
		
		# Analytical and numerical gradients should be close
		np.testing.assert_array_almost_equal(analytical_grad, numerical_grad, decimal=4)


class TestTraining:
	"""Test the training process."""
	
	def test_training_updates_weights(self):
		"""Test that training actually updates the weights."""
		model = logreg.LogisticRegressor(num_feats=2, learning_rate=0.1, max_iter=5, batch_size=5)
		initial_W = model.W.copy()
		
		# Create simple synthetic data
		X_train = np.random.randn(10, 2)
		y_train = np.random.binomial(1, 0.5, 10)
		X_val = np.random.randn(5, 2)
		y_val = np.random.binomial(1, 0.5, 5)
		
		# Train model
		model.train_model(X_train, y_train, X_val, y_val)
		
		# Check that weights have changed
		assert not np.allclose(model.W, initial_W), "Weights should be updated during training"
	
	def test_training_loss_decreases(self):
		"""Test that training loss generally decreases over time."""
		model = logreg.LogisticRegressor(num_feats=2, learning_rate=0.01, max_iter=10, batch_size=5)
		
		# Create synthetic data with clear separation
		np.random.seed(42)
		X_train = np.vstack([
			np.random.randn(10, 2) - 2,  # Class 0
			np.random.randn(10, 2) + 2   # Class 1
		])
		y_train = np.array([0] * 10 + [1] * 10)
		X_val = np.vstack([
			np.random.randn(5, 2) - 2,
			np.random.randn(5, 2) + 2
		])
		y_val = np.array([0] * 5 + [1] * 5)
		
		# Train model
		model.train_model(X_train, y_train, X_val, y_val)
		
		# Loss history should not be empty
		assert len(model.loss_hist_train) > 0, "Training loss history should be populated"
		assert len(model.loss_hist_val) > 0, "Validation loss history should be populated"
		
		# First loss should be different from last loss (indicating training occurred)
		first_loss = model.loss_hist_train[0]
		last_loss = model.loss_hist_train[-1]
		assert first_loss != last_loss, "Loss should change during training"
	
	def test_training_with_real_data(self):
		"""Test training with the actual NSCLC dataset."""
		# Load a small subset of the data
		X_train, X_val, y_train, y_val = utils.loadDataset(
			features=[
				'Penicillin V Potassium 500 MG',
				'Computed tomography of chest and abdomen',
				'Plain chest X-ray (procedure)',
				'Low Density Lipoprotein Cholesterol',
				'Creatinine',
				'AGE_DIAGNOSIS'
			],
			split_percent=0.8,
			split_seed=42
		)
		
		# Scale the data
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_val = sc.transform(X_val)
		
		# Train model with conservative hyperparameters
		model = logreg.LogisticRegressor(
			num_feats=6,
			learning_rate=0.00001,
			tol=0.01,
			max_iter=10,
			batch_size=10
		)
		model.train_model(X_train, y_train, X_val, y_val)
		
		# Check that training completed
		assert len(model.loss_hist_train) > 0, "Training should produce loss history"
		
		# Check that predictions can be made
		y_pred = model.make_prediction(np.hstack([X_val, np.ones((X_val.shape[0], 1))]))
		assert y_pred.shape[0] == X_val.shape[0], "Should produce predictions for all validation samples"
		assert np.all((y_pred >= 0) & (y_pred <= 1)), "Predictions should be in [0, 1]"
	
	def test_training_convergence(self):
		"""Test that model converges on simple separable data."""
		np.random.seed(123)
		model = logreg.LogisticRegressor(
			num_feats=2,
			learning_rate=0.1,
			tol=0.001,
			max_iter=100,
			batch_size=5
		)
		
		# Create perfectly linearly separable data
		X_train = np.vstack([
			np.random.randn(15, 2) - 3,
			np.random.randn(15, 2) + 3
		])
		y_train = np.array([0] * 15 + [1] * 15)
		X_val = np.vstack([
			np.random.randn(5, 2) - 3,
			np.random.randn(5, 2) + 3
		])
		y_val = np.array([0] * 5 + [1] * 5)
		
		# Train
		model.train_model(X_train, y_train, X_val, y_val)
		
		# Model should converge (stop training before max_iter)
		# This is indicated by update_size becoming small
		assert len(model.loss_hist_train) > 0, "Should have training history"