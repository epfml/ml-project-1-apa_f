import numpy as np

def calculate_loss(target_values, features_matrix, coefficients):
    """
    Calculates the quadratic loss based on the given target values, feature observations, and weight coefficients.

    Args:
        target_values (np.array): n-dimensional array of dependent variables.
        features_matrix (np.array): (n x d)-dimensional matrix of feature observations.
        coefficients (np.array): d-dimensional array of weight coefficients.

    Returns:
        float: The computed quadratic loss.
    """
    
    residuals = target_values - features_matrix.dot(coefficients)
    quadratic_loss = (1 / (2 * len(residuals))) * np.linalg.norm(residuals)**2

    return quadratic_loss

def calculate_gradient(target_values, feature_matrix, coefficients):
    """
    Calculates the gradient for the Gradient Descent method based on given target values, feature observations, and weight coefficients.

    Args:
        target_values (np.array): n-dimensional array of dependent variables.
        feature_matrix (np.array): (n x d)-dimensional matrix of feature observations.
        coefficients (np.array): d-dimensional array of weight coefficients.

    Returns:
        np.array: The computed gradient.
    """

    sample_size = len(target_values)
    
    residuals = target_values - feature_matrix.dot(coefficients)
    
    gradient = -(1 / sample_size) * feature_matrix.T.dot(residuals)

    return gradient

def gradient_descent_mse(targets, features, initial_coeffs, iterations, learning_rate):
    """
    Performs Gradient Descent optimization to minimize the mean squared error.

    Args:
        targets (np.array): n-dimensional array of dependent variables.
        features (np.array): (n x d)-dimensional matrix of feature observations.
        initial_coeffs (np.array): Initial guess for the optimal coefficients, d-dimensional.
        iterations (int): Maximum number of iterations for optimization.
        learning_rate (float): Step size for updating the coefficients.

    Returns:
        np.array: Optimal coefficients after the specified number of iterations.
        float: Mean squared error at the final iteration.
    """

    w = initial_coeffs
    mse = calculate_loss(targets, features, w)

    for iter_count in range(iterations):
        
        gradient = calculate_gradient(targets, features, w)
        
        w -= learning_rate * gradient

        mse = calculate_loss(targets, features, w)

        if iter_count % 100 == 0:
            print(f"GD iteration {iter_count}/{iterations-1}: MSE={mse}")

    return w, mse


#############------------------------------#############

### implementation of mean_squared_error_sgd
def stochastic_gradient(targets, features, coeffs, mini_batch_size):
    """
    Computes the gradient using a random subset (mini-batch) of the data for Stochastic Gradient Descent.

    Args:
        targets (np.array): n-dimensional array of dependent variables.
        features (np.array): (n x d)-dimensional matrix of feature observations.
        coeffs (np.array): d-dimensional array of coefficients.
        mini_batch_size (int): Size of the random subset of data to compute the gradient.

    Returns:
        np.array: Gradient computed using the mini-batch for the specified coefficients.
    """

    sample_indices = np.random.randint(0, len(targets), size=mini_batch_size)
    
    gradient = calculate_gradient(targets[sample_indices], features[sample_indices, :], coeffs)

    return gradient

def sgd_mean_squared_error(targets, features, start_coeffs, num_iterations, learning_rate, mini_batch_size=1):
    """
    Performs Stochastic Gradient Descent to minimize the mean squared error.

    Args:
        targets (np.array): n-dimensional array of dependent variables.
        features (np.array): (n x d)-dimensional matrix of feature observations.
        start_coeffs (np.array): d-dimensional array representing the initial guess of the optimal coefficients.
        num_iterations (int): Maximum number of iterations for the SGD loop.
        learning_rate (float): Learning rate of the SGD algorithm.
        mini_batch_size (int, optional): Size of the random subset of data used to compute the gradient. Defaults to 1.

    Returns:
        tuple: Optimized coefficients and the loss at the final iteration.
    """

    w = start_coeffs
    mse_loss = calculate_loss(targets, features, w)

    iteration = 0

    while iteration < num_iterations:
        
        gradient = stochastic_gradient(targets, features, w, mini_batch_size)
        
        w -= learning_rate * gradient

        mse_loss = calculate_loss(targets, features, w)

        if iteration % 500 == 0:
            print(f"SGD Iteration {iteration}/{num_iterations - 1}: Loss = {mse_loss}")

        iteration += 1

    return w, mse_loss


#############------------------------------#############

def solve_least_squares(targets, features):
    """
    Calculates the optimal weights using the Least Squares method.

    Args:
        targets (np.array): n-dimensional array of dependent variables.
        features (np.array): (n x d)-dimensional matrix of feature observations.

    Returns:
        tuple: Optimal weights (W) and the associated quadratic loss.
    """

    w = np.linalg.solve(features.T @ features, features.T @ targets)
    
    mse_loss = calculate_loss(targets, features, w)

    return w, mse_loss


#############------------------------------#############

### implementation of ridge regression

def ridge_regression(targets, features, regularization):
    """
    Calculates the optimal weights using the Ridge Regression method.

    Args:
        targets (np.array): n-dimensional array of dependent variables.
        features (np.array): (n x d)-dimensional matrix of feature observations.
        regularization (float): Regularization parameter for Ridge Regression.

    Returns:
        tuple: Optimal weights (W) and the associated quadratic loss.
    """

    reg_param = regularization * 2 * len(targets)
    identity_mat = np.eye(features.shape[1])

    w = np.linalg.solve(features.T @ features + reg_param * identity_mat, features.T @ targets)
    
    mse_loss = calculate_loss(targets, features, w)

    return w, mse_loss


#############------------------------------#############

### implementation of logistic regression

def sigmoid_activation(input_values):
    """
    Computes the sigmoid activation for the given input values.

    Args:
        input_values (np.array): n-dimensional array of input values.

    Returns:
        np.array: n-dimensional array of sigmoid values.
    """
    
    return 1 / (1 + np.exp(-input_values))

def logistic_loss(y, features, weights):
    """
    Computes the logistic regression loss.

    Args:
        y (np.array): n-dimensional array of dependent variables.
        features (np.array): (n x d)-dimensional matrix of feature observations.
        weights (np.array): d-dimensional array of model weights.

    Returns:
        float: Logistic regression loss computed on y, features, and weights.
    """
    
    num_samples = y.shape[0]
    predictions = sigmoid_activation(features @ weights)

    loss_value = (-1 / num_samples) * (y.T @ np.log(predictions) + (1 - y).T @ np.log(1 - predictions))

    return loss_value


def logistic_gradient(y, features, weights):
    """
    Computes the gradient for logistic regression.

    Args:
        y (np.array): n-dimensional array of dependent variables.
        features (np.array): (n x d)-dimensional matrix of feature observations.
        weights (np.array): d-dimensional array of model weights.

    Returns:
        np.array: Gradient of the logistic regression method.
    """
    
    num_samples = y.shape[0]
    predictions = sigmoid_activation(features @ weights)

    gradient = (1 / num_samples) * features.T @ (predictions - y)

    return gradient


def perform_logistic_regression(y, features, starting_weights, max_iterations, learning_rate):
    """
    Performs logistic regression using gradient descent.

    Args:
        y (np.array): n-dimensional array of dependent variables.
        features (np.array): (n x d)-dimensional matrix of feature observations.
        starting_weights (np.array): Initial guess for the weights of shape (d,).
        max_iterations (int): Maximum number of iterations for gradient descent.
        learning_rate (float): The learning rate for gradient descent.

    Returns:
        tuple: Optimal weights and the loss of the logistic regression at the last iteration.
    """
    
    w = starting_weights

    for iteration in range(max_iterations):
        gradient = logistic_gradient(y, features, w)
        w -= learning_rate * gradient

        if iteration % 100 == 0:
            loss = logistic_loss(y, features, w)
            print(f"GD iteration {iteration}/{max_iterations - 1}: loss={loss}")

    final_loss = logistic_loss(y, features, w)

    return w, final_loss


#############------------------------------##############

### implementation of regularized logistic regression

def regularized_logistic_loss(y, tx, w, lambda_):
    """
    Calculate the loss for regularized logistic regression.

    Parameters:
    - y (np.array): Vector of observed outcomes.
    - tx (np.array): Matrix of input features.
    - w (np.array): Vector of model weights.
    - lambda_ (float): Regularization strength.

    Returns:
    - float: The computed loss with regularization.
    """

    base_loss = logistic_loss(y, tx, w)

    # L2 regularization component
    regularization = lambda_ * np.sum(w**2)

    total_loss = base_loss + regularization

    return total_loss


def regularized_logistic_gradient(y, tx, w, lambda_):
    """
    Calculate the gradient for regularized logistic regression.

    Parameters:
    - y (np.array): Vector of observed outcomes.
    - tx (np.array): Matrix of input features.
    - w (np.array): Vector of model weights.
    - lambda_ (float): Regularization strength.

    Returns:
    - np.array: The gradient of the loss with respect to the weights, including the regularization term.
    """

    base_gradient = logistic_gradient(y, tx, w)

    # L2 regularization gradient
    regularization_gradient = 2 * lambda_ * w

    total_gradient = base_gradient + regularization_gradient

    return total_gradient


def regularized_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform logistic regression with L2 regularization.

    Parameters:
    - y (np.array): Vector of observed outcomes.
    - tx (np.array): Matrix of input features.
    - lambda_ (float): Regularization strength.
    - initial_w (np.array): Initial weights for the model.
    - max_iters (int): Maximum number of iterations for the gradient descent.
    - gamma (float): Learning rate.

    Returns:
    - (np.array, float): Tuple containing the optimal weights and the final loss value.
    """

    w = initial_w

    for iter in range(max_iters):

        gradient = regularized_logistic_gradient(y, tx, w, lambda_)

        # Update weights using the gradient
        w -= gamma * gradient

        # Compute regularized loss
        loss = regularized_logistic_loss(y, tx, w, lambda_)

        # Print progress every 100 iterations
        if iter % 100 == 0:
            print(f"Gradient Descent iteration {iter}/{max_iters - 1}: Loss={loss}")

    return w, loss

