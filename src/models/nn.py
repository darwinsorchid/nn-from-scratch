import numpy as np
import logging

# ------------------------------------------ Initialization --------------------------------------------------


def init_logger():
    # Configure logging and initialize logger
    logging.basicConfig(
        filename="logs/init.log",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger = logging.getLogger(__name__)
    return logger


# Initialize architecture
nn_architecture = [
    {"input_dim": 2, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 16, "activation": "relu"},
    {"input_dim": 16, "output_dim": 1, "activation": "sigmoid"},
]


def init_layers(nn_architecture, seed=42):
    """Initialize layers and parameter values per layer:
        • weight matrix (output_dim, input_dim)
        • bias vector   (output_dim, 1)
    Returns params dict and logs initial parameters to init.log file.

    :param nn_architecture: Network architecture as a list of dictionaries for each layer of the net.
    :param seed: Set seed
    """

    # Set seed and initialize params dict
    np.random.seed(seed)
    params = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        input_dim = layer["input_dim"]
        output_dim = layer["output_dim"]

        # Populate params with weight matrix of (output, input) dimensions and bias vector of (output, 1) dimensions
        params[f"W{layer_idx}"] = np.random.randn(output_dim, input_dim) * 0.1
        params[f"b{layer_idx}"] = np.random.randn(output_dim, 1) * 0.1

    # More readable output
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    logger = init_logger()
    # Log initial parameters to file
    logger.debug("Parameter Values:\n %s", params)

    return params


# ---------------------------- Activations Functions ---------------------------------


def relu(Z):
    return np.maximum(0, Z)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu_backprop(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backprop(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


# -------------------------------- Forward Pass --------------------------------------
def single_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    """
    Forward propagation for a single layer.

    :param A_prev: Activation vector of previous layer.
    :param W_curr: Weight matrix of current layer.
    :param b_curr: Bias vector of current layer.
    :param activation: Activation function of current layer.
    """
    # Calculate Z_curr
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation == "relu":
        act_func = relu
    elif activation == "sigmoid":
        act_func = sigmoid
    else:
        raise Exception("Non-valid activation function.")

    # Return calculated activation A and intermediate output matrix Z
    return act_func(Z_curr), Z_curr


def full_forward_propagation(X, params, nn_architecture):
    """
    Iterates through all network layers and does forward propagation.

    :param X: Input vector (activation of layer 0)
    :param nn_architecture: Network architecture
    """
    # Initialize temporary memory to store params and outputs - to be used later for backward prop.
    memory = {}

    # Activation of layer 0
    A_curr = X

    # Iterate through layers
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        # Transfer activation from previous iteration
        A_prev = A_curr

        # Get weight matrix, bias vector and activation function of current layer
        W_curr = params[f"W{layer_idx}"]
        b_curr = params[f"b{layer_idx}"]
        act_func = layer["activation"]

        # Do forward propagation for current layer
        A_curr, Z_curr = single_forward_propagation(A_prev, W_curr, b_curr, act_func)

        # Store intermediate values
        memory[f"A{idx}"] = A_prev
        memory[f"Z{layer_idx}"] = Z_curr

    # Return prediction vector and memory of intermediate prediction probabilities
    return A_curr, memory


# ---------------------------------- Loss & Accuracy ----------------------------------
def loss_function(y_hat, y):
    """
    Calculate loss function.
    Binary Cross-Entropy (BCE) loss for binary classification problems.

    :param y_hat: Vector of network prediction probabilities
    :param y: Vector of actual labels.
    """
    # Number of examples
    n = y_hat.shape[1]

    # Add clipping to avoid breaking gradient descent when np.log(0)
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)

    # Calculate loss according to BCE formula
    bce = -1 / n * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))

    return np.squeeze(bce)


def prob_to_label(probs):
    """
    Turns predicted probabilities to binary labels 0 / 1.

    :param probs: Vector containing predicted probabilities.
    """
    probs_ = np.copy(probs)

    # Threshold probabilities and turn to binary labels.
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0

    return probs_


def accuracy(y_hat, y):
    """
    Get model accuracy:
    1) Element-wise comparison of predicted vs true labels
    2) Get a True/False value for each column
    3) Get mean value = Accuracy

    :param y_hat: Predicted probability vector
    :param y: Real labels
    """
    # Turn predicted probabilities to binary labels
    y_hat_ = prob_to_label(y_hat)

    # Return accuracy
    return (y_hat_ == y).all(axis=0).mean()


# -------------------------------- Backpropagation -----------------------------------
def single_backprop(dA_curr, Z_curr, W_curr, b_curr, A_prev, activation="relu"):
    """
    Calculate partial derivatives w.r.t. weights and biases for current layer.

    :param dA_curr: Derivative of current layer activation
    :param Z_curr: Intermediate output of current layer
    :param W_curr: Weight matrix of current layer
    :param b_curr: Bias vector of current layer
    :param A_prev: Matrix of previous layer activation values
    :param activation: Activation function of current layer
    """

    # Get number of examples
    n = A_prev.shape[1]

    # Get activation function to apply derivative
    if activation == "relu":
        backwards_act = relu_backprop
    elif activation == "sigmoid":
        backwards_act = sigmoid_backprop
    else:
        raise Exception("Non-valid activation function.")

    # Calculate activation function derivative
    dZ_curr = backwards_act(dA_curr, Z_curr)

    # Calculate dW (derivative of matrix W)
    dW_curr = np.dot(dZ_curr, A_prev.T) / n

    # Calculate db (derivative of vector b)
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / n

    # Calculate dA_prev (derivative of matrix A of previous layer)
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backprop(y_hat, y, memory, params, nn_architecture):
    """
    Backward propagation of errors: iterates backwards through the layers
    and calculates partial derivatives.
    Returns dictionary of gradient values for weights and biases of each layer.

    :param y_hat: Prediction vector
    :param y: True labels
    :param memory: Dictionary with intermediate prediction vectors from each layer
    :param params: Weights and biases of each layer
    :param nn_architecture: Network architecture list of dicts
    """

    # Initialize dictionary of gradient values
    grads_values = {}

    # Number of examples
    n = y.shape[1]
    # Ensure same shape of prediction vector and labels vector
    y = y.reshape(y_hat.shape)

    # Derivative of binary cross-entropy loss function w.r.t. ŷ
    dA_prev = -(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1

        # extract activation func of current layer
        activation_func_curr = layer["activation"]

        dA_curr = dA_prev

        # Get params and derivatives
        A_prev = memory[f"A{layer_idx_prev}"]
        Z_curr = memory[f"Z{layer_idx_curr}"]

        W_curr = params[f"W{layer_idx_curr}"]
        b_curr = params[f"b{layer_idx_curr}"]

        # Do backprop for current layer
        dA_prev, dW_curr, db_curr = single_backprop(
            dA_curr, Z_curr, W_curr, b_curr, A_prev, activation_func_curr
        )

        # Save gradients
        grads_values[f"dW{layer_idx_curr}"] = dW_curr
        grads_values[f"db{layer_idx_curr}"] = db_curr

    return grads_values


def update_params(params, grads, nn_architecture, learning_rate):
    """
    Simple optimization algorithm; update network parameters using gradient optimization
    -> Try to bring target function to a minimum.

    :param params: Dictionary with parameter values (weights, biases) for each layer
    :param grads: Dictionary with cost function derivative values w.r.t. parameters (weights, biases) for each layer
    :param nn_architecture: List of dicts for network's architecture
    :param learning_rate: Defines how big of a parameter change to be applied
    """

    for idx, layer in enumerate(nn_architecture, 1):
        params[f"W{idx}"] -= learning_rate * grads[f"dW{idx}"]
        params[f"b{idx}"] -= learning_rate * grads[f"db{idx}"]

    return params


if __name__ == "__main__":
    params = init_layers(nn_architecture)
