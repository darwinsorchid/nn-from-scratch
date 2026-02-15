import numpy
from src.models import nn

def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    '''
    Standard training function for numpy network.
    
    :param X: Features matrix
    :param Y: Labels vector
    :param nn_architecture: List of dicts with network architecture and activation funcs
    :param epochs: Number of epochs (int)
    :param learning_rate: Learning rate value (float)
    :param verbose: Optional run info arg
    :param callback: Optional callback function arg
    '''
    # Initialize params
    params = nn.init_layers(nn_architecture)

    # Initialize lists for cost and accuracy values during training
    cost_history = []
    accuracy_history = []

    # Iterate over epochs
    for i in range(epochs):

        # Do forward pass
        Y_hat, cache = nn.full_forward_propagation(X, params, nn_architecture)

        # Calculate loss
        cost = nn.loss_function(Y_hat, Y)

        # Calculate accuracy
        accuracy = nn.accuracy(Y_hat, Y)

        # Save metrics to history
        cost_history.append(cost)
        accuracy_history.append(accuracy)

        # Do backprop
        grads_values = nn.full_backprop(Y_hat, Y, cache, params, nn_architecture)

        # Update params based on gradient values
        params = nn.update_params(params, grads_values, nn_architecture, learning_rate)

        
        # Progress every 50 epochs
        if i % 50 == 0:
            if verbose:
                print(f"Iteration: {i} - cost: {cost:.5f} - accuracy: {accuracy:.5f}")
            if callback is not None:
                callback(i, params)

    return params


