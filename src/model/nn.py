import numpy as np
import logging

# ------------------------------------------ Initialization --------------------------------------------------
# Configure logging and initialize logger
logging.basicConfig(
    filename="logs/init.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)



# Initialize architecture
nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"}
]


def init_layers(nn_architecture, seed=42):
    '''Initialize layers and parameter values per layer: 
        • weight matrix (output_dim, input_dim)
        • bias vector   (output_dim, 1)
    Returns params dict and logs initial parameters to init.log file.
    
    :param nn_architecture: Network architecture as a list of dictionaries for each layer of the net.
    :param seed: Set seed 
    '''

    # Set seed and initialize params dict
    np.random.seed(seed)
    params = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        input_dim = layer['input_dim']
        output_dim = layer['output_dim']

        # Populate params with weight matrix of (output, input) dimensions and bias vector of (output, 1) dimensions
        params["W" + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
        params["b" + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1


    # More readable output
    np.set_printoptions(
        precision = 4,
        suppress = True,
        linewidth = 120
    )
    # Log initial parameters to file
    logger.debug("Parameter Values:\n %s", params)

    return params


# -------------------------------- Activations Functions ---------------------------------

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu_backprop():
    pass

def sigmoid_backprop():
    pass


def single_forward_propagation():
    pass

def full_forward_propagation():
    pass


def loss_function():
    pass


def accuracy():
    pass




if __name__ == "__main__":

    params = init_layers(nn_architecture)