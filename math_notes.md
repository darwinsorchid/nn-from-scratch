## Formulas

### ReLU (forward propagation)

$$
f(x) = max(1,x)
$$

### Sigmoid (forward propagation)

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

### Forward Pass

#### 1. Calculate Weighted Sum $Z$ for each layer

$$
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}
$$

#### 2. Apply activation function to get output $A$

$$
A^{[l]} = a(Z^{[l]})
$$

### Binary Cross-Entropy (BCE) Loss Formula

$$
\mathrm{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y\log\hat{y} + (1 - y)\log(1- \hat{y})]
$$

### Backward Propagation of Errors (Backpropagation) vs Gradient Descent

After computing predictions and prediction error (loss function) during the forward pass, backpropagation uses derivatives to calculate how much each parameter (weights and biases) contributed to the error and adjusts them using gradient descent.

Backpropagation uses the _chain rule_ from calculus to "propagate" derivatives backward through the layers.
It computes the gradient of a loss function w.r.t. the weights of the network for a single input-output example.

1. Computes gradient of loss funcion w.r.t. each weight using the chain rule making it possible to update weights efficiently
2. Scales well to networks with multiple layers and complex architectures
3. Enables automization of learning process - the model adjusts itself to optimize performance

After computing the _delta terms_, gradient descent is used to update the weights, by taking small steps in the opposite direction of the gradient (i.e., in the direction of the steepest descent) in order to get closer to the minimum error.
