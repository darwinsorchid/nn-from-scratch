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

### Error and Binary Cross-Entropy (BCE) Loss

$$
J(W,b) = \frac{1}{n} \sum_{i=1}^{n} L(\hat{y}, y)
$$

$$
L(\hat{y}, y) = -y\log\hat{y} + (1 - y)\log(1- \hat{y})
$$

Goal is to find the parameters (weights, biases) that minimize the error function J(W,b). [^2]

### Backward Propagation of Errors [^1] (Backpropagation) vs Gradient Descent

#### In Theory:

After computing predictions and prediction error (loss function) during the forward pass, backpropagation uses derivatives to calculate how much each parameter (weights and biases) contributed to the error and adjusts them using gradient descent.

Backpropagation uses the _chain rule_ from calculus to "propagate" derivatives backward through the layers.
It computes the gradient of a loss function w.r.t. the weights of the network for a single input-output example.

1. Computes gradient of loss funcion w.r.t. each weight using the chain rule making it possible to update weights efficiently
2. Scales well to networks with multiple layers and complex architectures
3. Enables automization of learning process - the model adjusts itself to optimize performance

After computing the _delta terms_, gradient descent is used to update the weights, by taking small steps in the opposite direction of the gradient (i.e., in the direction of the steepest descent) in order to get closer to the minimum error.

#### In Practice:

In order to use gradient descent, one needs to compute the partial derivatives of the error function w.r.t. each one of the weights and biases for each example. The extension to n examples is implemented by the sum of the derivatives for each example.

$$
\frac{\partial J}{\partial w_{ij}^{l}} = \frac{\partial L(\hat{y}, y)}{\partial w_{ij}^{l}}
$$

The loss function is a composition of functions:

$$
L(A^{[l]}(Z^{[l]}(W^{[l]}))))
$$

Backpropagation is implemented by applying the chain rule:

1. Partial derivative of L w.r.t. the weights:

$$
dW^{[l]} = \frac{\partial L}{\partial W^{[l]}_{ij}} = \frac{\partial L}{\partial A^{[l]}} \frac{\partial A^{[l]}}{\partial Z^{[l]}} \frac{\partial Z^{[l]}}{\partial W^{[l]}_{ij}}
$$

[^1]: [Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. "Learning representations by back-propagating errors." nature 323.6088 (1986): 533–536](https://doi.org/10.1038/323533a0)

[^2]: This error function is _non-convex_: the non-linearity of the activation functions stacked with multiple hidden layers destroys convexity. This implies that the error function may have multiple local minima, which the network could get stuck in instead of reaching the global minimum. This can be avoided by using momentum.
