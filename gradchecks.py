# Observations: For simple activation functions (especially bounded ones), if the input is
#               too big or too large the gradient doesn't check.
import numpy as np

from computation_graph import *
np.random.seed(42)

n_hidden = 10
n_outputs = 1

# Training data (raw, not normalized)
# x_train = list(np.linspace(-100.0, 100.0, 100))  # TODO: change back to 100
x_train = np.random.rand(3, 5)
y_train = [-0.5 * x - 100 + np.random.randint(-100, 100) for x in range(x_train.shape[0])]  # for regression problems
# y_train = np.random.randint(0, n_outputs, np.shape(x_train)[0])

# X = TensorNode(learnable=False, shape=(len(x_train), 2), name='X_data')  # one extra feature for bias term
# y = TensorNode(learnable=False, shape=(len(y_train), 1), name='y_labels')
# W = TensorNode(learnable=True, shape=(2, n_hidden), name='Weights-Layer-1')  # 1 weight + 1 bias, 1 hidden layer with 2 neurons
# XW = MultiplicationNode(X, W)
# Z = SimpleActivationNode(XW, kind='sigmoid')
# W2 = TensorNode(learnable=True, shape=(n_hidden, n_outputs), name='Weights-Layer-2')  # 1 weight + 1 bias
# ZW = MultiplicationNode(Z, W2, 'poopy')
# # Loss = SquaredLossNode(mu=ZW, y=y)
# soft = SoftmaxNode(ZW)
# Loss = CrossEntropyLossNode(soft, y)
# # List of all nodes, for convenience
# node_list = [X, y, W, XW, soft, Loss]

# X = TensorNode(learnable=True, shape=(len(x_train), 2), name='X_data')  # one extra feature for bias term
# X_normed = LayerNormNode(X)
# y = TensorNode(learnable=False, shape=(len(y_train), 1), name='y_labels')
# W = TensorNode(learnable=True, shape=(2, n_hidden), name='Weights-Layer-1')  # 1 weight + 1 bias, 1 hidden layer with 2 neurons
# XW = MultiplicationNode(X_normed, W)
# Z = SimpleActivationNode(XW, kind='sigmoid')
# W2 = TensorNode(learnable=True, shape=(n_hidden, n_outputs), name='Weights-Layer-2')  # 1 weight + 1 bias
# ZW = MultiplicationNode(Z, W2, 'poopy')
# # Loss = SquaredLossNode(mu=ZW, y=y)
# soft = SoftmaxNode(ZW)
# Loss = CrossEntropyLossNode(soft, y)
# # List of all nodes, for convenience
# node_list = [X, X_normed, y, W, XW, soft, Loss]

X = TensorNode(learnable=True, shape=(len(x_train), 5), name='X_data')  # 5 total features
X_normed = LayerNormNode(X)
y = TensorNode(learnable=False, shape=(len(y_train), 1), name='y_labels')
W = TensorNode(learnable=True, shape=(5, n_outputs), name='Weights-Layer-1')  # 5 weights, 1 output
XW = MultiplicationNode(X_normed, W)
Loss = SquaredLossNode(XW, y)
# List of all nodes, for convenience
node_list = [X, X_normed, y, W, XW, Loss]

# Initialize weights & biases randomly
W.set_input(np.random.rand(*W.shape))
# W2.set_input(np.random.rand(*W2.shape))

# Preprocess input
# x_raw = np.array(x_train)
# x_input = np.reshape(x_raw, (len(x_raw), 1))
# x_input = np.hstack((x_input, 40.0 * np.ones((len(x_raw), 1))))  # bias trick TODO change back to 1
y_input = np.reshape(y_train, (len(y_train), 1))

# Set the inputs
X.set_input(x_train)
y.set_input(y_input)

# The gradient check
h = 1e-5
for param in [X]:
    Loss.reset()
    Loss.fire()
    # print(X)
    # print(X_normed)
    # print(W)
    # print(XW.value)
    Loss.backfire()
    old_param = copy.copy(param)
    for flat_idx in [0, 1, 2, 3, 4, 5]:
        old, actual_grad = old_param.get_values_from_flat(flat_idx)
        print('Checking gradient for', param.name, flat_idx, 'parameter.')
        Loss.reset()
        param.set_value_from_flat(old + h, flat_idx)
        Jplus = Loss.fire()
        # print(X)
        # print(X_normed)
        # print(W)
        # print(XW.value)
        Loss.reset()
        param.set_value_from_flat(old - h, flat_idx)
        Jminus = Loss.fire()
        Loss.reset()
        numerical_grad = ((Jplus - Jminus) / (2 * h))[0]
        param.set_value_from_flat(old, flat_idx)
        print('Jplus:', Jplus, 'Jminus:', Jminus)
        print('Actual:', actual_grad)
        print('Numerical:', numerical_grad)  # TODO: Implement a legit comparison, e.g. relative error.
        rel_error = abs(actual_grad - numerical_grad) / max(abs(actual_grad), abs(numerical_grad))
        print('Relative error between the two:', rel_error)
        print()