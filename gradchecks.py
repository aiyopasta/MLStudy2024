# Observations: For simple activation functions (especially bounded ones), if the input is
#               too big or too large the gradient doesn't check.
import numpy as np

from computation_graph import *

n_hidden = 3
n_outputs = 3

# Training data (raw, not normalized)
x_train = list(np.linspace(-50.0, 40.0, 100))
# y_train = [-0.5 * x - 100 + np.random.randint(-100, 100) for x in x_train]  # for regression problems
y_train = np.random.randint(0, n_outputs, np.shape(x_train)[0])

X = TensorNode(learnable=False, shape=(len(x_train), 2), name='X_data')  # one extra feature for bias term
y = TensorNode(learnable=False, shape=(len(y_train), 1), name='y_labels')
W = TensorNode(learnable=True, shape=(2, n_hidden), name='Weights-Layer-1')  # 1 weight + 1 bias, 1 hidden layer with 2 neurons
XW = MultiplicationNode(X, W)
Z = SimpleActivationNode(XW, kind='softplus')
W2 = TensorNode(learnable=True, shape=(n_hidden, n_outputs), name='Weights-Layer-2')  # 1 weight + 1 bias
ZW = MultiplicationNode(Z, W2, 'poopy')
# Loss = SquaredLossNode(mu=ZW, y=y)
soft = SoftmaxNode(ZW)
Loss = CrossEntropyLossNode(soft, y)
# List of all nodes, for convenience
node_list = [X, y, W, XW, soft, Loss]

# Initialize weights & biases randomly
W.set_input(np.random.rand(*W.shape))
W2.set_input(np.random.rand(*W2.shape))

# Preprocess input
x_raw = np.array(x_train)
x_input = np.reshape(x_raw, (len(x_raw), 1))
x_input = np.hstack((x_input, 1 * np.ones((len(x_raw), 1))))  # bias trick
y_input = np.reshape(y_train, (len(y_train), 1))

# Set the inputs
X.set_input(x_input)
y.set_input(y_input)

# The gradient check
h = 1e-5
for param in [W, W2]:
    Loss.reset()
    Loss.fire()
    Loss.backfire()
    old_param = copy.copy(param)
    for flat_idx in [0, 1, 2]:
        old, actual_grad = old_param.get_values_from_flat(flat_idx)
        print('Checking gradient for', param.name, flat_idx, 'parameter.')
        Loss.reset()
        param.set_value_from_flat(old + h, flat_idx)
        Jplus = Loss.fire()
        Loss.reset()
        param.set_value_from_flat(old - h, flat_idx)
        Jminus = Loss.fire()
        Loss.reset()
        numerical_grad = ((Jplus - Jminus) / (2 * h))[0]
        param.set_value_from_flat(old, flat_idx)
        # print('Jplus:', Jplus, 'Jminus:', Jminus)
        print('Actual:', actual_grad)
        print('Numerical:', numerical_grad)  # TODO: Implement a legit comparison, e.g. relative error.
        rel_error = abs(actual_grad - numerical_grad) / max(abs(actual_grad), abs(numerical_grad))
        print('Relative error between the two:', rel_error)
        print()