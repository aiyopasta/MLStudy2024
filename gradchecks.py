# Observations: For simple activation functions (especially bounded ones), if the input is
#               too big or too large the gradient doesn't check.
import numpy as np

from computation_graph import *
np.random.seed(42)

dataset = 2  # 0 = radial, 1 = affine, 2 = v-shaped
x_train, y_train = [], []
n_examples = 100
# Radial
if dataset == 0:
    max_radius = height / 3.
    cutoff = max_radius * 0.6
    for i in range(n_examples):
        r = max_radius * np.sqrt(np.random.random())
        theta = np.random.random() * 2.0 * np.pi
        x_train.append(r * np.array([np.cos(theta), np.sin(theta)]))
        y_train.append(0 if r < cutoff else 1)  # 0 = inside class, 1 = outside class
# Affine
elif dataset == 2:
    angle = np.radians(45)
    offset = np.array([100.0, 100.0])
    sidelen = height * 0.8
    for i in range(n_examples):
        pt = np.random.uniform(-sidelen / 2, +sidelen / 2, size=2)
        x_train.append(pt)
        nor = np.array([np.cos(angle), np.sin(angle)])
        label = int(np.round((np.sign(np.dot(nor, pt - offset)) / 2.0) + 0.5))
        y_train.append(label)
# V-shaped
elif dataset == 2:
    angle = np.radians(30)  # angle the normal vector of the decision boundary should make with horizontal
    sidelen = height * 0.8
    for i in range(n_examples):
        pt = np.random.uniform(-sidelen/2, +sidelen/2, size=2)
        x_train.append(pt)
        nor = np.array([np.cos(angle), np.sin(angle)])
        label1 = int(np.round((np.sign(np.dot(nor, pt)) / 2.0) + 0.5))
        nor2 = np.array([-np.sin(angle), np.cos(angle)])
        label2 = int(np.round((np.sign(np.dot(nor2, pt)) / 2.0) + 0.5))
        y_train.append(label1 & label2)
else:
    print('huh?')

n_hidden = 3  # number of neurons in single (for now) hidden layer.
n_outputs = 2  # 2 classes
# note: read "W_b" as "W and b", 'b' for bias vector. and 'XW_b" as 'XW+b", where b is the bias vector included in W.
X = BiasTrickNode(TensorNode(learnable=False, shape=(len(x_train), 2), name='X_data'))  # (x1, x2) + 1 bias
X_normed = LayerNormNode(X)
y = TensorNode(learnable=False, shape=(len(y_train), 1), name='y_labels')
W_b = TensorNode(learnable=True, shape=(3, n_hidden), name='WeightsBias-Layer-1')  # 3 weights per hidden neuron (for x1, x2, and bias)
XW_b = MultiplicationNode(X_normed, W_b)
Z = BiasTrickNode(SimpleActivationNode(XW_b, kind='relu'))
W2_b = TensorNode(learnable=True, shape=(n_hidden + 1, n_outputs), name='WeightsBias-Layer-2')  # num of hidden neurons + bias
ZW_b = MultiplicationNode(Z, W2_b)
Soft = SoftmaxNode(ZW_b)
Loss = CrossEntropyLossNode(Soft, y)
# List of all nodes, for convenience
node_list = [X, X_normed, y, W_b, XW_b, Z, W2_b, ZW_b, Soft, Loss]