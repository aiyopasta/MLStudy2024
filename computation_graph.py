# NOTE 1: Both forward & backwards passes are called recursively, starting with final loss node.
#         Each backwards pass onto a parent must include from which child it was sent, for summing.
#
# NOTE 2: The optimization step, e.g. x += alpha * dx for batch-GD, needs to be called for each node, non-recursively.
#         Each node will hold its own per-node optimization quantities, e.g. "velocity" for momentum.
#         It will not have its own alpha, mu, etc. Those need to be globally set outside, and passed in.
#
# NOTE 3: There are two separate procedures you can run with a computation graph: Gradient checking and training.
#         You decide how often you want to gradient check, or how to structure when it's called. But the
#         two algorithms are as follows.
#
# So the training loop will look like:
# (0) Fill input nodes (initialize weights, load mini-batch)
# (1) node.reset()     (mark each as "unvisited" again, set gradients to 0)
# (2) loss.fire()
# (3) loss.backfire()
# (4) for each LEARNABLE TensorNode:
#         node.update(alpha, mu)
#
# The gradcheck procedure will look like:
# (0) Load small batch (like 2-3) + initialize weights.
# (1) loss.reset()
# (2) loss.fire()
# (3) loss.backfire()
# (4) COPY each LEARNABLE TensorNode
# (5) for each LEARNABLE TensorNode:                   <-- TensorNodes are the ONLY kind of nodes that can be learnable
#         for flat_idx in [1, 2, 3]:                   <-- this loop is necessary as we ONLY want to perturb one param at a time
#             old, actual_grad = copy_of_this_node.get_values_from_flat(flat_idx)
#             loss.reset()
#             node.set_value_from_flat(old + epsilon, flat_idx)
#             Jplus = loss.fire()
#             loss.reset()
#             node.set_value_from_flat(old - epsilon, flat_idx)
#             Jminus = loss.fire()
#             loss.reset()
#             numerical_grad = (Jplus - Jminus) / (2 * epsilon)
#             node.set_value_from_flat(old, flat_idx)
#             compare(numerical_grad, actual_grad)
#
import copy
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

# Debugging prints
print_fired = False
reset_reminder = False
tensor_updates = False
flat_set_message = False


# A node representing a scalar, vector, matrix, or arbitrary dimension "tensor"
class TensorNode:
    def __init__(self, learnable, shape=(1,1), name='no_name'):
        ''' shape is a tuple (d1, d2, ..., dn) describing dimensions of input '''
        self.shape = shape
        self.value = np.zeros(shape)
        self.children = []  # add children from outside. it also has no parent; use hasattr() to see if it does
        self.gradient = np.zeros(shape)
        self.learnable = learnable

        # Additional params for certain optimization methids
        self.cache = 0.0  # for adagrad

        self.name = name

    def set_input(self, tensor:np.ndarray):
        ''' Use it to load in a new batch of training examples, or setting new parameter values. '''
        if not self.learnable and tensor_updates: print('Non-parameter tensor', self.name, 'was given a new value.')
        if self.learnable and tensor_updates: print('Parameter tensor', self.name, 'was updated.')
        self.value = tensor
        self.gradient = np.zeros_like(tensor)
        self.shape = tensor.shape

    def reset(self, hard=False):
        ''' hard reset sets all inputs (learnable or not) of the graph to 0s as well '''
        if hard: self.value = np.zeros(self.shape)
        self.gradient = np.zeros(self.shape)

    def __reset_shape__(self):
        pass  # base case of __reset_shape__() DFS

    def fire(self):
        if print_fired: print(self.name, 'was fired.')
        return self.value

    def backfire(self, from_child):
        assert from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')  # we assume child has list of gradients, indexed by parents
        if self.learnable:
            self.gradient += from_child.gradients[self]  # reverse DFS hits dead-end here

    def update(self, params: dict, method='GD'):
        ''' Run an optimization step, based on computed gradient. GD = gradient descent, TODO: Add more! '''
        if method == 'GD':
            assert 'alpha' in params
            alpha = params['alpha']
            self.set_input(self.value - (alpha * self.gradient))
        elif method == 'adagrad':
            assert 'alpha' in params
            alpha = params['alpha']
            self.cache += np.sum(self.gradient * self.gradient)
            self.set_input(self.value - ((alpha * self.gradient) / (np.sqrt(self.cache) + 1e-7)))

    def set_value_from_flat(self, new_val, flat_idx):
        assert self.learnable, str(self.name) + ' is not learnable, are you sure you want to be doing this?'
        if flat_set_message: print(self.name, 'node was explicitly set (outside of optimization).')
        idx = self.get_idx_from_flat(flat_idx)
        self.value[idx] = new_val

    def get_values_from_flat(self, flat_idx):
        idx = self.get_idx_from_flat(flat_idx)
        return copy.copy(self.value[idx]), copy.copy(self.gradient[idx])

    def get_idx_from_flat(self, flat_idx):
        return np.unravel_index(flat_idx, self.shape)

    def __call__(self, *args, **kwargs):
        return self.fire()

    def __repr__(self):
        return str(self.value)


# A node representing multiplication between two tensors  TODO: Extend to arbitrary dimension
class MultiplicationNode:
    def __init__(self, tensor1, tensor2, name='multiplication'):
        assert len(tensor1.shape) == len(tensor2.shape) == 2, 'multiplication has not been implemented for >2 tensors!'
        assert tensor1.shape[1] == tensor2.shape[0], 'input shapes don\'t match!'
        self.left_tensor, self.right_tensor = tensor1, tensor2  # called 'left' & 'right' as order matters
        self.children = []  # add children from outside
        self.parents = [tensor1, tensor2]
        tensor1.children.append(self)
        tensor2.children.append(self)

        self.value, self.gradients, self.has_cached, self.shape = None, None, None, None  # good practice to initialize in constructor
        self.reset()

        self.name = name

    def reset(self, hard=False):
        self.__reset_shape__()
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.left_tensor: np.zeros(self.left_tensor.shape),
                          self.right_tensor: np.zeros(self.right_tensor.shape)}
        for parent in self.parents: parent.reset(hard)  # clean parents, recursively

    def __reset_shape__(self):
        for parent in self.parents: parent.__reset_shape__()  # set the shape of the parents first
        self.shape = (self.left_tensor.shape[0], self.right_tensor.shape[1])  # shape of the output

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        if not self.has_cached: self.value = self.left_tensor.fire() @ self.right_tensor.fire()
        elif print_fired: print('<used cached> ', end='')
        self.has_cached = True
        if print_fired: print(self.name, 'was fired.')
        return self.value

    def backfire(self, from_child):
        assert from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')
        self.gradients[self.left_tensor] += from_child.gradients[self] @ self.right_tensor().T
        self.gradients[self.right_tensor] += self.left_tensor().T @ from_child.gradients[self]
        for parent in self.parents: parent.backfire(self)

    def __call__(self, *args, **kwargs):
        return self.fire()


# A node representing addition between two tensors
class AdditionNode:
    def __init__(self, tensor1, tensor2, name='addition'):
        assert tensor1.shape == tensor2.shape, 'input shapes don\'t match!'
        self.tensor1, self.tensor2 = tensor1, tensor2
        self.children = []  # add children from outside
        self.parents = [tensor1, tensor2]
        tensor1.children.append(self)
        tensor2.children.append(self)

        self.value, self.gradients, self.has_cached, self.shape = None, None, None, None  # good practice to initialize in constructor
        self.reset()

        self.name = name

    def reset(self, hard=False):
        self.__reset_shape__()
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.tensor1: np.zeros(self.tensor1.shape),
                          self.tensor2: np.zeros(self.tensor2.shape)}
        for parent in self.parents: parent.reset(hard)  # clean parents, recursively

    def __reset_shape__(self):
        for parent in self.parents: parent.__reset_shape__()  # set the shape of the parents first
        self.shape = self.tensor1.shape  # shape of the output. could have also made it tensor2.shape

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        if not self.has_cached: self.value = self.tensor1.fire() + self.tensor2.fire()
        elif print_fired: print('<used cached> ', end='')
        self.has_cached = True
        if print_fired: print(self.name, 'was fired.')
        return self.value

    def backfire(self, from_child):
        assert from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')
        self.gradients[self.tensor1] += from_child.gradients[self]  # the gradient just distributes
        self.gradients[self.tensor2] += from_child.gradients[self]
        for parent in self.parents: parent.backfire(self)

    def __call__(self, *args, **kwargs):
        return self.fire()


# A node representing the squared loss, given a mini-batch of activations, and corresponding labels.
# For homoskedastic regression, where we simply output a mu:   1/2m * |mu-y|^2. Calculate fixed variance as mean((mu-y)^2).
# For heteroskedastic regression, where we output the std: <TODO: fill in>
class SquaredLossNode:
    def __init__(self, mu, y, std=None, name='squared_loss'):
        self.parents = [mu, y]
        assert mu.shape == y.shape, 'mu & y shapes don\'t match!'
        if std is not None:
            assert std.shape == mu.shape, 'std shape doesn\'t match!'
            self.std = std
            self.parents.append(std)

        self.mu, self.y = mu, y
        mu.children.append(self)
        y.children.append(self)
        self.children = []  # add from outside. loss node can have children, like addition nodes, for adding losses

        self.value, self.gradients, self.has_cached, self.shape = None, None, None, None  # good practice to initialize in constructor
        self.reset()

        self.name = name

    def reset(self, hard=False):
        self.__reset_shape__()
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.mu: np.zeros(self.mu.shape)}
        if hasattr(self, 'std'): self.gradients[self.std] = np.zeros(self.std.shape)
        for parent in self.parents: parent.reset(hard)  # clean parents, recursively

    def __reset_shape__(self):
        for parent in self.parents: parent.__reset_shape__()  # set the shape of the parents first
        self.shape = (1, 1)

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        sigma = np.ones_like(self.mu.fire()) * (self.std.fire() if hasattr(self, 'std') else 1)
        if not self.has_cached: self.value = (0.5 * (np.linalg.norm((self.y.fire() - self.mu.fire()) / sigma) ** 2) + sum(np.log(sigma)))  # will reduce to MSE if sigma is just vector of ones
        elif print_fired: print('<used cached> ', end='')
        self.has_cached = True
        if print_fired: print(self.name, 'was fired.')
        return self.value

    def backfire(self, from_child=None):
        assert from_child is None or from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients') or from_child is None
        child_gradient = from_child.gradient[self] if from_child is not None else 1.  # note that it will be a real number!
        sigma = np.ones_like(self.mu()) * (self.std if hasattr(self, 'std') else 1)
        self.gradients[self.mu] += ((self.mu() - self.y()) / (sigma ** 2)) * child_gradient
        if hasattr(self, 'std'): self.gradients[self.std] += ((- ((self.y() - self.mu()) ** 2) / (self.std() ** 3)) + (1. / self.std())) * child_gradient
        for parent in self.parents: parent.backfire(self)
        # Dumb note about what highschool me was confused about LOL
        # (yi - ui)^2 = yi^2 - 2yiui + ui^2  --->  2ui - 2yi = 2(ui - yi)
        # (ui - yi)^2 = ui^2 - 2yiui + yi^2  --->  2ui - 2yi = 2(ui - yi)
        # chain rule on (1): -2(yi - ui) = 2(ui - yi)
        # chain rule on (2): 2(ui - yi)
        if reset_reminder: print('MAKE SURE YOU CALL RESET AFTER THIS!')

    def __call__(self, *args, **kwargs):
        return self.fire()


# LayerNorm node
# NOTE: This only gradchecks when the number of features is at least a handful. If there is only 1 features (e.g.
# in linear regression it's just a single input + bias, there will be a loss of precision when you perturb by any
# small amount).
# (use it at the beginning of an MLP so don't need to worry about rescaling features)
# (or use it in a transformer architecture)
class LayerNormNode:
    def __init__(self, tensor, gamma=None, beta=None, name='layernorm'):
        ''' this layer eats a mini-batch and rescales each individual example: (i) first to 0 mean
            and unit variance, and (ii) second, optionally, to a learned scaling gamma and learned offset.
        '''
        assert len(tensor.shape) == 2, 'layernorm hasn\'t been implemented for >2 tensors!'
        self.parents = [tensor]
        if gamma is not None:
            assert gamma.shape[0] == tensor.shape[1], 'gamma\'s shape does not match!'
            assert gamma.shape[1] == 1, 'gamma must have 1 column!'
            self.gamma = gamma
            self.parents.append(gamma)
            gamma.children.append(self)
        if beta is not None:
            assert beta.shape[0] == tensor.shape[1], 'gamma\'s shape does not match!'
            assert beta.shape[1] == 1, 'beta must have 1 column!'
            self.beta = beta
            self.parents.append(beta)
            beta.children.append(self)

        self.tensor = tensor
        tensor.children.append(self)
        self.children = []  # add from outside

        self.value, self.gradients, self.has_cached, self.shape = None, None, None, None  # good practice to initialize in constructor
        self.reset()

        self.name = name

    def set_input(self, tensor_input: np.ndarray):
        assert isinstance(self.tensor, TensorNode), 'parent is not a Tensor node, not sure what to set input to!'
        self.tensor.set_input(tensor_input)

    def reset(self, hard=False):
        self.__reset_shape__()
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.tensor: np.zeros(self.tensor.shape)}  # recall gradient should be able to flow through tensor too! (e.g. in a transformer)
        if hasattr(self, 'gamma'): self.gradients[self.gamma] = np.zeros(self.gamma.shape)
        if hasattr(self, 'beta'): self.gradients[self.beta] = np.zeros(self.beta.shape)
        for parent in self.parents: parent.reset(hard)  # clean parents, recursively

    def __reset_shape__(self):
        for parent in self.parents: parent.__reset_shape__()  # set the shape of the parents first
        self.shape = self.tensor.shape  # the output is still just the original mini-batch, just with different scaling

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        if not self.has_cached:
            mu, std = np.mean(self.tensor(), axis=1).reshape(-1, 1), np.std(self.tensor(), axis=1, ddof=1).reshape(-1, 1) + 1e-10
            self.value = (self.tensor() - mu) / std
            if hasattr(self, 'gamma'): self.value *= self.gamma().T
            if hasattr(self, 'beta'): self.value += self.beta().T
        elif print_fired: print('<used cached> ', end='')
        self.has_cached = True
        if print_fired: print(self.name, 'was fired.')
        return self.value

    def backfire(self, from_child):
        assert from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')
        child_gradient = from_child.gradients[self]
        assert child_gradient.shape == self.tensor.shape
        S = np.sum(child_gradient, axis=1, keepdims=True) * np.ones_like(child_gradient)
        mu, std = np.mean(self.tensor(), axis=1, keepdims=True), np.std(self.tensor(), axis=1, ddof=1, keepdims=True) + 1e-10
        XminusMu = self.tensor() - mu
        XminusMuoverSigma = XminusMu / std
        XminusMuoverSigmaCubed = XminusMu / np.power(std, 3.0)
        n = self.tensor.shape[1]
        colvector = np.sum(child_gradient * XminusMuoverSigmaCubed, axis=1, keepdims=True)
        self.gradients[self.tensor] += ((child_gradient / std) - ((1/n) * S / std) - (XminusMu * colvector / (n-1))) * self.gamma().T
        if hasattr(self, 'gamma'): self.gradients[self.gamma] += np.sum(XminusMuoverSigma * child_gradient, axis=0, keepdims=True).T
        if hasattr(self, 'beta'): self.gradients[self.beta] += np.sum(child_gradient, axis=0, keepdims=True).T  # gradient simply forks
        for parent in self.parents: parent.backfire(self)

    def __call__(self, *args, **kwargs):
        return self.fire()

    def __repr__(self):
        return str(self.value)



# Ridge, Lasso, and elastic
class RegularizationLossNode:
    def __init__(self, ):
        pass


# Bias trick node (it just adds an extra column of 1s at the end, but the gradient chops off the corresponding column)
class BiasTrickNode:
    def __init__(self, tensor, bias_val=1.0, name='bias_trick'):
        self.tensor = tensor
        self.parents = [tensor]
        tensor.children.append(self)
        self.children = []  # add from outside

        self.bias_val = bias_val  # ONLY use this if the number of neurons in the layer to which you're adding this is small.
                                  # otherwise, stick with 1.0 and apply layernorm after this layer!
        self.value, self.gradients, self.has_cached, self.shape = None, None, None, None  # good practice to initialize in constructor
        self.reset()

        self.name = name

    def set_input(self, tensor_input: np.ndarray):
        assert isinstance(self.tensor, TensorNode), 'parent is not a Tensor node, not sure what to set input to!'
        self.tensor.set_input(tensor_input)

    def reset(self, hard=False):
        self.__reset_shape__()
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.tensor: np.zeros(self.tensor.shape)}
        for parent in self.parents: parent.reset(hard)  # clean parents, recursively

    def __reset_shape__(self):
        for parent in self.parents: parent.__reset_shape__()  # set the shape of the parents first
        self.shape = (self.tensor.shape[0], self.tensor.shape[1] + 1)  # shape of the output. account for the extra column of ones

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        if not self.has_cached: self.value = np.hstack( ( self.tensor(), self.bias_val * np.ones( (self.tensor.shape[0], 1) ) ) )  # tack on the ones
        elif print_fired: print('<used cached> ', end='')
        self.has_cached = True
        if print_fired: print(self.name, 'was fired.')
        return self.value

    def backfire(self, from_child):
        assert from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')
        self.gradients[self.tensor] += from_child.gradients[self][:, :-1]  # remove the final column
        for parent in self.parents: parent.backfire(self)

    def __call__(self, *args, **kwargs):
        return self.fire()


# Component-wise activation function node
class SimpleActivationNode:
    def __init__(self, tensor, kind: str, name='none'):
        ''' append_bias controls whether the output of this node should tack on an extra column of ones '''
        activations = {'sigmoid': sigmoid, 'tanh': tanh, 'softplus': softplus, 'relu': relu}
        activations_prime = {'sigmoid': sigmoid_prime, 'tanh': tanh_prime, 'softplus': softplus_prime, 'relu': relu_prime}
        assert str not in activations.keys(), 'not a valid component-wise activation function!'
        self.parents = [tensor]
        self.tensor = tensor
        self.actfn = activations[kind]
        self.actfn_prime = activations_prime[kind]
        tensor.children.append(self)
        self.children = []  # add from outside

        self.value, self.gradients, self.has_cached, self.shape = None, None, None, None  # good practice to initialize in constructor
        self.reset()

        self.name = name if name != 'none' else kind + ' activation'

    def reset(self, hard=False):
        self.__reset_shape__()
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.tensor: np.zeros(self.tensor.shape)}
        for parent in self.parents: parent.reset(hard)  # clean parents, recursively

    def __reset_shape__(self):
        for parent in self.parents: parent.__reset_shape__()  # set the shape of the parents first
        self.shape = self.tensor.shape

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        if not self.has_cached: self.value = self.actfn(self.tensor())
        elif print_fired: print('<used cached> ', end='')
        self.has_cached = True
        if print_fired: print(self.name, 'was fired.')
        return self.value

    def backfire(self, from_child):
        assert from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')
        self.gradients[self.tensor] += from_child.gradients[self] * self.actfn_prime(self.tensor())
        for parent in self.parents: parent.backfire(self)

    def __call__(self, *args, **kwargs):
        return self.fire()


# Softmax activation node
class SoftmaxNode:
    def __init__(self, tensor, name='softmax activation'):
        self.parents = [tensor]
        self.tensor = tensor
        tensor.children.append(self)
        self.children = []  # add from outside

        self.value, self.gradients, self.has_cached, self.shape = None, None, None, None  # good practice to initialize in constructor
        self.reset()

        self.name = name

    def reset(self, hard=False):
        self.__reset_shape__()
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.tensor: np.zeros(self.tensor.shape)}
        for parent in self.parents: parent.reset(hard)  # clean parents, recursively

    def __reset_shape__(self):
        for parent in self.parents: parent.__reset_shape__()  # set the shape of the parents first
        self.shape = self.tensor.shape

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        if not self.has_cached: self.value = softmax(self.tensor())
        elif print_fired: print('<used cached> ', end='')
        self.has_cached = True
        if print_fired: print(self.name, 'was fired.')
        return self.value

    def backfire(self, from_child):
        assert from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')
        S = self()
        dLdS = from_child.gradients[self]
        self.gradients[self.tensor] += S * (dLdS - (dLdS * S).sum(axis=1, keepdims=True))
        for parent in self.parents: parent.backfire(self)

    def __call__(self, *args, **kwargs):
        return self.fire()


# Cross-entropy loss node
class CrossEntropyLossNode:
    def __init__(self, probs, y_labels, name='cross-entropy loss'):
        assert y_labels.shape[1] == 1, 'each row of y_labels must have 1 entry, the label index!'
        self.parents = [probs, y_labels]
        self.probs = probs
        self.y_labels = y_labels
        self.probs.children.append(self)
        self.y_labels.children.append(self)
        self.children = []  # add from outside

        self.value, self.gradients, self.has_cached, self.shape = None, None, None, None  # good practice to initialize in constructor
        self.reset()

        self.name = name

    def reset(self, hard=False):
        self.__reset_shape__()
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.probs: np.zeros(self.probs.shape)}  # labels have no gradient
        for parent in self.parents: parent.reset(hard)  # clean parents, recursively

    def __reset_shape__(self):
        for parent in self.parents: parent.__reset_shape__()  # set the shape of the parents first
        self.shape = (1, 1)

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        p_mat, y = self.probs(), self.y_labels()
        if not self.has_cached: self.value = -sum(np.log(p_mat[np.arange(p_mat.shape[0]), y.flatten()].reshape(-1, 1)))
        elif print_fired: print('<used cached> ', end='')
        self.has_cached = True
        if print_fired: print(self.name, 'was fired.')
        return self.value

    def backfire(self, from_child=None):
        assert from_child is None or from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients') or from_child is None
        child_gradient = from_child.gradient[self] if from_child is not None else 1.  # note that it will be a real number!
        p_mat, y = self.probs(), self.y_labels()
        grad = np.zeros_like(p_mat)
        row_idxs = np.arange(p_mat.shape[0])
        y_flat = y.flatten()
        grad[row_idxs, y_flat] = -1.0 / p_mat[row_idxs, y_flat]
        self.gradients[self.probs] += child_gradient * grad
        for parent in self.parents: parent.backfire(self)

    def __call__(self, *args, **kwargs):
        return self.fire()


# Simple R --> R functions to call
def sigmoid(tau):
    return 1.0 / (1.0 + np.exp(-tau))


def sigmoid_prime(tau):
    sig = sigmoid(tau)
    return sig * (1.0 - sig)


def tanh(tau):
    return np.tanh(tau)


def tanh_prime(tau):
    return 1.0 - np.power(np.tanh(tau), 2.0)


def softplus(tau):
    return np.log(1.0 + np.exp(tau))


def softplus_prime(tau):
    return sigmoid(tau)


def relu(tau):
    return np.maximum(0.0, tau)


def relu_prime(tau):
    return np.where(tau > 0.0, 1.0, 0.0)  # wherever tau > 0, set 1.0 else 0.0


def softmax(Tau: np.array):
    assert len(Tau.shape) == 2, 'Input must be a matrix!'
    maxVal = np.max(Tau, axis=1, keepdims=True)  # Find the max for each row
    Tau_adjusted = Tau - maxVal  # Subtract max for numerical stability
    exp = np.exp(Tau_adjusted)
    return exp / np.sum(exp, axis=1, keepdims=True)



# Single value softmax
# def softmax(i, tau: np.ndarray):
#     assert 0 <= i < len(tau)
#     return np.exp(tau[i]) / sum([np.exp(xj) for xj in tau])

# Single value softmax prime. For the general, matrix-based softmax, its derivative is given right in the backfire method.
# def softmax_prime(i, j, tau: np.ndarray):
#     ''' i, j are such that xi is in numerator, xj is what derivative is wrt '''
#     assert 0 <= i < len(tau) and 0 <= j < len(tau)
#     soft = softmax(i, tau)
#     dii = soft * (1.0 - soft)
#     return dii if i==j else -dii
