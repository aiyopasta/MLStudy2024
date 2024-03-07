# NOTE 1: Both forward & backwards passes are called recursively, starting with final loss node.
#         Each backwards pass onto a parent must include from which child it was sent, for summing.
#
# NOTE 2: The optimization step, e.g. x += alpha * dx needs to be called for each node, non-recursively.
#         Each node will hold its own per-node optimization quantities, e.g. "velocity" for momentum.
#         It will not have its own alpha, mu, etc. Those need to be globally set outside, and passed in.
#
# NOTE 3: There are two separate procedures you can run with a computation graph: Gradient checking and training.
#         You decide how often you want to gradient check, or how to structure when it's called. But the
#         two algorithms are as follows.
#
# So the training loop will look like:
# (0) Fill input nodes (initialize weights, load mini-batch)
# (1) loss.fire()
# (2) loss.backfire()
# (3) for each LEARNABLE TensorNode:
#         node.update(alpha, mu)
#         node.reset()                      <-- mark each as "unvisited" again, set gradients to 0
#
# The gradcheck procedure will look like:
# (1) Load small batch (like 2-3) + initialize weights.
# (2) loss.fire()
# (3) loss.backfire()
# (4) for each LEARNABLE TensorNode:                   <-- TensorNodes are the ONLY kind of nodes that can be learnable
#         for flat_idx in [1, 2, 3]:                   <-- this loop is necessary as we ONLY want to perturb one param at a time
#             old, actual_grad = node.get_values_from_flat(flat_idx)
#             node.set_value_from_flat(old + epsilon, flat_idx)
#             Jplus = loss.fire()
#             node.set_value_from_flat(old - epsilon, flat_idx)
#             Jminus = loss.fire()
#             numerical_grad = (Jplus + Jminus) / (2 * epsilon)
#             compare(numerical_grad, actual_grad)
#             node.set_value_from_flat(old, flat_idx)
#
import copy
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

# A node representing a scalar, vector, matrix, or arbitrary dimension "tensor"
class TensorNode:
    def __init__(self, learnable, shape=(1,1)):
        ''' shape is a tuple (d1, d2, ..., dn) describing dimensions of input '''
        self.shape = shape
        self.value = np.zeros(shape)
        self.children = []  # add children from outside. it also has no parent; use hasattr() to see if it does
        self.gradient = np.zeros(shape)
        self.learnable = learnable

    def reset(self):
        self.value = np.zeros(self.shape)

    def fire(self):
        return self.value

    def backfire(self, from_child):
        assert from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')  # we assume child has list of gradients, indexed by parents
        self.gradient += from_child.gradients[self]  # reverse DFS hits dead-end here

    def set_value_from_flat(self, new_val, flat_idx):
        idx = np.unravel_index(flat_idx, self.shape)
        self.value[idx] = new_val

    def get_values_from_flat(self, flat_idx):
        idx = np.unravel_index(flat_idx, self.shape)
        return self.value[idx], self.gradient[idx]


# A node representing multiplication between two tensors  TODO: Extend to arbitrary dimension
class MultiplyNode:
    def __init__(self, tensor1, tensor2):
        assert len(tensor1.value.shape) == len(tensor2.value.shape) == 2, 'multiplication has not been implemented for >2 tensors!'
        assert tensor1.value.shape[1] == tensor2.value.shape[0], 'input shapes don\'t match!'
        self.left_tensor, self.right_tensor = tensor1, tensor2  # called 'left' & 'right' as order matters
        self.children = []  # add children from outside
        self.parents = [tensor1, tensor2]
        self.shape = (tensor1.shape[0], tensor2.shape[1])  # shape of the output

        self.value, self.gradients, self.has_cached = None, None, None  # good practice to initialize in constructor
        self.reset()

    def reset(self):
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.left_tensor: np.zeros(self.left_tensor.value.shape),
                          self.right_tensor: np.zeros(self.right_tensor.value.shape)}
        for parent in self.parents: parent.reset()  # clean parents, recursively

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        if not self.has_cached: self.value = self.left_tensor.fire() @ self.right_tensor.fire()
        self.has_cached = True
        return self.value

    def backfire(self, from_child):
        assert from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')
        self.gradients[self.left_tensor] += from_child.gradients[self] @ self.right_tensor.value.T
        self.gradients[self.right_tensor] += self.left_tensor.value.T @ from_child.gradients[self]
        for parent in self.parents: parent.backfire(self)


# A node representing addition between two tensors
class AdditionNode:
    def __init__(self, tensor1, tensor2):
        assert tensor1.shape == tensor2.shape, 'input shapes don\'t match!'
        self.tensor1, self.tensor2 = tensor1, tensor2
        self.children = []  # add children from outside
        self.parents = [tensor1, tensor2]
        self.shape = tensor1.shape  # shape of the output. could have also made it tensor2.shape

        self.value, self.gradients, self.has_cached = None, None, None  # good practice to initialize in constructor
        self.reset()

    def reset(self):
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.tensor1: np.zeros(self.tensor1.shape),
                          self.tensor2: np.zeros(self.tensor2.shape)}
        for parent in self.parents: parent.reset()  # clean parents, recursively

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        if not self.has_cached: self.value = self.tensor1.fire() + self.tensor2.fire()
        self.has_cached = True
        return self.value

    def backfire(self, from_child):
        assert from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')
        self.gradients[self.tensor1] += from_child.gradients[self]  # the gradient just distributes
        self.gradients[self.tensor2] += from_child.gradients[self]
        for parent in self.parents: parent.backfire(self)


# A node representing the squared loss, given a mini-batch of activations, and corresponding labels.
# For homoskedastic regression, where we simply output a mu:   1/2m * |mu-y|^2. Calculate fixed variance as mean((mu-y)^2).
# For heteroskedastic regression, where we output the std:
class SquaredLossNode:
    def __init__(self, mu, y, std=None):
        assert mu.shape == y.shape, 'mu & y shapes don\'t match!'
        if std is not None:
            assert std.shape == mu.shape, 'std shape doesn\'t match!'
            self.std = std

        self.mu, self.y = mu, y
        self.parents = [mu, y]
        self.shape = (1, 1)
        self.children = []  # add from outside. loss node can have children, like addition nodes, for adding losses

        self.value, self.gradients, self.has_cached = None, None, None  # good practice to initialize in constructor
        self.reset()

    def reset(self):
        self.has_cached = False  # this will allow self.value to be overwritten automatically
        self.gradients = {self.mu: np.zeros(self.mu.shape)}
        if hasattr(self, 'std'): self.gradients[self.std] = np.zeros(self.std.shape)
        for parent in self.parents: parent.reset()  # clean parents, recursively

    def fire(self):
        # If we haven't already computed this, compute it. Otherwise use cached.
        sigma = np.ones_like(self.mu) * (self.std if hasattr(self, 'std') else 1)
        if not self.has_cached: self.value = 0.5 * (np.linalg.norm((self.y - self.mu) / sigma) ** 2) + sum(np.log(sigma))  # will reduce to MSE if sigma is just vector of ones
        self.has_cached = True
        return self.value

    def backfire(self, from_child=None):
        assert from_child is None or from_child in self.children, 'not a valid child of this node!'
        assert hasattr(from_child, 'gradients')
        child_gradient = from_child.gradient[self] if from_child is not None else 1.  # note that it will be a real number!
        sigma = np.ones_like(self.mu) * (self.std if hasattr(self, 'std') else 1)
        self.gradients[self.mu] += 0.5 * ((self.mu - self.y) / (sigma ** 2)) * child_gradient
        if hasattr(self, 'std'): self.gradients[self.std] += ((- ((self.y - self.mu) ** 2) / (self.std ** 3)) + (1. / self.std)) * child_gradient
        for parent in self.parents: parent.backfire(self)
        # Dumb note about what highschool me was confused about LOL
        # (yi - ui)^2 = yi^2 - 2yiui + ui^2  --->  2ui - 2yi = 2(ui - yi)
        # (ui - yi)^2 = ui^2 - 2yiui + yi^2  --->  2ui - 2yi = 2(ui - yi)
        # chain rule on (1): -2(yi - ui) = 2(ui - yi)
        # chain rule on (2): 2(ui - yi)


# Ridge, Lasso, and elastic
class RegularizationLossNode:
    def __init__(self, ):
        pass
    

# Cross-entropy loss
class CrossEntropyLossNode:
    def __init__(self):
        pass


# Activation function nodes
class LogisticSigmoidNode:
    def __init__(self):
        pass


class TanhNode:
    def __init__(self):
        pass


class SoftmaxNode:
    def __init__(self):
        pass


class ReLUNode:
    def __init__(self):
        pass


class SoftplusNode:
    def __init__(self):
        pass
