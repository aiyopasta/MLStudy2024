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

    def set_value_from_flat(self, new_val, flat_idx):
        assert self.learnable, str(self.name) + ' is not learnable, are you sure you want to be doing this?'
        if flat_set_message: print(self.name, 'node was explicitly set (outside of optimization).')
        idx = np.unravel_index(flat_idx, self.shape)
        self.value[idx] = new_val

    def get_values_from_flat(self, flat_idx):
        idx = np.unravel_index(flat_idx, self.shape)
        return copy.copy(self.value[idx]), copy.copy(self.gradient[idx])

    def __call__(self, *args, **kwargs):
        return self.fire()

    def __repr__(self):
        return str(self.value)


# A node representing multiplication between two tensors  TODO: Extend to arbitrary dimension
class MultiplicationNode:
    def __init__(self, tensor1, tensor2, name='multiplication'):
        assert len(tensor1.value.shape) == len(tensor2.value.shape) == 2, 'multiplication has not been implemented for >2 tensors!'
        assert tensor1.value.shape[1] == tensor2.value.shape[0], 'input shapes don\'t match!'
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
        self.gradients = {self.left_tensor: np.zeros(self.left_tensor.value.shape),
                          self.right_tensor: np.zeros(self.right_tensor.value.shape)}
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
        assert mu.shape == y.shape, 'mu & y shapes don\'t match!'
        if std is not None:
            assert std.shape == mu.shape, 'std shape doesn\'t match!'
            self.std = std

        self.mu, self.y = mu, y
        self.parents = [mu, y]
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
# (use it at the beginning of an MLP to not worry about rescaling features, or use in a transformer)
class LayerNormNode:
    def __init__(self, tensor, gamma, beta):
        pass



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
