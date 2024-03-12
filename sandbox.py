import numpy as np
# from computation_graph import *

T = np.array([[-0.07426286,  0.12243876],
              [-0.0481759 , -0.0481759 ],
              [ 0.12243876, -0.07426286]])

print(np.argmax(T, axis=1, keepdims=True))

# D = np.sum(T, axis=1, keepdims=True) * np.ones_like(T)
# print(D)

# D = T - np.mean(T, axis=1, keepdims=True)
# print(D)

# y = np.array([[1],
#               [0],
#               [1]])
#
# M2 = np.zeros_like(T)
# row_idxs = np.arange(T.shape[0])
# y_flat = y.flatten()
# M2[row_idxs, y_flat] = T[row_idxs, y_flat]
# print(M2)


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