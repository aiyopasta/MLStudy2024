# The idea here is to do a "grand tour" visualization of an MLP learning to classify the MNIST
# digits. One related vanilla approach to solving MNIST is using a "one vs. all" scheme where 
# we train 10 different Bernoulli dists (even just using the perceptron algorithm with Heaviside)
# that light up for each digit type. The problem with this is we don't get a rich representation
# of the input space in the hidden layers, so you can't do cool stuff like latent space interpolation
# or deep dream, autoencoders, cool saliency maps, etc, etc.
#
# Main tasks: (1) Load in dataset (so I should be able to set an "n_examples" int and
#                 have that many (x_train, y_train) loaded in from digits.txt.
#             (2) Given a particular idx and an (x, y) position of the screen, I want to
#                 be able to display the corresponding image (to the idx) at (x, y) with pygame.
#             (3) Train a 1-2 hidden layer MLP to classify between digits to get a high training accuracy.
#                 I don't quite care about having a validation set right now. I just want to extract cool
#                 information from the latent representations + visualize the training process.
#             (4) Display the training process using a Grand Tour interactive visualization.
#             (5) Save the learned weights into a file, in order to do saliency maps + deep dream,
#                 which'll be done in a separate file. The saliency map is purely for seeing which
#                 layers / neurons would be cool to dream at. I want to be able to start with an
#                 input digit (which I draw to the screen) and then "dream up" a "tree"-like thing
#                 from it that "looks like" a bunch of fours and nines.
#
import numpy as np

from computation_graph import *
import pygame
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

# Pygame + gameloop setup
width = 800
height = 800
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("MNIST Visualization")
pygame.init()


# Drawing Coordinate Shift Functions
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])


def A_inv(val):
    global width, height
    return np.array([val[0] - width / 2, -(val[1] - height / 2)])


def A_many(vals):
    return [A(v) for v in vals]   # Galaxy brain function


# THE ACTUAL CODE STARTS HERE —————————————————————————————————————————————————————————————————————————————————
# Load training data (raw, not normalized nor reshaped)
x_train, y_train = [], []
n_examples = 100
digits = []  # fill in if there are a particular subset of digits to exclusively train on
file = open('digits.txt', 'r')  # File structure. For each new line: digit1, pixel1, pixel2, ..., pixel784
for i in range(n_examples):
    line = np.array(file.readline().rstrip().rsplit(',')).astype(float)
    if len(digits)>0 and int(line[0]) not in digits:
        continue
    x_train.append(line[1:])
    y_train.append(line[0])

# Preprocess data to feed into MLP
x_train = np.array(x_train)
y_train = np.array([y_train], dtype=int).T

# Build computation graph for 2 hidden layer MLP
# (1) Decide on the architecture
shape = [784, 128, 128, 10]       # NOT including biases
bias_vals = [255.0, 100.0, 50.0]  # latter two are super arbitrary
actfn = 'relu'
assert len(bias_vals) == len(shape)-1
node_list, params_list = [], []
# (2) Create X and y
X = BiasTrickNode(TensorNode(learnable=False, shape=(n_examples, 784), name='X_data'), bias_val=bias_vals[0])  # 784 weights + 1 bias
y = TensorNode(learnable=False, shape=(n_examples, 1), name='y_labels')
node_list.extend([X, y])
# (3) Create hidden layers
out = X
for i in range(len(shape)-2):  # e.g. i=0,1 for 2 hidden layer MLP
    # Create the hidden layer
    W_b = TensorNode(learnable=True, shape=(shape[i]+1, shape[i+1]), name='WeightsBias-Layer-'+str(i+1))
    outW_b = MultiplicationNode(out, W_b)
    out = BiasTrickNode(SimpleActivationNode(outW_b, kind=actfn), bias_val=bias_vals[i])
    node_list.extend([W_b, outW_b, out])
    params_list.append(W_b)
# (4) Connect final hidden layer to output + compute loss
W_b = TensorNode(learnable=True, shape=(shape[-2]+1, shape[-1]), name='WeightsBias-Layer-'+str(len(shape)-1))
outW_b = MultiplicationNode(out, W_b)
Soft = SoftmaxNode(outW_b)
Loss = CrossEntropyLossNode(Soft, y)
node_list.extend([W_b, outW_b, Soft, Loss])
params_list.append(W_b)

# Training accuracy (percentage correctly classified)
accuracy = 0.0

# Initialize params for training.
for param in params_list:
    assert param.shape[0]>1 and param.shape[0]>1, 'are you sure all learnable parameters are of the form W_b?'
    for h_ in range(param.shape[1]): param.value[param.shape[0] - 1, h_] = 0.01
    if actfn in ['sigmoid', 'tanh']:
        a = np.sqrt(6.0 / (param.shape[0] + param.shape[1]))
        param.value[:-1, :] = np.random.uniform(-a, a, size=(param.shape[0]-1, param.shape[1]))
    elif actfn in ['relu', 'softplus']:
        mu = 0.0
        sigma = np.sqrt(2.0 / param.shape[0])
        param.value[:-1, :] = np.random.normal(mu, sigma, size=(param.shape[0] - 1, param.shape[1]))
    else:
        print('Not sure how to initialize weights for this actfn...')

# Training parameters
h = 1e-5  # for gradient checking
max_epochs = 1000
current_epoch = 0  # we'll increment by 1 before current the first epoch


# One step of training
def train_step(x_input, y_input):
    global X, y, Loss, Soft, accuracy, params_list, current_epoch, max_epochs
    X.set_input(x_input)  # In real ML problems, we'd iterate over mini-batches and set X's value to each mini-batch's value. 1 epoch = all mini-batches done.
    y.set_input(y_input)
    Loss.reset()
    Loss.fire()

    # Calculate training accuracy
    incorrect = np.where(np.argmax(Soft.value, axis=1, keepdims=True) == y_input, 0, 1)
    accuracy = (1.0 - (sum(incorrect) / x_input.shape[0]))[0]
    # Update weights
    Loss.backfire()
    for param in params_list:
        # param.update(params={'alpha': 0.0000001}, method='GD')
        param.update(params={'alpha': 0.02}, method='adagrad')

    # Print epochs, loss, accuracy
    print(current_epoch, Loss.value, accuracy)


# Optional gradient checking
def grad_check(x_input, y_input, n_samples_per_param=3):
    global X, y, Loss, params_list, node_list, h
    X.set_input(x_input)
    y.set_input(y_input)
    for param in params_list:
        Loss.reset()
        Loss.fire()
        Loss.backfire()

        old_param = copy.copy(param)
        k = n_samples_per_param
        for flat_idx in np.random.choice(np.prod(param.shape), k, replace=False):
            old, actual_grad = old_param.get_values_from_flat(flat_idx)
            print('Checking gradient for', param.name, param.get_idx_from_flat(flat_idx), 'parameter.')
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
            print('Numerical:', numerical_grad)
            denom = max(abs(actual_grad), abs(numerical_grad))
            rel_error = (abs(actual_grad - numerical_grad) / denom) if denom>0 else 0.0
            print('Relative error between the two:', rel_error)
            print()


# Convert a particular example in the dataset into a pygame surface for blitting it
def idx2surf(idx: int):
    global x_train
    assert 0 <= idx < x_train.shape[0]
    pixel_array = np.reshape(x_train[idx], (28, 28)).T
    surf = pygame.Surface((28, 28))
    pygame.surfarray.blit_array(surf, np.stack((pixel_array,)*3, axis=-1))  # the 3 is to turn it into RGB channels
    return surf


# Additional vars (for drawing and such)
colors = {
    'white': np.array([255., 255., 255.]),
    'black': np.array([0., 0., 0.]),
    'red': np.array([255, 66, 48]),
    'blue': np.array([30, 5, 252]),
    'fullred': np.array([255, 0, 0]),
    'fullblue': np.array([0, 0, 255]),
    'START': np.array([255, 255, 255])
}


# Keyhandling (for zooming in and out)
def handle_keys(keys_pressed):
    pass


def main():
    global colors, x_train, y_train, current_epoch, max_epochs

    # Pre-gameloop stuff
    run = True

    # Game loop
    count = 0
    while run:
        # Reset stuff
        window.fill((0, 0, 0))

        # (1) Optional gradcheck (comment out if done checking)
        # grad_check(x_train, y_train)
        # break
        # (2) Train for 1 epoch (for now, 1 epoch = full batch, as before) TODO: change this, and do true mini-batches
        if current_epoch <= max_epochs:
            current_epoch += 1
            train_step(x_train, y_train)


        # Image display test
        window.blit(idx2surf(4), pygame.mouse.get_pos())

        # Handle keys
        keys_pressed = pygame.key.get_pressed()
        handle_keys(keys_pressed)

        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.flip()
        count += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()


if __name__ == '__main__':
    main()