# A Tensorflow-playground styled visualization of multilayer perceptron learning
# in real-time. There's a visualizer for the live decision boundary, as well as a visual
# of the opposing POV, where the input space is warped to make the data linearly separable.
#
# NOTES / OBSERVATIONS: (1) Sometimes, the learning rate is too high even for Adagrad (I think?)
#
import moderngl
import numpy as np
import pygame
from computation_graph import *
np.random.seed(42)

pygame.init()

# Special configs required for OpenGL to work on my macOS system
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

# Pygame + gameloop setup
width = 2000
height = 800
# The pygame.OPENGL flag is what makes the ctx's create context method hook on to pygame automatically
window = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
# Thus we create a new surface to allow us to draw on it
screen = pygame.Surface((width, height), pygame.SRCALPHA)
pygame.display.set_caption("Pygame with shaders!")

# Create moderngl context
ctx = moderngl.create_context()

# The "shadertoy trick" of the quad
quad_buffer = ctx.buffer(data=np.array([
    -1.0, 1.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 0.0,
    -1.0, -1.0, 0.0, 1.0,
    1.0, -1.0, 1.0, 1.0,
], dtype='f').tobytes())

# Import vertex and frag shaders
with open('vert_db.glsl', 'r') as file:
    vert_shader = file.read()
with open('frag_db.glsl', 'r') as file:
    frag_shader = file.read()

# Kind of "bundle up" the two shader programs (vert & frag) together. This'll compile them too, I think.
program = ctx.program(vert_shader, frag_shader)
# Setup the VAO
render_object = ctx.vertex_array(program, [(quad_buffer, '2f 2f', 'vert', 'texcoord')])
# Function for converting the "screen" surface to a texture to send into the shader
def surf2tex(surf):
    tex = ctx.texture(surf.get_size(), 4)
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.swizzle = 'BGRA'
    tex.write(surf.get_view('1'))
    return tex


# MY HELPER FUNCTIONS —————————————————————————————————————————————————————————————————————————————————
# Coordinate Shift
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])


def A_inv(val):
    global width, height
    return np.array([val[0] - width / 2, -(val[1] - height / 2)])


def A_many(vals):
    return [A(v) for v in vals]   # Galaxy brain function


# THE ACTUAL CODE STARTS HERE —————————————————————————————————————————————————————————————————————————————————
# Generate training data (raw, not normalized or shifted to fit in visualization window).
# Points will shifted to fit into the little window used for visualization of the decision boundary at draw time.
# So here, create the points as if "height" and "width" refer simply to the visualization window's dimensions.
# TODO: Add a bit of jitter
dataset = 0  # 0 = radial, 1 = affine, 2 = v-shaped
x_train, y_train = [], []
n_examples = 150
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
elif dataset == 1:
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

# print(np.array(x_train))
assert np.all((np.array(y_train) == 0) | (np.array(y_train) == 1)), 'labels must be of the form 0, 1, 2...'

# Build computation graph (TODO: add functionality for many more hidden layers, neurons, and classes)
# Note that because of glsl's visualization limits, we can only have 3 neurons per hidden unit, as that
# would mean 3 weights + 1 bias = 4 maximum dimension of matrices, which is the max that glsl supports.
# NOTE: IF YOU CHANGE SOMETHING HERE, MAKE SURE TO CHANGE IT IN THE GLSL FILE TOO!!
n_hidden = 3  # number of neurons in single (for now) hidden layer.
n_outputs = 2  # 2 classes
bias_vals = [500.0, 500.0]  # used to be 500.0, 500.0
# note: read "W_b" as "W and b", 'b' for bias vector. and 'XW_b" as 'XW+b", where b is the bias vector included in W.
X = BiasTrickNode(TensorNode(learnable=False, shape=(len(x_train), 2), name='X_data'), bias_val=bias_vals[0])  # (x1, x2) + 1 bias
# gamma = TensorNode(learnable=True, shape=(3, 1), name='gamma')  # 3 inputs (= 2 + 1 bias)
# beta = TensorNode(learnable=True, shape=(3, 1), name='beta')    # 3 inputs (= 2 + 1 bias)
# X_normed = LayerNormNode(X, gamma=gamma, beta=beta)
y = TensorNode(learnable=False, shape=(len(y_train), 1), name='y_labels')
W_b = TensorNode(learnable=True, shape=(3, n_hidden), name='WeightsBias-Layer-1')  # 3 weights per hidden neuron (for x1, x2, and bias)
XW_b = MultiplicationNode(X, W_b)
Z = BiasTrickNode(SimpleActivationNode(XW_b, kind='relu'), bias_val=bias_vals[1])
W2_b = TensorNode(learnable=True, shape=(n_hidden + 1, n_outputs), name='WeightsBias-Layer-2')  # num of hidden neurons + bias
ZW_b = MultiplicationNode(Z, W2_b)
Soft = SoftmaxNode(ZW_b)
Loss = CrossEntropyLossNode(Soft, y)
# List of all nodes, for convenience
node_list = [X, y, W_b, XW_b, Z, W2_b, ZW_b, Soft, Loss]

# Training accuracy (percentage correctly classified)
accuracy = 0.0

# Initialize params for training.
# (1) For biases, initialize with 0.01, which is recommended
for h_ in range(n_hidden): W_b.value[W_b.shape[0]-1, h_] = 0.01
for o_ in range(n_outputs): W2_b.value[W2_b.shape[0]-1, o_] = 0.01
# (2) Set gamma equal to ones and beta equal to 0s
# gamma.set_input(np.ones(gamma.shape))
# beta.set_input(np.zeros(beta.shape))
# (3) For linear weights, use Glorot initialization (assuming activation function is sigmoid or tanh!)
for param in [W_b, W2_b]:
    actfn = Z.parents[0].actfn
    if actfn in [sigmoid, tanh]:
        a = np.sqrt(6.0 / (param.shape[0] + param.shape[1]))
        param.value[:-1, :] = np.random.uniform(-a, a, size=(param.shape[0]-1, param.shape[1]))
    elif actfn in [relu, softplus]:
        mu = 0.0
        sigma = np.sqrt(2.0 / param.shape[0])
        param.value[:-1, :] = np.random.normal(mu, sigma, size=(param.shape[0] - 1, param.shape[1]))
    else:
        print('Not sure how to initialize weights...')

# print(W_b.value)
# print()

# Training parameters
h = 1e-5  # for gradient checking
should_retrain = False
max_epochs = 10000
current_epoch = 0  # we'll increment by 1 before current the first epoch


# Preprocessing
def preprocess(x_raw=None, y_raw=None):
    assert x_raw is not None or y_raw is not None, 'what do u even want me to preprocess, fool?'
    x_input, y_input = None, None
    if x_raw is not None:
        x_raw = np.array(x_raw)
        x_input = x_raw
        # x_input = np.hstack((x_raw, 1 * np.ones((len(x_raw), 1))))  # bias trick  TODO: Note: The bias trick will now be implemented through a separate node
    if y_raw is not None:
        y_input = np.reshape(y_raw, (len(y_raw), 1))

    return x_input, y_input


# Postprocessing of the output
def postprocess(y_raw):
    pass  # TODO


# Optional gradient checking
def grad_check(x_input, y_input):
    global X, y, W_b, XW_b, Z, W2_b, ZW_b, Soft, Loss
    X.set_input(x_input)
    y.set_input(y_input)
    for param in [W_b, W2_b, gamma, beta]:
        Loss.reset()
        Loss.fire()
        Loss.backfire()

        old_param = copy.copy(param)
        for flat_idx in range(np.prod(param.shape)):
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
            rel_error = abs(actual_grad - numerical_grad) / max(abs(actual_grad), abs(numerical_grad))
            print('Relative error between the two:', rel_error)
            print()


# One step of training
def train_step(x_input, y_input):
    global X, y, W_b, XW_b, Z, W2_b, ZW_b, Soft, Loss, accuracy, current_epoch
    X.set_input(x_input)  # In real ML problems, we'd iterate over mini-batches and set X's value to each mini-batch's value. 1 epoch = all mini-batches done.
    y.set_input(y_input)
    Loss.reset()
    Loss.fire()

    # Calculate training accuracy
    incorrect = abs(np.argmax(Soft.value, axis=1, keepdims=True) - y_input)
    accuracy = (1.0 - (sum(incorrect) / len(x_train)))[0]
    # Update weights
    Loss.backfire()
    for param in [W_b, W2_b]:
        # param.update(params={'alpha': 0.0000001}, method='GD')
        param.update(params={'alpha': 0.02}, method='adagrad')

    # Print epochs, loss, accuracy
    # print(current_epoch, Loss.value, accuracy)


# Visualization parameters (for decision boundary visualization window and more)
# Diagram — LEFT: Live warping animation. MIDDLE: MLP live depiction. RIGHT: Live decision boundary visualization.
db_sidelen = 0.8 * height
db_offset = np.array([(width/2) - (db_sidelen/2 + 50.0), 0.0])

# Coordinate shift into the decision boundary window. Usage for drawing: A(DB(val))
def DB(val):
    global db_sidelen, db_offset
    value = np.array(val)
    factor = db_sidelen / height
    return (value * factor) + db_offset

def DB_inv(val):
    global db_sidelen, db_offset
    value = np.array(val)
    inv_factor = height / db_sidelen
    return (value - db_offset) * inv_factor

# Additional vars (for pygame drawing and such)
drag_idx = -1  # index of point being moved. -1 if none
point_radius = 10
n_samples = 100  # number of samples for drawing the learned curve
colors = {
    'white': np.array([255., 255., 255.]),
    'black': np.array([0., 0., 0.]),
    'red': np.array([255, 66, 48]),
    'blue': np.array([30, 5, 252]),
    'fullred': np.array([255, 0, 0]),
    'fullblue': np.array([0, 0, 255]),
    'START': np.array([255, 255, 255])
}
prev_mouse_pos = np.array([0.,0.])


# Key handling # Example: keys_pressed[pygame.K_p]
def handle_keys(keys_pressed):
    pass


# Mouse handling
def handle_mouse(event):
    global x_train, y_train, point_radius, drag_idx, should_retrain
    # Either add a new point or select existing one for dragging
    if drag_idx == -1:
        pos = DB_inv(A_inv(pygame.mouse.get_pos()))
        is_selecting = [np.linalg.norm(pos - x_) < point_radius for x_, y_ in zip(x_train, y_train)]
        if np.any(is_selecting): drag_idx = is_selecting.index(True)
        else:
            # x_train.append(pos[0])
            # y_train.append(pos[1])
            # should_retrain = True
            pass  # TODO add ability to add new points, of any desired color!
    # Stop tracking the dragging point
    else: drag_idx = -1


def main():
    global prev_mouse_pos, point_radius, n_samples, should_retrain, current_epoch, max_epochs, x_train, y_train, Loss, Soft, accuracy, db_sidelen, db_offset, bias_vals

    # Pre-gameloop stuff
    run = True
    font = pygame.font.Font('/Users/adityaabhyankar/Library/Fonts/cmunrm.ttf', 36)

    # Game loop
    count = 0
    while run:
        # Reset stuff
        screen.fill((0, 0, 0, 0.0))  # make background black + transparent (TODO: hmm... making it transparent messes with text that is blit to the screen)

        # UPDATE MODEL / DATA ————
        # Move point being dragged to mouse location. If it's a new drag location, need to restart training.
        mouse_pos = DB_inv(A_inv(pygame.mouse.get_pos()))
        if drag_idx != -1:
            x_train[drag_idx] = np.array(mouse_pos)
            should_retrain = should_retrain or np.linalg.norm(prev_mouse_pos - mouse_pos) > 1e-10
        # Set this mouse position to the old one
        prev_mouse_pos = mouse_pos

        # Actual training of the model!
        # (1) Preprocess
        x_input, y_input = preprocess(x_train, y_train)
        # (2) Optional gradcheck (comment out if done checking)
        # grad_check(x_input, y_input)
        # break
        # (3) Train step, if not run out of epochs
        if current_epoch <= max_epochs:
            current_epoch += 1
            train_step(x_input, y_input)
            # break

        # Track weights
        # print(W_b.value)
        # print()
        # break

        # Test current mouse point (debugging)
        # X.set_input(np.array([mouse_pos]))
        # Loss.reset()
        # Loss.fire()
        # print(current_epoch, Soft.value)

        # Essentially more epochs to the training if the data has changed
        if should_retrain:
            current_epoch = 0
            W_b.cache = 0.0    # TODO: SHOULD NOT BE DOING THIS HERE, MOVE SOMEWHERE ELSE LIKE IN RESET()!
            W2_b.cache = 0.0
            should_retrain = False

        # DRAW USING PYGAME COMMANDS ————
        # Draw decision boundary window bounding box
        pygame.draw.rect(screen, colors['white'], (*tuple(A([0,0]) + db_offset - np.array([db_sidelen/2, db_sidelen/2])), db_sidelen, db_sidelen), width=1)
        # Draw training points
        for x_, y_ in zip(x_train, y_train):
            col = colors['red'] if y_ == 0 else colors['blue']
            pygame.draw.circle(screen, col, A(DB(x_)), radius=point_radius, width=0)
            pygame.draw.circle(screen, (0, 0, 0), A(DB(x_)), radius=point_radius, width=2)

        # Draw training epochs
        # text = font.render('Epoch: '+str(min(current_epoch, max_epochs))+'/'+str(max_epochs), True, colors['red'])
        # screen.blit(text, (width/2 - 120, 40))

        # Handle keys + mouse
        keys_pressed = pygame.key.get_pressed()
        handle_keys(keys_pressed)

        # Send the pygame surface over to our shaders for post-processing
        # (1) Send the surface as a uniform called 'tex'
        frame_tex = surf2tex(screen)
        frame_tex.use(0)  # bind the texture to ctx's texture unit 0
        program['tex'] = 0  # send in the texture in texture unit 0 to the uniform tex variable
        # (2) Send the uniform 'time' variable
        program['time'] = count
        # (3) Send any more uniform variables here...
        # (a) For drawing / display
        program['window_dims'] = np.array([width, height])
        program['db_offset'] = db_offset.flatten()
        program['db_sidelen'] = db_sidelen
        # (b) The actual MLP params
        program['W_bT'] = W_b.value.T.flatten('F')
        program['W2_bT'] = W2_b.value.T.flatten('F')
        program['bias_vals'] = bias_vals
        # (4) Render the result to the screen
        render_object.render(mode=moderngl.TRIANGLE_STRIP)

        # END THE RUN
        count += 1
        pygame.display.flip()  # this flips the buffers of the WINDOW, not screen, which is a standalone surface
        frame_tex.release()  # VERY IMPORTANT
        # Check if needed to quit + handle mouse (ya it's weird but if we do pygame.event.get() separately above
        # it does not work because each call clears all the events leaving none for the second one to process).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONUP:
                handle_mouse(event)


if __name__ == '__main__':
    main()