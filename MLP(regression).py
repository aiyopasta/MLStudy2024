# Stuff I learned / observations:
# (1) Layernorm doesn't work well if the number of neurons in a layer is not large! Looks like a step-function.
#     Example: If you turn on the Layernorm layer (and bump up alpha), but only use normal linear regression
#              with the bias trick, it only roughly "sees" 2 kinds of inputs, e.g. [-0.70710678  0.70710678]
#              because ____ (TODO: finish).
#
# (2) However, we still need to deal with the bad statistics induced by using the bias trick. So what to do?
#     Only thing I can think of (and seems to work) is just bumping up the bias 1 by something else like 500.0.
#     Changing the initial WEIGHT for the bias term from 0.01 to something like 500.0 does NOT work, however.
#
#  (3) *OHH* we need to have gamma and beta for layernorm to work even in the toy examples I think! Let me try
#      this... hopefully it fixes the need of the "bias_value" parameters.

import moderngl
import numpy as np
import pygame
from computation_graph import *

pygame.init()

# Special configs required for OpenGL to work on my macOS system
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

# Pygame + gameloop setup
width = 800
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
with open('vert_linear.glsl', 'r') as file:
    vert_shader = file.read()
with open('frag_linear.glsl', 'r') as file:
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
# Training data (raw, not normalized)
x_train = list(np.linspace(-width/2, width/2, 100))
y_train = [-0.5 * x - 200 + np.random.randint(-100, 100) for x in x_train]

# Data normalization parameters (need to use the same ones at test time)
x_mean, x_std = np.mean(x_train), np.std(x_train)
y_mean, y_std = np.mean(y_train), np.std(y_train)

# Normal linear regression (using bias trick, but no layernorm)
# X = BiasTrickNode(TensorNode(learnable=False, shape=(len(x_train), 1), name='X_data'), bias_val=500.0)
# X_normed = X#LayerNormNode(X)
# y = TensorNode(learnable=False, shape=(len(y_train), 1), name='y_labels')
# W_b = TensorNode(learnable=True, shape=(2, 1), name='Weights')  # 1 weight + 1 bias
# XW_b = MultiplicationNode(X_normed, W_b)
# Loss = SquaredLossNode(mu=XW_b, y=y)
# List of all nodes, for convenience
# node_list = [X, X_normed, y, W_b, XW_b, Loss]

# MLP
n_hidden = 3  # number of neurons in single (for now) hidden layer.
n_outputs = 1  # 1 predicted y-value
X = BiasTrickNode(TensorNode(learnable=False, shape=(len(x_train), 1), name='X_data'), bias_val=100.0)
W_b = TensorNode(learnable=True, shape=(2, n_hidden), name='Weights')  # 1 weight + 1 bias
y = TensorNode(learnable=False, shape=(len(y_train), 1), name='y_labels')
XW_b = MultiplicationNode(X, W_b)
Z = BiasTrickNode(SimpleActivationNode(XW_b, kind='relu'), bias_val=500.0)
W2_b = TensorNode(learnable=True, shape=(n_hidden+1, 1), name='Weights')  # 1 weight + 1 bias
ZW_b = MultiplicationNode(Z, W2_b)
Loss = SquaredLossNode(mu=ZW_b, y=y)
# List of all nodes, for convenience
node_list = [X, W_b, y, XW_b, Z, W2_b, ZW_b, Loss]



# Initialize weights & biases for training.
# (1) For biases, initialize with 0.01, which is recommended
for h_ in range(1): W_b.value[W_b.shape[0]-1, h_] = 0.01
for o_ in range(1): W2_b.value[W2_b.shape[0]-1, o_] = 0.01
# print(W_b.value)
# print()
# (2) For weights, use Glorot initialization (assuming activation function is sigmoid or tanh!)
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

# Training parameters
h = 1e-5  # for gradient checking
should_retrain = False
max_epochs = 3000
current_epoch = 0  # we'll increment by 1 before current the first epoch


# Preprocessing
def preprocess(x_raw=None, y_raw=None):
    global x_mean, x_std, y_mean, y_std
    assert x_raw is not None or y_raw is not None, 'what do u even want me to preprocess, fool?'
    x_input, y_input = None, None
    if x_raw is not None:
        x_raw = np.array(x_raw)
        x_input = np.reshape(x_raw, (len(x_raw), 1))
        # x_input = (x_raw - x_mean) / x_std
        # x_input = np.reshape(x_input, (len(x_input), 1))
        # x_input = np.hstack((x_input, 1 * np.ones((len(x_raw), 1))))  # bias trick
    if y_raw is not None:
        # y_input = (y_raw - y_mean) / y_std
        y_input = np.reshape(y_raw, (len(y_raw), 1))

    return x_input, y_input


# Postprocessing of the output
def postprocess(y_raw):
    return np.reshape(y_raw, (1, y_raw.shape[0]))[0]
    # global y_mean, y_std  # very sloppy implementation, but bare with me
    # y_output = np.reshape(y_raw, (1, y_raw.shape[0]))[0]
    # return (y_output * y_std) + y_mean


# Optional grad-checking
def grad_check(x_input, y_input):
    global X, y, W_b, XW_b, Z, W2_b, ZW_b, Loss
    X.set_input(x_input)
    y.set_input(y_input)
    for param in [W_b, W2_b]:
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
    global X, y, W_b, XW_b, Z, W2_b, ZW_b, Loss
    X.set_input(x_input)  # In real ML problems, we'd iterate over mini-batches and set X's value to each mini-batch's value. 1 epoch = all mini-batches done.
    y.set_input(y_input)
    Loss.reset()
    Loss.fire()
    # print('X', X.value)
    # print('y', y.value)
    # print('W', W_b.value)
    # print('XW', XW.value)
    print('Loss', Loss.value)
    # print()

    # print('X', X.shape)
    # print('y', y.shape)
    # print('W', W.shape)
    # print('XW', XW.shape)
    # print('Loss', Loss.shape)
    # print()

    Loss.backfire()
    # print('W', W.get_values_from_flat(1))
    for param in [W_b, W2_b]:
        param.update(params={'alpha': 0.1}, method='adagrad')


# Additional vars (for drawing and such)
drag_idx = -1  # index of point being moved. -1 if none
point_radius = 5
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
        pos = A_inv(pygame.mouse.get_pos())
        is_selecting = [np.linalg.norm(pos - np.array([x,y])) < point_radius for x,y in zip(x_train, y_train)]
        if np.any(is_selecting): drag_idx = is_selecting.index(True)
        else:
            x_train.append(pos[0])
            y_train.append(pos[1])
            should_retrain = True
    # Stop tracking the dragging point
    else: drag_idx = -1


def main():
    global prev_mouse_pos, point_radius, n_samples, should_retrain, current_epoch, max_epochs,\
        X, y, W_b, XW_b, Z, W2_b, ZW_b, Loss

    # Pre-gameloop stuff
    run = True
    font = pygame.font.Font('/Users/adityaabhyankar/Library/Fonts/cmunrm.ttf', 36)

    # Game loop
    count = 0
    while run:
        # Reset stuff
        screen.fill((0, 0, 0, 1.0))  # make background black + transparent (TODO: hmm... making it transparent messes with text that is blit to the screen)

        # UPDATE MODEL / DATA ————
        # Move point being dragged to mouse location. If it's a new drag location, need to restart training.
        mouse_pos = A_inv(pygame.mouse.get_pos())
        if drag_idx != -1:
            x_train[drag_idx], y_train[drag_idx] = tuple(mouse_pos)
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

        # Essentially more epochs to the training if the data has changed
        if should_retrain:
            current_epoch = 0
            should_retrain = False

        # DRAW USING PYGAME COMMANDS ————
        # Draw training points
        for x_, y_ in zip(x_train, y_train):
            pygame.draw.circle(screen, colors['white'], A([x_,y_]), radius=point_radius, width=0)

        # Draw learned curve
        x_test = np.linspace(-width/2, width/2, n_samples)
        x_input, _ = preprocess(x_raw=x_test)
        X.set_input(x_input)
        Loss.reset()
        Loss.fire()
        # y_output = postprocess(ZW_b.value)
        y_output = postprocess(ZW_b.value)
        pts = [np.array([x_, y_]) for x_, y_ in zip(x_test, y_output)]
        # print(pts)
        pygame.draw.lines(screen, colors['red'], False, A_many(pts), width=2)

        # Draw training epochs
        text = font.render('Epoch: '+str(min(current_epoch, max_epochs))+'/'+str(max_epochs), True, colors['red'])
        screen.blit(text, (width/2 - 120, 40))

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
        # (3) TODO: Send any more uniform variables here...
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