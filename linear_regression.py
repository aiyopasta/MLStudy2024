# Steps for linear regression:
# (1) Set up pygame surface with TRANSPARENCY + opengl support (visualized output should be from shaders, which maybe set transparent pixels to white)
# (2) Allow for clicking + dragging of points
# (3) We'll train algos on the full-batch, with no validation set for now.
#     While iters < max_iters,
#     (i) Preprocess the inputs: Normalize, perhaps put through some basis functions like 1, x, x^2, x^3, ....
#     (ii) Use computation graph of the form: (TensorNode, TensorNode(learnable)) --> Multiplication Node --> SquaredLossNode
#     to do a forward pass + a backwards pass. Make sure it at least RUNS. (See algo given in other file.)
# (4) Use gradient checking (See algo given in other file) to verify gradients are being computed correctly.
# (5) If the user clicks / drags a point, set iters = 0.
# (6) For i in linspace(-window_width, window_width, increments), normalize i, query regression algo, plot line(s).

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
x_train, y_train = [0., 100., 300., -50.], [0., 100., 200., -40]

# Build computation graph (TensorNode, TensorNode(learnable)) --> Multiplication Node --> SquaredLossNode <-- TensorNode
# Create the nodes (for homoskedastic linear regression, for now)
X = TensorNode(learnable=False, shape=(len(x_train), 2), name='X_data')  # one extra feature for bias term
y = TensorNode(learnable=False, shape=(len(y_train), 1), name='y_labels')
W = TensorNode(learnable=True, shape=(2, 1), name='Weights')  # 1 weight + 1 bias
XW = MultiplicationNode(X, W)
Loss = SquaredLossNode(mu=XW, y=y)
# List of all nodes, for convenience
node_list = [X, y, W, XW, Loss]


# TODO <delete> Testing:
# Set the inputs of input nodes
# (1) Preprocess
# (a) Normalize
x_input = (x_train - np.mean(x_train)) / np.std(x_train)
y_input = (y_train - np.mean(y_train)) / np.std(y_train)
# (b) Reshape
x_input = np.reshape(x_input, (len(x_input), 1))
x_input = np.hstack((x_input, np.ones((len(x_train), 1))))  # bias trick
y_input = np.reshape(y_input, (len(y_train), 1))
# (2) Actually set the inputs
X.set_input(x_input)
y.set_input(y_input)
# (3) Initialize weights and biases (TODO: finish)
W.value[1,0] = 0.01

# Training parameters
should_retrain = True
max_epochs = 100
current_epoch = 0  # we'll increment by 1 before current the first epoch

# # Let's do gradient checking now
# h = 1e-5
# for flat_idx in [0, 1]:
#     Loss.fire()
#     Loss.backfire()
#     old, actual_grad = W.get_values_from_flat(flat_idx)
#     Loss.reset()
#     W.set_value_from_flat(old + h, flat_idx)
#     Jplus = Loss.fire()
#     Loss.reset()
#     W.set_value_from_flat(old - h, flat_idx)
#     Jminus = Loss.fire()
#     Loss.reset()
#     numerical_grad = ((Jplus - Jminus) / (2 * h))[0]
#     W.set_value_from_flat(old, flat_idx)
#     print('Jplus:', Jplus, 'Jminus:', Jminus)
#     print('Actual:', actual_grad)
#     print('Numerical:', numerical_grad)  # TODO: Implement a legit comparison, e.g. relative error.
#     print()


# The actual training
while current_epoch < max_epochs:
    X.set_input(x_input)  # In real ML problems, we'd iterate over mini-batches and set X's value to each mini-batch's value. 1 epoch = all mini-batches done.
    y.set_input(y_input)
    Loss.fire()
    # print('X', X.value)
    # print('y', y.value)
    # print('W', W.value)
    # print('XW', XW.value)
    print('Loss', Loss.value)
    print()
    Loss.backfire()
    W.update({'alpha': 0.01})
    Loss.reset()
    current_epoch += 1

# Additional vars (for drawing and such)
drag_idx = -1  # index of point being moved. -1 if none
point_radius = 10
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
    global prev_mouse_pos, should_retrain, current_epoch, max_epochs

    # Pre-gameloop stuff
    run = True

    # Game loop
    count = 0
    while run:
        # Reset stuff
        screen.fill((0, 0, 0, 0))  # make background black + transparent

        # UPDATE MODEL / DATA ————
        # Move point being dragged to mouse location. If it's a new drag location, need to restart training.
        mouse_pos = A_inv(pygame.mouse.get_pos())
        if drag_idx != -1:
            x_train[drag_idx], y_train[drag_idx] = tuple(mouse_pos)
            should_retrain = should_retrain or np.linalg.norm(prev_mouse_pos - mouse_pos) > 1e-10
        # Set this mouse position to the old one
        prev_mouse_pos = mouse_pos

        # Actual model training step. Restart if needed. TODO: Implement
        current_epoch += 1
        if current_epoch <= max_epochs:
            pass # we'll just implement the forward / backward passes and stuff right here!

        # if should_retrain:
        #     print('retrain', np.random.random())
        #     should_retrain = False


        # DRAW USING PYGAME COMMANDS ————
        # Draw training points
        for x, y in zip(x_train, y_train):
            pygame.draw.circle(screen, colors['white'], A([x,y]), radius=point_radius, width=0)


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