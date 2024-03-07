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
with open('vert_regression.glsl', 'r') as file:
    vert_shader = file.read()
with open('frag_regression.glsl', 'r') as file:
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


# Key handling # Example: keys_pressed[pygame.K_p]
def handle_keys(keys_pressed):
    pass


def main():
    # Pre-gameloop stuff
    run = True

    # Game loop
    count = 0
    while run:
        # Reset stuff
        screen.fill((0, 0, 0, 0))  # make background black + transparent

        # Draw stuff on screen using Pygame commands
        pygame.draw.rect(screen, (255, 0, 255, 128), (100, 100, 300, 300), width=10)

        # Handle keys
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
        # Check if needed to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()


if __name__ == '__main__':
    main()