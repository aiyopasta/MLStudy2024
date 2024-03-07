#version 410

in vec2 vert;      // the position of a vertex
in vec2 texcoord;  // the uv coordinate of the vertex
out vec2 uvs;      // the only "official" output is the uv, because we simply wanna port these to each fragment

void main() {
    uvs = texcoord;  // the uv will be used in the fragment shader to _____
    gl_Position = vec4(vert, 0.0, 1.0);   // set the position of the vertex in 3D to the input, with 0 depth
}