#version 410

uniform sampler2D tex;  // this is the way to send in large amounts of data to each fragment.
uniform float time;

in vec2 uvs;       // the ported, interpolated uv coords (from [0, 1]^2)
out vec4 f_color;  // the only output here is the fragment color, as usual :)

void main() {
    // For some reason, you have to use ALL the uniform variables in some capacity for it to run.
    time; tex;
    // The actual code
    f_color = vec4(texture(tex, uvs).rgb, 1.0);
}