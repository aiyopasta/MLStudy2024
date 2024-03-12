#version 410

// Generic uniforms
uniform sampler2D tex;  // this is the way to send in large amounts of data to each fragment.
uniform float time;
uniform vec2 window_dims;
// For decision boundary viz
uniform vec2 db_offset;
uniform float db_sidelen;

in vec2 uvs;       // the ported, interpolated uv coords (from [0, 1]^2)
out vec4 f_color;  // the only output here is the fragment color, as usual :)

// Coordinate shift
vec2 A(vec2 val) {
    return vec2(val.x + (window_dims.x / 2.0), -val.y + (window_dims.y / 2.0));
}
vec2 A_inv(vec2 val) {
    return vec2(val.x - (window_dims.x / 2.0), -(val.y - (window_dims.y / 2.0)));
}

void main() {
    // For some reason, you have to use ALL the uniform variables in some capacity for it to run.
    time; tex; db_sidelen; db_offset; window_dims;

    // THE ACTUAL CODE ——————————————————————————————————————————————
    float width = window_dims.x, height = window_dims.y;
    vec2 pt = A_inv(gl_FragCoord.xy);  // we'll do all our computations in the "good" coordinates as usual
    vec4 basecol = texture(tex, uvs);  // the base color, from pygame

    // Check whether our pixel is inside the DECISION BOUNDARY bounding box
    vec2 db_min = db_offset - vec2(db_sidelen / 2.0), db_max = db_offset + vec2(db_sidelen / 2.0);
    bool inside = (pt.x >= db_min.x && pt.x <= db_max.x) && (pt.y >= db_min.y && pt.y <= db_max.y);
    if (inside && basecol.a == 0.0) {
        f_color = vec4(1.0);
    } else {
        f_color = vec4(basecol);
    }
}