#version 410

// Generic uniforms
uniform sampler2D tex;  // this is the way to send in large amounts of data to each fragment.
uniform float time;
uniform vec2 window_dims;
// For decision boundary viz
uniform vec2 db_offset;
uniform float db_sidelen;
// MLP params
uniform mat3x3 W_b;
uniform mat4x2 W2_b;

in vec2 uvs;       // the ported, interpolated uv coords (from [0, 1]^2)
out vec4 f_color;  // the only output here is the fragment color, as usual :)

// Coordinate shift
vec2 A(vec2 val) {
    return vec2(val.x + (window_dims.x / 2.0), -val.y + (window_dims.y / 2.0));
}
vec2 A_inv(vec2 val) {
    return vec2(val.x - (window_dims.x / 2.0), -(val.y - (window_dims.y / 2.0)));
}
vec2 DB_inv(vec2 val) {
    float inv_factor = window_dims.y / db_sidelen;
    return (val - vec2(db_offset)) * inv_factor;
}

// Evaluation helper functions
vec2 mean_std(vec3 val) {
    float mean = (val.x + val.y + val.z) / 3.0;
    float std = sqrt(((val.x - mean) * (val.x - mean) +
                      (val.y - mean) * (val.y - mean) +
                      (val.z - mean) * (val.z - mean)) / 2.0);

    return vec2(mean, std);
}

float sigmoid(float tau) {
    return 1.0 / (1.0 + exp(-tau));
}

float soft(int i, vec2 arr) {
    return exp(arr[i]) / (exp(arr.x) + exp(arr.y));
}

vec2 softmax(vec2 arr) {
    return vec2(soft(0, arr), soft(1, arr));
}


// Evaluate MLP for current pixel (this outputs a soft value between 0 and 1).
// Note that "vec2 x" must be in the "good" coordinates & NON-DB-SHIFTED, just as the network was trained!
float MLP(vec2 x) {
    // Apply bias trick + normalize
    vec3 x_1 = vec3(x, 1.0);
    vec2 mu_sigma = mean_std(x_1);
    vec3 x_1normed = (x_1 - vec3(mu_sigma.x)) / vec3(mu_sigma.y);
    // Hidden layer
    vec3 wx_b = W_b * x_1normed;
    vec4 z_1 = vec4(vec3(sigmoid(wx_b.x), sigmoid(wx_b.y), sigmoid(wx_b.z)), 1.0);
    // Output
    vec2 zw_b = softmax(W2_b * z_1);
    return round(zw_b.x);  // just return one of the 2 probabilities
}


void main() {
    // For some reason, you have to use ALL the uniform variables in some capacity for it to run.
    time; tex; db_sidelen; db_offset; window_dims; W_b; W2_b;

    // THE ACTUAL CODE ——————————————————————————————————————————————
    float width = window_dims.x, height = window_dims.y;
    vec2 pt = A_inv(gl_FragCoord.xy);  // we'll do all our computations in the "good" coordinates as usual
    vec4 basecol = texture(tex, uvs);  // the base color, from pygame

    // Check whether our pixel is inside the DECISION BOUNDARY bounding box
    vec2 db_min = db_offset - vec2(db_sidelen / 2.0), db_max = db_offset + vec2(db_sidelen / 2.0);
    bool inside = (pt.x >= db_min.x && pt.x <= db_max.x) && (pt.y >= db_min.y && pt.y <= db_max.y);
    if (inside && basecol.a == 0.0) {
        float prob = MLP(DB_inv(pt));
        f_color = prob * vec4(1.0);
    } else {
        f_color = vec4(basecol);
    }
}