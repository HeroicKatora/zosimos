#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform FragmentColor {
    vec2 scale;
    vec2 position;
} u_fragmentParams;

vec2 stepMandelbrot(vec2 rf, vec2 c) {
    // At least some precision is nice.
    float real = dot(rf, vec2(rf.x, -rf.y));
    // (real, 2*rf.x*rf.y) + c
    return vec2(real + c.x, fma(2*rf.x, rf.y, c.y));
}

void main() {
    vec2 c = (uv - u_fragmentParams.position) * u_fragmentParams.scale;

    vec2 xy = vec2(0.0, 0.0);
    vec2 sum = xy;

    int steps = 2048;
    for (int i = 0; i < steps; i++) {
    	sum += xy;
        xy = stepMandelbrot(xy, c);
    }

    // Approximates the fixpoints / average of cyclic points
    sum /= float(steps);
    // And calculate fixpoint's distance to `c`.
    sum -= c;

    // Only points within the Mandelbrot set are colored.
    float len = length(xy);
    float light = clamp(2.0 - len, 0.0, 0.7);

    // Rescale chromaticity a bit.
    vec2 vis = sum / 2.0;

    f_color = vec4(light, vis.x - 0.0, vis.y, 1.0);
}
