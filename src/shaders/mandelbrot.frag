#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform FragmentColor {
    vec2 scale;
    vec2 position;
} u_fragmentParams;

vec2 stepMandelbrot(vec2 rf, vec2 c) {
    vec2 pow2 = vec2(rf.x*rf.x - rf.y*rf.y, 2*rf.x*rf.y);
    return pow2 + c;
}

void main() {
    vec2 c = (uv - u_fragmentParams.position) * u_fragmentParams.scale;

    vec2 xy = vec2(0.0, 0.0);
    for (int i = 0; i < 255; i++) {
        xy = stepMandelbrot(xy, c);
    }

    float rad = atan(xy.x, xy.y);
    float len = length(xy);
    f_color = vec4(sin(rad), cos(rad), clamp(3.0 - len, 0.0, 1.0), 1.0);
}
