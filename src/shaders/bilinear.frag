#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform FragmentColor {
    vec4 u_min;
    vec4 u_max;
    vec4 v_min;
    vec4 v_max;
} u_fragmentParams;

void main() {
    vec4 val_u = mix(u_fragmentParams.u_min, u_fragmentParams.u_max, uv.x);
    vec4 val_v = mix(u_fragmentParams.v_min, u_fragmentParams.v_max, uv.y);

    // TODO: other bilinear combination?
    f_color = val_u + val_v;
}
