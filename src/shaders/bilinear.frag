#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform FragmentColor {
    vec4 u_min;
    vec4 u_max;
    vec4 v_min;
    vec4 v_max;
    vec4 uv_min;
    vec4 uv_max;
} u_fragmentParams;

void main() {
    vec4 val_u = mix(u_fragmentParams.u_min, u_fragmentParams.u_max, uv.x);
    vec4 val_v = mix(u_fragmentParams.v_min, u_fragmentParams.v_max, uv.y);
    vec4 val_uv = mix(u_fragmentParams.uv_min, u_fragmentParams.uv_max, uv.x * uv.y);

    f_color = val_u + val_v + val_uv;
}
