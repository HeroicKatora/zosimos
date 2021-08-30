#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 2, binding = 0) uniform FragmentColor {
    vec4 color;
} u_fragmentColor;

void main() {
    f_color = u_fragmentColor.color;
}
