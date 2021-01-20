#version 450
layout (location = 0) in vec2 uv;

layout(binding = 0) uniform sampler2D in_texture;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = texture(in_texture, uv).rgba;
}
