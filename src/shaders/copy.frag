#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (binding = 0) uniform texture2D in_texture;
layout (binding = 1) uniform sampler texture_sampler;

void main() {
    f_color = texture(sampler2D(in_texture, texture_sampler), uv).rgba;
}
