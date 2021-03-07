#version 450
layout (location = 0) in vec2 uv;

layout (push_constant) uniform FragmentPushConstants {
    layout (offset = 0) vec4 select;
} u_pushConstants;

// The base texture.
layout(binding = 0) uniform sampler2D in_texture;
// The one which inserts some channels.
layout(binding = 1) uniform sampler2D inject_texture;

layout(location = 0) out vec4 f_color;

void main() {
    vec4 bg = texture(in_texture, uv).rgba;
    vec4 fg = texture(inject_texture, uv).rgba;
    f_color = mix(bg, fg, u_pushConstants.select);
}
