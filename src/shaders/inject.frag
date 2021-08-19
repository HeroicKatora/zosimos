#version 450
layout (location = 0) in vec2 uv;

// The base texture.
layout(set = 1, binding = 0) uniform sampler2D in_texture;
// The one which inserts some channels.
layout(set = 1, binding = 1) uniform sampler2D inject_texture;
// The parameters..
layout (set = 1, binding = 2) uniform FragmentPushConstants {
    vec4 select;
} u_pushConstants;

layout(location = 0) out vec4 f_color;

void main() {
    vec4 bg = texture(in_texture, uv).rgba;
    vec4 fg = texture(inject_texture, uv).rgba;
    f_color = mix(bg, fg, u_pushConstants.select);
}
