#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

// The base texture.
layout (set = 1, binding = 0) uniform sampler texture_sampler;
layout (set = 1, binding = 1) uniform texture2D lhs;
layout (set = 1, binding = 2) uniform texture2D rhs;

// The parameters..
layout (set = 2, binding = 0) uniform FragmentPushConstants {
    vec4 select;
} u_pushConstants;

void main() {
    vec4 bg = texture(sampler2D(lhs, texture_sampler), uv).rgba;
    vec4 fg = texture(sampler2D(rhs, texture_sampler), uv).rgba;
    f_color = mix(bg, fg, u_pushConstants.select);
}
