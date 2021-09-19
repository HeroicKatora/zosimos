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
    // An after-the-fact matrix to collapse the color to its value.
    // This avoids 'transmuting' and duplicating the rhs input texture.
    // If we want to support more than one injected channel then we can use a
    // mat4x4 here in the future.
    vec4 color;
} u_pushConstants;

void main() {
    vec4 bg = texture(sampler2D(lhs, texture_sampler), uv).rgba;
    vec4 fg = texture(sampler2D(rhs, texture_sampler), uv).rgba;
    vec4 inject = vec4(dot(fg, u_pushConstants.color));
    f_color = mix(bg, inject, u_pushConstants.select);
}
