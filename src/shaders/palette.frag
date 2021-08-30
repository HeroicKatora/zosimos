#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform sampler texture_sampler;
layout (set = 1, binding = 1) uniform texture2D lhs;
layout (set = 1, binding = 2) uniform texture2D rhs;

layout (set = 2, binding = 0) uniform FragmentPushConstants {
  mat4x2 channels;
} u_platte;

void main() {
    vec4 basis = texture(sampler2D(rhs, texture_sampler), uv).rgba;
    vec2 paletteuv = u_platte.channels * basis;

    f_color = texture(sampler2D(lhs, texture_sampler), paletteuv).rgba;
}
