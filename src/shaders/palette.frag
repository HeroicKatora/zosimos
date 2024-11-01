#version 450
// FIXME(naga): see below, required for std430 which is not enabled
// #extension GL_EXT_scalar_block_layout : require                                                                                   │    │
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform sampler texture_sampler;
layout (set = 1, binding = 1) uniform texture2D lhs;
layout (set = 1, binding = 2) uniform texture2D rhs;

layout (set = 2, binding = 0) uniform FragmentPushConstants {
  // FIXME(naga) mat4x2 with std430 gets miscompiled on WebGL.
  // Hence, we use the equivalent layout vec4 [2];
  vec4 channels[2];
} u_platte;

mat4x2 channel_matrix(vec4 a, vec4 b) {
  return mat4x2(a.xy, a.zw, b.xy, b.zw);
}

void main() {
    vec4 basis = texture(sampler2D(rhs, texture_sampler), uv).rgba;

    ivec2 sz = textureSize(sampler2D(rhs, texture_sampler), 0);
    vec2 bias = 0.5 / vec2(sz);

    // FIXME(naga): see above
    mat4x2 mat = channel_matrix(u_platte.channels[0], u_platte.channels[1]);
    vec2 paletteuv = mat * basis + bias;

    f_color = texture(sampler2D(lhs, texture_sampler), paletteuv).rgba;
}
