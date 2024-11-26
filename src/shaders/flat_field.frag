#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform sampler texture_sampler;
layout (set = 1, binding = 1) uniform texture2D in_texture;
layout (set = 1, binding = 2) uniform texture2D flat_texture;

layout (set = 2, binding = 0) uniform FragmentColor {
    float mean;
} u_fragmentParams;

vec2 stepMandelbrot(vec2 rf, vec2 c) {
    // At least some precision is nice.
    float real = dot(rf, vec2(rf.x, -rf.y));
    // (real, 2*rf.x*rf.y) + c
    return vec2(real + c.x, fma(2*rf.x, rf.y, c.y));
}

void main() {
    const vec4 rgba = texture(sampler2D(in_texture, texture_sampler), uv);

    const vec4 field = texture(sampler2D(flat_texture, texture_sampler), uv);
    const float relative_efficiency = u_fragmentParams.mean / field.x;

    f_color = vec4(rgba.xyz * relative_efficiency, rgba.w);
}
