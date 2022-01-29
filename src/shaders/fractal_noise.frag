#version 450
#extension GL_EXT_scalar_block_layout : require

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0, std430) uniform FractalNoise {
    // Every image is split into a number of cells
    vec2 initial_scale;
    // Amplitude used for the first iteration
    float amplitude;
    // Damping of the amplitude for further iterations
    float damping;
    // The number of iterations to add
    uint num_octaves;
} u_fragmentParams;

// From https://jcgt.org/published/0009/03/02/paper.pdf
uvec4 pcg4d(uvec4 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    v = v ^ (v >> 16u);
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    return v;
}

vec4 hash(uvec2 v) {
    const float MAX = float(0xFFFFFFFFu);
    uvec4 hashed = pcg4d(uvec4(v, 0, 0));
    return vec4(hashed) / MAX;
}

vec4 noise(vec3 x) {
    vec2 pt = x.xy * u_fragmentParams.initial_scale;
    vec2 f = fract(pt.xy);
    uvec2 seed = uvec2(ivec2(floor(pt.xy)));

    // Four corners in 2D of a tile
    vec4 a = hash(seed);
    vec4 b = hash(seed + uvec2(1, 0));
    vec4 c = hash(seed + uvec2(0, 1));
    vec4 d = hash(seed + uvec2(1, 1));

    // smoothstep with some common subexpressions optimized away.
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

vec3 next_point(vec3 x) {
    // Rotate to reduce axial bias
    const mat3 rot = 2 *
        mat3( cos(0.5), sin(0.5), 1,
                -sin(0.5), cos(0.5), 1,
                0, 0, 1);
    return rot * x;
}

vec4 fbm(vec2 x) {
    float a = u_fragmentParams.amplitude;
    float damping = u_fragmentParams.damping;

    vec3 it = vec3(x, 1.0);
    vec4 v = vec4(0.0);
    for (int i = 0; i < u_fragmentParams.num_octaves; ++i) {
        v += a * noise(it);

        it = next_point(it);
        a *= damping;
    }
    return v;
}

void main() {
    f_color = fbm(uv);
}
