#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform sampler texture_sampler;
layout (set = 1, binding = 1) uniform texture2D in_texture;

layout (set = 2, binding = 0) uniform Box3 {
    mat3x3 weights;
} box_params;

float weighted_sum(mat3x3 w, mat3x3 c) {
    return dot(matrixCompMult(c, w) * vec3(1.0), vec3(1.0));
}

void main() {
    vec4 p00 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(-1, -1));
    vec4 p01 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(-1, 0));
    vec4 p02 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(-1, 1));
    vec4 p10 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(0, -1));
    vec4 p11 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(0, 0));
    vec4 p12 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(0, 1));
    vec4 p20 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(1, -1));
    vec4 p21 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(1, 0));
    vec4 p22 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(1, 1));

    // Remember: column-major initializer.
    mat3x3 c0 = mat3x3(
        p00.x, p01.x, p02.x,
        p10.x, p11.x, p12.x,
        p20.x, p21.x, p22.x
    );

    mat3x3 c1 = mat3x3(
        p00.y, p01.y, p02.y,
        p10.y, p11.y, p12.y,
        p20.y, p21.y, p22.y
    );

    mat3x3 c2 = mat3x3(
        p00.z, p01.z, p02.z,
        p10.z, p11.z, p12.z,
        p20.z, p21.z, p22.z
    );

    // Wait, shouldn't we have three sets of weights?
    float x = weighted_sum(c0, box_params.weights);
    float y = weighted_sum(c1, box_params.weights);
    float z = weighted_sum(c2, box_params.weights);

    f_color = vec4(vec3(x, y, z), 1.0);
}
