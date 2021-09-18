#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform sampler texture_sampler;
layout (set = 1, binding = 1) uniform texture2D in_texture;

layout (set = 2, binding = 0) uniform Box3 {
    mat3x3 weights;
} box_params;

void main() {
    vec4 p00 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(0, 0));
    vec4 p01 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(0, 1));
    vec4 p02 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(0, 2));
    vec4 p10 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(1, 0));
    vec4 p11 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(1, 1));
    vec4 p12 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(1, 2));
    vec4 p20 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(2, 0));
    vec4 p21 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(2, 1));
    vec4 p22 = textureOffset(sampler2D(in_texture, texture_sampler), uv, ivec2(2, 2));

    // Remember: column-major initializer.
    mat3x3 c0 = mat3x3(
        p00.x, p01.x, p02.x,
        p10.x, p11.x, p12.x,
        p20.x, p21.x, p22.x
    );

    float x = dot(matrixCompMult(c0, box_params.weights) * vec3(1.0), vec3(1.0));
    f_color = vec4(vec3(x), 1.0);
}
