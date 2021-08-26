#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform sampler texture_sampler;
layout (set = 1, binding = 1) uniform texture2D in_texture;

layout (set = 2, binding = 0, std140) uniform Matrix {
  mat3 rgb_matrix;
} color_matrix;

void main() {
    mat3 color_mat = mat3(color_matrix.rgb_matrix);
	
    vec4 rgba = texture(sampler2D(in_texture, texture_sampler), uv).rgba;
    f_color = vec4(color_mat * rgba.rgb, rgba.a);
}
