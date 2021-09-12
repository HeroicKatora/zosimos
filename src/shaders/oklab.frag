#version 440
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform sampler texture_sampler;
layout (set = 1, binding = 1) uniform texture2D in_texture;

layout (set = 2, binding = 0) uniform FragmentColor {
    mat3x3 xyz_transform;
} u_fragmentColor;

// The canonical Oklab matrices, but GLSL constructs matrices column-wise.
// Therefore we will need to transpose before actual use.
const mat3x3 M1 = mat3x3(
        +0.8189330101, +0.0329845436, +0.0482003018,
        +0.3618667424, +0.9293118715, +0.2643662691,
        -0.1288597137, +0.0361456387, +0.6338517070
    );

const mat3x3 M2 = mat3x3(
        +0.2104542553, +1.9779984951, +0.0259040371,
        +0.7936177850, -2.4285922050, +0.7827717662,
        -0.0040720468, +0.4505937099, -0.8086757660
    );


#ifndef OKLAB_ENCODE_AS_MAIN
#define OKLAB_ENCODE_AS_MAIN oklab_encode
#endif
#ifndef OKLAB_DECODE_AS_MAIN
#define OKLAB_DECODE_AS_MAIN oklab_decode
#endif

void OKLAB_ENCODE_AS_MAIN() {
    // Assuming the input is some linear rgb space.
    const vec4 rgba = texture(sampler2D(in_texture, texture_sampler), uv);
    const vec3 xyz = u_fragmentColor.xyz_transform * rgba.rgb;

    // The OKLab transformation.
    const vec3 lms = M1 * xyz;
    // We can't use pow outright for negative components.
    const vec3 lms_star = pow(abs(lms), vec3(1.0 / 3.0)) * sign(lms);
    const vec3 Lab = M2 * lms_star;

    // Write this as our 'linear color' (preserve alpha).
    f_color = vec4(Lab, rgba.a);
}


void OKLAB_DECODE_AS_MAIN() {
    // Assuming the input is some linear rgb space.
    const vec4 lab_a = texture(sampler2D(in_texture, texture_sampler), uv);
    const vec3 Lab = lab_a.xyz;

    // The OKLab transformation.
    const vec3 lms_star = inverse(M2) * Lab;
    // Not using pow because that would be undefined for negative components.
    const vec3 lms = lms_star * lms_star * lms_star;
    const vec3 xyz = inverse(M1) * lms;

    // Write this as our 'linear color' (preserve alpha).
    const vec3 rgb = u_fragmentColor.xyz_transform * xyz;
    f_color = vec4(clamp(rgb, 0.0, 1.0), lab_a.a);
}
