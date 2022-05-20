#version 440
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform sampler texture_sampler;
layout (set = 1, binding = 1) uniform texture2D in_texture;

layout (set = 2, binding = 0) uniform FragmentColor {
    vec4 whitepoint_xyz;
    mat3x3 xyz_transform;
} u_fragmentColor;

// The canonical CIECAM02 matrices, but GLSL constructs matrices column-wise.
const mat3x3 M_CAT02 = mat3x3(
         0.7328,-0.7036, 0.0030,
         0.4296, 1.6975, 0.0136,
        -0.1624, 0.0061, 0.9834
    );

const mat3x3 M_HPE = mat3x3(
         0.38971,-0.22981, 0.00000,
         0.68898, 1.18340, 0.00000,
        -0.07868, 0.04641, 1.00000
    );

vec3 srlab2_non_linearity(vec3);
vec3 srlab2_non_linearity_inv(vec3);

#ifndef SRLAB2_ENCODE_AS_MAIN
#define SRLAB2_ENCODE_AS_MAIN srlab2_encode
#endif
#ifndef SRLAB2_DECODE_AS_MAIN
#define SRLAB2_DECODE_AS_MAIN srlab2_decode
#endif

void SRLAB2_ENCODE_AS_MAIN() {
    const vec3 wp_rgb = vec3(1.0); // M_CAT02 * u_fragmentColor.whitepoint_xyz.xyz;

    // Assuming the input is some linear rgb space.
    const vec4 rgba = texture(sampler2D(in_texture, texture_sampler), uv);
    const vec3 xyz = u_fragmentColor.xyz_transform * rgba.rgb;

    // Correct van Kries whitepoint correction.
    const vec3 rgb_w = (M_CAT02 * xyz); //  / wp_rgb;

    // An in-cone-space non-linearity
    const vec3 lms = M_HPE * inverse(M_CAT02) * rgb_w;
    const vec3 xyz_e = inverse(M_HPE) * srlab2_non_linearity(lms);

    // Followed by the difference-scheme in xyz'
    f_color = vec4(
        vec3(
            xyz_e.y,
            (xyz_e.x - xyz_e.y) * 5.0 / 1.16,
            (xyz_e.z - xyz_e.y) * 2.0 / 1.16
        ),
        rgba.a
    );
}


void SRLAB2_DECODE_AS_MAIN() {
    const vec3 wp_rgb = M_CAT02 *  u_fragmentColor.whitepoint_xyz.xyz;

    // Assuming the input is some linear rgb space.
    const vec4 lab_a = texture(sampler2D(in_texture, texture_sampler), uv);
    const vec3 Lab = lab_a.xyz;

    // Undo difference scheme
    const vec3 xyz_e = vec3(
        Lab.y * 1.16  / 5.0 + Lab.x,
        Lab.x,
        Lab.z * 1.16  / 2.0 + Lab.x
    );
    // Undo non_linearity
    const vec3 lms = srlab2_non_linearity_inv(M_HPE * xyz_e);
    // Into whitepoint adapted rgb
    const vec3 rgb_w = M_CAT02 * inverse(M_HPE) * lms;
    // Undo whitepoint adaptation
    const vec3 xyz = inverse(M_CAT02) * (rgb_w * wp_rgb);

    // Write this as our 'linear color' (preserve alpha).
    const vec3 rgb = u_fragmentColor.xyz_transform * xyz;
    f_color = vec4(clamp(rgb, 0.0, 1.0), lab_a.a);
}

float srlab2_non_linearity_component(float v) {
    // 6**3 / 29**3
    if (abs(v) < 216.0 / 24389.0) {
        // Limited to 0.08 precisely
        return v * 24389.0 / 2700.0;
    } else {
        return 1.16 * pow(v, 1.0 / 3.0) - 0.16;
    }
}

vec3 srlab2_non_linearity(vec3 lms) {
    return vec3(
        srlab2_non_linearity_component(lms.x),
        srlab2_non_linearity_component(lms.y),
        srlab2_non_linearity_component(lms.z)
    );
}

float srlab2_non_linearity_inv_component(float v) {
    if (abs(v) < 0.08) {
        return v * 2700.0 / 24389.0;
    } else {
        const float vp = ((v + 0.16) / 1.16);
        return vp * vp * vp;
    }
}

vec3 srlab2_non_linearity_inv(vec3 lms) {
    return vec3(
        srlab2_non_linearity_inv_component(lms.x),
        srlab2_non_linearity_inv_component(lms.y),
        srlab2_non_linearity_inv_component(lms.z)
    );
}
