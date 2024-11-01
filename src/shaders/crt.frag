#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform sampler texture_sampler;
layout (set = 1, binding = 1) uniform texture2D in_texture;

layout (set = 2, binding = 0) uniform FragmentColor {
    uvec2 scale;
} u_fragmentParams;

vec2 stepMandelbrot(vec2 rf, vec2 c) {
    // At least some precision is nice.
    float real = dot(rf, vec2(rf.x, -rf.y));
    // (real, 2*rf.x*rf.y) + c
    return vec2(real + c.x, fma(2*rf.x, rf.y, c.y));
}

void main() {
    uvec2 screen_xy = uvec2(
        uint(uv.x * float(u_fragmentParams.scale.x)),
        uint(uv.y * float(u_fragmentParams.scale.y)));
    const vec4 rgba = texture(sampler2D(in_texture, texture_sampler), uv);
    
    vec4 muls = vec4(0.0, 0.0, 0.0, 1.0);

    uint bias = ((screen_xy.x / 3) % 2) * 3;
    uint cell = (screen_xy.y + bias) % 6;
    float off_center = float(cell) * float(5 - cell);

    muls[screen_xy.x % 3] = 0.16 * off_center;

/*
    for (int i = 0; i < 9; i++) {
        ivec2 off = ivec2(i % 3, i / 3);
    }
*/

    f_color =  rgba * muls;
}
