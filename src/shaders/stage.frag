#version 450
/** This is a special shader to convert to/from color spaces and texture
 * formats that are not natively supported. This works by introducing a staging
 * texture that is in the correct byte representation of the supposed format
 * but whose texel format is some chosen, supported format with the same byte
 * layout. This is then rendered to and from the final texture that is in
 * linear RGB space (or some other supported reference space).
 *
 * For example, consider Oklab, a recent perceptual color space combining good
 * aspects from HCL but with transfer functions that optimized towards
 * uniformity as measured in CIEDE2000. This is (of course) not yet supported
 * natively and even less so if we don't have any available native GL features.
 * Instead, when we need it for painting we load it from the staging texture,
 * where it is in Lab form, as a linear RGB texture. Using this fragment shader
 * you're viewing we then paint this onto an intermediate texture and in this
 * step we calculate the actual linear RGB values.
 *
 * In other conversions we might load a texture as u32 gray scale and demux
 * individual components. Note that this also allows us to do our own
 * quantization. In this way our calculation can happen in full floating point
 * precision but the result appears as if it performed in u8 or whatever the
 * input precision might be.
 *
 * Since the conversion share some amount of code this is a single source file
 * with multiple entry points.
 */
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0) uniform texture2D in_texture;
layout (set = 1, binding = 1) uniform sampler texture_sampler;

/** Forward declarations.
 *
 * For all signals in these functions we assume normalized values.
 */

float transfer_oe_bt709(float val);
float transfer_eo_bt709(float val);

float transfer_oe_bt470m(float val);
float transfer_eo_bt470m(float val);

float transfer_oe_bt601(float val);
float transfer_eo_bt601(float val);

float transfer_oe_smpte240(float val);
float transfer_eo_smpte240(float val);

float transfer_oe_srgb(float val);
float transfer_eo_srgb(float val);

float transfer_oe_bt2020_10b(float val);
float transfer_eo_bt2020_10b(float val);

// Used in Bt.2100, this differentiates between scene, electrical, display light.
float transfer_eo_smpte2084(float val);
float transfer_eo_inv_smpte2084(float val);
float transfer_scene_display_smpte2084(float val);
float transfer_display_scene_smpte2084(float val);
float transfer_oe_smpte2084(float val);
float transfer_oe_inv_smpte2084(float val);

// Used Reference: BT.709-6, Section 1.2
float transfer_oe_bt709(float val) {
  // TODO: is there a numerically better way?
  if (val >= 0.018)
    return 1.099 * pow(val, 0.45) - 0.099;
  else
    return 4.500 * val;
}

// Used Reference: BT.709-6, Section 1.2, inverted.
float transfer_eo_bt709(float val) {
  // TODO: is there a numerically better way?
  if (val >= transfer_oe_bt709(0.018))
    return pow((val + 0.099) / 1.099, 1.0 / 0.45);
  else
    return val / 4.500;
}

// Used Reference: BT.470, Table 1, Item 5
float transfer_oe_bt470m(float val) {
  return pow(val, 1.0 / 2.200);
}

// Used Reference: BT.470, Table 1, Item 5
float transfer_eo_bt470m(float val) {
  return pow(val, 2.200); 
}

// Used Reference: BT.601-7, Section 2.6.4
float transfer_oe_bt601(float val) {
  return transfer_eo_bt709(val);
}

// Used Reference: BT.601-7, Section 2.6.4
float transfer_eo_bt601(float val) {
  return transfer_oe_bt709(val);
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-smpte-240m-v4l2-colorspace-smpte240m
float transfer_oe_smpte240(float val) {
  if (val < 0.0228)
    return 4.0 * val;
  else
    return 1.1115 * pow(val, 0.45);
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-smpte-240m-v4l2-colorspace-smpte240m
float transfer_eo_smpte240(float val) {
  if (val < 0.0913)
    return val / 4.0;
  else
    return pow(val / 1.1115, 1.0 / 0.045);
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#
// Transfer function. Note that negative values for L are only used by the Y’CbCr conversion.
float transfer_oe_srgb(float val) {
  if (val < -0.0031308)
    return -1.055 * pow(-val, 1.0 / 2.4) + 0.055;
  else if (val <= 0.04045)
    return val * 12.92;
  else
    return 1.055 * pow(val, 1.0 / 2.4) - 0.055;
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html
float transfer_eo_srgb(float val) {
  if (val < -0.04045)
    return -pow((-val + 0.055) / 1.055, 2.4);
  else if (val <= 0.04045)
    return val / 12.92;
  else
    return pow((val + 0.055) / 1.055, 2.4);
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-bt-2020-v4l2-colorspace-bt2020
float transfer_oe_bt2020_10b(float val) {
  return transfer_oe_bt709(val);
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-bt-2020-v4l2-colorspace-bt2020
float transfer_eo_bt2020_10b(float val) {
  return transfer_eo_bt709(val);
}


// Used Reference: BT.2100-2, Table 4, Reference PQ EOTF
#define SMPTE2084_M1 (2610.0/16384.0)
#define SMPTE2084_M2 (2523.0/4096.0)
#define SMPTE2084_C1 (3424.0/4096.0)
#define SMPTE2084_C2 (2413.0/128.0)
#define SMPTE2084_C3 (2392.0/128.0)

// Used Reference: BT.2100-2, Table 4, Reference PQ EOTF
// Note: the output is _display_ color value Y and _not_ scene luminance.
float transfer_eo_smpte2084(float val) {
  float N = pow(val, 1.0 / SMPTE2084_M2);
  float nom = max(N - SMPTE2084_C1, 0.0);
  float denom = SMPTE2084_C2 - SMPTE2084_C3 * N;
  return pow(nom / denom, 1.0 / SMPTE2084_M1);
}
// Used Reference: BT.2100-2, Table 4, Reference PQ OETF
// Note: the input is _display_ color value Y and _not_ scene luminance.
float transfer_eo_inv_smpte2084(float val) {
  float Y = pow(val, SMPTE2084_M1);
  float nom = SMPTE2084_C1 + SMPTE2084_C2 * Y;
  float denom = SMPTE2084_C3 * Y + 1.0;
  return pow(nom / denom, SMPTE2084_M2);
}

// Used Reference: BT.2100-2, Table 4, Reference PQ OOTF
// Used Reference: Python `colour science`: https://github.com/colour-science/colour/blob/a196f9536c44e2101cde53446550d64303c0ab46/colour/models/rgb/transfer_functions/itur_bt_2100.py#L276
// IMPORTANT: we map to a normalized linear color range Y, and _not_ to display luminance F_D.
float transfer_scene_display_smpte2084(float val) {
  float e_prime = transfer_oe_bt709(59.5208 * val);
  return pow(e_prime, 2.4) / 100.0;
}

// Used Reference: BT.2100-2, Table 4, Reference PQ OOTF
float transfer_display_scene_smpte2084(float val) {
  float e_prime = pow(val * 100.0, 1.0 / 2.4);
  return transfer_eo_bt709(e_prime) / 59.5208;
}

float transfer_oe_smpte2084(float val) {
  return transfer_eo_inv_smpte2084(transfer_scene_display_smpte2084(val));
}
float transfer_oe_inv_smpte2084(float val) {
  return transfer_display_scene_smpte2084(transfer_eo_smpte2084(val));
}

// TODO: https://github.com/colour-science/colour/blob/a196f9536c44e2101cde53446550d64303c0ab46/colour/models/rgb/transfer_functions/arib_std_b67.py#L108
vec3 transfer_scene_display_bt2100hlg(vec3 rgb) {
  return vec3(0.0);
}

void main() {}
