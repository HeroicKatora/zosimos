#version 450

/** Hacky way because shaderc only supports `main` for the name of the entry point.
 *
 * Hence we rename the current entry point to `main` through a macro in the
 * compiler options while falling back to a non-main name... This is stupid and
 * shaderc must have been written by monkeys with type writers. Especially its
 * documentation which doesn't mention this detail at all apart from an internal
 * method.
 */
#if (defined(DECODE_RGBA32UI_AS_MAIN) || defined(DECODE_RGBA16UI_AS_MAIN) || defined(DECODE_R32UI_AS_MAIN) || defined(DECODE_R16UI_AS_MAIN) || defined(DECODE_R8UI_AS_MAIN))
#define STAGE_DECODE
#else
#define STAGE_ENCODE
#endif

#ifndef DECODE_R8UI_AS_MAIN
#define DECODE_R8UI_AS_MAIN decode_r8ui
#endif
#ifndef ENCODE_R8UI_AS_MAIN
#define ENCODE_R8UI_AS_MAIN encode_r8ui
#endif
#ifndef DECODE_R16UI_AS_MAIN
#define DECODE_R16UI_AS_MAIN decode_r16ui
#endif
#ifndef ENCODE_R16UI_AS_MAIN
#define ENCODE_R16UI_AS_MAIN encode_r16ui
#endif
#ifndef DECODE_R32UI_AS_MAIN
#define DECODE_R32UI_AS_MAIN decode_r32ui
#endif
#ifndef ENCODE_R32UI_AS_MAIN
#define ENCODE_R32UI_AS_MAIN encode_r32ui
#endif
#ifndef DECODE_RGBA16UI_AS_MAIN
#define DECODE_RGBA16UI_AS_MAIN decode_rgba16ui
#endif
#ifndef ENCODE_RGBA16UI_AS_MAIN
#define ENCODE_RGBA16UI_AS_MAIN encode_rgba16ui
#endif
#ifndef DECODE_RGBA32UI_AS_MAIN
#define DECODE_RGBA32UI_AS_MAIN decode_rgba32ui
#endif
#ifndef ENCODE_RGBA32UI_AS_MAIN
#define ENCODE_RGBA32UI_AS_MAIN encode_rgba32ui
#endif
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
/* Hack on hack: we define two output colors (float output for decode, int for encode)
 * Each entry point writes to a fictional variable that is either a local or
 * the true shader output.
 */
#ifdef STAGE_DECODE
layout (location = 0) out vec4 f_color;
#define decode_color f_color
#define encode_color fake_local_noop
#else
layout (location = 0) out uvec4 f_color;
#define decode_color fake_local_noop
#define encode_color f_color
#endif

/* Not all those bindings will be bound!
 */
layout (set = 1, binding = 0) uniform utexture2D image_r8ui;
layout (set = 1, binding = 1) uniform utexture2D image_r16ui;
layout (set = 1, binding = 2) uniform utexture2D image_r32ui;
layout (set = 1, binding = 3) uniform utexture2D image_rgba16ui;
layout (set = 1, binding = 4) uniform utexture2D image_rgba32ui;

/** For encoding, this is the input frame buffer.
 */
layout (set = 1, binding = 32) uniform texture2D in_texture;
layout (set = 1, binding = 33) uniform sampler texture_sampler;
layout (set = 1, binding = 34) uniform sampler read_sampler;

layout (set = 2, binding = 0, std140) uniform Parameter {
  uvec4 space;
} parameter;

// FIXME: this could and should be an auto-generated header with cbindgen

const uint TRANSFER_Bt709 = 0;
const uint TRANSFER_Bt470M = 1;
const uint TRANSFER_Bt601 = 2;
const uint TRANSFER_Smpte240 = 3;
const uint TRANSFER_Linear = 4;
const uint TRANSFER_Srgb = 5;
const uint TRANSFER_Bt2020_10bit = 6;
const uint TRANSFER_Bt2020_12bit = 7;
const uint TRANSFER_Smpte2084 = 8;
const uint TRANSFER_Bt2100Pq = 9;
const uint TRANSFER_Bt2100Hlg = 10;
const uint TRANSFER_LinearScene = 11;
const uint TRANSFER_LabLch = 0x100;

uint get_transfer() {
  return parameter.space.x;
}

const uint SAMPLE_PARTS_A = 0;
const uint SAMPLE_PARTS_R = 1;
const uint SAMPLE_PARTS_G = 2;
const uint SAMPLE_PARTS_B = 3;
const uint SAMPLE_PARTS_Luma = 4;
const uint SAMPLE_PARTS_LumaA = 5;
const uint SAMPLE_PARTS_Rgb = 6;
const uint SAMPLE_PARTS_Bgr = 7;
const uint SAMPLE_PARTS_Rgba = 8;
const uint SAMPLE_PARTS_Rgbx = 9;
const uint SAMPLE_PARTS_Bgra = 10;
const uint SAMPLE_PARTS_Bgrx = 11;
const uint SAMPLE_PARTS_Argb = 12;
const uint SAMPLE_PARTS_Xrgb = 13;
const uint SAMPLE_PARTS_Abgr = 14;
const uint SAMPLE_PARTS_Xbgr = 15;
const uint SAMPLE_PARTS_Yuv = 16;
const uint SAMPLE_PARTS_Lab = 17;
const uint SAMPLE_PARTS_LabA = 18;
const uint SAMPLE_PARTS_LCh = 19;
const uint SAMPLE_PARTS_LChA = 20;

uint get_sample_parts() {
  return parameter.space.y;
}

const uint SAMPLE_BITS_Int8 = 0;
const uint SAMPLE_BITS_Int332 = 1;
const uint SAMPLE_BITS_Int233 = 2;
const uint SAMPLE_BITS_Int16 = 3;
const uint SAMPLE_BITS_Int4x4 = 4;
const uint SAMPLE_BITS_Inti444 = 5;
const uint SAMPLE_BITS_Int444i = 6;
const uint SAMPLE_BITS_Int565 = 7;
const uint SAMPLE_BITS_Int8x2 = 8;
const uint SAMPLE_BITS_Int8x3 = 9;
const uint SAMPLE_BITS_Int8x4 = 10;
const uint SAMPLE_BITS_Int16x2 = 11;
const uint SAMPLE_BITS_Int16x3 = 12;
const uint SAMPLE_BITS_Int16x4 = 13;
const uint SAMPLE_BITS_Int1010102 = 14;
const uint SAMPLE_BITS_Int2101010 = 15;
const uint SAMPLE_BITS_Int101010i = 16;
const uint SAMPLE_BITS_Inti101010 = 17;
const uint SAMPLE_BITS_Float16x4 = 18;
const uint SAMPLE_BITS_Float32x4 = 19;

/** Define the constants for division/multiplication for bits.
 * This ensures we don't typo the exact constant.
 */
const float BITS2 = 3.;
const uint MASK2 = 3;
const float BITS3 = 7.;
const uint MASK3 = 7;
const float BITS4 = 15.;
const uint MASK4 = 15;
const float BITS5 = 31.;
const uint MASK5 = 31;
const float BITS6 = 63.;
const uint MASK6 = 63;
const float BITS8 = 255.;
const uint MASK8 = 255;
const float BITS10 = 1023.;
const uint MASK10 = 1023;
const float BITS16 = 65535.;
const uint MASK16 = 65535;

// Failure indicator (for debugging)
const vec4 BIT_DECODE_FAIL = vec4(1.0, 0.0, 0.0, 1.0);
const uint BIT_ENCODE_FAIL = 0x55445544;

uint get_sample_bits() {
  return parameter.space.z;
}

/** How many texels each call is responsible for (horizontally).
 */
uint get_horizontal_workload() {
  return max(parameter.space.a, 1) & 0xff;
}

/** The 'position' in the input texel to retrieve one of the actual texels of the input image.
 *
 * Since, on WebGPU, we are only allowed to Load/Store at 32-bit granularity,
 * we must aggregate multiple texels of the underlying image binary format into
 * one logical texel in the GL binding.
 *
 * During decoding, our gl_FragCoord points at the _pixel_ index of an output
 * image. During encoding, the gl_FragCoord points at a _texel_ index in the
 * granular, artificial stage texture format.
 *
 * This method derives the texel coord of the pixel within the stage texture.
 */
ivec2 decodeStageTexelCoord() {
// FIXME: block texel?
  return ivec2(gl_FragCoord) / ivec2(get_horizontal_workload(), 1);
}

/** Derives the coord of the image texel within the stage texture texel.
 * See `decodeStageTexelCoord` for explanation.
 */
uint decodeSubtexelCoord() {
  return ivec2(gl_FragCoord).x % get_horizontal_workload();
}

/** Derives the base pixel coord of the image stored to this stage texture texel.
 */
ivec2 encodePixelCoord() {
  return ivec2(gl_FragCoord) * ivec2(get_horizontal_workload(), 1);
}

/** Forward declarations.
 *
 * For all signals in transfer functions we assume normalized values.
 */

vec4 demux_uint(uint, uint kind);
uint mux_uint(vec4, uint kind);

vec4 parts_normalize(vec4, uint);
vec4 parts_denormalize(vec4, uint);

vec4 parts_transfer(vec4, uint);
vec4 parts_untransfer(vec4, uint);

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

vec3 transfer_lab_to_lch(vec3);
vec3 transfer_lch_to_lab(vec3);

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
    return 1.1115 * pow(val, 0.45) - 0.1115;
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-smpte-240m-v4l2-colorspace-smpte240m
float transfer_eo_smpte240(float val) {
  if (val < 0.0913)
    return val / 4.0;
  else
    return pow((val - 0.1115) / 1.1115, 1.0 / 0.45);
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#
// Transfer function. Note that negative values for L are only used by the Y’CbCr conversion.
float transfer_oe_srgb(float val) {
  if (val < -0.0031308)
    return -1.055 * pow(-val, 1.0 / 2.4) + 0.055;
  else if (val <= 0.0031308)
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

vec3 transfer_lab_to_lch(vec3 lab) {
  float c = length(lab.yz);
  // Angle but scaled to [0; 1]
  float h = (degrees(atan(lab.z, lab.y)) / 360.0) + 0.5;
  return vec3(lab.x, c, h);
}

vec3 transfer_lch_to_lab(vec3 lch) {
  float angle = radians(360.0 * (lch.z - 0.5));
  return vec3(lch.x, lch.y*cos(angle), lch.y*sin(angle));
}

/** All decode methods work in several stages:
 *
 * 1. Demux the bit-encoded components into a vector.
 * 2. Reorder the components into a normalized form for the color type.
 * 3. Apply transfer function (and primary transform such as YUV).
 * 4. We now hold a vector of floating point linear color encoding, write it.
 *
 * The encoding works the other way around. Note that there are some invalid
 * combinations (Bits::Int332 and Parts::A for example) and it is expected that
 * the calling layer handles those.
 */

void DECODE_R8UI_AS_MAIN() {
  uint num = texelFetch(usampler2D(image_r8ui, read_sampler), decodeStageTexelCoord(), 0).x;
  uint work = (num >> 8*decodeSubtexelCoord()) & 0xff;
  vec4 components = demux_uint(work, get_sample_bits());

  // FIXME: YUV transform and accurate YUV transform.
  vec4 electrical = parts_normalize(components, get_sample_parts());
  vec4 primaries = parts_untransfer(electrical, get_transfer());

  vec4 fake_local_noop = vec4(0);
  decode_color = primaries;
}

void ENCODE_R8UI_AS_MAIN() {
  ivec2 baseCoord = encodePixelCoord();
  uint num = 0;
  for (int i = 0; i < get_horizontal_workload(); i++) {
    ivec2 pixelCoord = baseCoord + ivec2(i, 0);
    vec4 primaries = texelFetch(sampler2D(in_texture, texture_sampler), pixelCoord, 0);

    vec4 electrical = parts_transfer(primaries, get_transfer());
    // FIXME: YUV transform and accurate YUV transform.
    vec4 components = parts_denormalize(electrical, get_sample_parts());

    uint texelNum = mux_uint(clamp(components, 0.0, 1.0), get_sample_bits());
    num |= (texelNum & 0xff) << (8*i);
  }

  uvec4 fake_local_noop = uvec4(0);
  encode_color = uvec4(num);
}

void DECODE_R16UI_AS_MAIN() {
  uint num = texelFetch(usampler2D(image_r16ui, read_sampler), decodeStageTexelCoord(), 0).x;
  uint work = (num >> 16*decodeSubtexelCoord()) & 0xffff;
  vec4 components = demux_uint(work, get_sample_bits());

  // FIXME: YUV transform and accurate YUV transform.
  vec4 electrical = parts_normalize(components, get_sample_parts());
  vec4 primaries = parts_untransfer(electrical, get_transfer());

  vec4 fake_local_noop = vec4(0);
  decode_color = primaries;
}

void ENCODE_R16UI_AS_MAIN() {
  ivec2 baseCoord = encodePixelCoord();
  uint num = 0;
  for (int i = 0; i < get_horizontal_workload(); i++) {
    ivec2 pixelCoord = baseCoord + ivec2(i, 0);
    vec4 primaries = texelFetch(sampler2D(in_texture, texture_sampler), pixelCoord, 0);

    vec4 electrical = parts_transfer(primaries, get_transfer());
    // FIXME: YUV transform and accurate YUV transform.
    vec4 components = parts_denormalize(electrical, get_sample_parts());

    uint texelNum = mux_uint(clamp(components, 0.0, 1.0), get_sample_bits());
    num |= (texelNum & 0xffff) << (16*i);
  }

  uvec4 fake_local_noop = uvec4(0);
  encode_color = uvec4(num);
}

void DECODE_R32UI_AS_MAIN() {
  uint num = texelFetch(usampler2D(image_r32ui, read_sampler), decodeStageTexelCoord(), 0).x;
  vec4 components = demux_uint(num, get_sample_bits());

  // FIXME: YUV transform and accurate YUV transform.
  vec4 electrical = parts_normalize(components, get_sample_parts());
  vec4 primaries = parts_untransfer(electrical, get_transfer());

  vec4 fake_local_noop = vec4(0);
  decode_color = primaries;
}

void ENCODE_R32UI_AS_MAIN() {
  vec4 primaries = texture(sampler2D(in_texture, texture_sampler), uv).rgba;

  vec4 electrical = parts_transfer(primaries, get_transfer());
  // FIXME: YUV transform and accurate YUV transform.
  vec4 components = parts_denormalize(electrical, get_sample_parts());

  uint num = mux_uint(clamp(components, 0.0, 1.0), get_sample_bits());
  uvec4 fake_local_noop = uvec4(0);
  encode_color = uvec4(num);
}

// The bit decoding used by 8bit, 16bit, 32bit staging.
// Returns the parts in a canonical order:
// - 1 part: (x, 0., 0., 1.)
// - 2 parts: (x, 0., 0., y)
// - 3 parts: (x, y, z, 1.)
// - 4 parts: (x, y, z, a)
vec4 demux_uint(uint num, uint kind) {
  switch (kind) {
  case SAMPLE_BITS_Int8:
    return vec4(num) / BITS8;
  case SAMPLE_BITS_Int332:
    return vec4(num & MASK2, (num >> 2) & MASK3, num >> 5, 1.0) / vec4(BITS2, BITS3, BITS3, 1.0);
  case SAMPLE_BITS_Int233:
    return vec4(num & MASK3, (num >> 3) & MASK3, num >> 6, 1.0) / vec4(BITS3, BITS3, BITS2, 1.0);
  case SAMPLE_BITS_Int16:
    return vec4(num) / BITS16;
  case SAMPLE_BITS_Int4x4:
    return vec4(num & MASK4, (num >> 4) & MASK4, (num >> 8) & MASK4, num >> 12) / BITS4;
  case SAMPLE_BITS_Inti444:
    return vec4(num & MASK4, (num >> 4) & MASK4, (num >> 8) & MASK4, MASK4) / BITS4;
  case SAMPLE_BITS_Int444i:
    return vec4((num >> 4) & MASK4, (num >> 9) & MASK4, (num >> 12) & MASK4, MASK4) / BITS4;
  case SAMPLE_BITS_Int565:
    return vec4(num & MASK5, (num >> 5) & MASK6, num >> 11, 1.0) / vec4(BITS5, BITS6, BITS5, 1.0);
  case SAMPLE_BITS_Int8x2:
    return vec4(num & MASK8, 0., 0., (num >> 8) & MASK8) / BITS8;
  case SAMPLE_BITS_Int8x3:
    return vec4(num & 0xff, (num >> 8) & 0xff, (num >> 16) & 0xff, BITS8) / BITS8;
  case SAMPLE_BITS_Int8x4:
    return vec4(num & 0xff, (num >> 8) & 0xff, (num >> 16) & 0xff, num >> 24) / BITS8;
  case SAMPLE_BITS_Int16x2:
    return vec4(num & MASK16, (num >> 16) & MASK16, 0, BITS16) / BITS16;
  case SAMPLE_BITS_Int16x3:
  case SAMPLE_BITS_Int16x4:
    // FAILURE case, above 32-bits.
    return BIT_DECODE_FAIL;
  case SAMPLE_BITS_Int1010102:
    return vec4(num & MASK2, (num >> 2) & MASK10, (num >> 12) & MASK10, num >> 22)
      / vec4(BITS2, BITS10, BITS10, BITS10);
  case SAMPLE_BITS_Int2101010:
    return vec4(num & MASK10, (num >> 10) & MASK10, (num >> 20) & MASK10, num >> 30)
      / vec4(BITS10, BITS10, BITS10, BITS2);
  case SAMPLE_BITS_Int101010i:
    return vec4((num >> 2) & MASK10, (num >> 12) & MASK10, num >> 22, 1.0)
      / vec4(BITS10, BITS10, BITS10, 1.0);
  case SAMPLE_BITS_Inti101010:
    return vec4(num & MASK10, (num >> 10) & MASK10, (num >> 20) & MASK10, 1.0)
      / vec4(BITS10, BITS10, BITS10, 1.0);
  case SAMPLE_BITS_Float16x4:
  case SAMPLE_BITS_Float32x4:
    // FAILURE case, above 32-bits.
    return BIT_DECODE_FAIL;
  }
  // Oops, we missed some cases.
  return BIT_DECODE_FAIL;
}

// The bit encoding used by 8bit, 16bit, 32bit staging.
// Reverses demux_uint.
// Defines a bunch of macros to make this easier.
// Each position takes a number of bits to skip (sn) and bits to quantize to (bn).
// The inputs are expected in the canonical positions according to demux_uint
// and will be put in the low-order bits of the output.
uint mux_uint(vec4 c, uint kind) {
#define STEALTH_MUX_UINT1(s0, b0) uint(c.x * BITS##b0)
#define STEALTH_MUX_UINT2(s0, b0, s1, b1) ((uint(c.x * BITS##b0) << s0) + (uint(c.w * BITS##b1) << (s0+b0+s1)))
#define STEALTH_MUX_UINT3(s0, b0, s1, b1, s2, b2) ((uint(c.x * BITS##b0) << s0) + (uint(c.y * BITS##b1) << (s0+b0+s1))+ (uint(c.z * BITS##b2) << (s0+b0+s1+b1+s2)))
#define STEALTH_MUX_UINT4(s0, b0, s1, b1, s2, b2, s3, b3) ((uint(c.x * BITS##b0) << s0) + (uint(c.y * BITS##b1) << (s0+b0+s1))+ (uint(c.z * BITS##b2) << (s0+b0+s1+b1+s2))+ (uint(c.w * BITS##b3) << (s0+b0+s1+b1+s2+b2+s3)))
  switch (kind) {
  case SAMPLE_BITS_Int8:
    return STEALTH_MUX_UINT1(0, 8);
  case SAMPLE_BITS_Int332:
    return STEALTH_MUX_UINT3(0, 2, 0, 3, 0, 3);
  case SAMPLE_BITS_Int233:
    return STEALTH_MUX_UINT3(0, 3, 0, 3, 0, 2);
  case SAMPLE_BITS_Int16:
    return STEALTH_MUX_UINT1(0, 16);
  case SAMPLE_BITS_Int4x4:
    return STEALTH_MUX_UINT4(0, 4, 0, 4, 0, 4, 0, 4);
  case SAMPLE_BITS_Inti444:
    return STEALTH_MUX_UINT3(0, 4, 0, 4, 0, 4);
  case SAMPLE_BITS_Int444i:
    return STEALTH_MUX_UINT3(4, 4, 0, 4, 0, 4);
  case SAMPLE_BITS_Int565:
    return STEALTH_MUX_UINT3(0, 5, 0, 6, 0, 5);
  case SAMPLE_BITS_Int8x2:
    return STEALTH_MUX_UINT2(0, 8, 0, 8);
  case SAMPLE_BITS_Int8x3:
    return STEALTH_MUX_UINT3(0, 8, 0, 8, 0, 8);
  case SAMPLE_BITS_Int8x4:
    return STEALTH_MUX_UINT4(0, 8, 0, 8, 0, 8, 0, 8);
  case SAMPLE_BITS_Int16x2:
    return STEALTH_MUX_UINT2(0, 16, 0, 16);
  case SAMPLE_BITS_Int16x3:
  case SAMPLE_BITS_Int16x4:
    return BIT_ENCODE_FAIL;
  case SAMPLE_BITS_Int1010102:
    return STEALTH_MUX_UINT4(0, 2, 0, 10, 0, 10, 0, 10);
  case SAMPLE_BITS_Int2101010:
    return STEALTH_MUX_UINT4(0, 10, 0, 10, 0, 10, 0, 2);
  case SAMPLE_BITS_Int101010i:
    return STEALTH_MUX_UINT3(2, 10, 0, 10, 0, 10);
  case SAMPLE_BITS_Inti101010:
    return STEALTH_MUX_UINT3(0, 10, 0, 10, 0, 10);
  case SAMPLE_BITS_Float16x4:
  case SAMPLE_BITS_Float32x4:
    return BIT_ENCODE_FAIL;
  // FIXME: other bits.
  }
  return BIT_ENCODE_FAIL;
#undef STEALTH_MUX_UINT1
#undef STEALTH_MUX_UINT2
#undef STEALTH_MUX_UINT3
#undef STEALTH_MUX_UINT4
}

// Swap the parts into the canonical location for the color representation.
// The order of channels in the inputs depends on the channel count, and only
// on the input count, as normalized by the used demux_* method.
//
// - The alpha channel, if used, is always in a otherwise `1.0`.
// - Unused channels are generally zero-filled.
// - rgb parts will be assigned to `.rgb`.
// - XYZ observer will be assigned to `rgb`.
// - xyY triplet will be assigned to `bgr`.
// - LMS cone responses will be assigned to `rgb`.
// - La*b* will be assigned to `rgb`
vec4 parts_normalize(vec4 components, uint parts) {
  switch (parts) {
  case SAMPLE_PARTS_A:
    return vec4(0.0, 0.0, 0.0, components.x);
  case SAMPLE_PARTS_R:
    return vec4(components.x, 0.0, 0.0, 1.0);
  case SAMPLE_PARTS_G:
    return vec4(0.0, components.x, 0.0, 1.0);
  case SAMPLE_PARTS_B:
    return vec4(0.0, 0.0, components.x, 1.0);
  case SAMPLE_PARTS_Luma:
    return vec4(vec3(components.x), 1.0);
  case SAMPLE_PARTS_LumaA:
    return vec4(vec3(components.x), components.w);
  case SAMPLE_PARTS_Rgb:
  case SAMPLE_PARTS_Rgbx:
    return vec4(components.xyz, 1.0);
  case SAMPLE_PARTS_Bgr:
  case SAMPLE_PARTS_Bgrx:
    return vec4(components.zyx, 1.0);
  case SAMPLE_PARTS_Rgba:
    return components.xyzw;
  case SAMPLE_PARTS_Bgra:
    return components.xyzw;
  case SAMPLE_PARTS_Argb:
    return components.yzwx;
  case SAMPLE_PARTS_Abgr:
    return components.wzyx;
  case SAMPLE_PARTS_Xrgb:
    return vec4(components.yzw, 1.0);
  case SAMPLE_PARTS_Xbgr:
    return vec4(components.wzy, 1.0);
  // HACK(naga-1403):
  // this could be unified with the branch below but Naga has a bug.
  case SAMPLE_PARTS_Lab:
    return vec4(components.xyz, 1.0);
  case SAMPLE_PARTS_LCh:
    return vec4(components.xyz, 1.0);
  // HACK(naga-1403)
  case SAMPLE_PARTS_LabA:
    return components.xyzw;
  case SAMPLE_PARTS_LChA:
    return components.xyzw;
  }
  return BIT_DECODE_FAIL;
}

// Invert parts_normalize.
// Moves the components to the location expected by the count of channels.
vec4 parts_denormalize(vec4 c, uint parts) {
  switch (parts) {
  case SAMPLE_PARTS_Rgba:
    return c.xyzw;
  case SAMPLE_PARTS_A:
    return vec4(c.w, 0.0, 0.0, 1.0);
  case SAMPLE_PARTS_R:
    return vec4(c.x, 0.0, 0.0, 1.0);
  case SAMPLE_PARTS_G:
    return vec4(c.y, 0.0, 0.0, 1.0);
  case SAMPLE_PARTS_B:
    return vec4(c.z, 0.0, 0.0, 1.0);
  case SAMPLE_PARTS_Luma:
    return vec4(vec3(c.x), 1.0);
  case SAMPLE_PARTS_LumaA:
    return vec4(vec3(c.x), c.w);
  case SAMPLE_PARTS_Rgb:
  case SAMPLE_PARTS_Rgbx:
    return vec4(c.rgb, 1.0);
  case SAMPLE_PARTS_Bgr:
  case SAMPLE_PARTS_Bgrx:
    return vec4(c.bgr, 1.0);
  case SAMPLE_PARTS_Bgra:
    return c.bgra;
  case SAMPLE_PARTS_Argb:
    return c.argb;
  case SAMPLE_PARTS_Abgr:
    return c.abgr;
  case SAMPLE_PARTS_Xrgb:
    return vec4(1.0, c.rgb);
  case SAMPLE_PARTS_Xbgr:
    return vec4(1.0, c.bgr);
  // HACK(naga-1403)
  case SAMPLE_PARTS_Lab:
    return vec4(c.xyz, 1.0);
  case SAMPLE_PARTS_LCh:
    return vec4(c.xyz, 1.0);
  // HACK(naga-1403)
  case SAMPLE_PARTS_LabA:
    return c.xyzw;
  case SAMPLE_PARTS_LChA:
    return c.xyzw;
  }
  return c.xyzw;
}

// HACK(naga-1403) not sure if anything here is affected.
vec4 parts_transfer(vec4 linear, uint fnk) {
#define TRANSFER_WITH_XYZ(E, FN) vec4(FN(E.x), FN(E.y), FN(E.z), E.a)
  switch (fnk) {
  case TRANSFER_Bt709:
  return TRANSFER_WITH_XYZ(linear, transfer_oe_bt709);
  case TRANSFER_Bt470M:
  return TRANSFER_WITH_XYZ(linear, transfer_oe_bt470m);
  case TRANSFER_Bt601:
  return TRANSFER_WITH_XYZ(linear, transfer_oe_bt601);
  case TRANSFER_Smpte240:
  return TRANSFER_WITH_XYZ(linear, transfer_oe_smpte240);
  case TRANSFER_Linear:
  return linear;
  case TRANSFER_Srgb:
  return TRANSFER_WITH_XYZ(linear, transfer_oe_srgb);
  case TRANSFER_Bt2020_10bit:
  case TRANSFER_Bt2020_12bit:
  return TRANSFER_WITH_XYZ(linear, transfer_oe_bt2020_10b);
  case TRANSFER_Smpte2084:
  return TRANSFER_WITH_XYZ(linear, transfer_oe_smpte2084);
  return TRANSFER_WITH_XYZ(linear, transfer_oe_smpte2084);
  case TRANSFER_Bt2100Hlg:
  // FIXME: unimplemented.
  return linear;
  case TRANSFER_LabLch:
  return vec4(transfer_lab_to_lch(linear.xyz), linear.a);
  }
  return linear;
}

// HACK(naga-1403) not sure if anything here is affected.
vec4 parts_untransfer(vec4 nonlin, uint fnk) {
#define TRANSFER_WITH_XYZ(E, FN) vec4(FN(E.x), FN(E.y), FN(E.z), E.a)
  switch (fnk) {
  case TRANSFER_Bt709:
  return TRANSFER_WITH_XYZ(nonlin, transfer_eo_bt709);
  case TRANSFER_Bt470M:
  return TRANSFER_WITH_XYZ(nonlin, transfer_eo_bt470m);
  case TRANSFER_Bt601:
  return TRANSFER_WITH_XYZ(nonlin, transfer_eo_bt601);
  case TRANSFER_Smpte240:
  return TRANSFER_WITH_XYZ(nonlin, transfer_eo_smpte240);
  case TRANSFER_Linear:
  return nonlin;
  case TRANSFER_Srgb:
  return TRANSFER_WITH_XYZ(nonlin, transfer_eo_srgb);
  case TRANSFER_Bt2020_10bit:
  case TRANSFER_Bt2020_12bit:
  return TRANSFER_WITH_XYZ(nonlin, transfer_eo_bt2020_10b);
  case TRANSFER_Smpte2084:
  return TRANSFER_WITH_XYZ(nonlin, transfer_eo_smpte2084);
  return TRANSFER_WITH_XYZ(nonlin, transfer_eo_smpte2084);
  case TRANSFER_Bt2100Hlg:
  // FIXME: unimplemented.
  return nonlin;
  case TRANSFER_LabLch:
  return vec4(transfer_lch_to_lab(nonlin.xyz), nonlin.a);
  }
  return nonlin;
}
