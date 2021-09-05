/** Expands different half-float/compressed float formats.
 */

float fp_decode_ieee_half(uint bits) {
  bool sign = (bits & 0x80) != 0;
  int e = int((bits >> 10) & 0x1f) - 15;
  uint m = 0x400 + (bits & 0x3ff);
  return (sign?-1.0:1.0)* float(m) * pow(2.0, float(e - 11));
}
