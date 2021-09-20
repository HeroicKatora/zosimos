#version 450
#extension GL_EXT_scalar_block_layout : require
layout (location = 0) in vec2 vertPosition;
layout (location = 0) out vec2 uv;

layout (set = 0, binding = 0, std430) uniform PaintCoordinates {
  /* Positions in clockwise rotation starting at top-left. */
  // HACK(naga-1400) if we use arrays vec2[4], we violate Vulkan requirements of alignment.
  // if we use mat4x2 then naga miscalculates the layout and we end up getting overlapping fields.
  // So we use a poor mans array .. .. .
  vec2 rect_selection0;
  vec2 rect_selection1;
  vec2 rect_selection2;
  vec2 rect_selection3;
  vec2 rect_position0;
  vec2 rect_position1;
  vec2 rect_position2;
  vec2 rect_position3;
} paint_coordinates;

// HACK(naga-1400)
vec2 rect_selection(int i) {
  vec2 v [4] = { 
    paint_coordinates.rect_selection0,
    paint_coordinates.rect_selection1,
    paint_coordinates.rect_selection2,
    paint_coordinates.rect_selection3,
  };

  return v[i];
}
vec2 rect_position(int i) {
  vec2 v [4] = { 
    paint_coordinates.rect_position0,
    paint_coordinates.rect_position1,
    paint_coordinates.rect_position2,
    paint_coordinates.rect_position3,
  };

  return v[i];
}

void main() {
  // Top-left based position (coordinates as in the image package).
  vec2 innerUv = mix(
    mix(rect_selection(0), rect_selection(1), vertPosition.x),
    mix(rect_selection(3), rect_selection(2), vertPosition.x),
    vertPosition.y);

  vec2 innerPos = mix(
    mix(rect_position(0), rect_position(1), vertPosition.x),
    mix(rect_position(3), rect_position(2), vertPosition.x),
    vertPosition.y);

  vec2 glslPos = 2.0*vec2(innerPos.x, 1.0 - innerPos.y) - 1.0;

  gl_Position = vec4(glslPos, 0.0, 1.0);
  uv = innerUv;
}
