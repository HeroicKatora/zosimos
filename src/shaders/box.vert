#version 450
layout (location = 0) in vec2 vertPosition;
layout (location = 0) out vec2 uv;

layout (set = 0, binding = 0, std140) uniform PaintCoordinates {
  /* Positions in clockwise rotation starting at top-left. */
  vec2 rect_selection[4];
  vec2 rect_position[4];
} paint_coordinates;

void main() {
  // Top-left based position (coordinates as in the image package).
  vec2 innerUv = mix(
    mix(paint_coordinates.rect_selection[0], paint_coordinates.rect_selection[1], vertPosition.x),
    mix(paint_coordinates.rect_selection[3], paint_coordinates.rect_selection[2], vertPosition.x),
    vertPosition.y);

  vec2 innerPos = mix(
    mix(paint_coordinates.rect_position[0], paint_coordinates.rect_position[1], vertPosition.x),
    mix(paint_coordinates.rect_position[3], paint_coordinates.rect_position[2], vertPosition.x),
    vertPosition.y);

  vec2 glslPos = 2.0*vec2(innerPos.x, 1.0 - innerPos.y) - 1.0;

  gl_Position = vec4(glslPos, 0.0, 1.0);
  uv = innerUv;
}
