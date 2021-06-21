#version 450
layout (location = 0) in vec2 vertPosition;
layout (location = 0) out vec2 uv;

void main() {
  gl_Position = vec4(vertPosition, 0.0, 1.0);
  uv = (vertPosition + 1.0) * 0.5;
}
