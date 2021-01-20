#version 450
layout (location = 0) in vec3 vertPosition;
layout (location = 1) in vec2 vertUv;

layout (location = 0) out vec2 uv;

void main() {
  gl_Position = vec4(vertPosition, 1.0);
  uv = vertUv;
}
