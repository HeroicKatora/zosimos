#version 450
#extension GL_EXT_scalar_block_layout : require
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 f_color;

layout (set = 1, binding = 0, std430) uniform FragmentColor {
    // The expected value of each coordinate (often denoted mu).
    vec2 expectation;
    // The covariance matrix of random values (often denoted Sigma). This is
    // the/a pseudo inverse of it. If the covariance matrix is full rank, such
    // as diagonal, then this is simple the inverse.
    // This ensures the caller is aware of degenerate cases. You may choose to
    // use the Pseudo-Inverse to have this shader model 1-dimensional
    // distributions instead.
    mat2x2 covariance_inverse;
    // The pseudo determinant of the covariance matrix, i.e. the product of all
    // non-zero eigen values.
    float pseudo_determinant;
} u_fragmentParams;

/*
double pseudoDeterminant(mat2 m) {
    double c = determinant(m);
    double b = -(m[0].x + m[1].y);
    // We want the product of all non-zero solutions to
    //  x² + bx + c
    // The set is given by:
    //  (b +- sqrt(b²-4c)) / 2
    // a) b == 0 and b² == 4c then there are no non-zero solutions
    // b) b = +-sqrt(b²-4c) then there is one non-zero solution
    // c) the product is the determinant c otherwise
    return determinant(m);
}
*/

#define PI 3.1415926538

void main() {
    vec2 screenSpace = 2.0*(uv - vec2(0.5));
    vec2 pos = (screenSpace - vec2(u_fragmentParams.expectation));
    mat2x2 covinv = mat2x2(
    	vec2(u_fragmentParams.covariance_inverse[0]),
	vec2(u_fragmentParams.covariance_inverse[1]));
    float exponent = 0.5 * dot(pos, covinv * pos);
    float value = exp(-exponent) / sqrt(u_fragmentParams.pseudo_determinant);

    // TODO: can we provide useful information in other channels?
    f_color = vec4(vec3(value), 1.0);
}
