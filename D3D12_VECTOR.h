#ifndef D3D12_VECTOR_H
#define D3D12_VECTOR_H

#include <DirectXMath.h>

/**
 * @brief A simple structure to define a 3D vector with x, y, and z components.
 */
struct D3D12_VECTOR {
    float x;
    float y;
    float z;

    // Default constructor
    D3D12_VECTOR() : x(0.0f), y(0.0f), z(0.0f) {}

    // Parameterized constructor
    D3D12_VECTOR(float x, float y, float z) : x(x), y(y), z(z) {}

    // Vector addition
    D3D12_VECTOR operator+(const D3D12_VECTOR& other) const {
        return D3D12_VECTOR(x + other.x, y + other.y, z + other.z);
    }

    // Vector subtraction
    D3D12_VECTOR operator-(const D3D12_VECTOR& other) const {
        return D3D12_VECTOR(x - other.x, y - other.y, z - other.z);
    }

    // Scalar multiplication
    D3D12_VECTOR operator*(float scalar) const {
        return D3D12_VECTOR(x * scalar, y * scalar, z * scalar);
    }

    // Magnitude of the vector
    float Magnitude() const {
        return sqrtf(x * x + y * y + z * z);
    }

    // Normalize the vector
    D3D12_VECTOR Normalize() const {
        float mag = Magnitude();
        if (mag == 0.0f) return D3D12_VECTOR(0.0f, 0.0f, 0.0f);
        return D3D12_VECTOR(x / mag, y / mag, z / mag);
    }
};

#endif // D3D12_VECTOR_H
