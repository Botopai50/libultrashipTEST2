#pragma once

#include <stdint.h>

// Shared display-list payload used by both game-side GBI commands and the Fast3D interpreter.
#define TOON_SHADOW_RECEIVER_MAX_TRIANGLES 16
#define TOON_SHADOW_RECEIVER_FLAG_WALL (1U << 0)
#define TOON_SHADOW_RECEIVER_FLAG_LOWER_FLOOR (1U << 1)

typedef struct ToonShadowReceiverMesh {
    uint8_t triangleCount;
    uint8_t pad[3];
    uint8_t triangleFlags[TOON_SHADOW_RECEIVER_MAX_TRIANGLES];
    float vertices[TOON_SHADOW_RECEIVER_MAX_TRIANGLES][3][3];
} ToonShadowReceiverMesh;
