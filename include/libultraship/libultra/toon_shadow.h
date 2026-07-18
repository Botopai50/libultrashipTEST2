#pragma once

#include <stdint.h>

// Shared display-list payload used by both game-side GBI commands and the Fast3D interpreter.
#define TOON_SHADOW_RECEIVER_MAX_TRIANGLES 16

typedef struct ToonShadowReceiverMesh {
    uint8_t triangleCount;
    uint8_t pad[3];
    float vertices[TOON_SHADOW_RECEIVER_MAX_TRIANGLES][3][3];
} ToonShadowReceiverMesh;
