#ifndef FAST_SHADOW_LIGHT_FRAME_H
#define FAST_SHADOW_LIGHT_FRAME_H
#include <math.h>

// A rotation-minimizing light frame. Rebuilding cross(worldUp, light) every frame introduces
// unnecessary roll near the zenith and a discontinuity when the fallback up axis changes.
typedef struct ShadowLightFrame {
    int valid;
    float x[3], y[3], z[3];
} ShadowLightFrame;

static inline void ShadowLightCross(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

// direction must be normalized, as it is by RenderShadowMap before calling this function.
static inline void ShadowLightFrameUpdate(ShadowLightFrame* frame, const float direction[3]) {
    if (frame->valid && frame->z[0] == direction[0] && frame->z[1] == direction[1] && frame->z[2] == direction[2]) {
        return; // identical light must retain bit-identical matrices and cached maps
    }
    float x[3] = { frame->x[0], frame->x[1], frame->x[2] };
    if (frame->valid) {
        const float cosine = frame->z[0] * direction[0] + frame->z[1] * direction[1] + frame->z[2] * direction[2];
        if (cosine > -0.9999f) {
            float axis[3], first[3], second[3];
            ShadowLightCross(frame->z, direction, axis);
            ShadowLightCross(axis, frame->x, first);
            ShadowLightCross(axis, first, second);
            for (int i = 0; i < 3; ++i)
                x[i] += first[i] + second[i] / (1.0f + cosine);
        }
        // Remove accumulated floating-point drift; also supplies the antipodal fallback.
        const float along = x[0] * direction[0] + x[1] * direction[1] + x[2] * direction[2];
        for (int i = 0; i < 3; ++i)
            x[i] -= along * direction[i];
    }
    float length = sqrtf(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    if (!frame->valid || length < 1e-6f) {
        float up[3] = { 0, 1, 0 };
        if (fabsf(direction[1]) > 0.99f) {
            up[0] = 1;
            up[1] = 0;
        }
        ShadowLightCross(up, direction, x);
        length = sqrtf(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    }
    for (int i = 0; i < 3; ++i) {
        frame->x[i] = x[i] / length;
        frame->z[i] = direction[i];
    }
    ShadowLightCross(frame->z, frame->x, frame->y);
    frame->valid = 1;
}
#endif
