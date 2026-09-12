#ifndef FAST_TOON_SHADING_H
#define FAST_TOON_SHADING_H

// Shared default parameters for the per-pixel toon-lighting effect.
//
// The effect lives in the toon variant of the Fast3D fragment shaders (a distinct shader compiled
// when the SHADER_OPT(TOON) bit is set). The application emits a gSPToon(true/false) marker around
// the draws it wants relit; the interpreter forwards the world-space vertex normal and the single
// dominant light, and the fragment shader ramps N·L. The ramp shape (center/softness/highlight/
// shadow) is frame-global tuning the application pushes via GfxRenderingAPI::SetToonRamp(); until it
// does, the backends use the defaults below. The framework never reads the application's config, so
// no app-specific CVar keys live here.

// Defaults shared by every rendering backend (half-Lambert N·L mapped to 0..1).
#define TOON_SHADING_DEFAULT_RAMP_CENTER 0.5f
#define TOON_SHADING_DEFAULT_RAMP_SOFTNESS 0.1f
// Highlight = brightness of the lit band; Shadow = how dark the shadow band gets (1 = ambient).
// Both default to 1.0, which reproduces the plain "ambient + ramp*lightColor" two-tone.
#define TOON_SHADING_DEFAULT_HIGHLIGHT 1.0f
#define TOON_SHADING_DEFAULT_SHADOW 1.0f

// Four local contributions supplement the directional key. Direction and attenuation
// are evaluated at the object origin; the ramp remains per pixel. Float4 arrays
// match HLSL constant-buffer array packing and GL/Metal transport.
#include <stdint.h>
#include <string.h>
#include <math.h>
#define TOON_LOCAL_LIGHT_MAX 4
typedef struct ToonLocalLights {
    float direction[TOON_LOCAL_LIGHT_MAX][4];
    float color[TOON_LOCAL_LIGHT_MAX][4];
    float enabled;
} ToonLocalLights;

static inline float ToonLocalLightAttenuation(float distanceSquared, float radius) {
    if (!(radius > 0.0f) || !(distanceSquared >= 0.0f)) return 0.0f;
    float a = 1.0f - distanceSquared / (radius * radius);
    return a > 0.0f ? a * a : 0.0f;
}

// 0x4c: slot 0 resets the object snapshot (w0 bit 0 enables multi-light).
// Slots 1..4 carry the same signed direction / RGB bytes as the key command.
// The payload lives in the display list, so replay never accesses live actors.
static inline void ToonLocalLightsDecode(ToonLocalLights* lights, uint32_t w0, uint32_t w1) {
    unsigned slot = w1 >> 24;
    if (slot == 0) {
        memset(lights, 0, sizeof(*lights));
        lights->enabled = (w0 & 1) ? 1.0f : 0.0f;
        return;
    }
    if (slot > TOON_LOCAL_LIGHT_MAX || !lights->enabled) return;
    --slot;
    float len2 = 0.0f;
    for (int c = 0; c < 3; ++c) {
        int shift = 16 - 8 * c;
        int byte = (w0 >> shift) & 255;
        float d = (float)(byte >= 128 ? byte - 256 : byte);
        lights->direction[slot][c] = d;
        len2 += d * d;
        lights->color[slot][c] = ((w1 >> shift) & 255) / 255.0f;
    }
    if (len2 > 0.0f) {
        float invLen = 1.0f / sqrtf(len2);
        for (int c = 0; c < 3; ++c) lights->direction[slot][c] *= invLen;
    } else {
        // Invalid direction must not introduce a NaN or illuminate an object.
        memset(lights->color[slot], 0, sizeof(lights->color[slot]));
    }
}

#endif // FAST_TOON_SHADING_H
