// Standalone transport test: cc -std=c11 -Iinclude tests/toon_local_lights.c -lm
#include <assert.h>
#include <stdio.h>
#include "fast/toon_shading.h"
#include "libultraship/libultra/gbi.h"

int main(void) {
    ToonLocalLights lights = {0};
    Gfx commands[6];
    Gfx* cursor = commands;
    gSPToonLocalLightsReset(cursor++, 1);
    for (int i = 0; i < TOON_LOCAL_LIGHT_MAX; ++i) {
        gSPToonLocalLight(cursor++, i, -127, 0, 0, 255, 64, 0);
    }
    assert(cursor == commands + 5); // macro must evaluate pkt exactly once
    for (int i = 0; i < 5; ++i) {
        assert((commands[i].words.w0 >> 24) == G_SETTOONLOCAL);
        ToonLocalLightsDecode(&lights, commands[i].words.w0, commands[i].words.w1);
    }
    for (int i = 0; i < 4; ++i) {
        assert(lights.direction[i][0] == -1.0f);
        assert(lights.color[i][0] == 1.0f);
        assert(fabsf(lights.color[i][1] - 64.0f / 255) < 1e-6f);
    }
    ToonLocalLights before = lights;
    ToonLocalLightsDecode(&lights, 0, 255u << 24); // malformed slot cannot write outside arrays
    assert(memcmp(&before, &lights, sizeof(lights)) == 0);
    gSPToonLocalLightsReset(cursor++, 1);
    ToonLocalLightsDecode(&lights, commands[5].words.w0, commands[5].words.w1);
    assert(lights.enabled == 1.0f);
    assert(lights.color[3][0] == 0.0f); // next object, zero sources
    ToonLocalLightsDecode(&lights, 0, (1u << 24) | 0xffffff); // zero direction
    assert(lights.color[0][0] == 0.0f && isfinite(lights.direction[0][0]));
    ToonLocalLightsDecode(&lights, 0, 0); // toon off / legacy reset
    ToonLocalLightsDecode(&lights, 127, (1u << 24) | 0xffffff);
    assert(lights.enabled == 0.0f && lights.color[0][0] == 0.0f);
    assert(ToonLocalLightAttenuation(0, 100) == 1.0f);
    assert(ToonLocalLightAttenuation(10000, 100) == 0.0f);
    assert(ToonLocalLightAttenuation(20000, 100) == 0.0f);
    assert(ToonLocalLightAttenuation(0, -1) == 0.0f); // Navi's inactive auxiliary light
    assert(ToonLocalLightAttenuation(0, 0) == 0.0f);
    assert(fabsf(ToonLocalLightAttenuation(2500, 100) - 0.5625f) < 1e-6f);
    puts("PASS: C/C++ command packing, 4 lights, invalid slot, reset, zero direction, attenuation");
    return 0;
}
