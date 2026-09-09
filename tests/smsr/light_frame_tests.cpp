#include "fast/shadow_light_frame.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void Check(bool ok, const char* why) {
    if (!ok)
        throw std::runtime_error(why);
}
static float Dot(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static void Normalize(float* a) {
    const float n = std::sqrt(Dot(a, a));
    for (int i = 0; i < 3; ++i)
        a[i] /= n;
}
static float Distance(const float* a, const float* b) {
    float d[3] = { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
    return std::sqrt(Dot(d, d));
}
static void Validate(const ShadowLightFrame& f) {
    Check(std::abs(Dot(f.x, f.x) - 1) < 1e-5 && std::abs(Dot(f.y, f.y) - 1) < 1e-5 &&
              std::abs(Dot(f.z, f.z) - 1) < 1e-5,
          "unit light frame");
    Check(std::abs(Dot(f.x, f.z)) < 1e-5 && std::abs(Dot(f.x, f.y)) < 1e-5 && std::abs(Dot(f.y, f.z)) < 1e-5,
          "orthogonal light frame");
    float cross[3];
    ShadowLightCross(f.x, f.y, cross);
    Check(Distance(cross, f.z) < 1e-5, "handedness unchanged");
}
int main() {
    try {
        ShadowLightFrame frame{};
        float oldX[3]{}, previousX[3]{};
        float oldTravel = 0, newTravel = 0;
        for (int i = 0; i <= 1000; ++i) {
            const float angle = -0.03f + 0.00006f * i;
            // The actual solar orbit near noon, normalized and negated for light travel.
            float z[3] = { std::sin(angle) * 120, -std::cos(angle) * 120, -std::cos(angle) * 20 };
            Normalize(z);
            float up[3] = { 0, 1, 0 }, rebuiltX[3];
            ShadowLightCross(up, z, rebuiltX);
            Normalize(rebuiltX);
            ShadowLightFrameUpdate(&frame, z);
            Validate(frame);
            if (i) {
                oldTravel += Distance(oldX, rebuiltX);
                newTravel += Distance(previousX, frame.x);
            }
            std::memcpy(oldX, rebuiltX, sizeof(oldX));
            std::memcpy(previousX, frame.x, sizeof(previousX));
        }
        Check(newTravel < oldTravel / 4, "solar motion does not introduce amplified grid roll");
        const auto held = frame;
        for (int i = 0; i < 100; ++i)
            ShadowLightFrameUpdate(&frame, held.z);
        Check(std::memcmp(&frame, &held, sizeof(frame)) == 0, "static light preserves exact cache identity");
        frame = {};
        for (int i = 0; i <= 1000; ++i) {
            const float x = 0.13f + i * 0.00002f;
            float z[3] = { x, -std::sqrt(1 - x * x), 0.001f };
            Normalize(z);
            const auto before = frame;
            ShadowLightFrameUpdate(&frame, z);
            Validate(frame);
            if (before.valid)
                Check(Distance(before.x, frame.x) < 0.001f, "no up-axis threshold flip");
        }
        float opposite[3] = { -frame.z[0], -frame.z[1], -frame.z[2] };
        ShadowLightFrameUpdate(&frame, opposite);
        Validate(frame);
        std::cout << "Solar grid-axis travel: " << oldTravel << " before, " << newTravel
                  << " after. Pole crossing, antipodes, orthogonality and exact stationary reuse passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
