#include <d3d11.h>
#include <wrl/client.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "fast/shadow_map.h"
using Microsoft::WRL::ComPtr;
#define SPDLOG_ERROR(...) ((void)0)

// Real production upload code and WARP buffers. No game or scene assets are needed.
class GfxRenderingAPIDX11 {
  public:
    static constexpr int Slots = SHADOW_MAP_LAYERS * SHADOW_MAP_CASTER_SLOTS;
    ComPtr<ID3D11Device> mDevice;
    ComPtr<ID3D11DeviceContext> mContext;
    ComPtr<ID3D11InputLayout> mShadowDepthLayout;
    ComPtr<ID3D11VertexShader> mShadowDepthVs;
    ComPtr<ID3D11Buffer> mShadowDepthCb, mShadowCasterVb[Slots], mShadowAlphaVb[Slots];
    size_t mShadowCasterVbVertices[Slots]{}, mShadowAlphaVbVertices[Slots]{};
    const float* mShadowLastCasterPtr[Slots]{};
    const float* mShadowAlphaLastPtr[Slots]{};
    size_t mShadowLastCasterCount[Slots]{}, mShadowAlphaLastCount[Slots]{};
    uint64_t mShadowWorldUploadGeneration = UINT64_MAX;
    bool mShadowPassActive = true, mShadowAlphaBound = false, mShadowAlphaPipelineReady = true;
    int mShadowAlphaBoundIndex = -1, mShadowCurrentLayer = SHADOW_MAP_LAYER_WORLD;
    void ShadowMapInvalidateOpenSlice() {
        throw std::runtime_error("unexpected upload failure");
    }
    void ShadowMapSetWorldGeneration(uint64_t);
    void ShadowMapDrawCasters(const float*, size_t, int, size_t, size_t);
    void ShadowMapUploadAlphaCasters(const float*, size_t, int);
    int ShadowAlphaSlotIndex(int) const;
};
#include "shadow_upload.inc"

static void Check(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

static float ReadFirst(GfxRenderingAPIDX11& api, ID3D11Buffer* source) {
    D3D11_BUFFER_DESC desc{};
    source->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> staging;
    Check(SUCCEEDED(api.mDevice->CreateBuffer(&desc, nullptr, &staging)), "staging buffer");
    api.mContext->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    Check(SUCCEEDED(api.mContext->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped)), "read GPU buffer");
    float result = *static_cast<float*>(mapped.pData);
    api.mContext->Unmap(staging.Get(), 0);
    return result;
}

int main() {
    try {
        GfxRenderingAPIDX11 api;
        Check(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0,
                                         D3D11_SDK_VERSION, &api.mDevice, nullptr, &api.mContext)), "WARP device");
        constexpr int slot = SHADOW_MAP_CASTER_SLOT_MAIN;
        constexpr int index = SHADOW_MAP_LAYER_WORLD * SHADOW_MAP_CASTER_SLOTS + slot;
        float opaque[9] = { 1 }, alpha[15] = { 1 };
        auto upload = [&] {
            // No render target is needed: this test reads uploaded bytes, not rendered pixels.
            api.ShadowMapDrawCasters(opaque, 3, slot, 0, 3);
            api.ShadowMapUploadAlphaCasters(alpha, 3, slot);
        };
        api.ShadowMapSetWorldGeneration(1);
        upload();
        Check(ReadFirst(api, api.mShadowCasterVb[index].Get()) == 1, "initial opaque upload");
        Check(ReadFirst(api, api.mShadowAlphaVb[index].Get()) == 1, "initial cutout upload");
        api.ShadowMapSetWorldGeneration(1);
        Check(api.mShadowLastCasterPtr[index] == opaque && api.mShadowAlphaLastPtr[index] == alpha,
              "unchanged generation preserves cached uploads");

        // The intermediate room capture is culled: no upload. A later capture reuses the original
        // allocation with the same vertex count. Pointer/count matching alone retains the old bytes.
        api.ShadowMapSetWorldGeneration(2);
        opaque[0] = alpha[0] = 3;
        api.ShadowMapSetWorldGeneration(3);
        upload();
        Check(ReadFirst(api, api.mShadowCasterVb[index].Get()) == 3, "reused allocation uploads current opaque geometry");
        Check(ReadFirst(api, api.mShadowAlphaVb[index].Get()) == 3, "reused allocation uploads current cutout geometry");

        // Emulate the old pointer-only policy to demonstrate the regression is sensitive to stale data.
        opaque[0] = alpha[0] = 4;
        upload();
        Check(ReadFirst(api, api.mShadowCasterVb[index].Get()) == 3, "control reproduces stale opaque upload");
        Check(ReadFirst(api, api.mShadowAlphaVb[index].Get()) == 3, "control reproduces stale cutout upload");
        api.ShadowMapSetWorldGeneration(4);
        upload();
        Check(ReadFirst(api, api.mShadowCasterVb[index].Get()) == 4, "generation repairs stale opaque upload");
        Check(ReadFirst(api, api.mShadowAlphaVb[index].Get()) == 4, "generation repairs stale cutout upload");
        std::cout << "GPU readback: stale room uploads reproduced and generation invalidation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
