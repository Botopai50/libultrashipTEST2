#include <d3d11.h>
#include <wrl/client.h>
#include <iostream>
#include <stdexcept>
#include "fast/shadow_map.h"
using Microsoft::WRL::ComPtr;
#define SPDLOG_ERROR(...) ((void)0)

// Compile the production allocation/configuration code. Only pipeline and optional
// static-cache creation are stubbed: the test checks real D3D11 texture and view descriptors.
class GfxRenderingAPIDX11 {
  public:
    ComPtr<ID3D11Device> mDevice;
    ComPtr<ID3D11Texture2D> mShadowMapTexture, mShadowActorTexture;
    ComPtr<ID3D11ShaderResourceView> mShadowMapSrv, mShadowActorSrv;
    ComPtr<ID3D11DepthStencilView> mShadowMapDsv[SHADOW_MAP_MAX_SLICES];
    bool mShadowSliceValid[SHADOW_MAP_MAX_SLICES]{};
    bool mShadowActorSplit = false;
    int mShadowCascadeCount = 0, mShadowResolution = 0, mShadowActorResolution = 0;
    ShadowMapQuality mShadowQuality = ShadowMapQualityDefaults();
    bool CreateShadowMapPipeline() {
        return true;
    }
    bool CreateShadowStaticTargets(int, int) {
        return true;
    }
    void ShadowStaticRelease() {
    }
    bool CreateShadowMapTargets(int, int, int);
    bool ShadowMapConfigure(int, int, int);
};
#include "shadow_targets.inc"

struct Ladder {
    int mShadowMapCascadeCount = 10;
    ShadowMapQuality mShadowMapQuality = ShadowMapQualityDefaults();
    float mShadowMapSplitsRequested[3] = { 190, 2000, 6000 };
    float requestedGuard = 12345;
    float mShadowMapSplits[3]{};
    float outputGuard[8] = { 42, 42, 42, 42, 42, 42, 42, 42 };
#include "shadow_ladder.inc"
};

static void Check(bool ok, const char* why) {
    if (!ok)
        throw std::runtime_error(why);
}
int main() {
    try {
        GfxRenderingAPIDX11 renderer;
        Check(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                          &renderer.mDevice, nullptr, nullptr)),
              "WARP");
        for (int layout : { SHADOW_MAP_LAYOUT_CASCADE, SHADOW_MAP_LAYOUT_CLIPMAP }) {
            renderer.mShadowQuality.layout = layout;
            for (int requested : { 0, 1, 3, 8, 10, 99 }) {
                for (int actorResolution : { 256, 512 }) {
                    Check(renderer.ShadowMapConfigure(requested, 512, actorResolution), "configure shadow maps");
                    const int bound = layout == SHADOW_MAP_LAYOUT_CLIPMAP ? 10 : 3;
                    const int count = requested < 1 ? 1 : (requested > bound ? bound : requested);
                    Check(renderer.mShadowCascadeCount == count, "active layout must preserve its actual level count");
                    D3D11_TEXTURE2D_DESC texture{};
                    renderer.mShadowMapTexture->GetDesc(&texture);
                    Check(texture.ArraySize == UINT(SHADOW_MAP_SLICES_FOR(count)), "all level slices allocated");
                    for (int i = 0; i < SHADOW_MAP_SLICES_FOR(count); ++i) {
                        D3D11_DEPTH_STENCIL_VIEW_DESC view{};
                        Check(renderer.mShadowMapDsv[i] != nullptr, "depth view exists for every requested level");
                        renderer.mShadowMapDsv[i]->GetDesc(&view);
                        Check(view.Texture2DArray.FirstArraySlice == UINT(i), "logical world/actor index");
                    }
                    D3D11_SHADER_RESOURCE_VIEW_DESC actorView{};
                    renderer.mShadowActorSrv->GetDesc(&actorView);
                    Check(actorView.Texture2DArray.ArraySize == texture.ArraySize,
                          "actor SRV stride matches level count");
                }
            }
        }
        Ladder ladder;
        ladder.mShadowMapQuality.ladderMode = SHADOW_MAP_LADDER_PRACTICAL;
        ladder.ApplyShadowLadder();
        Check(ladder.mShadowMapSplits[2] == 6000 && ladder.requestedGuard == 12345,
              "ladder uses three requested splits");
        for (float guard : ladder.outputGuard)
            Check(guard == 42, "clipmap must not overrun three-entry cascade ladder");
        std::cout << "Cascade/clipmap counts, real DSV/SRV arrays, actor strides and ladder bounds passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
