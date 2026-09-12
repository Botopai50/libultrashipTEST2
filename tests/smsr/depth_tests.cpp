#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "fast/shadow_map.h"
using Microsoft::WRL::ComPtr;
#define SPDLOG_ERROR(...) ((void)0)
static void Check(bool ok, const char* why) {
    if (!ok)
        throw std::runtime_error(why);
}

class GfxRenderingAPIDX11 {
  public:
    ComPtr<ID3D11Device> mDevice;
    ComPtr<ID3D11RasterizerState> mShadowRasterizerState, mShadowRasterizerCascade[SHADOW_MAP_MAX_SLICES];
    float mShadowRasterizerCascadeSlope[SHADOW_MAP_MAX_SLICES]{};
    float mShadowRasterizerCascadeClamp[SHADOW_MAP_MAX_SLICES]{};
    ID3D11RasterizerState* ShadowRasterizerForCascade(int, int, const float[16]);
};
#include "shadow_rasterizer.inc"

int main() {
    try {
        GfxRenderingAPIDX11 api;
        ComPtr<ID3D11DeviceContext> context;
        Check(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                          &api.mDevice, nullptr, &context)),
              "WARP");
        const float unit = 1.0f / 65535.0f;
        for (float radius : { 100.0f, 2000.0f, 50000.0f, 100000.0f, 500000.0f }) {
            float matrix[16]{};
            matrix[0] = 1.0f / radius;
            auto* state = api.ShadowRasterizerForCascade(0, 1024, matrix);
            Check(state != nullptr, "production rasterizer");
            D3D11_RASTERIZER_DESC desc{};
            state->GetDesc(&desc);
            Check(desc.DepthBias == SHADOW_MAP_DEPTH_BIAS_UNITS, "D16 quantization margin");
            Check(desc.DepthBiasClamp >= SHADOW_MAP_DEPTH_BIAS_UNITS * unit, "clamp never becomes unlimited");
            const float ceiling =
                std::max(SHADOW_MAP_DEPTH_BIAS_UNITS * unit, SHADOW_MAP_MAX_SLOPE_BIAS_WORLD / (5.0f * radius));
            Check(desc.DepthBiasClamp <= ceiling + 1e-8f, "bounded world-space displacement");
            Check(api.ShadowRasterizerForCascade(0, 1024, matrix) == state, "stable state reuse");
        }
        float matrix[16]{};
        matrix[0] = 1.0f / 2000;
        ComPtr<ID3D11RasterizerState> fixed = api.ShadowRasterizerForCascade(0, 1024, matrix);
        D3D11_RASTERIZER_DESC oldDesc{};
        fixed->GetDesc(&oldDesc);
        oldDesc.DepthBias = 0;
        ComPtr<ID3D11RasterizerState> old;
        Check(SUCCEEDED(api.mDevice->CreateRasterizerState(&oldDesc, &old)), "old control state");

        const char* source = R"(
cbuffer Params : register(b0) { float4 depth; };
float4 main(uint id : SV_VertexID) : SV_Position {
    float2 p = id == 0 ? float2(-1,-1) : (id == 1 ? float2(-1,3) : float2(3,-1));
    return float4(p, depth.x, 1);
})";
        ComPtr<ID3DBlob> shader, errors;
        Check(SUCCEEDED(D3DCompile(source, strlen(source), nullptr, nullptr, nullptr, "main", "vs_4_0", 0, 0, &shader,
                                   &errors)),
              "depth shader");
        ComPtr<ID3D11VertexShader> vs;
        Check(SUCCEEDED(
                  api.mDevice->CreateVertexShader(shader->GetBufferPointer(), shader->GetBufferSize(), nullptr, &vs)),
              "vertex shader");
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        D3D11_BUFFER_DESC cbDesc{};
        cbDesc.ByteWidth = 16;
        cbDesc.Usage = D3D11_USAGE_DEFAULT;
        cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        ComPtr<ID3D11Buffer> cb;
        Check(SUCCEEDED(api.mDevice->CreateBuffer(&cbDesc, nullptr, &cb)), "depth parameter buffer");
        context->VSSetConstantBuffers(0, 1, cb.GetAddressOf());
        D3D11_TEXTURE2D_DESC texDesc{};
        texDesc.Width = texDesc.Height = 16;
        texDesc.MipLevels = texDesc.ArraySize = texDesc.SampleDesc.Count = 1;
        texDesc.Format = DXGI_FORMAT_R16_TYPELESS;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        ComPtr<ID3D11Texture2D> target, staging;
        Check(SUCCEEDED(api.mDevice->CreateTexture2D(&texDesc, nullptr, &target)), "D16 target");
        D3D11_DEPTH_STENCIL_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_D16_UNORM;
        view.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        ComPtr<ID3D11DepthStencilView> dsv;
        Check(SUCCEEDED(api.mDevice->CreateDepthStencilView(target.Get(), &view, &dsv)), "D16 view");
        texDesc.Usage = D3D11_USAGE_STAGING;
        texDesc.BindFlags = 0;
        texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        Check(SUCCEEDED(api.mDevice->CreateTexture2D(&texDesc, nullptr, &staging)), "readback");
        D3D11_VIEWPORT viewport{ 0, 0, 16, 16, 0, 1 };
        context->RSSetViewports(1, &viewport);
        int oldAcne = 0, fixedAcne = 0;
        for (int mode = 0; mode < 2; ++mode) {
            context->RSSetState(mode == 0 ? old.Get() : fixed.Get());
            for (int frame = 0; frame < 64; ++frame) {
                float depth[4] = { (32768.0f + float(frame) / 64) * unit };
                context->UpdateSubresource(cb.Get(), 0, nullptr, depth, 0, 0);
                context->OMSetRenderTargets(0, nullptr, dsv.Get());
                context->ClearDepthStencilView(dsv.Get(), D3D11_CLEAR_DEPTH, 1, 0);
                context->Draw(3, 0);
                context->OMSetRenderTargets(0, nullptr, nullptr);
                context->CopyResource(staging.Get(), target.Get());
                D3D11_MAPPED_SUBRESOURCE mapped{};
                Check(SUCCEEDED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped)), "read D16 depth");
                const float stored = *static_cast<const uint16_t*>(mapped.pData) * unit;
                context->Unmap(staging.Get(), 0);
                Check(stored < 0.51f, "test triangle actually rasterized");
                if (depth[0] > stored)
                    (mode == 0 ? oldAcne : fixedAcne)++;
                if (mode == 1) {
                    Check(depth[0] + 8 * unit > stored, "separate occluder remains shadowed");
                    Check(stored - depth[0] < 3 * unit, "quantization margin stays small");
                }
            }
        }
        Check(oldAcne > 0 && fixedAcne == 0, "self-shadow flicker reproduced and removed");
        std::cout << "D16 temporal self-shadow failures: " << oldAcne << "/64 before, " << fixedAcne
                  << "/64 after. Large-level bias clamp and occlusion checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
