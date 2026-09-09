#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "fast/shadow_map.h"
#include "shadow_cbuffer.inc"

using Microsoft::WRL::ComPtr;
using Pixel = std::array<float, 4>;
static void Check(bool ok, const char* why) {
    if (!ok)
        throw std::runtime_error(why);
}
static std::string Read(const char* path) {
    std::ifstream f(path);
    Check(bool(f), "read shader");
    return std::string(std::istreambuf_iterator<char>(f), {});
}
static ComPtr<ID3DBlob> Compile(const std::string& source, const char* entry, const char* profile) {
    ComPtr<ID3DBlob> blob, errors;
    HRESULT hr = D3DCompile(source.data(), source.size(), "smsr-test", nullptr, nullptr, entry, profile,
                            D3DCOMPILE_OPTIMIZATION_LEVEL2 | D3DCOMPILE_ENABLE_STRICTNESS, 0, &blob, &errors);
    if (errors)
        std::cerr << (const char*)errors->GetBufferPointer();
    Check(SUCCEEDED(hr), "native HLSL compilation");
    return blob;
}

struct Params {
    Pixel dc{}, don{}, p{}, uv{}, plane{ 0, 0, 1, 0 };
    Pixel smsr{ 1, SHADOW_MAP_DEFAULT_SMSR_STEPS, SHADOW_MAP_DEFAULT_SMSR_EPSILON, 0 };
    Pixel config{}; // mode, map size, logical slice, actor flag
};

static const char* kPrefix = R"(
Texture2DArray<float> g_shadowMap : register(t6);
Texture2DArray<float> g_shadowMapActors : register(t7);
cbuffer TestCB : register(b0) {
    float4 testDC, testDon, testP, testUV, testPlane, shadow_smsr, testConfig;
};
static const float4 shadow_params = float4(3,0.1,0,1);
static const float4 shadow_splits = float4(100,300,1000,0);
static const float4 shadow_range = float4(0,0,0,0);
float SampleShadowJittered(float2 uv, float z, float slice, float texelUv, bool actor, float2 pixel, float2 gradient) {
    return 0.375; // distinguish the existing filtered path from binary SMSR
}
)";
static const char* kEntry = R"(
float4 TestVS(uint id : SV_VertexID) : SV_POSITION {
    float w = testConfig.x == 8.0 ? (id == 0 ? testDC.x : (id == 1 ? testDC.y : testDC.z)) : 1.0;
    return float4((id == 2 ? 3.0 : -1.0) * w, (id == 1 ? 3.0 : -1.0) * w, 0.5 * w, w);
}
float4 TestPS(float4 position : SV_POSITION) : SV_TARGET {
    if (testConfig.x == 9.0) {
        return ShadowAnalyticCoverage(testDC, 0.0, position.xy / testUV.w, 2.0);
    }
    if (testConfig.x == 8.0) {
        float depth = position.w;
        return float4(position.w, depth, ShadowCascadeIndex(depth), ShadowCascadeIndex(rcp(position.w)));
    }
    if (testConfig.x == 7.0) {
        float4 receiver = ShadowReceiverDepths(testUV.xy, testUV.z, 1.0 / testConfig.y, testPlane.xy);
        return step(receiver, testDC);
    }
    if (testConfig.x == 0.0) return float4(vSMSR(testDC, testDon, testP.xy), 0.0, 0.0, 1.0);
    if (testConfig.x == 1.0) return float4(SmsrNormalize(testDC.xy, testDC.zw, testP.x), 0.0, 1.0);
    if (testConfig.x == 5.0) return float4(SmsrDepthPlane(testP.xyz, float3(1,0,0),
                                                      float3(0,1,0), float3(0,0,0.2)), 1.0);
    ShadowProjection projection;
    projection.uv = (testConfig.x == 3.0 || testConfig.x == 4.0) ? position.xy / testUV.w : testUV.xy;
    projection.z = testUV.z + dot(testPlane.xy, projection.uv - float2(0.5, 0.5));
    projection.texelUv = 1.0 / testConfig.y;
    projection.slice = testConfig.z;
    projection.inside = 1.0;
    projection.depthPlane = testPlane.xyz;
    bool actor = testConfig.w > 0.5;
    float value;
    if (testConfig.x == 6.0) {
        value = ShadowSample(projection, actor, position.xy);
    } else if (testConfig.x == 4.0) {
        value = SmsrLitAt(projection, int2(floor(projection.uv * testConfig.y)), (int)testConfig.y, actor);
    } else {
        value = SampleShadowSMSR(projection, actor);
    }
    return float4(value, value, value, 1.0);
}
)";

class Fixture {
  public:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11Buffer> cb;
    explicit Fixture(const std::string& source) {
        Check(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                          &device, nullptr, &context)),
              "WARP device");
        auto vertex = Compile(source, "TestVS", "vs_4_0");
        auto pixel = Compile(source, "TestPS", "ps_4_0");
        Check(SUCCEEDED(device->CreateVertexShader(vertex->GetBufferPointer(), vertex->GetBufferSize(), nullptr, &vs)),
              "create VS");
        Check(SUCCEEDED(device->CreatePixelShader(pixel->GetBufferPointer(), pixel->GetBufferSize(), nullptr, &ps)),
              "create PS");
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = sizeof(Params);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        Check(SUCCEEDED(device->CreateBuffer(&desc, nullptr, &cb)), "constant buffer");
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        context->PSSetConstantBuffers(0, 1, cb.GetAddressOf());
        context->VSSetConstantBuffers(0, 1, cb.GetAddressOf());
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    }
    ComPtr<ID3D11ShaderResourceView> Map(int size, int slices, const std::vector<float>& depths) {
        Check(depths.size() == size_t(size * size * slices), "map dimensions");
        std::vector<uint16_t> encoded(depths.size());
        for (size_t i = 0; i < depths.size(); ++i)
            encoded[i] = uint16_t(std::round(depths[i] * 65535.0f));
        std::vector<D3D11_SUBRESOURCE_DATA> data(slices);
        for (int i = 0; i < slices; ++i)
            data[i] = { encoded.data() + i * size * size, UINT(size * 2), 0 };
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = desc.Height = size;
        desc.MipLevels = 1;
        desc.ArraySize = slices;
        desc.Format = DXGI_FORMAT_R16_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        ComPtr<ID3D11Texture2D> texture;
        Check(SUCCEEDED(device->CreateTexture2D(&desc, data.data(), &texture)), "shadow fixture texture");
        D3D11_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format = desc.Format;
        view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
        view.Texture2DArray.MipLevels = 1;
        view.Texture2DArray.ArraySize = slices;
        ComPtr<ID3D11ShaderResourceView> srv;
        Check(SUCCEEDED(device->CreateShaderResourceView(texture.Get(), &view, &srv)), "shadow fixture SRV");
        return srv;
    }
    std::vector<Pixel> Draw(const Params& params, int size = 1) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = desc.Height = size;
        desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
        desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        ComPtr<ID3D11Texture2D> target, staging;
        Check(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &target)), "target");
        ComPtr<ID3D11RenderTargetView> rtv;
        Check(SUCCEEDED(device->CreateRenderTargetView(target.Get(), nullptr, &rtv)), "RTV");
        D3D11_VIEWPORT viewport{ 0, 0, float(size), float(size), 0, 1 };
        context->RSSetViewports(1, &viewport);
        context->OMSetRenderTargets(1, rtv.GetAddressOf(), nullptr);
        const float clear[4] = { -9, -9, -9, -9 };
        context->ClearRenderTargetView(rtv.Get(), clear);
        context->UpdateSubresource(cb.Get(), 0, nullptr, &params, 0, 0);
        context->Draw(3, 0);
        context->OMSetRenderTargets(0, nullptr, nullptr);
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        Check(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &staging)), "staging");
        context->CopyResource(staging.Get(), target.Get());
        D3D11_MAPPED_SUBRESOURCE mapped{};
        Check(SUCCEEDED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped)), "read result");
        std::vector<Pixel> output(size * size);
        for (int y = 0; y < size; ++y) {
            const Pixel* row = (const Pixel*)((const char*)mapped.pData + y * mapped.RowPitch);
            std::copy(row, row + size, output.begin() + y * size);
        }
        context->Unmap(staging.Get(), 0);
        return output;
    }
};

static void VisibilityCases(Fixture& fixture) {
    struct Case {
        Pixel dc, don;
        std::array<float, 2> p;
        float expected;
    };
    const Case cases[] = {
        { { .5f, 0, 1, 0 }, { 0, .8f, 0, -1 }, { .1f, .2f }, 1 },     // 1 dual negative
        { { .5f, 0, 1, 0 }, { 0, .2f, 0, 1 }, { .9f, .2f }, 0 },      // 2 dual positive
        { { .75f, 0, 1, 0 }, { 0, .2f, 0, 0 }, { .9f, .2f }, 0 },     // 3 opposite edges
        { { .5f, .25f, 1, 0 }, { .1f, .8f, 0, 0 }, { .2f, .9f }, 0 }, // 4 X dominates
        { { .25f, .5f, 0, 1 }, { .8f, .1f, 0, 0 }, { .1f, .8f }, 0 }, // 5 Y dominates
        { { .5f, .5f, 0, 0 }, { .7f, 0, 0, 0 }, { .2f, .6f }, 0 },    // 6 corner, bottom
        { { .25f, .25f, 0, 0 }, { .7f, 0, 0, 0 }, { .2f, .2f }, 0 },  // 7 corner, top
        { { .5f, .25f, 1, 1 }, { .1f, .8f, 0, 0 }, { .2f, .9f }, 0 }, // 8 intersection: min(0,1)
        { { 0, .5f, 0, 1 }, { .7f, 0, 0, 0 }, { .1f, .5f }, 0 },      // 9 bottom
        { { 0, .25f, 0, 1 }, { .7f, 0, 0, 0 }, { .1f, .5f }, 0 },     // 10 top
        { { .5f, 0, 1, 0 }, { 0, .7f, 0, 0 }, { .5f, .1f }, 0 },      // 11 left
        { { .25f, 0, 1, 0 }, { 0, .7f, 0, 0 }, { .5f, .1f }, 0 },     // 12 right
        { { 0, 0, 0, 0 }, { 1, 1, 1, 1 }, { .1f, .1f }, 1 },          // no discontinuity
        { { .5f, 0, 1, 0 }, { 0, .7f, 0, 0 }, { .9f, .1f }, 1 },      // lit side of reconstructed line
        { { .5f, 0, 1, 0 }, { 0, .5f, 0, 0 }, { .5f, .1f }, 1 },      // equality: strict comparison
    };
    for (size_t i = 0; i < std::size(cases); ++i) {
        Params p;
        p.dc = cases[i].dc;
        p.don = cases[i].don;
        p.p = { cases[i].p[0], cases[i].p[1], 0, 0 };
        if (fixture.Draw(p)[0][0] != cases[i].expected) {
            throw std::runtime_error("SMSR visibility case " + std::to_string(i + 1));
        }
    }
    Params p;
    p.config[0] = 1;
    p.dc = { -2, 1, 3, 1 }; // L=4, forward endpoint
    p.p[0] = .4f;
    auto value = fixture.Draw(p)[0];
    Check(std::abs(value[0] - .35f) < 1e-5f && value[1] == 0, "ONDS forward equation");
    p.dc = { 3, 1, -2, 1 };
    value = fixture.Draw(p)[0];
    Check(std::abs(value[0] - .4f) < 1e-5f && value[1] == 0, "ONDS oriented reverse equation");
    p.dc = { -2, 0, 3, 1 };
    Check(fixture.Draw(p)[0][1] == -1, "truncated traversal must not invent an endpoint");
    p.config[0] = 5;
    p.p = { 1, 2, -2, 0 };
    value = fixture.Draw(p)[0];
    Check(std::abs(value[0] - .2f) < 1e-5f && std::abs(value[1] + .4f) < 1e-5f && value[2] == 1,
          "receiver-plane slope and Y convention");
    std::cout << "12 visibility cases, equality, ONDS orientation and depth-plane projection passed\n";
}

static void AnalyticOccluderDepth(Fixture& fixture) {
    Params params;
    params.config[0] = 9;
    params.uv[3] = 32;
    params.dc = { -0.05f, 0.05f, 0.05f, -0.05f };
    const auto reference = fixture.Draw(params, 32);
    // Same vertical visibility edge. Only the occluder/receiver separation changes.
    for (const auto& depths : { Pixel{-.4f,.0001f,.0001f,-.1f},
                               Pixel{-.0001f,.1f,.4f,-.0001f} }) {
        params.dc = depths;
        const auto changed = fixture.Draw(params, 32);
        for (size_t i = 0; i < changed.size(); ++i) {
            Check(std::abs(changed[i][0]-reference[i][0]) < 1e-6f,
                  "unchanged visibility must not move the reconstructed edge when surface depths vary");
        }
    }
    for (float depth : { -.4f, .0001f }) {
        params.dc.fill(depth);
        for (const auto& pixel : fixture.Draw(params, 32))
            Check(pixel[0] == (depth < 0 ? 0.0f : 1.0f), "uniform lit/shadow interiors preserved");
    }
    std::cout << "Analytic coverage: depth discontinuities preserve the visibility contour and uniform interiors\n";
}

static void PerspectiveCascadeSelection(Fixture& fixture) {
    Params params;
    params.config[0] = 8;
    for (const auto& depths : { Pixel{50,50,50,0}, Pixel{150,150,150,0},
                               Pixel{500,500,500,0}, Pixel{50,500,900,0} }) {
        params.dc = depths;
        const auto image = fixture.Draw(params, 16);
        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 16; ++x) {
                const float b1 = .5f - (y + .5f) / 32.0f; // viewport Y points down
                const float b2 = (x + .5f) / 32.0f;
                const float expected = 1.0f / ((1-b1-b2)/depths[0] + b1/depths[1] + b2/depths[2]);
                const auto& pixel = image[y*16+x];
                if (std::abs(pixel[1]-expected) >= .001f) {
                    throw std::runtime_error("perspective depth: expected " + std::to_string(expected) +
                                             ", received " + std::to_string(pixel[1]));
                }
                const int cascade = expected <= 100 ? 0 : (expected <= 300 ? 1 : 2);
                Check(pixel[2] == cascade, "perspective receivers select the correct cascade");
                Check(pixel[3] == 0, "inverted-W control incorrectly selects the nearest cascade everywhere");
            }
        }
    }
    std::cout << "Perspective rasterization: 1024 pixels select correct cascades; inverted-W control stays in cascade zero\n";
}

static void ReceiverPlaneComparisons(Fixture& fixture) {
    Params params;
    params.config = { 7, 32, 0, 0 };
    for (const auto& gradient : { std::array<float, 2>{ .8f, -.6f }, { -.7f, .9f } }) {
        params.plane = { gradient[0], gradient[1], 1, 0 };
        for (int phase = 0; phase < 32; ++phase) {
            const float u = (16.0f + float(phase) / 32.0f) / 32.0f;
            const float v = (15.0f + float(31-phase) / 32.0f) / 32.0f;
            const auto depth = [&](float x, float y) { return .5f + gradient[0]*(x-.5f) + gradient[1]*(y-.5f); };
            params.uv = { u, v, depth(u,v), 0 };
            const float x = (std::floor(u*32-.5f)+.5f)/32;
            const float y = (std::floor(v*32-.5f)+.5f)/32;
            params.dc = { depth(x,y+1.0f/32), depth(x+1.0f/32,y+1.0f/32), depth(x+1.0f/32,y), depth(x,y) };
            int legacyShadowed = 0;
            for (float stored : params.dc) legacyShadowed += params.uv[2] > stored;
            Check(legacyShadowed > 0, "control reproduces false occlusion on tilted receiver");
            for (float& stored : params.dc) stored += 2.0f/65535.0f;
            const auto lit = fixture.Draw(params)[0];
            for (float value : lit) Check(value == 1, "all four taps compare on the receiver plane");
            for (float& stored : params.dc) stored -= .1f;
            const auto shadowed = fixture.Draw(params)[0];
            for (float value : shadowed) Check(value == 0, "real occluder is preserved");
        }
    }
    std::cout << "Receiver-plane quad comparisons: 64 phases without false occlusion; real occluders preserved\n";
}

static void Silhouettes(Fixture& fixture, const char* outputPath) {
    constexpr int mapSize = 32, scale = 8, renderSize = mapSize * scale;
    std::vector<float> depths(mapSize * mapSize * 5, .8f);
    for (int y = 0; y < mapSize; ++y)
        for (int x = 0; x < mapSize; ++x) {
            if (x < 8 + y / 4)
                depths[y * mapSize + x] = .2f;
        }
    auto world = fixture.Map(mapSize, 5, depths);
    fixture.context->PSSetShaderResources(6, 1, world.GetAddressOf());
    Params params;
    params.config = { 4, mapSize, 0, 0 };
    params.uv = { 0, 0, .5f, renderSize };
    const auto original = fixture.Draw(params, renderSize);
    params.config[0] = 3;
    const auto result = fixture.Draw(params, renderSize);
    int changed = 0, originalErrors = 0, smsrErrors = 0;
    for (int y = 0; y < renderSize; ++y)
        for (int x = 0; x < renderSize; ++x) {
            const int i = y * renderSize + x;
            Check(result[i][0] == 0 || result[i][0] == 1, "SMSR must have no fractional penumbra");
            Check(result[i][0] <= original[i][0], "SMSR must not lighten an existing shadow");
            changed += result[i][0] != original[i][0];
            if (y >= 4 * scale && y < 28 * scale) {
                const float reference = (x + .5f) / scale < 8 + (y + .5f) / (4 * scale) ? 0.0f : 1.0f;
                originalErrors += original[i][0] != reference;
                smsrErrors += result[i][0] != reference;
            }
        }
    Check(changed > 0, "end-to-end silhouette must reconstruct texel interiors");
    Check(smsrErrors < originalErrors, "reconstructed stair silhouette must approach analytic diagonal");
    std::cout << "Diagonal errors vs analytic line: original=" << originalErrors << ", SMSR=" << smsrErrors
              << "; reconstructed pixels=" << changed << '\n';
    std::ofstream ppm(outputPath, std::ios::binary);
    ppm << "P6\n" << renderSize * 2 << ' ' << renderSize << "\n255\n";
    for (int y = 0; y < renderSize; ++y)
        for (const auto* image : { &original, &result }) {
            for (int x = 0; x < renderSize; ++x) {
                unsigned char v = ((*image)[y * renderSize + x][0] > .5f) ? 235 : 30;
                for (int k = 0; k < 3; ++k)
                    ppm.put((char)v);
            }
        }
    // A search cap is a safe fallback, not a substitute endpoint for a shorter line.
    params.smsr[1] = 1;
    const auto capped = fixture.Draw(params, renderSize);
    int cappedChanges = 0;
    for (size_t i = 0; i < capped.size(); ++i)
        cappedChanges += capped[i][0] != original[i][0];
    Check(cappedChanges < changed, "short search limit must reduce reconstruction");

    // Separate actor array, its own texel size, logical actor slice 3. Other slices are lit.
    std::vector<float> actorDepths(16 * 16 * 5, .8f);
    std::fill(actorDepths.begin() + 3 * 16 * 16, actorDepths.begin() + 4 * 16 * 16, .2f);
    auto actors = fixture.Map(16, 5, actorDepths);
    fixture.context->PSSetShaderResources(7, 1, actors.GetAddressOf());
    params.config = { 2, 16, 3, 1 };
    params.uv = { .6f, .6f, .5f, 1 };
    Check(fixture.Draw(params)[0][0] == 0, "actor resolution and logical slice");
    params.config[2] = 4;
    Check(fixture.Draw(params)[0][0] == 1, "must not sample a neighbouring actor slice");
    params.config[0] = 6;
    params.smsr[0] = 0;
    Check(fixture.Draw(params)[0][0] == .375f, "disabled SMSR preserves the original filter path");
    params.smsr[0] = 1;
    Check(fixture.Draw(params)[0][0] == 1, "enabled SMSR bypasses the original filter path");

    // A sloped receiver fills the map with its own depth. No false edge may appear.
    for (int y = 0; y < mapSize; ++y)
        for (int x = 0; x < mapSize; ++x) {
            depths[y * mapSize + x] = .5f + .3f * ((x + .5f) / mapSize - .5f) - .2f * ((y + .5f) / mapSize - .5f);
        }
    world = fixture.Map(mapSize, 5, depths);
    fixture.context->PSSetShaderResources(6, 1, world.GetAddressOf());
    params.config = { 3, mapSize, 0, 0 };
    params.uv = { 0, 0, .5f, renderSize };
    params.plane = { .3f, -.2f, 1, 0 };
    params.smsr[1] = 16;
    const auto slope = fixture.Draw(params, renderSize);
    for (const auto& pixel : slope)
        Check(pixel[0] == 1, "sloped receiver must not self-shadow during traversal");
    // Mirror and transpose the same analytical edge to exercise all scan/visibility directions.
    params.plane = { 0, 0, 1, 0 };
    for (int orientation = 0; orientation < 8; ++orientation) {
        auto transform = [orientation](float x, float y, float size) {
            if (orientation & 1)
                std::swap(x, y);
            if (orientation & 2)
                x = size - x;
            if (orientation & 4)
                y = size - y;
            return std::array<float, 2>{ x, y };
        };
        for (int y = 0; y < mapSize; ++y)
            for (int x = 0; x < mapSize; ++x) {
                auto p = transform(x + .5f, y + .5f, float(mapSize));
                depths[y * mapSize + x] = int(p[0]) < 8 + int(p[1]) / 4 ? .2f : .8f;
            }
        world = fixture.Map(mapSize, 5, depths);
        fixture.context->PSSetShaderResources(6, 1, world.GetAddressOf());
        const auto image = fixture.Draw(params, renderSize);
        for (int y = 0; y < renderSize; ++y)
            for (int x = 0; x < renderSize; ++x) {
                auto p = transform((x + .5f) / scale, (y + .5f) / scale, float(mapSize));
                if (p[1] >= 4 && p[1] < 28) {
                    float expected = p[0] < 8 + p[1] / 4 ? 0.0f : 1.0f;
                    if (image[y * renderSize + x][0] != expected) {
                        throw std::runtime_error("analytical edge orientation " + std::to_string(orientation));
                    }
                }
            }
    }
    std::cout << "Binary output, search cap, actor slices, map boundaries and sloped R16 receiver passed\n";
    std::cout << "Eight mirrored/transposed diagonal reconstructions passed\n";
}

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--compile") {
            std::string source = Read(argv[2]);
            auto vs = Compile(source, "VSMain", (std::string("vs_") + argv[3]).c_str());
            auto ps = Compile(source, "PSMain", (std::string("ps_") + argv[3]).c_str());
            ComPtr<ID3D11ShaderReflection> reflection;
            Check(SUCCEEDED(D3DReflect(ps->GetBufferPointer(), ps->GetBufferSize(), IID_PPV_ARGS(&reflection))),
                  "shader reflection");
            D3D11_SHADER_VARIABLE_DESC variable{};
            Check(SUCCEEDED(reflection->GetConstantBufferByName("PerShadowCB")
                                ->GetVariableByName("shadow_smsr")
                                ->GetDesc(&variable)),
                  "SMSR cbuffer reflected");
            Check(variable.Size == 16 && variable.StartOffset == offsetof(PerShadowCB, shadow_smsr),
                  "SMSR cbuffer offset");
            std::cout << "Native " << argv[3] << " VS/PS + SMSR cbuffer layout passed\n";
            return 0;
        }
        Check(argc == 3, "usage: smsr_tests production.hlsl comparison.ppm");
        ShadowMapQuality q = ShadowMapQualityDefaults();
        Check(!q.smsr && q.smsrMaxSteps == 16, "original mode and bounded search defaults");
        q.smsr = 5;
        q.smsrMaxSteps = 900;
        q.smsrEpsilon = std::numeric_limits<float>::quiet_NaN();
        ShadowMapQualityClamp(&q);
        Check(q.smsr == 1 && q.smsrMaxSteps == 64 && q.smsrEpsilon == SHADOW_MAP_DEFAULT_SMSR_EPSILON, "config clamps");
        Fixture fixture(std::string(kPrefix) + Read(argv[1]) + kEntry);
        VisibilityCases(fixture);
        PerspectiveCascadeSelection(fixture);
        ReceiverPlaneComparisons(fixture);
        AnalyticOccluderDepth(fixture);
        Silhouettes(fixture, argv[2]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
