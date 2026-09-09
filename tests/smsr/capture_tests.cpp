#include "fast/backends/shadow_capture.h"
#include "fast/shadow_map.h"
#include "shadow_cbuffer.inc"
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
using Microsoft::WRL::ComPtr;
#define SPDLOG_INFO(...) ((void)0)
#define SPDLOG_ERROR(...) ((void)0)

namespace Ship {
struct CaptureVariables {
    int request = 0;
    std::string status;
    int GetInteger(const char*, int) {
        return request;
    }
    void SetInteger(const char*, int value) {
        request = value;
    }
    void SetString(const char*, const char* value) {
        status = value;
    }
};
struct Context {
    CaptureVariables variables;
    static Context* GetInstance() {
        static Context context;
        return &context;
    }
    CaptureVariables* GetConsoleVariables() {
        return &variables;
    }
    static std::string GetPathRelativeToAppDirectory(const std::string& path) {
        return path;
    }
};
} // namespace Ship
struct Fixture {
    ComPtr<ID3D11Device> mDevice;
    ComPtr<ID3D11DeviceContext> mContext;
    ComPtr<ID3D11Texture2D> mShadowMapTexture;
    PerShadowCB mPerShadowCbData{};
    bool mShadowSliceValid[SHADOW_MAP_MAX_SLICES]{};
    float mShadowSliceMatrix[SHADOW_MAP_MAX_SLICES][16]{};
    void Capture(int count) {
#include "shadow_capture.inc"
    }
};
static void Check(bool ok, const char* why) {
    if (!ok)
        throw std::runtime_error(why);
}
template <class T> static T Read(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    Check(bool(in), "truncated capture");
    return value;
}
int main() {
    try {
        Fixture fixture;
        Check(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                          &fixture.mDevice, nullptr, &fixture.mContext)),
              "WARP device");
        constexpr UINT width = 37, height = 13, slices = 10;
        std::vector<std::vector<uint16_t>> pixels(slices, std::vector<uint16_t>(width * height));
        std::vector<D3D11_SUBRESOURCE_DATA> initial(slices);
        for (UINT s = 0; s < slices; ++s) {
            for (UINT i = 0; i < width * height; ++i)
                pixels[s][i] = i % 5 ? 65535 : uint16_t(i + s);
            initial[s] = { pixels[s].data(), width * 2, 0 };
            fixture.mShadowSliceValid[s] = true;
            fixture.mShadowSliceMatrix[s][0] = float(s + 1);
        }
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = width;
        desc.Height = height;
        desc.ArraySize = slices;
        desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Format = DXGI_FORMAT_R16_TYPELESS;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
        Check(SUCCEEDED(fixture.mDevice->CreateTexture2D(&desc, initial.data(), &fixture.mShadowMapTexture)),
              "texture array");
        auto* vars = Ship::Context::GetInstance()->GetConsoleVariables();
        fixture.Capture(slices);
        Check(vars->status.empty(), "no automatic capture");
        vars->request = 1;
        fixture.Capture(slices);
        Check(vars->request == 0, "one-shot request consumed");
        std::filesystem::path dir(vars->status);
        std::ifstream in(dir / "world.sds", std::ios::binary);
        Check(Read<uint32_t>(in) == 0x31534453, "SDS1 magic");
        Check(Read<uint32_t>(in) == width && Read<uint32_t>(in) == height && Read<uint32_t>(in) == slices,
              "dimensions");
        for (UINT s = 0; s < slices; ++s)
            for (UINT y = 0; y < height; ++y) {
                const auto runs = Read<uint32_t>(in);
                UINT x = 0;
                for (UINT r = 0; r < runs; ++r) {
                    const auto value = Read<uint16_t>(in);
                    const auto length = Read<uint32_t>(in);
                    Check(length > 0 && length <= width - x, "bounded row run");
                    for (UINT end = x + length; x < end; ++x)
                        Check(value == pixels[s][y * width + x], "lossless GPU depth including padded rows and slices");
                }
                Check(x == width, "complete row");
            }
        Check(in.peek() == std::char_traits<char>::eof(), "no extra depth data");
        in.close();
        std::ifstream info(dir / "capture.json");
        nlohmann::json metadata;
        info >> metadata;
        info.close();
        Check(metadata["complete"] == true && metadata["slice_matrices"][9][0] == 10.0f,
              "actual slice matrix metadata");
        Check(metadata["shadow_smsr"].size() == 4, "receiver settings exported");
        const auto saved = vars->status;
        fixture.Capture(slices);
        Check(vars->status == saved, "capture not repeated next frame");
        vars->request = 1;
        fixture.Capture(0);
        Check(vars->request == 0 && vars->status.find("Capture failed:") == 0, "inactive map failure reported once");
        std::filesystem::remove(dir / "world.sds");
        std::filesystem::remove(dir / "capture.json");
        std::filesystem::remove(dir);
        std::cout << "Lossless ten-slice WARP capture, metadata, one-shot behavior and inactive-map failure passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
