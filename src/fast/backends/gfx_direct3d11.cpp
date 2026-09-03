#ifdef ENABLE_DX11

#include <cstdio>
#include <vector>
#include <fstream>
#include <filesystem>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cmath>

#include <map>
#include <unordered_map>

#include <windows.h>
#include <versionhelpers.h>
#include <wrl/client.h>

#include <dxgi1_3.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#ifndef _LANGUAGE_C
#define _LANGUAGE_C
#endif

#include "fast/backends/gfx_window_manager_api.h"
#include "fast/backends/gfx_direct3d_common.h"

#define DECLARE_GFX_DXGI_FUNCTIONS
#include "fast/backends/gfx_dxgi.h"

#include "fast/backends/gfx_screen_config.h"
#include "ship/window/gui/Gui.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"
#include "ship/window/Window.h"

#include "fast/backends/gfx_rendering_api.h"
#include "fast/interpreter.h"

#include <prism/processor.h>
#include "ship/config/ConsoleVariable.h"
#include <ship/resource/factory/ShaderFactory.h>
#include <ship/resource/ResourceManager.h>
#include "spdlog/spdlog.h"
#include "nlohmann/json.hpp"

#define DEBUG_D3D 0

using namespace Microsoft::WRL; // For ComPtr

namespace Fast {

GfxRenderingAPIDX11::~GfxRenderingAPIDX11() {
    // Before anything else: the prewarm threads read a member of this object, so none may still be running.
    JoinPrewarm();

}

GfxRenderingAPIDX11::GfxRenderingAPIDX11(GfxWindowBackendDXGI* backend) {
    mWindowBackend = backend;
}

void GfxRenderingAPIDX11::CreateDepthStencilObjects(uint32_t width, uint32_t height, uint32_t msaa_count,
                                                    ID3D11DepthStencilView** view, ID3D11ShaderResourceView** srv) {
    D3D11_TEXTURE2D_DESC texture_desc;
    texture_desc.Width = width;
    texture_desc.Height = height;
    texture_desc.MipLevels = 1;
    texture_desc.ArraySize = 1;
    // SOH [Enhancement] World light casting: use a combined depth+stencil format on ALL feature levels
    // so the stencil light-volume technique always has a stencil plane.
    texture_desc.Format = DXGI_FORMAT_R24G8_TYPELESS;
    texture_desc.SampleDesc.Count = msaa_count;
    texture_desc.SampleDesc.Quality = 0;
    texture_desc.Usage = D3D11_USAGE_DEFAULT;
    texture_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | (srv != nullptr ? D3D11_BIND_SHADER_RESOURCE : 0);
    texture_desc.CPUAccessFlags = 0;
    texture_desc.MiscFlags = 0;

    ComPtr<ID3D11Texture2D> texture;
    ThrowIfFailed(mDevice->CreateTexture2D(&texture_desc, nullptr, texture.GetAddressOf()));

    D3D11_DEPTH_STENCIL_VIEW_DESC view_desc;
    view_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; // SOH [Enhancement] world light casting (depth+stencil)
    view_desc.Flags = 0;
    if (msaa_count > 1) {
        view_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
        view_desc.Texture2DMS.UnusedField_NothingToDefine = 0;
    } else {
        view_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        view_desc.Texture2D.MipSlice = 0;
    }

    ThrowIfFailed(mDevice->CreateDepthStencilView(texture.Get(), &view_desc, view));

    if (srv != nullptr) {
        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
        // SOH [Enhancement] World light casting: depth read view matching the R24G8 depth+stencil texture
        // (the X8 part is the stencil byte, ignored by GetPixelDepth's compute shader).
        srv_desc.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        srv_desc.ViewDimension = msaa_count > 1 ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MostDetailedMip = 0;
        srv_desc.Texture2D.MipLevels = -1;

        ThrowIfFailed(mDevice->CreateShaderResourceView(texture.Get(), &srv_desc, srv));
    }
}
namespace {
// SOH [Enhancement] Shader model the scene shaders are built against.
//
// Texture2DArray.Gather returns a whole 2x2 texel footprint in ONE instruction, which is what lets the
// shadow kernel cost four texture instructions instead of sixteen for identical output. It first exists in
// Shader Model 4.1, and this renderer accepts adapters down to feature level 10_0, which only has 4.0 -- so
// the shader carries both kernels behind o_shadow_gather and this picks between them.
//
// Written exactly once, in Init, straight after the device comes up and before any shader is built; read
// from the prewarm threads afterwards, which is why it must never move again. Declared up here so the probe
// below can sit beside it; the profile strings it selects live with the rest of the compile machinery
// further down.
bool sShadowGather = false;

// Does this machine actually build and load that kernel?
//
// The feature level says Shader Model 4.1 is available and that is the documented home of
// Texture2DArray.Gather -- but "documented" is not "verified on the machine in front of us", and the cost of
// being wrong here is not a slower shader. Every receiver shader is compiled at RUNTIME, and a compile
// failure on that path is unhandled: a dialog, and the process goes down with it. So the question is asked
// once, up front, by the smallest shader that can ask it, rather than by the first material the player
// happens to walk past.
//
// Both halves are asked. The compiler settles whether the intrinsic exists at this profile;
// CreatePixelShader settles whether the driver accepts the bytecode it produced. Either one saying no drops
// the session to the 4.0 kernel, which draws exactly the same picture out of four times the fetches.
// SOH [Enhancement] FXAA -- an alternative to MSAA that works on the finished image instead of on geometry.
//
// The two are not variations of one setting; they attack different things and cost differently. MSAA takes
// several depth/coverage samples per pixel along POLYGON EDGES, so it costs in proportion to the geometry
// and to the render target's size -- at 4x it is four times the depth work and four times the colour
// bandwidth -- and it does nothing at all for an edge that is inside a texture rather than between two
// triangles. FXAA is one full-screen pass over the finished frame that finds high-contrast edges and blurs
// ALONG them, so it costs a fixed few milliseconds regardless of scene complexity, catches texture and alpha
// edges MSAA cannot see, and softens the image slightly because it cannot tell a real edge from a drawn one.
//
// Kept as a source string here rather than in the shader template because it shares nothing with it: no
// combiner, no prism options, one variant forever.
// SOH [Enhancement] The shadow-map viewer. Draws one slice of the cascade array into a corner of the screen.
//
// Every other diagnostic this renderer has looks at the RECEIVER -- what the shading pixel was handed. None
// of them shows what the depth pass STORED, and a shadow artefact can live in either half. When receiver-side
// reasoning stops making progress, this is the half left to look at.
//
// Depth is remapped rather than shown raw. A cascade's stored values crowd into a narrow band near the near
// plane, so drawn directly the whole slice is one flat shade and reveals nothing; the range is stretched
// around whatever this slice actually contains. Cleared texels (nothing was drawn there) come out BLUE so an
// empty slice cannot be mistaken for a dark one, which is exactly the confusion that makes a missing caster
// look like a present one.
static const char kShadowMapViewShader[] = R"HLSL(
Texture2DArray<float> t_shadow : register(t0);
SamplerState s_shadow : register(s0);

cbuffer ShadowViewCB : register(b0) {
    float slice;      // which slice of the array to draw
    float depth_lo;   // stretch the visible range: everything at or below this is black
    float depth_hi;   // ...and at or above this is white
    float view_pad;
};

struct VOut {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

// The same vertex-id triangle the FXAA pass uses; the viewport is what confines it to a corner.
VOut VSMain(uint id : SV_VertexID) {
    VOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv = uv;
    o.pos = float4((uv * float2(2.0, -2.0)) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

float4 PSMain(VOut i) : SV_TARGET {
    float d = t_shadow.SampleLevel(s_shadow, float3(i.uv, slice), 0);
    // A texel at the far plane was never written by any caster. Blue, so "nothing here" reads differently
    // from "something far away" -- the two are the same number and mean opposite things.
    if (d >= 0.99999) {
        return float4(0.05, 0.10, 0.45, 1.0);
    }
    float t = saturate((d - depth_lo) / max(depth_hi - depth_lo, 1e-6));
    return float4(t, t, t, 1.0);
}
)HLSL";

static const char kFxaaShader[] = R"HLSL(
Texture2D t_src : register(t0);
SamplerState s_src : register(s0);

cbuffer FxaaCB : register(b0) {
    float2 rcp_frame; // 1 / render target size, in pixels
    float2 fxaa_pad;
};

struct VOut {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

// One oversized triangle covering the screen, built from the vertex id, so the pass needs no vertex buffer
// and no input layout at all.
VOut VSMain(uint id : SV_VertexID) {
    VOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv = uv;
    o.pos = float4((uv * float2(2.0, -2.0)) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

float FxaaLuma(float3 c) {
    return dot(c, float3(0.299, 0.587, 0.114));
}

float4 PSMain(VOut i) : SV_TARGET {
    float3 rgbM = t_src.Sample(s_src, i.uv).rgb;
    float lumaM = FxaaLuma(rgbM);
    float lumaNW = FxaaLuma(t_src.Sample(s_src, i.uv + (float2(-1.0, -1.0) * rcp_frame)).rgb);
    float lumaNE = FxaaLuma(t_src.Sample(s_src, i.uv + (float2( 1.0, -1.0) * rcp_frame)).rgb);
    float lumaSW = FxaaLuma(t_src.Sample(s_src, i.uv + (float2(-1.0,  1.0) * rcp_frame)).rgb);
    float lumaSE = FxaaLuma(t_src.Sample(s_src, i.uv + (float2( 1.0,  1.0) * rcp_frame)).rgb);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    float range = lumaMax - lumaMin;

    // Flat neighbourhood: leave it exactly as it was. This early-out is most of why the pass is cheap, and
    // it is also what keeps large areas of flat colour from being touched at all.
    if (range < max(1.0 / 16.0, lumaMax * (1.0 / 8.0))) {
        return float4(rgbM, 1.0);
    }

    // The edge's direction, as the gradient of luma across the four corners, turned ninety degrees.
    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    // Damp the direction where the whole neighbourhood is dark: luma differences there are small in absolute
    // terms but large in ratio, and without this they steer the blur wildly.
    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * (1.0 / 8.0), 1.0 / 128.0);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, -8.0, 8.0) * rcp_frame;

    // Two taps inside the edge, then two further out. If the wider pair strays outside the luma range the
    // neighbourhood had, it has wandered onto something that is not this edge, so the narrow pair is used.
    float3 rgbA = 0.5 * (t_src.Sample(s_src, i.uv + (dir * ((1.0 / 3.0) - 0.5))).rgb +
                         t_src.Sample(s_src, i.uv + (dir * ((2.0 / 3.0) - 0.5))).rgb);
    float3 rgbB = (rgbA * 0.5) + (0.25 * (t_src.Sample(s_src, i.uv + (dir * -0.5)).rgb +
                                          t_src.Sample(s_src, i.uv + (dir *  0.5)).rgb));
    float lumaB = FxaaLuma(rgbB);
    return float4(((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB, 1.0);
}
)HLSL";

bool ShadowGatherProbe(pD3DCompile compileFn, ID3D11Device* device) {
    if (compileFn == nullptr || device == nullptr) {
        return false;
    }
    static const char kProbe[] = "Texture2DArray<float> t : register(t0);\n"
                                 "SamplerState s : register(s0);\n"
                                 "float4 PSMain(float4 p : SV_POSITION) : SV_TARGET {\n"
                                 "    return t.Gather(s, float3(p.xy, 0.0));\n"
                                 "}\n";
    ComPtr<ID3DBlob> ps, err;
    if (FAILED(compileFn(kProbe, sizeof(kProbe) - 1, nullptr, nullptr, nullptr, "PSMain", "ps_4_1", 0, 0,
                         ps.GetAddressOf(), err.GetAddressOf())) ||
        ps == nullptr) {
        SPDLOG_WARN("Shadow map: Texture2DArray.Gather did not compile at ps_4_1 ({}). Using the 4.0 kernel.",
                    err != nullptr ? (const char*)err->GetBufferPointer() : "(no message)");
        return false;
    }
    ComPtr<ID3D11PixelShader> shader;
    if (FAILED(
            device->CreatePixelShader(ps->GetBufferPointer(), ps->GetBufferSize(), nullptr, shader.GetAddressOf()))) {
        SPDLOG_WARN("Shadow map: the driver would not load a ps_4_1 gather shader. Using the 4.0 kernel.");
        return false;
    }
    return true;
}
} // namespace

static bool CreateDeviceFunc(class GfxRenderingAPIDX11* self, bool SoftwareRenderer) {
#if DEBUG_D3D
    UINT device_creation_flags = D3D11_CREATE_DEVICE_DEBUG;
#else
    UINT device_creation_flags = 0;
#endif
    bool CreationFailed = false;
    const char HardwareText[320] = "\nUsing software renderer. Performance issues are to be expected.\n\n"
                                   "Please check your preferred GPU in Windows graphics or GPU driver settings and "
                                   "make sure you have the correct GPU drivers installed.\n\n"
                                   "You can also try to change the graphic backend of the port in its config file.\n"
                                   "Window->Backend->Id\n"
                                   "0 = DX11, 1 = OpenGL";
    const char SoftwareText[33] = "\nUsing software renderer failed.";

    if (SoftwareRenderer) {
        SPDLOG_INFO("Using software renderer.");
    }

    HRESULT res = self->mDX11CreateDevice(
        NULL, SoftwareRenderer ? D3D_DRIVER_TYPE_WARP : D3D_DRIVER_TYPE_HARDWARE, nullptr, device_creation_flags, NULL,
        NULL, D3D11_SDK_VERSION, self->mDevice.GetAddressOf(), &self->mFeatureLevel, self->mContext.GetAddressOf());

    // Get and log name of adapter
    IDXGIDevice* DXGIDevice = nullptr;
    IDXGIAdapter* Adapter = nullptr;
    DXGI_ADAPTER_DESC adapterDesc;
    std::wstring adapterName;
    char adapterNameCStr[128] = "";
    char error_message[512];
    HRESULT res2;

    res2 = self->mDevice->QueryInterface(__uuidof(IDXGIDevice), (void**)&DXGIDevice);
    if (SUCCEEDED(res2)) {
        res2 = DXGIDevice->GetAdapter(&Adapter);
        if (SUCCEEDED(res2)) {
            res2 = Adapter->GetDesc(&adapterDesc);
            if (SUCCEEDED(res2)) {
                adapterName = adapterDesc.Description;
                wcstombs(adapterNameCStr, adapterName.c_str(), 128);
            }
            Adapter->Release();
        }
        DXGIDevice->Release();
    }
    SPDLOG_INFO("Using D3D adapter: {0}", adapterNameCStr);

    if (FAILED(res)) {
        CreationFailed = true;
        SPDLOG_WARN("Failed to create a D3D device. HRESULT: 0x{0:08x}", res);
        snprintf(error_message, sizeof(error_message), "Failed to create a D3D device on %s\nHRESULT: 0x%08X%s",
                 adapterNameCStr, res, SoftwareRenderer ? SoftwareText : HardwareText);
    }

    else if (self->mFeatureLevel < D3D_FEATURE_LEVEL_10_0) {
        CreationFailed = true;
        SPDLOG_WARN("D3D adapter doesn't support D3D feature level 10_0 or greater.");
        snprintf(error_message, sizeof(error_message), "%s doesn't support D3D feature level 10_0 or greater.%s",
                 adapterNameCStr, SoftwareRenderer ? SoftwareText : HardwareText);
    }

    else if (self->mFeatureLevel < D3D_FEATURE_LEVEL_10_1) {
        SPDLOG_WARN("D3D adapter doesn't support D3D feature level 10_1 or greater. MSAA setting will be ignored.");

    } else {
        // Check for Compute Shader support
        if (self->mDevice != NULL) {
            D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS features;
            self->mDevice->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &features,
                                               sizeof(D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS));
            if (features.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x == false) {
                CreationFailed = true;
                SPDLOG_WARN("D3D adapter doesn't support compute shaders.");
                snprintf(error_message, sizeof(error_message), "%s doesn't support compute shaders.%s", adapterNameCStr,
                         SoftwareRenderer ? SoftwareText : HardwareText);
            }
        }
    }

    if (CreationFailed) {
        if (self->mContext) {
            self->mContext->Release();
        }
        if (self->mDevice) {
            self->mDevice->Release();
        }
        MessageBoxA(self->mWindowBackend->GetWindowHandle(), error_message, "Warning", MB_OK | MB_ICONWARNING);
        return false;
    }
    return true;
};

void GfxRenderingAPIDX11::Init() {
    // Load d3d11.dll
    mDX11Module = LoadLibraryW(L"d3d11.dll");
    if (mDX11Module == nullptr) {
        ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()), mWindowBackend->GetWindowHandle(), "d3d11.dll not found");
    }
    mDX11CreateDevice = (PFN_D3D11_CREATE_DEVICE)GetProcAddress(mDX11Module, "D3D11CreateDevice");

    // Load D3DCompiler_47.dll
    mCompilerModule = LoadLibraryW(L"D3DCompiler_47.dll");
    if (mCompilerModule == nullptr) {
        ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()), mWindowBackend->GetWindowHandle(),
                      "D3DCompiler_47.dll not found");
    }
    mD3dCompile = (pD3DCompile)GetProcAddress(mCompilerModule, "D3DCompile");

    // Create D3D11 mDevice

    mWindowBackend->CreateFactoryAndDevice(DEBUG_D3D, 11, this, CreateDeviceFunc);

    // SOH [Enhancement] Which shadow kernel this session gets, settled here and never again.
    //
    // It decides the profile every scene shader is compiled against and the seed their disk cache is keyed
    // by, so it has to be fixed before the first shader is built and must not move while any of them are
    // alive. This is the earliest point where both things it depends on exist: the device (for its feature
    // level, and to be asked whether it will load the result) and the compiler.
    //
    // Feature level 10_1 is Shader Model 4.1, which is where Texture2DArray.Gather begins. Below it -- or if
    // the probe says no -- the kernel falls back to fetching its four texels one at a time, for the same
    // picture at four times the texture instructions.
    sShadowGather = mFeatureLevel >= D3D_FEATURE_LEVEL_10_1 && ShadowGatherProbe(mD3dCompile, mDevice.Get());
    SPDLOG_INFO("Shadow map kernel: {} (feature level {:#x})",
                sShadowGather ? "gathered 2x2 footprints, Shader Model 4.1" : "one fetch per texel, Shader Model 4.0",
                (unsigned)mFeatureLevel);

    // Create the swap chain
    mWindowBackend->CreateSwapChain(mDevice.Get(), [this]() {
        mFrameBuffers[0].render_target_view.Reset();
        mTextures[mFrameBuffers[0].texture_id].texture.Reset();
        mContext->ClearState();
        mContext->Flush();

        mLastShaderProgram = nullptr;
        mLastVertexBufferStride = 0;
        mLastBlendState.Reset();
        for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
            mLastResourceViews[i].Reset();
            mLastSamplerStates[i].Reset();
        }
        mLastDepthTest = -1;
        mLastDepthMask = -1;
        mLastZmodeDecal = -1;
        mLastStencilMode = -1; // SOH [Enhancement] world light casting / actor shadows
        mLastPrimitaveTopology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
        // SOH [Enhancement] ClearState above unbinds the constant buffers too, so what the toon buffer was
        // believed to hold is no longer a reason to skip re-uploading it.
        mPerToonCbValid = false;
        mShadowCbDirty = true;
    });

    // Create D3D Debug mDevice if in debug mode

#if DEBUG_D3D
    ThrowIfFailed(mDevice->QueryInterface(__uuidof(ID3D11Debug), (void**)debug.GetAddressOf()),
                  mWindowBackend->GetWindowHandle(), "Failed to get ID3D11Debug device.");
#endif

    // Create the default framebuffer which represents the window
    FramebufferDX11& fb = mFrameBuffers[CreateFramebuffer()];

    // Check the size of the window
    DXGI_SWAP_CHAIN_DESC1 swap_chain_desc;
    ThrowIfFailed(mWindowBackend->GetSwapChain()->GetDesc1(&swap_chain_desc));
    mTextures[fb.texture_id].width = swap_chain_desc.Width;
    mTextures[fb.texture_id].height = swap_chain_desc.Height;
    fb.msaa_level = 1;

    for (uint32_t sample_count = 1; sample_count <= D3D11_MAX_MULTISAMPLE_SAMPLE_COUNT; sample_count++) {
        ThrowIfFailed(mDevice->CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM, sample_count,
                                                             &mMsaaNumQualityLevels[sample_count - 1]));
    }

    // Create main vertex buffer

    D3D11_BUFFER_DESC vertex_buffer_desc;
    ZeroMemory(&vertex_buffer_desc, sizeof(D3D11_BUFFER_DESC));

    vertex_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
    // A ring of batches rather than one. DrawTriangles appends into this instead of renaming it per draw, so
    // it no longer has to match the CPU mBufVbo allocation -- it only has to be able to take the largest
    // single flush, which kVertexRingBytes derives from the same two constants mBufVbo does.
    vertex_buffer_desc.ByteWidth = kVertexRingBytes;
    vertex_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vertex_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    vertex_buffer_desc.MiscFlags = 0;

    ThrowIfFailed(mDevice->CreateBuffer(&vertex_buffer_desc, nullptr, mVertexBuffer.GetAddressOf()),
                  mWindowBackend->GetWindowHandle(), "Failed to create vertex buffer.");
    // A fresh buffer holds nothing, so the ring starts at the beginning and the binding record must not
    // claim an offset is still bound from the buffer this one replaces.
    mVertexRingOffset = 0;
    mLastVertexBufferOffset = UINT32_MAX;
    mLastVertexBufferStride = 0;

    // Create per-frame constant buffer

    D3D11_BUFFER_DESC constant_buffer_desc;
    ZeroMemory(&constant_buffer_desc, sizeof(D3D11_BUFFER_DESC));

    constant_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
    constant_buffer_desc.ByteWidth = sizeof(PerFrameCB);
    constant_buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    constant_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    constant_buffer_desc.MiscFlags = 0;

    ThrowIfFailed(mDevice->CreateBuffer(&constant_buffer_desc, nullptr, mPerFrameCb.GetAddressOf()),
                  mWindowBackend->GetWindowHandle(), "Failed to create per-frame constant buffer.");

    // Create per-draw constant buffer

    constant_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
    constant_buffer_desc.ByteWidth = sizeof(PerDrawCB);
    constant_buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    constant_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    constant_buffer_desc.MiscFlags = 0;

    ThrowIfFailed(mDevice->CreateBuffer(&constant_buffer_desc, nullptr, mPerDrawCb.GetAddressOf()),
                  mWindowBackend->GetWindowHandle(), "Failed to create per-draw constant buffer.");

    // SOH [Enhancement] Create the toon-lighting constant buffer (register b2), uploaded per toon draw.
    constant_buffer_desc.ByteWidth = sizeof(PerToonCB);
    ThrowIfFailed(mDevice->CreateBuffer(&constant_buffer_desc, nullptr, mPerToonCb.GetAddressOf()),
                  mWindowBackend->GetWindowHandle(), "Failed to create toon-lighting constant buffer.");

    // SOH [Enhancement] Create the shadow-cascade constant buffer (register b3). Created unconditionally
    // (it is a few hundred bytes) so the binding in StartFrame never has to branch; it stays zeroed --
    // cascade count 0, which the shader reads as "no shadow map" -- until a depth pass fills it.
    static_assert(sizeof(PerShadowCB) % 16 == 0, "constant buffers must be a multiple of 16 bytes");
    constant_buffer_desc.ByteWidth = sizeof(PerShadowCB);
    ThrowIfFailed(mDevice->CreateBuffer(&constant_buffer_desc, nullptr, mPerShadowCb.GetAddressOf()),
                  mWindowBackend->GetWindowHandle(), "Failed to create shadow-cascade constant buffer.");

    // Create compute shader that can be used to retrieve depth buffer values

    const char* shader_source = R"(
sampler my_sampler : register(s0);
Texture2D<float> tex : register(t0);
StructuredBuffer<int2> coord : register(t1);
RWStructuredBuffer<float> output : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID) {
    output[DTid.x] = tex.Load(int3(coord[DTid.x], 0));
}
)";

    const char* shader_source_msaa = R"(
sampler my_sampler : register(s0);
Texture2DMS<float, 2> tex : register(t0);
StructuredBuffer<int2> coord : register(t1);
RWStructuredBuffer<float> output : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID) {
    output[DTid.x] = tex.Load(coord[DTid.x], 0);
}
)";

#if DEBUG_D3D
    UINT compile_flags = D3DCOMPILE_DEBUG;
#else
    UINT compile_flags = D3DCOMPILE_OPTIMIZATION_LEVEL2;
#endif

    ComPtr<ID3DBlob> cs, error_blob;
    HRESULT hr;

    hr = mD3dCompile(shader_source, strlen(shader_source), nullptr, nullptr, nullptr, "CSMain", "cs_4_0", compile_flags,
                     0, cs.GetAddressOf(), error_blob.GetAddressOf());

    if (FAILED(hr)) {
        char* err = (char*)error_blob->GetBufferPointer();
        MessageBoxA(mWindowBackend->GetWindowHandle(), err, "Error", MB_OK | MB_ICONERROR);
        throw hr;
    }

    ThrowIfFailed(mDevice->CreateComputeShader(cs->GetBufferPointer(), cs->GetBufferSize(), nullptr,
                                               mComputeShader.GetAddressOf()));

    hr = mD3dCompile(shader_source_msaa, strlen(shader_source_msaa), nullptr, nullptr, nullptr, "CSMain", "cs_4_1",
                     compile_flags, 0, mComputeShaderMsaaBlob.GetAddressOf(), error_blob.ReleaseAndGetAddressOf());

    if (FAILED(hr)) {
        char* err = (char*)error_blob->GetBufferPointer();
        MessageBoxA(mWindowBackend->GetWindowHandle(), err, "Error", MB_OK | MB_ICONERROR);
        throw hr;
    }

    // Create ImGui

    Ship::GuiWindowInitData window_impl;
    window_impl.Dx11 = { mWindowBackend->GetWindowHandle(), mContext.Get(), mDevice.Get() };
    Ship::Context::GetInstance()->GetWindow()->GetGui()->Init(window_impl);
}

int GfxRenderingAPIDX11::GetMaxTextureSize() {
    return D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION;
}

const char* GfxRenderingAPIDX11::GetName() {
    return "DirectX 11";
}

struct GfxClipParameters GfxRenderingAPIDX11::GetClipParameters() {
    return { true, false };
}

void GfxRenderingAPIDX11::UnloadShader(struct ShaderProgram* old_prg) {
}

void GfxRenderingAPIDX11::LoadShader(struct ShaderProgram* new_prg) {
    mShaderProgram = (struct ShaderProgramD3D11*)new_prg;
}

// SOH [Enhancement] Compiled-shader cache on disk.
//
// The renderer builds its shaders at run time, one variant per material, and compiles each the first time a
// draw needs it -- synchronously, inside the frame. That is the hitch felt when new geometry rotates into
// view, and it is felt worst when the shadow map is switched on, because every receiver in the scene needs a
// variant it has never needed before and they all arrive at once.
//
// Compiling is the expensive part and its result never changes, so it belongs on disk. After the first run
// the compiler is not invoked at all: the bytecode is read back and handed straight to the device.
//
// The key is a hash of the EXPANDED source. That is what makes invalidation automatic and total -- the
// expansion already encodes the variant, the template and every option that produced it, so editing the
// shader, adding an option or changing the combiner all produce different text and therefore different
// keys. There is no version number to remember to bump. Two independent hashes are stored rather than one,
// with the source length beside them, because a collision would mean running the wrong shader.
//
// Entries orphaned by a shader edit are left behind rather than swept: they are a few kilobytes each, and a
// sweep would have to know which keys are still reachable, which is only knowable after every variant the
// game can build has been built.
namespace {

#if DEBUG_D3D
constexpr UINT kShaderCompileFlags = D3DCOMPILE_DEBUG;
#else
constexpr UINT kShaderCompileFlags = D3DCOMPILE_OPTIMIZATION_LEVEL2;
#endif

// Profile the shaders are compiled against, and the seed their disk cache is keyed by. Both follow
// sShadowGather: only the shadow kernel needs Shader Model 4.1, but FXC parses the whole file for each entry
// point, so an intrinsic the profile does not have is an error whether or not that entry point can reach it
// -- which is exactly how a shadow-only mistake takes down the vertex compile first. And the two builds must
// never share a cache entry: the SOURCE differs only for shadow variants, but the BYTECODE differs for every
// one of them, and 4_1 bytecode handed to a 10_0 device fails to create.
const char* ShaderProfileVs() {
    return sShadowGather ? "vs_4_1" : "vs_4_0";
}
const char* ShaderProfilePs() {
    return sShadowGather ? "ps_4_1" : "ps_4_0";
}
uint64_t ShaderCacheSeed() {
    return (uint64_t)kShaderCompileFlags ^ (sShadowGather ? 0x9E37u : 0u);
}

constexpr uint32_t kShaderCacheMagic = 0x53535546; // 'FUSS'
constexpr uint32_t kShaderCacheVersion = 1;

struct ShaderCacheHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t hashA;
    uint64_t hashB;
    uint64_t sourceLength;
    uint64_t vsSize;
    uint64_t psSize;
};

uint64_t ShaderHash(const char* data, size_t len, uint64_t seed) {
    uint64_t h = seed;
    for (size_t i = 0; i < len; i++) {
        h = (h ^ (uint8_t)data[i]) * 0x100000001B3ull;
        h ^= h >> 31;
    }
    return h;
}

// Empty when the cache directory cannot be established, which switches the cache off rather than failing.
std::string ShaderCachePath(uint64_t hashA, uint64_t hashB) {
    try {
        const std::filesystem::path dir = Ship::Context::GetPathRelativeToAppDirectory("shadercache-dx11");
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            return "";
        }
        char name[40];
        snprintf(name, sizeof(name), "%016llx%016llx.bin", (unsigned long long)hashA, (unsigned long long)hashB);
        return (dir / name).string();
    } catch (...) {
        return "";
    }
}

bool ShaderCacheLoad(const std::string& path, uint64_t hashA, uint64_t hashB, size_t sourceLength,
                     std::vector<uint8_t>& vs, std::vector<uint8_t>& ps) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    ShaderCacheHeader h{};
    f.read((char*)&h, sizeof(h));
    // Every field is checked, including the ones the filename already encodes: a truncated write, a stale
    // format or a name collision must all read as "not cached" and fall through to compiling.
    if (!f || h.magic != kShaderCacheMagic || h.version != kShaderCacheVersion || h.hashA != hashA ||
        h.hashB != hashB || h.sourceLength != sourceLength || h.vsSize == 0 || h.psSize == 0 ||
        h.vsSize > (1u << 24) || h.psSize > (1u << 24)) {
        return false;
    }
    vs.resize((size_t)h.vsSize);
    ps.resize((size_t)h.psSize);
    f.read((char*)vs.data(), vs.size());
    f.read((char*)ps.data(), ps.size());
    return (bool)f;
}

void ShaderCacheStore(const std::string& path, uint64_t hashA, uint64_t hashB, size_t sourceLength,
                      const void* vs, size_t vsSize, const void* ps, size_t psSize) {
    // Written to a temporary and renamed, so a crash or a second instance mid-write cannot leave a
    // half-file that later reads as a valid header with a truncated body.
    static std::atomic<uint32_t> counter{ 0 };
    const std::string tmp = path + "." + std::to_string(counter.fetch_add(1)) + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            return;
        }
        ShaderCacheHeader h{ kShaderCacheMagic, kShaderCacheVersion, hashA, hashB,
                             (uint64_t)sourceLength, (uint64_t)vsSize, (uint64_t)psSize };
        f.write((const char*)&h, sizeof(h));
        f.write((const char*)vs, vsSize);
        f.write((const char*)ps, psSize);
        if (!f) {
            return;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
    }
}

} // namespace

// SOH [Enhancement] Compile the variants an option turning ON is about to need, several at a time.
//
// Shaders are built per material and compiled the first time a draw asks for one, inside the frame that
// asks. Switching the shadow map on hands every receiver in the scene a variant it has never needed, and
// they arrive one after another -- so the wait is the sum of them all, on one thread, while the frame is
// held. The set is knowable in advance, though: it is exactly the variants already in the pool with the new
// bit set, and each is independent of the others.
//
// The threads only compile and write cache entries. They never touch the device, the shader pool or any
// renderer state, and they are all joined before this returns -- so nothing here outlives the call and
// nothing races a frame. The main thread takes a share of the work rather than idling.
//
// Deliberately best effort. Anything that fails or is skipped -- an unwritable cache, a compile error, a
// variant the pool did not predict -- is compiled on demand exactly as it was before, and a variant warmed
// but never drawn costs a cache entry nobody reads.
void GfxRenderingAPIDX11::PrewarmShaderVariants(uint32_t extraOptionBits) {
    if (extraOptionBits == 0 || mD3dCompile == nullptr) {
        return;
    }

    // The sources are built here rather than in the workers: building one runs the preprocessor against a
    // template fetched through the resource manager, which is not something to call off the main thread.
    std::vector<std::string> sources;
    {
        std::vector<std::pair<uint64_t, uint32_t>> wanted;
        for (const auto& entry : mShaderProgramPool) {
            const uint64_t id0 = entry.first.first;
            const uint32_t id1 = entry.first.second | extraOptionBits;
            if (id1 == entry.first.second) {
                continue; // already a variant with the option set
            }
            wanted.emplace_back(id0, id1);
        }
        for (const auto& w : wanted) {
            if (mShaderProgramPool.count(std::make_pair(w.first, w.second)) != 0) {
                continue; // already compiled this session
            }
            CCFeatures cc_features;
            gfx_cc_get_features(w.first, w.second, &cc_features);
            size_t numFloats = 0;
            sources.push_back(gfx_direct3d_common_build_shader(numFloats, cc_features, false,
                                                               mCurrentFilterMode == FILTER_THREE_POINT, mSrgbMode));
        }
    }
    if (sources.empty()) {
        return;
    }

    // Held on the object rather than on the stack: the threads outlive this call now, and what they read
    // has to outlive them. Any previous batch is finished first, so there is only ever one live set.
    JoinPrewarm();
    mPrewarmSources = std::move(sources);
    mPrewarmNext.store(0);
    mPrewarmRemaining.store((uint32_t)mPrewarmSources.size());

    auto work = [this]() {
        for (;;) {
            const size_t i = mPrewarmNext.fetch_add(1);
            if (i >= mPrewarmSources.size()) {
                return;
            }
            const std::string& src = mPrewarmSources[i];
            const uint64_t hashA = ShaderHash(src.data(), src.size(), 0xCBF29CE484222325ull ^ ShaderCacheSeed());
            const uint64_t hashB = ShaderHash(src.data(), src.size(), 0x9E3779B97F4A7C15ull ^ ShaderCacheSeed());
            const std::string path = ShaderCachePath(hashA, hashB);
            if (path.empty()) {
                mPrewarmRemaining.fetch_sub(1);
                continue;
            }
            std::vector<uint8_t> haveVs, havePs;
            if (ShaderCacheLoad(path, hashA, hashB, src.size(), haveVs, havePs)) {
                mPrewarmRemaining.fetch_sub(1);
                continue; // a previous run already paid for this one
            }
            ComPtr<ID3DBlob> vs, ps, err;
            if (FAILED(mD3dCompile(src.data(), src.size(), nullptr, nullptr, nullptr, "VSMain", ShaderProfileVs(),
                                   kShaderCompileFlags, 0, vs.GetAddressOf(), err.GetAddressOf())) ||
                FAILED(mD3dCompile(src.data(), src.size(), nullptr, nullptr, nullptr, "PSMain", ShaderProfilePs(),
                                   kShaderCompileFlags, 0, ps.GetAddressOf(), err.GetAddressOf()))) {
                // Left for the on-demand path, which reports it properly and takes the process down. Warming
                // must not be where a broken shader is discovered, and must never be where it is hidden.
                mPrewarmRemaining.fetch_sub(1);
                continue;
            }
            ShaderCacheStore(path, hashA, hashB, src.size(), vs->GetBufferPointer(), vs->GetBufferSize(),
                             ps->GetBufferPointer(), ps->GetBufferSize());
            mPrewarmRemaining.fetch_sub(1);
        }
    };

    unsigned int threads = std::thread::hardware_concurrency();
    threads = threads == 0 ? 2u : (threads > 8u ? 8u : threads);
    threads = (unsigned int)std::min<size_t>(threads, mPrewarmSources.size());

    // Started and left running. The calling thread does NOT take a share any more -- it is the frame, and
    // the whole point is to give it back. The application holds the option off until these finish, so what
    // used to be a frozen frame per material is now the feature arriving a moment late.
    mPrewarmThreads.reserve(threads);
    for (unsigned int t = 0; t < threads; t++) {
        mPrewarmThreads.emplace_back(work);
    }
    SPDLOG_INFO("Shader prewarm: {} variants queued on {} threads", mPrewarmSources.size(), threads);
}

bool GfxRenderingAPIDX11::ShaderPrewarmInProgress() {
    if (mPrewarmThreads.empty()) {
        return false;
    }
    if (mPrewarmRemaining.load() != 0) {
        return true;
    }
    // The work is done; collect the threads so the next batch starts from a clean slate.
    JoinPrewarm();
    return false;
}

// Joining is the only thing that has to be right about these threads: they read mPrewarmSources, which
// belongs to this object, so none of them may still be running when it goes away. Called before starting a
// batch, when one finishes, and from the destructor.
void GfxRenderingAPIDX11::JoinPrewarm() {
    for (std::thread& t : mPrewarmThreads) {
        if (t.joinable()) {
            t.join();
        }
    }
    mPrewarmThreads.clear();
}

struct ShaderProgram* GfxRenderingAPIDX11::CreateAndLoadNewShader(uint64_t shader_id0, uint32_t shader_id1) {
    CCFeatures cc_features;
    gfx_cc_get_features(shader_id0, shader_id1, &cc_features);

    char* buf;
    size_t len, numFloats;

    auto shader = gfx_direct3d_common_build_shader(numFloats, cc_features, false,
                                                   mCurrentFilterMode == FILTER_THREE_POINT, mSrgbMode);

    buf = shader.data();
    len = shader.size();

    ComPtr<ID3DBlob> vs, ps;
    ComPtr<ID3DBlob> error_blob;

    UINT compile_flags = kShaderCompileFlags;

    // Try the disk cache before the compiler. The flags and the target profile are folded into the seed so a
    // debug build never reads a release build's bytecode, and a 4_1 build never reads a 4_0 one's.
    const uint64_t cacheHashA = ShaderHash(buf, len, 0xCBF29CE484222325ull ^ ShaderCacheSeed());
    const uint64_t cacheHashB = ShaderHash(buf, len, 0x9E3779B97F4A7C15ull ^ ShaderCacheSeed());
    const std::string cachePath = ShaderCachePath(cacheHashA, cacheHashB);
    std::vector<uint8_t> cachedVs, cachedPs;
    bool fromCache = !cachePath.empty() && ShaderCacheLoad(cachePath, cacheHashA, cacheHashB, len, cachedVs, cachedPs);

    // Whichever path produced them, the bytecode is used the same way below -- including by the input
    // layout, which is validated against the vertex shader's signature.
    const void* vsData = nullptr;
    const void* psData = nullptr;
    size_t vsSize = 0, psSize = 0;

    if (fromCache) {
        vsData = cachedVs.data();
        vsSize = cachedVs.size();
        psData = cachedPs.data();
        psSize = cachedPs.size();
    } else {

        HRESULT hr = mD3dCompile(buf, len, nullptr, nullptr, nullptr, "VSMain", ShaderProfileVs(), compile_flags, 0,
                                 vs.GetAddressOf(), error_blob.GetAddressOf());

        if (FAILED(hr)) {
            // Log before the box. The throw below is unhandled and takes the process with it, and the crash
            // handler records a stack -- which names this line and tells you nothing about WHY the compile
            // failed. The compiler's own message is the only thing that does, and it was going solely to a
            // dialog that vanishes with the process. Anyone reading a log afterwards had a crash with no cause.
            const char* err = error_blob != nullptr ? (const char*)error_blob->GetBufferPointer() : "(no message)";
            SPDLOG_CRITICAL("Vertex shader failed to compile (id0 {:#x} id1 {:#x}): {}", shader_id0, shader_id1, err);
            MessageBoxA(mWindowBackend->GetWindowHandle(), err, "Error", MB_OK | MB_ICONERROR);
            throw hr;
    }

    hr = mD3dCompile(buf, len, nullptr, nullptr, nullptr, "PSMain", ShaderProfilePs(), compile_flags, 0,
                     ps.GetAddressOf(), error_blob.GetAddressOf());

    if (FAILED(hr)) {
        const char* err = error_blob != nullptr ? (const char*)error_blob->GetBufferPointer() : "(no message)";
        SPDLOG_CRITICAL("Pixel shader failed to compile (id0 {:#x} id1 {:#x}): {}", shader_id0, shader_id1, err);
        MessageBoxA(mWindowBackend->GetWindowHandle(), err, "Error", MB_OK | MB_ICONERROR);
        throw hr;
    }

    vsData = vs->GetBufferPointer();
    vsSize = vs->GetBufferSize();
    psData = ps->GetBufferPointer();
    psSize = ps->GetBufferSize();
    // Written only after both compiles succeeded, so a failed variant is never cached as though it were
    // good. A failure to write is not a failure to render: the shader is in hand either way.
    if (!cachePath.empty()) {
        ShaderCacheStore(cachePath, cacheHashA, cacheHashB, len, vsData, vsSize, psData, psSize);
    }

    } // end of the compile path; the cache hit above skipped all of it

    struct ShaderProgramD3D11* prg = &mShaderProgramPool[std::make_pair(shader_id0, shader_id1)];

    ThrowIfFailed(mDevice->CreateVertexShader(vsData, vsSize, nullptr, prg->vertex_shader.GetAddressOf()));
    ThrowIfFailed(mDevice->CreatePixelShader(psData, psSize, nullptr, prg->pixel_shader.GetAddressOf()));

    // Input Layout

    D3D11_INPUT_ELEMENT_DESC ied[16];
    uint8_t ied_index = 0;
    ied[ied_index++] = {
        "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0
    };
    for (UINT i = 0; i < 2; i++) {
        if (cc_features.usedTextures[i]) {
            ied[ied_index++] = {
                "TEXCOORD", i, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0
            };
            if (cc_features.clamp[i][0]) {
                ied[ied_index++] = { "TEXCLAMPS",
                                     i,
                                     DXGI_FORMAT_R32_FLOAT,
                                     0,
                                     D3D11_APPEND_ALIGNED_ELEMENT,
                                     D3D11_INPUT_PER_VERTEX_DATA,
                                     0 };
            }
            if (cc_features.clamp[i][1]) {
                ied[ied_index++] = { "TEXCLAMPT",
                                     i,
                                     DXGI_FORMAT_R32_FLOAT,
                                     0,
                                     D3D11_APPEND_ALIGNED_ELEMENT,
                                     D3D11_INPUT_PER_VERTEX_DATA,
                                     0 };
            }
        }
    }
    if (cc_features.opt_fog) {
        ied[ied_index++] = {
            "FOG", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0
        };
    }
    if (cc_features.opt_grayscale) {
        ied[ied_index++] = { "GRAYSCALE",
                             0,
                             DXGI_FORMAT_R32G32B32A32_FLOAT,
                             0,
                             D3D11_APPEND_ALIGNED_ELEMENT,
                             D3D11_INPUT_PER_VERTEX_DATA,
                             0 };
    }
    // SOH [Enhancement] Toon lighting world-space normal (order must match the vbo packing). The shadow map
    // takes the same attribute for its normal-offset bias, so it rides either option.
    if (cc_features.opt_toon || cc_features.opt_shadow_map) {
        ied[ied_index++] = {
            "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0
        };
    }
    // SOH [Enhancement] Cascaded shadow maps: receiver world position (order must match the vbo packing).
    if (cc_features.opt_shadow_map) {
        ied[ied_index++] = { "WORLDPOS",
                             0,
                             DXGI_FORMAT_R32G32B32A32_FLOAT, // xyz world position, w the receiver kind
                             0,
                             D3D11_APPEND_ALIGNED_ELEMENT,
                             D3D11_INPUT_PER_VERTEX_DATA,
                             0 };
    }
    for (unsigned int i = 0; i < cc_features.numInputs; i++) {
        // SOH [Enhancement] Four normalised bytes rather than three or four floats -- a quarter of the room
        // for exactly the precision the source has, since every one of these comes from an 8-bit RDP colour
        // register or an 8-bit vertex colour and was being widened to float only to be narrowed again by the
        // blender. The shader still declares float3 or float4 and still reads 0..1: the unpacking is the
        // input assembler's, at no cost. A float3 declaration against a four-component slot simply ignores
        // the fourth, which is what the no-alpha case wants.
        ied[ied_index++] = {
            "INPUT", i, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0
        };
    }

    // vsData, not vs: on a cache hit the blob was never created and the bytecode lives in the vector.
    ThrowIfFailed(mDevice->CreateInputLayout(ied, ied_index, vsData, vsSize,
                                             prg->input_layout.GetAddressOf()));

    // Blend state

    D3D11_BLEND_DESC blend_desc;
    ZeroMemory(&blend_desc, sizeof(D3D11_BLEND_DESC));

    if (cc_features.opt_alpha) {
        blend_desc.RenderTarget[0].BlendEnable = true;
        blend_desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
        blend_desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        blend_desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        blend_desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;
        blend_desc.RenderTarget[0].DestBlendAlpha =
            D3D11_BLEND_ONE; // We initially clear alpha to 1.0f and want to keep it at 1.0f
        blend_desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        blend_desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    } else {
        blend_desc.RenderTarget[0].BlendEnable = false;
        blend_desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    }

    ThrowIfFailed(mDevice->CreateBlendState(&blend_desc, prg->blend_state.GetAddressOf()));

    // Save some values

    prg->shader_id0 = shader_id0;
    prg->shader_id1 = shader_id1;
    prg->numInputs = cc_features.numInputs;
    prg->numFloats = numFloats;
    prg->opt_toon = cc_features.opt_toon;             // SOH [Enhancement] toon lighting
    prg->opt_shadow_map = cc_features.opt_shadow_map; // SOH [Enhancement] cascaded shadow maps
    prg->usedTextures[0] = cc_features.usedTextures[0];
    prg->usedTextures[1] = cc_features.usedTextures[1];
    prg->usedTextures[2] = cc_features.used_masks[0];
    prg->usedTextures[3] = cc_features.used_masks[1];
    prg->usedTextures[4] = cc_features.used_blend[0];
    prg->usedTextures[5] = cc_features.used_blend[1];

    return (struct ShaderProgram*)(mShaderProgram = prg);
}

struct ShaderProgram* GfxRenderingAPIDX11::LookupShader(uint64_t shader_id0, uint32_t shader_id1) {
    auto it = mShaderProgramPool.find(std::make_pair(shader_id0, shader_id1));
    return it == mShaderProgramPool.end() ? nullptr : (struct ShaderProgram*)&it->second;
}

void GfxRenderingAPIDX11::ShaderGetInfo(struct ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    struct ShaderProgramD3D11* p = (struct ShaderProgramD3D11*)prg;

    *numInputs = p->numInputs;
    usedTextures[0] = p->usedTextures[0];
    usedTextures[1] = p->usedTextures[1];
}

size_t GfxRenderingAPIDX11::GetVertexStrideFloats(struct ShaderProgram* prg) {
    return prg != nullptr ? ((struct ShaderProgramD3D11*)prg)->numFloats : 0;
}

uint32_t GfxRenderingAPIDX11::NewTexture() {
    mTextures.resize(mTextures.size() + 1);
    return (uint32_t)(mTextures.size() - 1);
}

void GfxRenderingAPIDX11::DeleteTexture(uint32_t texID) {
    // glDeleteTextures(1, &texID);
}

void GfxRenderingAPIDX11::SelectTexture(int tile, uint32_t texture_id) {
    mCurrentTile = tile;
    mCurrentTextureIds[tile] = texture_id;
}

static D3D11_TEXTURE_ADDRESS_MODE gfx_cm_to_d3d11(uint32_t val) {
    // TODO: handle G_TX_MIRROR | G_TX_CLAMP
    if (val & G_TX_CLAMP) {
        return D3D11_TEXTURE_ADDRESS_CLAMP;
    }
    return (val & G_TX_MIRROR) ? D3D11_TEXTURE_ADDRESS_MIRROR : D3D11_TEXTURE_ADDRESS_WRAP;
}

void GfxRenderingAPIDX11::UploadTexture(const uint8_t* rgba32_buf, uint32_t width, uint32_t height) {
    // Create texture

    TextureData* texture_data = &mTextures[mCurrentTextureIds[mCurrentTile]];
    texture_data->width = width;
    texture_data->height = height;

    D3D11_TEXTURE2D_DESC texture_desc;
    ZeroMemory(&texture_desc, sizeof(D3D11_TEXTURE2D_DESC));

    // SOH [Enhancement] Only textures big enough for minification to actually cost something get a chain,
    // which is what keeps this off the thousands of tiny stock textures that would pay for it and gain
    // nothing (see GFX_MIPMAP_MIN_TEXTURE_SIZE). Sub-1 dimensions are excluded because GenerateMips has
    // nothing to halve.
    const bool want_mips = mMipmapEnabled && width >= GFX_MIPMAP_MIN_TEXTURE_SIZE &&
                           height >= GFX_MIPMAP_MIN_TEXTURE_SIZE;
    texture_data->has_mips = want_mips;

    texture_desc.Width = width;
    texture_desc.Height = height;
    texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture_desc.CPUAccessFlags = 0;
    texture_desc.ArraySize = 1;
    texture_desc.SampleDesc.Count = 1;
    texture_desc.SampleDesc.Quality = 0;

    if (want_mips) {
        // A chain generated on the GPU, which needs the texture to be writable and bindable as a render
        // target -- so it cannot be IMMUTABLE with initial data the way the single-level path is. The levels
        // are filled by UpdateSubresource + GenerateMips below instead.
        texture_desc.Usage = D3D11_USAGE_DEFAULT;
        texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        texture_desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
        texture_desc.MipLevels = 0; // 0 asks for the full chain down to 1x1

        ThrowIfFailed(
            mDevice->CreateTexture2D(&texture_desc, nullptr, texture_data->texture.ReleaseAndGetAddressOf()));
        ThrowIfFailed(mDevice->CreateShaderResourceView(texture_data->texture.Get(), nullptr,
                                                        texture_data->resource_view.ReleaseAndGetAddressOf()));
        mContext->UpdateSubresource(texture_data->texture.Get(), 0, nullptr, rgba32_buf, width * 4, 0);
        mContext->GenerateMips(texture_data->resource_view.Get());
        return;
    }

    texture_desc.Usage = D3D11_USAGE_IMMUTABLE;
    texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    texture_desc.MiscFlags = 0;
    texture_desc.MipLevels = 1;

    D3D11_SUBRESOURCE_DATA resource_data;
    resource_data.pSysMem = rgba32_buf;
    resource_data.SysMemPitch = width * 4;
    resource_data.SysMemSlicePitch = resource_data.SysMemPitch * height;

    ThrowIfFailed(
        mDevice->CreateTexture2D(&texture_desc, &resource_data, texture_data->texture.ReleaseAndGetAddressOf()));

    // Create shader resource view from texture

    ThrowIfFailed(mDevice->CreateShaderResourceView(texture_data->texture.Get(), nullptr,
                                                    texture_data->resource_view.ReleaseAndGetAddressOf()));
}

// Which of a texture's two samplers this draw wants: the clamped one for screen-space geometry, the full
// one otherwise. Falls back to the full sampler whenever the clamped one was not built, which is every
// texture without a chain -- so nothing that had one sampler before now depends on having two.
const Microsoft::WRL::ComPtr<ID3D11SamplerState>& GfxRenderingAPIDX11::SamplerFor(uint32_t textureId) {
    TextureData& t = mTextures[textureId];
    return (mTextureLodClamped && t.sampler_state_lod0 != nullptr) ? t.sampler_state_lod0 : t.sampler_state;
}

void GfxRenderingAPIDX11::SetSamplerParameters(int tile, bool linear_filter, uint32_t cms, uint32_t cmt) {
    D3D11_SAMPLER_DESC sampler_desc;
    ZeroMemory(&sampler_desc, sizeof(D3D11_SAMPLER_DESC));

    TextureData* texture_data = &mTextures[mCurrentTextureIds[tile]];
    const bool linear = linear_filter && mCurrentFilterMode == FILTER_LINEAR;
    // SOH [Enhancement] Only a texture that actually HAS a chain may ask for mip filtering, and only then is
    // the bias meaningful. Everything without one keeps exactly the filter it had before.
    const bool mips = texture_data->has_mips;

    if (linear) {
        // Anisotropic only where there is a chain to walk: it is a minification filter, and on a
        // single-level texture it costs extra taps to reach the same texel.
        sampler_desc.Filter =
            (mips && mMipmapAnisotropy > 1) ? D3D11_FILTER_ANISOTROPIC : D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    } else {
        // Point magnification is the look the game is drawn for, so it survives the chain: only the choice
        // BETWEEN levels goes linear, which is what stops a visible seam where one level takes over.
        sampler_desc.Filter = mips ? D3D11_FILTER_MIN_MAG_POINT_MIP_LINEAR : D3D11_FILTER_MIN_MAG_MIP_POINT;
    }
    sampler_desc.MaxAnisotropy = (mips && linear) ? (UINT)mMipmapAnisotropy : 1;
    sampler_desc.MipLODBias = mips ? mMipmapLodBias : 0.0f;

    sampler_desc.AddressU = gfx_cm_to_d3d11(cms);
    sampler_desc.AddressV = gfx_cm_to_d3d11(cmt);
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

    texture_data->linear_filtering = linear_filter;

    // This function is called twice per texture, the first one only to set default values.
    // Maybe that could be skipped? Anyway, make sure to release the first default sampler
    // state before setting the actual one.
    texture_data->sampler_state.Reset();
    texture_data->sampler_state_lod0.Reset();

    ThrowIfFailed(mDevice->CreateSamplerState(&sampler_desc, texture_data->sampler_state.GetAddressOf()));

    // The interface's variant, built only where there is a chain for it to differ from. Same filter and same
    // addressing -- only the level is pinned to the top and the bias dropped, so a HUD element drawn from a
    // large texture reads exactly the pixels it read before mipmapping existed.
    if (mips) {
        sampler_desc.MaxLOD = 0.0f;
        sampler_desc.MipLODBias = 0.0f;
        ThrowIfFailed(mDevice->CreateSamplerState(&sampler_desc, texture_data->sampler_state_lod0.GetAddressOf()));
    }
}

void GfxRenderingAPIDX11::SetDepthTestAndMask(bool depth_test, bool depth_mask) {
    mCurrentDepthTest = depth_test;
    mCurrentDepthMask = depth_mask;
}

void GfxRenderingAPIDX11::SetZmodeDecal(bool zmode_decal) {
    mCurrentZmodeDecal = zmode_decal;
}

void GfxRenderingAPIDX11::SetViewport(int x, int y, int width, int height) {
    D3D11_VIEWPORT viewport;
    viewport.TopLeftX = x;
    viewport.TopLeftY = mRenderTargetHeight - y - height;
    viewport.Width = width;
    viewport.Height = height;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;

    mContext->RSSetViewports(1, &viewport);
}

void GfxRenderingAPIDX11::SetScissor(int x, int y, int width, int height) {
    D3D11_RECT rect;
    rect.left = x;
    rect.top = mRenderTargetHeight - y - height;
    rect.right = x + width;
    rect.bottom = mRenderTargetHeight - y;

    mContext->RSSetScissorRects(1, &rect);
}

void GfxRenderingAPIDX11::SetUseAlpha(bool use_alpha) {
    // Already part of the pipeline state from shader info
}

void GfxRenderingAPIDX11::DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {

    // SOH [Enhancement] World light casting: also rebuild when the stencil mode changes. The decal
    // flag participates because DepthFunc below depends on it; without it a decal-only change kept the
    // previous compare function (mLastZmodeDecal itself is updated by the rasterizer block further down).
    if (mLastDepthTest != mCurrentDepthTest || mLastDepthMask != mCurrentDepthMask ||
        mLastZmodeDecal != mCurrentZmodeDecal || mLastStencilMode != mStencilMode) {
        mLastDepthTest = mCurrentDepthTest;
        mLastDepthMask = mCurrentDepthMask;
        mLastStencilMode = mStencilMode;

        // SOH [Enhancement] Depth-stencil states are cached (created lazily, kept for the device's
        // lifetime): the stencil features flip the mode many times per frame, and re-running
        // CreateDepthStencilState on every flip churned a driver object per change. Same key scheme as
        // the Metal backend's cache.
        uint32_t dsKey = (mCurrentDepthTest ? 1u : 0u) | ((mCurrentDepthMask ? 1u : 0u) << 1) |
                         ((mCurrentZmodeDecal ? 1u : 0u) << 2) | (((uint32_t)mStencilMode & 7u) << 3);
        if (mDepthStencilStates[dsKey] == nullptr) {
            D3D11_DEPTH_STENCIL_DESC depth_stencil_desc;
            ZeroMemory(&depth_stencil_desc, sizeof(D3D11_DEPTH_STENCIL_DESC));

            depth_stencil_desc.DepthEnable = mCurrentDepthTest || mCurrentDepthMask;
            depth_stencil_desc.DepthWriteMask =
                mCurrentDepthMask ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
            depth_stencil_desc.DepthFunc =
                mCurrentDepthTest ? (mCurrentZmodeDecal ? D3D11_COMPARISON_LESS_EQUAL : D3D11_COMPARISON_LESS)
                                  : D3D11_COMPARISON_ALWAYS;

            // SOH [Enhancement] World light casting / actor shadows stencil volume state. Off leaves the
            // stencil disabled (rendering unchanged). The single-sided mask modes (z-fail) use identical
            // front/back ops because those passes cull game-side; VolumeIncrDecr is the two-sided single
            // pass — opposite WRAP ops per facing (D3D11_STENCIL_OP_INCR/DECR wrap; the _SAT variants
            // clamp), order/polarity independent. The composite draws where stencil != 0 and zeroes it as
            // it goes (self-clearing).
            if (mStencilMode == (int)StencilMode::Off) {
                depth_stencil_desc.StencilEnable = false;
            } else {
                depth_stencil_desc.StencilEnable = true;
                depth_stencil_desc.StencilReadMask = 0xFF;
                depth_stencil_desc.StencilWriteMask = 0xFF;

                D3D11_DEPTH_STENCILOP_DESC op;
                op.StencilFailOp = D3D11_STENCIL_OP_KEEP;
                if (mStencilMode == (int)StencilMode::VolumeIncr) {
                    op.StencilDepthFailOp = D3D11_STENCIL_OP_INCR_SAT;
                    op.StencilPassOp = D3D11_STENCIL_OP_KEEP;
                    op.StencilFunc = D3D11_COMPARISON_ALWAYS;
                } else if (mStencilMode == (int)StencilMode::VolumeDecr) {
                    op.StencilDepthFailOp = D3D11_STENCIL_OP_DECR_SAT;
                    op.StencilPassOp = D3D11_STENCIL_OP_KEEP;
                    op.StencilFunc = D3D11_COMPARISON_ALWAYS;
                } else if (mStencilMode == (int)StencilMode::VolumeIncrDecr) {
                    op.StencilDepthFailOp = D3D11_STENCIL_OP_INCR; // wrap
                    op.StencilPassOp = D3D11_STENCIL_OP_KEEP;
                    op.StencilFunc = D3D11_COMPARISON_ALWAYS;
                } else { // Composite: draw where stencil != ref(0), zeroing it
                    op.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
                    op.StencilPassOp = D3D11_STENCIL_OP_ZERO;
                    op.StencilFunc = D3D11_COMPARISON_NOT_EQUAL;
                }
                depth_stencil_desc.FrontFace = op;
                depth_stencil_desc.BackFace = op;
                if (mStencilMode == (int)StencilMode::VolumeIncrDecr) {
                    depth_stencil_desc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_DECR; // wrap
                }
            }

            ThrowIfFailed(
                mDevice->CreateDepthStencilState(&depth_stencil_desc, mDepthStencilStates[dsKey].GetAddressOf()));
        }
        mContext->OMSetDepthStencilState(mDepthStencilStates[dsKey].Get(), 0);
    }

    if (mLastZmodeDecal != mCurrentZmodeDecal) {
        mLastZmodeDecal = mCurrentZmodeDecal;

        mRasterizerState.Reset();

        D3D11_RASTERIZER_DESC rasterizer_desc;
        ZeroMemory(&rasterizer_desc, sizeof(D3D11_RASTERIZER_DESC));

        rasterizer_desc.FillMode = D3D11_FILL_SOLID;
        rasterizer_desc.CullMode = D3D11_CULL_NONE;
        rasterizer_desc.FrontCounterClockwise = true;
        rasterizer_desc.DepthBias = 0;
        // SSDB = SlopeScaledDepthBias 120 leads to -2 at 240p which is the same as N64 mode which has very little
        // fighting
        const int n64modeFactor = 120;
        const int noVanishFactor = 100;
        float SSDB = -2;

        switch (Ship::Context::GetInstance()->GetConsoleVariables()->GetInteger(CVAR_Z_FIGHTING_MODE, 0)) {
            case 1: // scaled z-fighting (N64 mode like)
                SSDB = -1.0f * (float)mRenderTargetHeight / n64modeFactor;
                break;
            case 2: // no vanishing paths
                SSDB = -1.0f * (float)mRenderTargetHeight / noVanishFactor;
                break;
            case 0: // disabled
            default:
                SSDB = -2;
        }
        rasterizer_desc.SlopeScaledDepthBias = mCurrentZmodeDecal ? SSDB : 0.0f;
        rasterizer_desc.DepthBiasClamp = 0.0f;
        rasterizer_desc.DepthClipEnable = false;
        rasterizer_desc.ScissorEnable = true;
        rasterizer_desc.MultisampleEnable = false;
        rasterizer_desc.AntialiasedLineEnable = false;

        ThrowIfFailed(mDevice->CreateRasterizerState(&rasterizer_desc, mRasterizerState.GetAddressOf()));
        mContext->RSSetState(mRasterizerState.Get());
    }

    bool textures_changed = false;

    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        if (mShaderProgram->usedTextures[i]) {
            if (mLastResourceViews[i].Get() != mTextures[mCurrentTextureIds[i]].resource_view.Get()) {
                mLastResourceViews[i] = mTextures[mCurrentTextureIds[i]].resource_view.Get();
                mContext->PSSetShaderResources(i, 1, mTextures[mCurrentTextureIds[i]].resource_view.GetAddressOf());

                if (mCurrentFilterMode == FILTER_THREE_POINT) {
                    mPerDrawCbData.mTextures[i].width = mTextures[mCurrentTextureIds[i]].width;
                    mPerDrawCbData.mTextures[i].height = mTextures[mCurrentTextureIds[i]].height;
                    mPerDrawCbData.mTextures[i].linear_filtering = mTextures[mCurrentTextureIds[i]].linear_filtering;
                    textures_changed = true;
                }
            }
        }
        // SOH [Enhancement] Bound only when it is not already bound.
        //
        // This used to record what the sampler was and then set it anyway, every slot on every draw --
        // six calls per draw call, almost all of them setting the state that was already there. The record
        // was kept and never consulted, so it was half a cache with the half that saves the work missing.
        //
        // Comparing the state OBJECT rather than the texture id is what makes it correct: two different
        // textures can share a sampler and want no rebind, and one texture's sampler changes under it when
        // its filtering or wrap mode is set (see SetSamplerParameters) or when the LOD clamp flips, which
        // the id alone would not notice.
        //
        // Every other pass that binds into these slots invalidates the record on its way out -- the FXAA
        // pass clears all of them, the shadow map's cutout pass clears slot 0 -- or the first draw after
        // one of those would skip a rebind it needs.
        ID3D11SamplerState* const sampler = SamplerFor(mCurrentTextureIds[i]).Get();
        if (mLastSamplerStates[i].Get() != sampler) {
            mLastSamplerStates[i] = sampler;
            mContext->PSSetSamplers(i, 1, &sampler);
        }
    }

    // Set per-draw constant buffer

    if (textures_changed) {
        D3D11_MAPPED_SUBRESOURCE ms;
        ZeroMemory(&ms, sizeof(D3D11_MAPPED_SUBRESOURCE));
        mContext->Map(mPerDrawCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
        memcpy(ms.pData, &mPerDrawCbData, sizeof(PerDrawCB));
        mContext->Unmap(mPerDrawCb.Get(), 0);
    }

    // SOH [Enhancement] Toon lighting: per-object dominant light + ramp shape into the dedicated toon
    // CB (b2), re-uploaded per toon draw (WRITE_DISCARD makes this safe). Only the toon pixel shader
    // reads it, and PerFrameCB is left untouched so it stays frame-global.
    // Uploaded only when the values differ from the ones already in the buffer. The key light is per
    // OBJECT, not per draw -- the interpreter flushes at each object boundary so every toon object becomes
    // its own batch -- so a single object that takes several draws was re-uploading the same sixteen floats
    // for each of them, and a run of objects sharing a key light re-uploaded them across the whole run.
    //
    // Built into a local and compared whole rather than field by field: the struct is all floats with its
    // padding named, and both sides are value-initialised, so the comparison sees no indeterminate bytes.
    // mPerToonCbData is now what the buffer HOLDS rather than a scratch area, which is what makes it usable
    // as the thing to compare against.
    if (mShaderProgram->opt_toon) {
        PerToonCB toon{};
        for (int j = 0; j < 3; j++) {
            toon.toon_light_dir[j] = mToonLightDir[j];
            toon.toon_light_color[j] = mToonLightColor[j];
            toon.toon_ambient[j] = mToonAmbient[j];
        }
        toon.toon_ramp_center = mToonRampCenter;
        toon.toon_ramp_softness = mToonRampSoftness;
        toon.toon_highlight_intensity = mToonHighlightIntensity;
        toon.toon_shadow_intensity = mToonShadowIntensity;
        toon.toon_debug = mToonDebug;

        if (!mPerToonCbValid || memcmp(&toon, &mPerToonCbData, sizeof(PerToonCB)) != 0) {
            D3D11_MAPPED_SUBRESOURCE toon_ms;
            ZeroMemory(&toon_ms, sizeof(D3D11_MAPPED_SUBRESOURCE));
            mContext->Map(mPerToonCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &toon_ms);
            memcpy(toon_ms.pData, &toon, sizeof(PerToonCB));
            mContext->Unmap(mPerToonCb.Get(), 0);
            mPerToonCbData = toon;
            mPerToonCbValid = true;
        }
    }

    // SOH [Enhancement] Cascaded shadow maps: the cascade transforms are frame-global, so upload them once
    // per frame on the first receiver draw rather than per draw like the toon CB.
    if (mShaderProgram->opt_shadow_map && mShadowCbDirty) {
        D3D11_MAPPED_SUBRESOURCE shadow_ms;
        ZeroMemory(&shadow_ms, sizeof(D3D11_MAPPED_SUBRESOURCE));
        if (SUCCEEDED(mContext->Map(mPerShadowCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &shadow_ms))) {
            memcpy(shadow_ms.pData, &mPerShadowCbData, sizeof(PerShadowCB));
            mContext->Unmap(mPerShadowCb.Get(), 0);
            mShadowCbDirty = false;
        }
    }

    // Set vertex buffer data

    // Appended into the ring, not written over the front of the buffer.
    //
    // MAP_WRITE_DISCARD renames the buffer: the driver hands back a fresh block of the FULL ByteWidth out of
    // a pool it then has to recycle against the GPU's progress. Doing that per draw made the price of a draw
    // call a function of how big the buffer is rather than of how much it writes -- which is why raising
    // MAX_TRI_BUFFER cost more than it saved the last time it was tried.
    //
    // MAP_WRITE_NO_OVERWRITE promises the opposite: that nothing already submitted is being touched. That
    // holds here by construction, because the offset only moves forwards over ground no draw has been
    // pointed at since the last rename. Running out is exactly when the promise would stop being true, and
    // that is where DISCARD comes back -- the rename leaves the draws already in flight reading the old
    // block, so wrapping cannot disturb them.
    const uint32_t size = (uint32_t)(buf_vbo_len * sizeof(float));
    D3D11_MAP mapType = D3D11_MAP_WRITE_NO_OVERWRITE;
    if (mVertexRingOffset + size > kVertexRingBytes) {
        mVertexRingOffset = 0;
        mapType = D3D11_MAP_WRITE_DISCARD;
    }

    D3D11_MAPPED_SUBRESOURCE ms;
    ZeroMemory(&ms, sizeof(D3D11_MAPPED_SUBRESOURCE));
    mContext->Map(mVertexBuffer.Get(), 0, mapType, 0, &ms);
    memcpy((char*)ms.pData + mVertexRingOffset, buf_vbo, size);
    mContext->Unmap(mVertexBuffer.Get(), 0);

    const uint32_t stride = mShaderProgram->numFloats * sizeof(float);
    const uint32_t offset = mVertexRingOffset;
    mVertexRingOffset += size;

    // The offset moves every draw now, so this can no longer be skipped on the stride alone. What replaces
    // the rename is a binding, which is the whole of the trade being made here.
    if (mLastVertexBufferStride != stride || mLastVertexBufferOffset != offset) {
        mLastVertexBufferStride = stride;
        mLastVertexBufferOffset = offset;
        mContext->IASetVertexBuffers(0, 1, mVertexBuffer.GetAddressOf(), &stride, &offset);
    }

    if (mLastShaderProgram != mShaderProgram) {
        mLastShaderProgram = mShaderProgram;
        mContext->IASetInputLayout(mShaderProgram->input_layout.Get());
        mContext->VSSetShader(mShaderProgram->vertex_shader.Get(), 0, 0);
        mContext->PSSetShader(mShaderProgram->pixel_shader.Get(), 0, 0);

        if (mLastBlendState.Get() != mShaderProgram->blend_state.Get()) {
            mLastBlendState = mShaderProgram->blend_state.Get();
            mContext->OMSetBlendState(mShaderProgram->blend_state.Get(), 0, 0xFFFFFFFF);
        }
    }

    if (mLastPrimitaveTopology != D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) {
        mLastPrimitaveTopology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        mContext->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    }

    mContext->Draw(buf_vbo_num_tris * 3, 0);
}

void GfxRenderingAPIDX11::OnResize() {
    // create_render_target_views(true);
}

void GfxRenderingAPIDX11::StartFrame() {
    ShadowTimerFrameBegin();
    // Set per-frame constant buffer
    // SOH [Enhancement] mPerToonCb bound at slot b2 for the toon pixel shader and mPerShadowCb at b3 for
    // the shadow-map receiver variant; both are ignored by shaders that do not declare them.
    ID3D11Buffer* buffers[4] = { mPerFrameCb.Get(), mPerDrawCb.Get(), mPerToonCb.Get(), mPerShadowCb.Get() };
    mContext->PSSetConstantBuffers(0, 4, buffers);

    mPerFrameCbData.noise_frame++;
    if (mPerFrameCbData.noise_frame > 150) {
        // No high values, as noise starts to look ugly
        mPerFrameCbData.noise_frame = 0;
    }
}

void GfxRenderingAPIDX11::EndFrame() {
    // Drawn here because here is where the scene is finished and its target is still bound. Anywhere inside
    // the frame would be drawing into a picture that later passes overwrite, and anywhere later would mean
    // this pass having to know which framebuffer is the final one -- which it has no business knowing.
    DrawShadowMapView();
    ShadowTimerFrameEnd();
    mContext->Flush();
}

void GfxRenderingAPIDX11::FinishRender() {
}

int GfxRenderingAPIDX11::CreateFramebuffer() {
    uint32_t texture_id = NewTexture();
    TextureData& t = mTextures[texture_id];

    size_t index = mFrameBuffers.size();
    mFrameBuffers.resize(mFrameBuffers.size() + 1);
    FramebufferDX11& data = mFrameBuffers.back();
    data.texture_id = texture_id;

    uint32_t tile = 0;
    uint32_t saved = mCurrentTextureIds[tile];
    mCurrentTextureIds[tile] = texture_id;
    SetSamplerParameters(0, true, G_TX_WRAP, G_TX_WRAP);
    mCurrentTextureIds[tile] = saved;

    return (int)index;
}

void GfxRenderingAPIDX11::UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                                      bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                                      bool can_extract_depth) {
    FramebufferDX11& fb = mFrameBuffers[fb_id];
    TextureData& tex = mTextures[fb.texture_id];

    width = ((width) > (1U) ? (width) : (1U));
    height = ((height) > (1U) ? (height) : (1U));
    // We can't use MSAA the way we are using it on feature level 10_0 hardware, so disable it altogether.
    msaa_level = mFeatureLevel < D3D_FEATURE_LEVEL_10_1 ? 1 : msaa_level;
    while (msaa_level > 1 && mMsaaNumQualityLevels[msaa_level - 1] == 0) {
        --msaa_level;
    }

    bool diff = tex.width != width || tex.height != height || fb.msaa_level != msaa_level;

    if (diff || (fb.render_target_view.Get() != nullptr) != render_target) {
        if (fb_id != 0) {
            D3D11_TEXTURE2D_DESC texture_desc;
            texture_desc.Width = width;
            texture_desc.Height = height;
            texture_desc.Usage = D3D11_USAGE_DEFAULT;
            texture_desc.BindFlags =
                (msaa_level <= 1 ? D3D11_BIND_SHADER_RESOURCE : 0) | (render_target ? D3D11_BIND_RENDER_TARGET : 0);
            texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            texture_desc.CPUAccessFlags = 0;
            texture_desc.MiscFlags = 0;
            texture_desc.ArraySize = 1;
            texture_desc.MipLevels = 1;
            texture_desc.SampleDesc.Count = msaa_level;
            texture_desc.SampleDesc.Quality = 0;

            ThrowIfFailed(mDevice->CreateTexture2D(&texture_desc, nullptr, tex.texture.ReleaseAndGetAddressOf()));

            if (msaa_level <= 1) {
                ThrowIfFailed(mDevice->CreateShaderResourceView(tex.texture.Get(), nullptr,
                                                                tex.resource_view.ReleaseAndGetAddressOf()));
            }
        } else if (diff || (render_target && tex.texture.Get() == nullptr)) {
            DXGI_SWAP_CHAIN_DESC1 desc1;
            IDXGISwapChain1* swap_chain = mWindowBackend->GetSwapChain();
            ThrowIfFailed(swap_chain->GetDesc1(&desc1));
            if (desc1.Width != width || desc1.Height != height) {
                fb.render_target_view.Reset();
                tex.texture.Reset();
                ThrowIfFailed(swap_chain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, desc1.Flags));
            }
            ThrowIfFailed(
                swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)tex.texture.ReleaseAndGetAddressOf()));
        }
        if (render_target) {
            ThrowIfFailed(mDevice->CreateRenderTargetView(tex.texture.Get(), nullptr,
                                                          fb.render_target_view.ReleaseAndGetAddressOf()));
        }

        tex.width = width;
        tex.height = height;
    }

    if (has_depth_buffer &&
        (diff || !fb.has_depth_buffer || (fb.depth_stencil_srv.Get() != nullptr) != can_extract_depth)) {
        fb.depth_stencil_srv.Reset();
        CreateDepthStencilObjects(width, height, msaa_level, fb.depth_stencil_view.ReleaseAndGetAddressOf(),
                                  can_extract_depth ? fb.depth_stencil_srv.GetAddressOf() : nullptr);
    }
    if (!has_depth_buffer) {
        fb.depth_stencil_view.Reset();
        fb.depth_stencil_srv.Reset();
    }

    fb.has_depth_buffer = has_depth_buffer;
    fb.msaa_level = msaa_level;
}

void GfxRenderingAPIDX11::StartDrawToFramebuffer(int fb_id, float noise_scale) {
    FramebufferDX11& fb = mFrameBuffers[fb_id];
    mRenderTargetHeight = mTextures[fb.texture_id].height;

    mContext->OMSetRenderTargets(1, fb.render_target_view.GetAddressOf(),
                                 fb.has_depth_buffer ? fb.depth_stencil_view.Get() : nullptr);

    mCurrentFramebuffer = fb_id;

    if (noise_scale != 0.0f) {
        mPerFrameCbData.noise_scale = 1.0f / noise_scale;
    }

    D3D11_MAPPED_SUBRESOURCE ms;
    ZeroMemory(&ms, sizeof(D3D11_MAPPED_SUBRESOURCE));
    mContext->Map(mPerFrameCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    memcpy(ms.pData, &mPerFrameCbData, sizeof(PerFrameCB));
    mContext->Unmap(mPerFrameCb.Get(), 0);
}

void GfxRenderingAPIDX11::ClearFramebuffer(bool color, bool depth) {
    FramebufferDX11& fb = mFrameBuffers[mCurrentFramebuffer];
    if (color) {
        const float clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        mContext->ClearRenderTargetView(fb.render_target_view.Get(), clearColor);
    }
    if (depth && fb.has_depth_buffer) {
        // SOH [Enhancement] Stencil volumes: also clear the stencil plane (the combined D24S8 format always
        // has one). The z-fail masks assume stencil starts at 0 each frame, and the composite only
        // self-zeroes pixels it draws.
        mContext->ClearDepthStencilView(fb.depth_stencil_view.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL,
                                        1.0f, 0);
    }
}

bool GfxRenderingAPIDX11::EnsureFxaaPipeline() {
    if (mFxaaPs != nullptr) {
        return true;
    }
    if (mFxaaFailed || mD3dCompile == nullptr || mDevice == nullptr) {
        return false;
    }
    // Latched up front: every path out of here that is not success leaves it set, so a driver that cannot
    // build this is asked once rather than once a frame forever.
    mFxaaFailed = true;

    ComPtr<ID3DBlob> vs_blob, ps_blob, err;
    UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL2;
    if (FAILED(mD3dCompile(kFxaaShader, sizeof(kFxaaShader) - 1, nullptr, nullptr, nullptr, "VSMain", "vs_4_0", flags,
                           0, vs_blob.GetAddressOf(), err.GetAddressOf()))) {
        SPDLOG_ERROR("FXAA: vertex shader failed to compile: {}",
                     err != nullptr ? (const char*)err->GetBufferPointer() : "no message");
        return false;
    }
    err.Reset();
    if (FAILED(mD3dCompile(kFxaaShader, sizeof(kFxaaShader) - 1, nullptr, nullptr, nullptr, "PSMain", "ps_4_0", flags,
                           0, ps_blob.GetAddressOf(), err.GetAddressOf()))) {
        SPDLOG_ERROR("FXAA: pixel shader failed to compile: {}",
                     err != nullptr ? (const char*)err->GetBufferPointer() : "no message");
        return false;
    }
    if (FAILED(mDevice->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr,
                                           mFxaaVs.GetAddressOf())) ||
        FAILED(mDevice->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr,
                                          mFxaaPs.GetAddressOf()))) {
        SPDLOG_ERROR("FXAA: could not create the shaders.");
        return false;
    }

    // Bilinear and clamped. The blur reads at fractional offsets, so the filtering is doing real work here
    // rather than being a formality, and clamping keeps the taps at the screen edge from wrapping around.
    D3D11_SAMPLER_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(mDevice->CreateSamplerState(&sd, mFxaaSampler.GetAddressOf()))) {
        return false;
    }

    D3D11_BUFFER_DESC cbd;
    ZeroMemory(&cbd, sizeof(cbd));
    cbd.Usage = D3D11_USAGE_DYNAMIC;
    cbd.ByteWidth = 16; // float2 + padding, and constant buffers come in multiples of 16
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(mDevice->CreateBuffer(&cbd, nullptr, mFxaaCb.GetAddressOf()))) {
        return false;
    }

    // No depth, no blend, no culling: the pass covers the target exactly once and replaces it.
    D3D11_DEPTH_STENCIL_DESC dsd;
    ZeroMemory(&dsd, sizeof(dsd));
    dsd.DepthEnable = false;
    dsd.StencilEnable = false;
    if (FAILED(mDevice->CreateDepthStencilState(&dsd, mFxaaDepthStencilState.GetAddressOf()))) {
        return false;
    }
    D3D11_BLEND_DESC bd;
    ZeroMemory(&bd, sizeof(bd));
    bd.RenderTarget[0].BlendEnable = false;
    bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(mDevice->CreateBlendState(&bd, mFxaaBlendState.GetAddressOf()))) {
        return false;
    }
    D3D11_RASTERIZER_DESC rd;
    ZeroMemory(&rd, sizeof(rd));
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = false;
    if (FAILED(mDevice->CreateRasterizerState(&rd, mFxaaRasterizerState.GetAddressOf()))) {
        return false;
    }

    mFxaaFailed = false;
    SPDLOG_INFO("FXAA: post-process pass ready.");
    return true;
}

bool GfxRenderingAPIDX11::EnsureShadowMapViewPipeline() {
    if (mShadowViewPs != nullptr) {
        return true;
    }
    if (mShadowViewFailed || mD3dCompile == nullptr || mDevice == nullptr) {
        return false;
    }
    // Latched up front, like the FXAA pass: a driver that cannot build this is asked once, not once a frame.
    mShadowViewFailed = true;

    ComPtr<ID3DBlob> vs_blob, ps_blob, err;
    UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL2;
    if (FAILED(mD3dCompile(kShadowMapViewShader, sizeof(kShadowMapViewShader) - 1, nullptr, nullptr, nullptr, "VSMain",
                           "vs_4_0", flags, 0, vs_blob.GetAddressOf(), err.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map view: vertex shader failed to compile: {}",
                     err != nullptr ? (const char*)err->GetBufferPointer() : "no message");
        return false;
    }
    err.Reset();
    if (FAILED(mD3dCompile(kShadowMapViewShader, sizeof(kShadowMapViewShader) - 1, nullptr, nullptr, nullptr, "PSMain",
                           "ps_4_0", flags, 0, ps_blob.GetAddressOf(), err.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map view: pixel shader failed to compile: {}",
                     err != nullptr ? (const char*)err->GetBufferPointer() : "no message");
        return false;
    }
    if (FAILED(mDevice->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr,
                                           mShadowViewVs.GetAddressOf())) ||
        FAILED(mDevice->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr,
                                          mShadowViewPs.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map view: could not create the shaders.");
        return false;
    }

    // Point sampling, deliberately. The whole purpose is to see what is IN the map, and a bilinear filter
    // would blend neighbouring texels into a picture the depth comparison never sees.
    D3D11_SAMPLER_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(mDevice->CreateSamplerState(&sd, mShadowViewSampler.GetAddressOf()))) {
        return false;
    }

    D3D11_BUFFER_DESC cbd;
    ZeroMemory(&cbd, sizeof(cbd));
    cbd.Usage = D3D11_USAGE_DYNAMIC;
    cbd.ByteWidth = 16;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(mDevice->CreateBuffer(&cbd, nullptr, mShadowViewCb.GetAddressOf()))) {
        return false;
    }

    D3D11_DEPTH_STENCIL_DESC dsd;
    ZeroMemory(&dsd, sizeof(dsd));
    dsd.DepthEnable = false;
    dsd.StencilEnable = false;
    if (FAILED(mDevice->CreateDepthStencilState(&dsd, mShadowViewDepthStencilState.GetAddressOf()))) {
        return false;
    }
    D3D11_BLEND_DESC bd;
    ZeroMemory(&bd, sizeof(bd));
    bd.RenderTarget[0].BlendEnable = false;
    bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(mDevice->CreateBlendState(&bd, mShadowViewBlendState.GetAddressOf()))) {
        return false;
    }
    D3D11_RASTERIZER_DESC rd;
    ZeroMemory(&rd, sizeof(rd));
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = false;
    if (FAILED(mDevice->CreateRasterizerState(&rd, mShadowViewRasterizerState.GetAddressOf()))) {
        return false;
    }

    mShadowViewFailed = false;
    SPDLOG_INFO("Shadow map view: overlay ready.");
    return true;
}

void GfxRenderingAPIDX11::DrawShadowMapView() {
    const int slice = mShadowViewSlice;
    if (slice <= 0 || mShadowMapSrv == nullptr || mContext == nullptr) {
        return;
    }
    // The array holds the world layer's cascades first and then the actor layer's, and the actor layer is
    // the shorter one -- so the count is not a product. Asking for a slice past the end would sample
    // whatever the clamp lands on and show a picture that is not there.
    const int worldSlices = mShadowCascadesActive;
    const int actorSlices = SHADOW_MAP_ACTOR_CASCADES_FOR(mShadowCascadesActive);
    const int totalSlices = worldSlices + actorSlices;
    const int index = slice - 1; // the setting is 1-based so that 0 can mean off
    if (index >= totalSlices) {
        return;
    }
    if (!EnsureShadowMapViewPipeline()) {
        return;
    }

    // Whatever is bound right now, which at this point in the frame is the scene's target. Taken from the
    // context rather than from a framebuffer id because this pass has no business knowing which id that is,
    // and it is put back before returning.
    ComPtr<ID3D11RenderTargetView> prev_rtv;
    ComPtr<ID3D11DepthStencilView> prev_dsv;
    mContext->OMGetRenderTargets(1, prev_rtv.GetAddressOf(), prev_dsv.GetAddressOf());
    if (prev_rtv == nullptr) {
        return;
    }
    UINT prev_vp_count = 1;
    D3D11_VIEWPORT prev_vp;
    mContext->RSGetViewports(&prev_vp_count, &prev_vp);
    if (prev_vp_count == 0) {
        return;
    }

    // The actor layer may live in its own texture when the two resolutions differ; when they match it is
    // null and both halves are slices of the one array.
    const bool isActorHalf = index >= worldSlices;
    ID3D11ShaderResourceView* srv = mShadowMapSrv.Get();
    float sliceIndex = (float)index;
    if (isActorHalf && mShadowActorSrv != nullptr) {
        srv = mShadowActorSrv.Get();
        sliceIndex = (float)(index - worldSlices);
    }

    // The stretch. A cascade's depths crowd near its near plane, so the raw range is nearly flat on screen
    // and shows nothing; this opens it up around where the values actually sit. Fixed rather than measured
    // because measuring would need a readback, and a readback here would stall the frame being looked at.
    D3D11_MAPPED_SUBRESOURCE ms;
    if (FAILED(mContext->Map(mShadowViewCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
        return;
    }
    float* cb = (float*)ms.pData;
    cb[0] = sliceIndex;
    cb[1] = 0.0f;
    cb[2] = 0.25f;
    cb[3] = 0.0f;
    mContext->Unmap(mShadowViewCb.Get(), 0);

    // A quarter of the shorter screen edge, in the bottom-left corner, kept square so the map's own aspect
    // is not distorted -- a stretched depth map hides exactly the kind of directional structure this is for.
    const float side = (prev_vp.Width < prev_vp.Height ? prev_vp.Width : prev_vp.Height) * 0.25f;
    D3D11_VIEWPORT vp;
    vp.TopLeftX = prev_vp.TopLeftX + 8.0f;
    vp.TopLeftY = prev_vp.TopLeftY + prev_vp.Height - side - 8.0f;
    vp.Width = side;
    vp.Height = side;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;

    mContext->OMSetRenderTargets(1, prev_rtv.GetAddressOf(), nullptr);
    mContext->OMSetDepthStencilState(mShadowViewDepthStencilState.Get(), 0);
    mContext->OMSetBlendState(mShadowViewBlendState.Get(), nullptr, 0xFFFFFFFF);
    mContext->RSSetState(mShadowViewRasterizerState.Get());
    mContext->RSSetViewports(1, &vp);
    mContext->IASetInputLayout(nullptr);
    mContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    mContext->VSSetShader(mShadowViewVs.Get(), nullptr, 0);
    mContext->PSSetShader(mShadowViewPs.Get(), nullptr, 0);
    mContext->PSSetConstantBuffers(0, 1, mShadowViewCb.GetAddressOf());
    mContext->PSSetShaderResources(0, 1, &srv);
    mContext->PSSetSamplers(0, 1, mShadowViewSampler.GetAddressOf());
    mContext->Draw(3, 0);

    // Unbind, or the next pass that wants to WRITE this array finds it still bound as input and the runtime
    // drops the binding with a warning.
    ID3D11ShaderResourceView* null_srv = nullptr;
    mContext->PSSetShaderResources(0, 1, &null_srv);
    mContext->OMSetRenderTargets(1, prev_rtv.GetAddressOf(), prev_dsv.Get());
    mContext->RSSetViewports(1, &prev_vp);

    // Same caches the FXAA pass invalidates, for the same reason: the per-draw path skips a rebind when it
    // believes a slot already holds what it wants, and this pass has just made that belief wrong.
    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        mLastSamplerStates[i] = nullptr;
    }
    mLastResourceViews[0] = nullptr;
}

bool GfxRenderingAPIDX11::ApplyFxaa(int fb_dst_id, int fb_src_id) {
    if (fb_src_id < 0 || fb_dst_id < 0 || fb_src_id >= (int)mFrameBuffers.size() ||
        fb_dst_id >= (int)mFrameBuffers.size() || fb_src_id == fb_dst_id) {
        return false;
    }
    if (!EnsureFxaaPipeline()) {
        return false;
    }

    FramebufferDX11& fb_dst = mFrameBuffers[fb_dst_id];
    FramebufferDX11& fb_src = mFrameBuffers[fb_src_id];
    TextureData& td_src = mTextures[fb_src.texture_id];
    TextureData& td_dst = mTextures[fb_dst.texture_id];
    // A multisampled source cannot be read as an ordinary texture; it has to be resolved first, and the
    // caller is expected to have done that. Refusing is better than binding a view that does not exist.
    if (fb_src.msaa_level > 1 || td_src.resource_view == nullptr || fb_dst.render_target_view == nullptr) {
        return false;
    }

    // Whatever is bound now has to come back. The caller binds the screen and clears it BEFORE asking for
    // this pass -- the filtered texture is then handed to the interface layer, which draws it into that
    // binding -- so a pass that leaves the target unbound costs the entire frame, not just the filtering.
    ComPtr<ID3D11RenderTargetView> prev_rtv;
    ComPtr<ID3D11DepthStencilView> prev_dsv;
    mContext->OMGetRenderTargets(1, prev_rtv.GetAddressOf(), prev_dsv.GetAddressOf());
    D3D11_VIEWPORT prev_vp;
    UINT prev_vp_count = 1;
    mContext->RSGetViewports(&prev_vp_count, &prev_vp);

    D3D11_MAPPED_SUBRESOURCE ms;
    ZeroMemory(&ms, sizeof(ms));
    if (FAILED(mContext->Map(mFxaaCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
        return false;
    }
    // One over the SOURCE's size: the offsets the shader steps by are texels of what it is reading.
    float rcp[4] = { td_src.width > 0 ? 1.0f / (float)td_src.width : 0.0f,
                     td_src.height > 0 ? 1.0f / (float)td_src.height : 0.0f, 0.0f, 0.0f };
    memcpy(ms.pData, rcp, sizeof(rcp));
    mContext->Unmap(mFxaaCb.Get(), 0);

    D3D11_VIEWPORT vp;
    vp.TopLeftX = 0.0f;
    vp.TopLeftY = 0.0f;
    vp.Width = (float)td_dst.width;
    vp.Height = (float)td_dst.height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;

    mContext->OMSetRenderTargets(1, fb_dst.render_target_view.GetAddressOf(), nullptr);
    mContext->OMSetDepthStencilState(mFxaaDepthStencilState.Get(), 0);
    mContext->OMSetBlendState(mFxaaBlendState.Get(), nullptr, 0xFFFFFFFF);
    mContext->RSSetState(mFxaaRasterizerState.Get());
    mContext->RSSetViewports(1, &vp);
    // Vertex-id geometry: one triangle larger than the screen, so there is no vertex buffer and no input
    // layout to bind or to put back afterwards.
    mContext->IASetInputLayout(nullptr);
    mContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    mContext->VSSetShader(mFxaaVs.Get(), nullptr, 0);
    mContext->PSSetShader(mFxaaPs.Get(), nullptr, 0);
    mContext->PSSetConstantBuffers(0, 1, mFxaaCb.GetAddressOf());
    mContext->PSSetShaderResources(0, 1, td_src.resource_view.GetAddressOf());
    mContext->PSSetSamplers(0, 1, mFxaaSampler.GetAddressOf());
    mContext->Draw(3, 0);

    // Unbind the source, or the next pass that wants to draw INTO it finds it still bound as input and the
    // runtime drops the binding with a warning.
    ID3D11ShaderResourceView* null_srv = nullptr;
    mContext->PSSetShaderResources(0, 1, &null_srv);
    // Put the caller's target and viewport back, exactly as they were.
    mContext->OMSetRenderTargets(1, prev_rtv.GetAddressOf(), prev_dsv.Get());
    if (prev_vp_count > 0) {
        mContext->RSSetViewports(1, &prev_vp);
    }

    // Everything this touched is set again from scratch by whatever draws next: StartDrawToFramebuffer
    // re-binds the target unconditionally, LoadShader re-binds the shaders and the input layout, and the
    // per-draw setup re-binds the states and the viewport. The exceptions are the two cached records of
    // what each texture slot holds, which would otherwise skip a rebind they now need.
    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        mLastSamplerStates[i] = nullptr;
    }
    // Slot 0's resource view, for the same reason and with worse consequences. DrawTriangles only calls
    // PSSetShaderResources when the view differs from this record, so leaving it saying "texture X is in
    // slot 0" after this pass has just put NULL there means the next draw that uses texture X skips the
    // rebind and samples nothing -- black, with no error anywhere, because a null view is a legal thing to
    // sample in D3D11.
    //
    // Which is why it showed up on the debug map select and almost nowhere else. That screen draws every
    // glyph from a single font texture, so once the record goes stale nothing ever forces a rebind and the
    // whole screen stays blank behind the untextured clear; ordinary gameplay changes texture constantly,
    // so the very next different texture repairs the binding before anyone sees it.
    //
    // The shadow map's alpha pass already does exactly this, for exactly this reason (see the note beside
    // its own mLastResourceViews[0] reset). This pass was the one that did not.
    mLastResourceViews[0] = nullptr;
    return true;
}

void GfxRenderingAPIDX11::ResolveMSAAColorBuffer(int fb_id_target, int fb_id_source) {
    FramebufferDX11& fb_dst = mFrameBuffers[fb_id_target];
    FramebufferDX11& fb_src = mFrameBuffers[fb_id_source];

    mContext->ResolveSubresource(mTextures[fb_dst.texture_id].texture.Get(), 0,
                                 mTextures[fb_src.texture_id].texture.Get(), 0, DXGI_FORMAT_R8G8B8A8_UNORM);
}

void* GfxRenderingAPIDX11::GetFramebufferTextureId(int fb_id) {
    return (void*)mTextures[mFrameBuffers[fb_id].texture_id].resource_view.Get();
}

void GfxRenderingAPIDX11::SelectTextureFb(int fbID) {
    int tile = 0;
    SelectTexture(tile, mFrameBuffers[fbID].texture_id);
}

void GfxRenderingAPIDX11::CopyFramebuffer(int fb_dst_id, int fb_src_id, int srcX0, int srcY0, int srcX1, int srcY1,
                                          int dstX0, int dstY0, int dstX1, int dstY1) {
    if (fb_src_id >= (int)mFrameBuffers.size() || fb_dst_id >= (int)mFrameBuffers.size()) {
        return;
    }

    FramebufferDX11& fb_dst = mFrameBuffers[fb_dst_id];
    FramebufferDX11& fb_src = mFrameBuffers[fb_src_id];

    TextureData& td_dst = mTextures[fb_dst.texture_id];
    TextureData& td_src = mTextures[fb_src.texture_id];

    // Textures are the same size so we can do a direct copy or resolve
    if (td_src.height == td_dst.height && td_src.width == td_dst.width) {
        if (fb_src.msaa_level <= 1) {
            mContext->CopyResource(td_dst.texture.Get(), td_src.texture.Get());
        } else {
            mContext->ResolveSubresource(td_dst.texture.Get(), 0, td_src.texture.Get(), 0, DXGI_FORMAT_R8G8B8A8_UNORM);
        }
        return;
    }

    if (srcY1 > (int)td_src.height || srcX1 > (int)td_src.width || srcX0 < 0 || srcY0 < 0 ||
        dstY1 > (int)td_dst.height || dstX1 > (int)td_dst.width || dstX0 < 0 || dstY0 < 0) {
        // Using a source region larger than the source resource or copy outside of the destination resource is
        // considered undefined behavior and could lead to removal of the rendering mDevice
        return;
    }

    D3D11_BOX region;
    region.left = srcX0;
    region.right = srcX1;
    region.top = srcY0;
    region.bottom = srcY1;
    region.front = 0;
    region.back = 1;

    // We can't region copy a multi-sample texture to a single sample texture
    if (fb_src.msaa_level <= 1) {
        mContext->CopySubresourceRegion(td_dst.texture.Get(), dstX0, dstY0, 0, 0, td_src.texture.Get(), 0, &region);
    } else {
        // Setup a temporary texture
        TextureData td_resolved;
        td_resolved.width = td_src.width;
        td_resolved.height = td_src.height;

        D3D11_TEXTURE2D_DESC texture_desc;
        texture_desc.Width = td_src.width;
        texture_desc.Height = td_src.height;
        texture_desc.Usage = D3D11_USAGE_DEFAULT;
        texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture_desc.CPUAccessFlags = 0;
        texture_desc.MiscFlags = 0;
        texture_desc.ArraySize = 1;
        texture_desc.MipLevels = 1;
        texture_desc.SampleDesc.Count = 1;
        texture_desc.SampleDesc.Quality = 0;

        ThrowIfFailed(mDevice->CreateTexture2D(&texture_desc, nullptr, td_resolved.texture.GetAddressOf()));

        // Resolve multi-sample to temporary
        mContext->ResolveSubresource(td_resolved.texture.Get(), 0, td_src.texture.Get(), 0, DXGI_FORMAT_R8G8B8A8_UNORM);
        // Then copy the region to the destination
        mContext->CopySubresourceRegion(td_dst.texture.Get(), dstX0, dstY0, 0, 0, td_resolved.texture.Get(), 0,
                                        &region);
    }
}

void GfxRenderingAPIDX11::ReadFramebufferToCPU(int fb_id, uint32_t width, uint32_t height, uint16_t* rgba16_buf) {
    if (fb_id >= (int)mFrameBuffers.size()) {
        return;
    }

    FramebufferDX11& fb = mFrameBuffers[fb_id];
    TextureData& td = mTextures[fb.texture_id];

    ID3D11Texture2D* staging = nullptr;

    // Create an staging texture with cpu read access
    D3D11_TEXTURE2D_DESC texture_desc;
    texture_desc.Width = width;
    texture_desc.Height = height;
    texture_desc.Usage = D3D11_USAGE_STAGING;
    texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    texture_desc.BindFlags = 0;
    texture_desc.MiscFlags = 0;
    texture_desc.ArraySize = 1;
    texture_desc.MipLevels = 1;
    texture_desc.SampleDesc.Count = 1;
    texture_desc.SampleDesc.Quality = 0;

    ThrowIfFailed(mDevice->CreateTexture2D(&texture_desc, nullptr, &staging));

    // Copy the framebuffer texture to the staging texture
    mContext->CopyResource(staging, td.texture.Get());

    // Map the staging texture to a resource that we can read
    D3D11_MAPPED_SUBRESOURCE resource = {};
    ThrowIfFailed(mContext->Map(staging, 0, D3D11_MAP_READ, 0, &resource));

    if (!resource.pData) {
        return;
    }

    // Copy the mapped values to a temp array that we can process later
    uint32_t* temp = new uint32_t[width * height]();
    for (size_t i = 0; i < height; i++) {
        memcpy((uint8_t*)temp + (resource.RowPitch * i), (uint8_t*)resource.pData + (resource.RowPitch * i),
               resource.RowPitch);
    }

    mContext->Unmap(staging, 0);

    // Convert the RGBA32 values to RGBA16
    for (size_t i = 0; i < width; i++) {
        for (size_t j = 0; j < height; j++) {
            uint32_t pixel = temp[i + (j * width)];
            uint8_t r = (((pixel & 0xFF) + 4) * 0x1F) / 0xFF;
            uint8_t g = ((((pixel >> 8) & 0xFF) + 4) * 0x1F) / 0xFF;
            uint8_t b = ((((pixel >> 16) & 0xFF) + 4) * 0x1F) / 0xFF;
            uint8_t a = ((pixel >> 24) & 0xFF) ? 1 : 0;

            rgba16_buf[i + (j * width)] = (r << 11) | (g << 6) | (b << 1) | a;
        }
    }

    // Cleanup
    staging->Release();
    staging = nullptr;

    delete[] temp;
}

void GfxRenderingAPIDX11::SetTextureFilter(FilteringMode mode) {
    mCurrentFilterMode = mode;
    gfx_texture_cache_clear();
}

FilteringMode GfxRenderingAPIDX11::GetTextureFilter() {
    return mCurrentFilterMode;
}

std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIDX11::GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) {
    FramebufferDX11& fb = mFrameBuffers[fb_id];
    TextureData& td = mTextures[fb.texture_id];

    if (coordinates.size() > mCoordBufferSize) {
        mCoordBuffer.Reset();
        mCoordBufferSrv.Reset();
        mDepthValueOutputBuffer.Reset();
        mDepthValueOutputUav.Reset();
        mDepthValueOutputBufferCopy.Reset();

        D3D11_BUFFER_DESC coord_buf_desc;
        coord_buf_desc.Usage = D3D11_USAGE_DYNAMIC;
        coord_buf_desc.ByteWidth = sizeof(Coord) * coordinates.size();
        coord_buf_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        coord_buf_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        coord_buf_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        coord_buf_desc.StructureByteStride = sizeof(Coord);

        ThrowIfFailed(mDevice->CreateBuffer(&coord_buf_desc, nullptr, mCoordBuffer.GetAddressOf()));

        D3D11_SHADER_RESOURCE_VIEW_DESC coord_buf_srv_desc;
        coord_buf_srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        coord_buf_srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        coord_buf_srv_desc.Buffer.FirstElement = 0;
        coord_buf_srv_desc.Buffer.NumElements = coordinates.size();

        ThrowIfFailed(
            mDevice->CreateShaderResourceView(mCoordBuffer.Get(), &coord_buf_srv_desc, mCoordBufferSrv.GetAddressOf()));

        D3D11_BUFFER_DESC output_buffer_desc;
        output_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
        output_buffer_desc.ByteWidth = sizeof(float) * coordinates.size();
        output_buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        output_buffer_desc.CPUAccessFlags = 0;
        output_buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        output_buffer_desc.StructureByteStride = sizeof(float);
        ThrowIfFailed(mDevice->CreateBuffer(&output_buffer_desc, nullptr, mDepthValueOutputBuffer.GetAddressOf()));

        D3D11_UNORDERED_ACCESS_VIEW_DESC output_buffer_uav_desc;
        output_buffer_uav_desc.Format = DXGI_FORMAT_UNKNOWN;
        output_buffer_uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        output_buffer_uav_desc.Buffer.FirstElement = 0;
        output_buffer_uav_desc.Buffer.NumElements = coordinates.size();
        output_buffer_uav_desc.Buffer.Flags = 0;
        ThrowIfFailed(mDevice->CreateUnorderedAccessView(mDepthValueOutputBuffer.Get(), &output_buffer_uav_desc,
                                                         mDepthValueOutputUav.GetAddressOf()));

        output_buffer_desc.Usage = D3D11_USAGE_STAGING;
        output_buffer_desc.BindFlags = 0;
        output_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ThrowIfFailed(mDevice->CreateBuffer(&output_buffer_desc, nullptr, mDepthValueOutputBufferCopy.GetAddressOf()));

        mCoordBufferSize = coordinates.size();
    }

    D3D11_MAPPED_SUBRESOURCE ms;

    if (fb.msaa_level > 1 && mComputeShaderMsaa.Get() == nullptr) {
        ThrowIfFailed(mDevice->CreateComputeShader(mComputeShaderMsaaBlob->GetBufferPointer(),
                                                   mComputeShaderMsaaBlob->GetBufferSize(), nullptr,
                                                   mComputeShaderMsaa.GetAddressOf()));
    }

    // ImGui overwrites these values, so we cannot set them once at init
    mContext->CSSetShader(fb.msaa_level > 1 ? mComputeShaderMsaa.Get() : mComputeShader.Get(), nullptr, 0);
    mContext->CSSetUnorderedAccessViews(0, 1, mDepthValueOutputUav.GetAddressOf(), nullptr);

    ThrowIfFailed(mContext->Map(mCoordBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms));
    Coord* coord_cb = (Coord*)ms.pData;
    {
        size_t i = 0;
        for (const auto& coord : coordinates) {
            coord_cb[i].x = coord.first;
            // We invert y because the gfx_pc assumes OpenGL coordinates (bottom-left corner is origin), while DX's
            // origin is top-left corner
            coord_cb[i].y = td.height - 1 - coord.second;
            ++i;
        }
    }
    mContext->Unmap(mCoordBuffer.Get(), 0);

    // The depth stencil texture can only have one mapping at a time, so unbind from the OM
    ID3D11RenderTargetView* null_arr1[1] = { nullptr };
    mContext->OMSetRenderTargets(1, null_arr1, nullptr);

    ID3D11ShaderResourceView* srvs[2] = { fb.depth_stencil_srv.Get(), mCoordBufferSrv.Get() };
    mContext->CSSetShaderResources(0, 2, srvs);

    mContext->Dispatch(coordinates.size(), 1, 1);

    mContext->CopyResource(mDepthValueOutputBufferCopy.Get(), mDepthValueOutputBuffer.Get());
    ThrowIfFailed(mContext->Map(mDepthValueOutputBufferCopy.Get(), 0, D3D11_MAP_READ, 0, &ms));
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res;
    {
        size_t i = 0;
        for (const auto& coord : coordinates) {
            res.emplace(coord, ((float*)ms.pData)[i++] * 65532.0f);
        }
    }
    mContext->Unmap(mDepthValueOutputBufferCopy.Get(), 0);

    ID3D11ShaderResourceView* null_arr[2] = { nullptr, nullptr };
    mContext->CSSetShaderResources(0, 2, null_arr);

    return res;
}

ImTextureID GfxRenderingAPIDX11::GetTextureById(int id) {
    return mTextures[id].resource_view.Get();
}

void GfxRenderingAPIDX11::SetSrgbMode() {
    mSrgbMode = true;
}

// ===========================================================================================
// SOH [Enhancement] Cascaded shadow maps (see fast/shadow_map.h).
//
// Casters are rendered depth-only into one D16 texture array -- no textures, no combiner, no
// lighting -- so this path deliberately bypasses the normal shader pipeline instead of pushing
// display lists through it a second time. Because it binds its own shader, layout, states and
// viewport outside the per-draw path, it must invalidate that path's "last state" caches on the way
// out, or the next ordinary draw would keep whatever the depth pass left bound.
//
// Every failure here degrades to "no shadow map" rather than throwing: an unsupported or
// out-of-memory device must still render the game.
// ===========================================================================================

// Depth-only vertex shader. Position is world-space and the matrix is the cascade's light view-proj.
// row_major matches the interpreter's CPU convention (row vector times row-major matrix); HLSL packs
// constant-buffer matrices column-major by default, so leaving this off would silently transpose it.
static const char* kShadowDepthShaderSource = R"(
cbuffer ShadowDepthCB : register(b0) {
    row_major float4x4 lightViewProj;
};
float4 VSMain(float3 pos : POSITION) : SV_POSITION {
    return mul(float4(pos, 1.0), lightViewProj);
}
)";

// SOH [Enhancement] Cascaded shadow maps: the alpha-cutout caster pipeline. Foliage is a billboard with a
// leaf texture, so the depth-only pipeline above records the whole quad and a tree casts a rectangle. This
// one carries the material's texture coordinate through and clips against its alpha, which is the only way
// the depth map can hold the shape of the leaves.
//
// It still writes nothing but depth -- the pixel shader returns void and exists purely so clip() has
// somewhere to live. Sampling at mip 0 rather than letting the hardware pick: there are no derivatives
// worth trusting at shadow-map resolution, and a blurred mip would eat the cutout.
// The cutout threshold is a shared constant (fast/shadow_map.h) spliced into the shader text, so the value
// cannot drift between the two places it would otherwise be written.
#define SHADOW_MAP_STR2(x) #x
#define SHADOW_MAP_STR(x) SHADOW_MAP_STR2(x)
#define SHADOW_MAP_ALPHA_CUTOUT_STR SHADOW_MAP_STR(SHADOW_MAP_ALPHA_CUTOUT)

static const char* kShadowAlphaDepthShaderSource = R"(
cbuffer ShadowDepthCB : register(b0) {
    row_major float4x4 lightViewProj;
};
Texture2D g_casterTex : register(t0);
SamplerState g_casterSampler : register(s0);

struct VSOutput {
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

VSOutput VSMain(float3 pos : POSITION, float2 uv : TEXCOORD0) {
    VSOutput o;
    o.position = mul(float4(pos, 1.0), lightViewProj);
    o.uv = uv;
    return o;
}

void PSMain(VSOutput input) {
    float alpha = g_casterTex.SampleLevel(g_casterSampler, input.uv, 0).a;
    clip(alpha - )" SHADOW_MAP_ALPHA_CUTOUT_STR R"();
}
)";

// Matches kShadowDepthShaderSource's cbuffer. Constant buffers must be a multiple of 16 bytes; a
// float4x4 already is.
struct ShadowDepthCB {
    float lightViewProj[16];
};

bool GfxRenderingAPIDX11::SupportsShadowMap() {
    return true;
}

bool GfxRenderingAPIDX11::CreateShadowMapPipeline() {
    if (mShadowPipelineReady) {
        return true;
    }
    if (mShadowPipelineFailed) {
        return false; // already tried and failed; do not recompile every frame
    }
    mShadowPipelineFailed = true; // cleared again only on full success

#if DEBUG_D3D
    UINT compile_flags = D3DCOMPILE_DEBUG;
#else
    UINT compile_flags = D3DCOMPILE_OPTIMIZATION_LEVEL2;
#endif

    ComPtr<ID3DBlob> vs, error_blob;
    HRESULT hr = mD3dCompile(kShadowDepthShaderSource, strlen(kShadowDepthShaderSource), nullptr, nullptr, nullptr,
                             "VSMain", "vs_4_0", compile_flags, 0, vs.GetAddressOf(), error_blob.GetAddressOf());
    if (FAILED(hr)) {
        SPDLOG_ERROR("Shadow map: depth vertex shader failed to compile: {}",
                     error_blob ? (const char*)error_blob->GetBufferPointer() : "no error blob");
        return false;
    }
    if (FAILED(mDevice->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(), nullptr,
                                           mShadowDepthVs.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map: could not create the depth vertex shader.");
        return false;
    }

    const D3D11_INPUT_ELEMENT_DESC ied[1] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    if (FAILED(mDevice->CreateInputLayout(ied, 1, vs->GetBufferPointer(), vs->GetBufferSize(),
                                          mShadowDepthLayout.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map: could not create the depth input layout.");
        return false;
    }

    D3D11_BUFFER_DESC cb_desc;
    ZeroMemory(&cb_desc, sizeof(cb_desc));
    cb_desc.Usage = D3D11_USAGE_DYNAMIC;
    cb_desc.ByteWidth = sizeof(ShadowDepthCB);
    cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(mDevice->CreateBuffer(&cb_desc, nullptr, mShadowDepthCb.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map: could not create the depth constant buffer.");
        return false;
    }

    // Depth bias lives in the rasterizer rather than the shader so it scales with the depth format
    // automatically. Slope-scaled carries most of the load: a constant bias large enough for the
    // steepest polygon would detach contact shadows everywhere else ("peter panning").
    D3D11_RASTERIZER_DESC rast_desc;
    ZeroMemory(&rast_desc, sizeof(rast_desc));
    rast_desc.FillMode = D3D11_FILL_SOLID;
    // Two-sided. Culling front faces is the other classic answer to a surface shadowing itself, and it was
    // tried here -- but it records the FAR side of a wall, a whole wall thickness behind the side being
    // shaded, and that offset is peter panning by construction: shadows visibly detached from what cast
    // them. The self-shadowing it was covering for is handled properly now, by a normal-offset bias that
    // works on the room mesh (the shader recovers a face normal from screen derivatives where no vertex
    // normal exists). Two-sided also keeps single-sided geometry casting at all, which front-face culling
    // silently dropped.
    rast_desc.CullMode = D3D11_CULL_NONE;
    // Clamp depth instead of clipping it. With clipping on, any caster in front of the near plane is
    // discarded outright -- it writes no depth and therefore casts nothing. The eye is pulled back by three
    // times the cascade radius, so a tall caster clears that margin in a small cascade while easily fitting
    // in a large one: the same object would cast in one cascade and not in another, and which cascade a
    // receiver used changed with the camera angle. That is why large shadows blinked as the camera tilted
    // while small ones, whose caster sits right next to the receiver, never did.
    // Clamped, an out-of-range caster still rasterizes at the limit depth and still occludes, which is what
    // it should do. The far side is safe too: a caster clamped to the far value never wins a comparison it
    // should lose, so nothing gains a shadow it should not have.
    rast_desc.DepthClipEnable = FALSE;
    // No constant bias here. It is applied in the shader instead, in world units divided by each
    // cascade's own depth range -- the rasterizer's units are depth increments, which mean a different
    // physical distance in every cascade. The slope term stays: being relative to the polygon's own
    // gradient is exactly right, and it is the same relative amount whatever the range.
    rast_desc.DepthBias = 0;
    rast_desc.SlopeScaledDepthBias = SHADOW_MAP_SLOPE_BIAS;
    if (FAILED(mDevice->CreateRasterizerState(&rast_desc, mShadowRasterizerState.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map: could not create the depth rasterizer state.");
        return false;
    }

    D3D11_DEPTH_STENCIL_DESC ds_desc;
    ZeroMemory(&ds_desc, sizeof(ds_desc));
    ds_desc.DepthEnable = TRUE;
    ds_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    ds_desc.DepthFunc = D3D11_COMPARISON_LESS;
    ds_desc.StencilEnable = FALSE;
    if (FAILED(mDevice->CreateDepthStencilState(&ds_desc, mShadowDepthStencilState.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map: could not create the depth-stencil state.");
        return false;
    }

    // Plain point sampler, not a comparison one. The shader fetches the stored depths and compares them
    // itself, and that is a choice rather than a limitation now that the profile can be 4_1: a comparison
    // sampler tests all four texels of its footprint against ONE depth, and the per-texel depth here is the
    // receiver-plane bias -- the thing that keeps a sixteen-tap kernel free of acne without paying for it in
    // peter panning. Gather buys back the fetch count without giving that up (see SampleShadowPCF4).
    //
    // Point filtering is also the only correct setting for a hand-rolled comparison: blending stored depths
    // and comparing once is not the same as comparing per texel and averaging, and only the latter produces
    // a real penumbra. Gather ignores the filter mode entirely -- it always returns the bilinear footprint
    // -- but it does honour the addressing below, which is why the border still reads as "nothing occludes".
    D3D11_SAMPLER_DESC samp_desc;
    ZeroMemory(&samp_desc, sizeof(samp_desc));
    samp_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    // Clamp to a "nothing occludes" border: outside a cascade's footprint nothing is known to occlude, and
    // wrapping would fold a distant part of the map back over the edge. 1.0 is the far plane, so any
    // receiver compares as lit against it.
    samp_desc.AddressU = D3D11_TEXTURE_ADDRESS_BORDER;
    samp_desc.AddressV = D3D11_TEXTURE_ADDRESS_BORDER;
    samp_desc.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
    samp_desc.BorderColor[0] = samp_desc.BorderColor[1] = 1.0f;
    samp_desc.BorderColor[2] = samp_desc.BorderColor[3] = 1.0f;
    samp_desc.ComparisonFunc = D3D11_COMPARISON_NEVER; // unused without a comparison filter
    samp_desc.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(mDevice->CreateSamplerState(&samp_desc, mShadowMapSampler.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map: could not create the comparison sampler.");
        return false;
    }

    // Alpha-cutout caster pipeline. Failing to build it is NOT fatal: the opaque path still works, and
    // foliage falls back to casting its quad -- worse looking, but a scene with shadows.
    {
        ComPtr<ID3DBlob> avs, aps, aerr;
        HRESULT ahr = mD3dCompile(kShadowAlphaDepthShaderSource, strlen(kShadowAlphaDepthShaderSource), nullptr,
                                  nullptr, nullptr, "VSMain", "vs_4_0", compile_flags, 0, avs.GetAddressOf(),
                                  aerr.GetAddressOf());
        if (SUCCEEDED(ahr)) {
            ahr = mD3dCompile(kShadowAlphaDepthShaderSource, strlen(kShadowAlphaDepthShaderSource), nullptr, nullptr,
                              nullptr, "PSMain", "ps_4_0", compile_flags, 0, aps.GetAddressOf(),
                              aerr.ReleaseAndGetAddressOf());
        }
        if (FAILED(ahr)) {
            // Logged loudly on purpose: this is the difference between foliage casting its leaf shape and
            // foliage casting the quad it is painted on, and nothing else in the frame reports it.
            SPDLOG_ERROR("Shadow map: alpha caster shader failed to compile, cutout casters fall back to "
                         "their quad: {}",
                         aerr ? (const char*)aerr->GetBufferPointer() : "no error blob");
        } else {
            const D3D11_INPUT_ELEMENT_DESC aied[2] = {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            };
            D3D11_SAMPLER_DESC casterSamp;
            ZeroMemory(&casterSamp, sizeof(casterSamp));
            // Linear, and wrapping. Wrapping rather than clamping because a repeated material (a vine
            // sheet, a canopy) genuinely tiles, and clamping it would smear the edge row across the whole
            // repeat. Cutout geometry keeps its coordinates inside the tile either way.
            casterSamp.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            casterSamp.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
            casterSamp.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
            casterSamp.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
            casterSamp.ComparisonFunc = D3D11_COMPARISON_NEVER;
            casterSamp.MaxLOD = D3D11_FLOAT32_MAX;
            if (SUCCEEDED(mDevice->CreateVertexShader(avs->GetBufferPointer(), avs->GetBufferSize(), nullptr,
                                                      mShadowAlphaVs.GetAddressOf())) &&
                SUCCEEDED(mDevice->CreatePixelShader(aps->GetBufferPointer(), aps->GetBufferSize(), nullptr,
                                                     mShadowAlphaPs.GetAddressOf())) &&
                SUCCEEDED(mDevice->CreateInputLayout(aied, 2, avs->GetBufferPointer(), avs->GetBufferSize(),
                                                     mShadowAlphaLayout.GetAddressOf())) &&
                SUCCEEDED(mDevice->CreateSamplerState(&casterSamp, mShadowAlphaSampler.GetAddressOf()))) {
                mShadowAlphaPipelineReady = true;
                SPDLOG_INFO("Shadow map: alpha caster pipeline ready; cutout foliage will cast its leaf "
                            "shape.");
            } else {
                SPDLOG_ERROR("Shadow map: could not create the alpha caster pipeline objects.");
            }
        }
    }

    mShadowPipelineFailed = false;
    mShadowPipelineReady = true;
    return true;
}

bool GfxRenderingAPIDX11::CreateShadowMapTargets(int cascadeCount, int resolution, int actorResolution) {
    if (mShadowMapTexture != nullptr && cascadeCount == mShadowCascadeCount && resolution == mShadowResolution &&
        actorResolution == mShadowActorResolution) {
        return true; // already the right shape
    }

    // Drop the old array first so the driver can reuse its memory for the new one.
    for (int i = 0; i < SHADOW_MAP_MAX_SLICES; i++) {
        mShadowMapDsv[i].Reset();
        // Whatever each slice held goes with the texture. Leaving the records behind would let the
        // reuse check below match against contents that no longer exist, and that slice would then
        // never be drawn at all.
        mShadowSliceValid[i] = false;
    }
    mShadowMapSrv.Reset();
    mShadowMapTexture.Reset();
    mShadowActorSrv.Reset();
    mShadowActorTexture.Reset();
    mShadowActorSplit = false;
    mShadowCascadeCount = 0;
    mShadowResolution = 0;
    mShadowActorResolution = 0;

    // TYPELESS so the same slices can be a depth target (D16_UNORM) while writing and a texture
    // (R16_UNORM) while sampling.
    D3D11_TEXTURE2D_DESC tex_desc;
    ZeroMemory(&tex_desc, sizeof(tex_desc));
    tex_desc.Width = (UINT)resolution;
    tex_desc.Height = (UINT)resolution;
    tex_desc.MipLevels = 1;
    // Twice the slices: the world layer occupies [0, cascadeCount) and the actor layer the rest.
    tex_desc.ArraySize = (UINT)SHADOW_MAP_SLICES_FOR(cascadeCount);
    tex_desc.Format = DXGI_FORMAT_R16_TYPELESS;
    tex_desc.SampleDesc.Count = 1;
    tex_desc.Usage = D3D11_USAGE_DEFAULT;
    tex_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(mDevice->CreateTexture2D(&tex_desc, nullptr, mShadowMapTexture.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map: could not create a {}x{} x{} depth array.", resolution, resolution, cascadeCount);
        return false;
    }

    const int sliceCount = SHADOW_MAP_SLICES_FOR(cascadeCount);

    // The actor layer's own array, when it was asked to be a different size. Deliberately the SAME ArraySize
    // as the world one, even though only its upper half is ever drawn into: the slice number is then the same
    // arithmetic in both arrays, so every index downstream -- the DSV table, mShadowSliceValid/Key/Matrix, and
    // the shader's sliceBase + cascade + layerStride -- keeps working untouched, and the split is purely a
    // question of WHICH texture a slice lives in. The unused half costs a few MB at the small resolutions this
    // exists for, which is a good trade for not having two different slice numberings to keep in step.
    //
    // Failure is not fatal. Falling back to the shared array gives the arrangement that existed before the
    // layers could be sized apart, which is worse than asked for but is not "no shadows".
    if (actorResolution != resolution) {
        D3D11_TEXTURE2D_DESC actor_desc = tex_desc;
        actor_desc.Width = (UINT)actorResolution;
        actor_desc.Height = (UINT)actorResolution;
        if (FAILED(mDevice->CreateTexture2D(&actor_desc, nullptr, mShadowActorTexture.GetAddressOf()))) {
            SPDLOG_ERROR("Shadow map: could not create the {}x{} actor-layer depth array; "
                         "falling back to the world layer's array.",
                         actorResolution, actorResolution);
            mShadowActorTexture.Reset();
        } else {
            mShadowActorSplit = true;
        }
    }
    for (int i = 0; i < sliceCount; i++) {
        // Slices at or past cascadeCount are the actor layer (see shadow_map.h), so they come out of the actor
        // array when there is one.
        ID3D11Texture2D* sliceTexture =
            (mShadowActorSplit && i >= cascadeCount) ? mShadowActorTexture.Get() : mShadowMapTexture.Get();
        D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc;
        ZeroMemory(&dsv_desc, sizeof(dsv_desc));
        dsv_desc.Format = DXGI_FORMAT_D16_UNORM;
        dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
        dsv_desc.Texture2DArray.MipSlice = 0;
        dsv_desc.Texture2DArray.FirstArraySlice = (UINT)i;
        dsv_desc.Texture2DArray.ArraySize = 1;
        if (FAILED(mDevice->CreateDepthStencilView(sliceTexture, &dsv_desc, mShadowMapDsv[i].GetAddressOf()))) {
            SPDLOG_ERROR("Shadow map: could not create the depth view for slice {}.", i);
            for (int j = 0; j < i; j++) {
                mShadowMapDsv[j].Reset(); // do not leave views pointing at a texture we are dropping
            }
            mShadowMapTexture.Reset();
            return false;
        }
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
    ZeroMemory(&srv_desc, sizeof(srv_desc));
    srv_desc.Format = DXGI_FORMAT_R16_UNORM;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    srv_desc.Texture2DArray.MostDetailedMip = 0;
    srv_desc.Texture2DArray.MipLevels = 1;
    srv_desc.Texture2DArray.FirstArraySlice = 0;
    srv_desc.Texture2DArray.ArraySize = (UINT)sliceCount;
    if (FAILED(mDevice->CreateShaderResourceView(mShadowMapTexture.Get(), &srv_desc, mShadowMapSrv.GetAddressOf()))) {
        SPDLOG_ERROR("Shadow map: could not create the cascade array resource view.");
        for (int i = 0; i < sliceCount; i++) {
            mShadowMapDsv[i].Reset();
        }
        mShadowMapTexture.Reset();
        mShadowActorTexture.Reset();
        mShadowActorSplit = false;
        return false;
    }

    // The actor slot. When the layers share one array this is the very same view, so the shader reads the
    // same texels through both slots and the split costs nothing but a second binding.
    if (mShadowActorSplit) {
        if (FAILED(mDevice->CreateShaderResourceView(mShadowActorTexture.Get(), &srv_desc,
                                                     mShadowActorSrv.GetAddressOf()))) {
            SPDLOG_ERROR("Shadow map: could not create the actor-layer resource view; "
                         "falling back to the world layer's array.");
            mShadowActorSrv.Reset();
            mShadowActorTexture.Reset();
            mShadowActorSplit = false;
            // The actor slices' depth views point into a texture we just dropped, so they have to be rebuilt
            // against the shared array before anything renders through them.
            for (int i = cascadeCount; i < sliceCount; i++) {
                D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc;
                ZeroMemory(&dsv_desc, sizeof(dsv_desc));
                dsv_desc.Format = DXGI_FORMAT_D16_UNORM;
                dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
                dsv_desc.Texture2DArray.MipSlice = 0;
                dsv_desc.Texture2DArray.FirstArraySlice = (UINT)i;
                dsv_desc.Texture2DArray.ArraySize = 1;
                mShadowMapDsv[i].Reset();
                if (FAILED(mDevice->CreateDepthStencilView(mShadowMapTexture.Get(), &dsv_desc,
                                                           mShadowMapDsv[i].GetAddressOf()))) {
                    SPDLOG_ERROR("Shadow map: could not rebuild the depth view for slice {}.", i);
                    for (int j = 0; j < sliceCount; j++) {
                        mShadowMapDsv[j].Reset();
                    }
                    mShadowMapSrv.Reset();
                    mShadowMapTexture.Reset();
                    return false;
                }
            }
        }
    }
    if (!mShadowActorSplit) {
        mShadowActorSrv = mShadowMapSrv;
    }

    mShadowCascadeCount = cascadeCount;
    mShadowResolution = resolution;
    mShadowActorResolution = mShadowActorSplit ? actorResolution : resolution;
    return true;
}

void GfxRenderingAPIDX11::ShadowStaticRelease() {
    for (int i = 0; i < SHADOW_MAP_MAX_SLICES; i++) {
        mShadowStaticDsv[i].Reset();
        mShadowStaticValid[i] = false;
        mShadowStaticKey[i] = 0;
    }
    mShadowStaticTexture.Reset();
    mShadowStaticReady = false;
    mShadowStaticResolution = 0;
    mShadowStaticSlices = 0;
    mShadowStaticOpenSlice = -1;
}

// SOH [Enhancement] Static caster cache (see fast/shadow_map.h). A second array holding the WORLD layer's
// slices with only the casters that do not move in them.
//
// World layer only, and the array is only as long as that layer: the actor layer is characters, whose
// content changes every frame by definition, so there is no static half of it to keep.
bool GfxRenderingAPIDX11::CreateShadowStaticTargets(int cascadeCount, int resolution) {
    if (mShadowStaticTexture != nullptr && cascadeCount == mShadowStaticSlices &&
        resolution == mShadowStaticResolution) {
        return true;
    }
    ShadowStaticRelease();
    if (cascadeCount <= 0 || resolution <= 0 || mShadowMapTexture == nullptr) {
        return false;
    }

    // Deliberately the same desc as the live array, because the copy below requires it: D3D11 will only
    // copy between resources of identical type, format and dimensions.
    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.Width = (UINT)resolution;
    desc.Height = (UINT)resolution;
    desc.MipLevels = 1;
    desc.ArraySize = (UINT)cascadeCount;
    desc.Format = DXGI_FORMAT_R16_TYPELESS;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    // A depth target because the static half is rasterised straight into it; never read as a texture, only
    // copied out of, so it needs no shader resource view.
    desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    if (FAILED(mDevice->CreateTexture2D(&desc, nullptr, mShadowStaticTexture.GetAddressOf()))) {
        SPDLOG_WARN("Shadow map: could not create the {}x{} x{} static caster cache; the split is off.",
                    resolution, resolution, cascadeCount);
        ShadowStaticRelease();
        return false;
    }
    for (int i = 0; i < cascadeCount; i++) {
        D3D11_DEPTH_STENCIL_VIEW_DESC dsv;
        ZeroMemory(&dsv, sizeof(dsv));
        dsv.Format = DXGI_FORMAT_D16_UNORM;
        dsv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
        dsv.Texture2DArray.MipSlice = 0;
        dsv.Texture2DArray.FirstArraySlice = (UINT)i;
        dsv.Texture2DArray.ArraySize = 1;
        if (FAILED(mDevice->CreateDepthStencilView(mShadowStaticTexture.Get(), &dsv,
                                                   mShadowStaticDsv[i].GetAddressOf()))) {
            SPDLOG_WARN("Shadow map: could not create the static cache's view for slice {}.", i);
            ShadowStaticRelease();
            return false;
        }
    }
    mShadowStaticResolution = resolution;
    mShadowStaticSlices = cascadeCount;
    mShadowStaticReady = true;
    SPDLOG_INFO("Shadow map: static caster cache ready at {}x{} across {} slices.", resolution, resolution,
                cascadeCount);
    return true;
}

// Copy the static half of `slice` into the live slice. Neither may be bound as a target while it happens,
// which is why the target is dropped and put back around it.
//
// No source box: D3D11 will not copy a sub-region of a depth-stencil resource, and a whole slice is what is
// wanted anyway.
void GfxRenderingAPIDX11::ShadowStaticBlit(int slice) {
    mContext->OMSetRenderTargets(0, nullptr, nullptr);
    mContext->CopySubresourceRegion(mShadowMapTexture.Get(), (UINT)slice, 0, 0, 0, mShadowStaticTexture.Get(),
                                    (UINT)slice, nullptr);
    mContext->OMSetRenderTargets(0, nullptr, mShadowMapDsv[slice].Get());
}

int GfxRenderingAPIDX11::ShadowMapBeginCascadeSplit(int layer, int cascadeIndex, const float lightViewProj[16],
                                                   uint64_t staticKey, uint64_t dynamicKey) {
    // The two keys identify the slice together, and the live reuse test is unchanged by the split: a slice
    // still holds what its whole caster set drew. Mixed rather than xored so a static change cannot be
    // cancelled out by a dynamic one landing on the same bits.
    const uint64_t combined = (staticKey * 0x100000001B3ull) ^ dynamicKey;

    // The actor layer has no static half, and a cache that failed to build is simply not there.
    if (layer != SHADOW_MAP_LAYER_WORLD || !mShadowStaticReady || mShadowStaticTexture == nullptr ||
        !mShadowQuality.staticCache) {
        return ShadowMapBeginCascade(layer, cascadeIndex, lightViewProj, combined) ? SHADOW_MAP_SLICE_FULL
                                                                                  : SHADOW_MAP_SLICE_REUSED;
    }
    if (cascadeIndex < 0 || cascadeIndex >= mShadowCascadeCount) {
        return SHADOW_MAP_SLICE_REUSED;
    }
    const int slice = layer * mShadowCascadeCount + cascadeIndex;
    if (slice >= mShadowStaticSlices || mShadowStaticDsv[slice] == nullptr) {
        return ShadowMapBeginCascade(layer, cascadeIndex, lightViewProj, combined) ? SHADOW_MAP_SLICE_FULL
                                                                                  : SHADOW_MAP_SLICE_REUSED;
    }

    // Is the cached half still the right image? The same casters through a different matrix is a different
    // image, so both have to match.
    const bool staticUsable = mShadowStaticValid[slice] && mShadowStaticKey[slice] == staticKey &&
                              memcmp(mShadowStaticMatrix[slice], lightViewProj, 16 * sizeof(float)) == 0;

    if (staticUsable) {
        // The clear is what the blit replaces, so it is suppressed and the copy lands on a slice nobody
        // has written to yet this frame.
        mShadowSkipSliceClear = true;
        const bool opened = ShadowMapBeginCascade(layer, cascadeIndex, lightViewProj, combined);
        mShadowSkipSliceClear = false;
        if (!opened) {
            return SHADOW_MAP_SLICE_REUSED; // the live slice already held this exact image
        }
        ShadowStaticBlit(slice);
        return SHADOW_MAP_SLICE_DYNAMIC;
    }

    // The cached half has to be rebuilt, so the same opening body is aimed at the static copy instead. The
    // caller draws the static casters, then says so, and ShadowMapEndStaticCasters blits and switches over.
    mShadowStaticTargetSlice = slice;
    const bool opened = ShadowMapBeginCascade(layer, cascadeIndex, lightViewProj, combined);
    mShadowStaticTargetSlice = -1;
    if (!opened) {
        return SHADOW_MAP_SLICE_REUSED;
    }
    mShadowStaticOpenSlice = slice;
    mShadowStaticKey[slice] = staticKey;
    memcpy(mShadowStaticMatrix[slice], lightViewProj, 16 * sizeof(float));
    // Claimed only once the draws have gone in; see ShadowMapEndStaticCasters.
    mShadowStaticValid[slice] = false;
    return SHADOW_MAP_SLICE_FULL;
}

void GfxRenderingAPIDX11::ShadowMapEndStaticCasters() {
    if (mShadowStaticOpenSlice < 0) {
        return;
    }
    const int slice = mShadowStaticOpenSlice;
    mShadowStaticOpenSlice = -1;
    // The static half is complete and may now be trusted for later frames.
    mShadowStaticValid[slice] = true;
    ShadowStaticBlit(slice);
}

bool GfxRenderingAPIDX11::ShadowMapConfigure(int cascadeCount, int resolution, int actorResolution) {
    if (cascadeCount < 1) {
        cascadeCount = 1;
    } else if (cascadeCount > SHADOW_MAP_MAX_CASCADES) {
        cascadeCount = SHADOW_MAP_MAX_CASCADES;
    }
    if (resolution < SHADOW_MAP_MIN_RESOLUTION) {
        resolution = SHADOW_MAP_MIN_RESOLUTION;
    } else if (resolution > SHADOW_MAP_MAX_RESOLUTION) {
        resolution = SHADOW_MAP_MAX_RESOLUTION;
    }
    // Clamped against the same bounds, and never above the world layer's: a character map finer than the
    // scenery it is cast onto buys nothing, and allowing it would only spend the fill this split exists to
    // save.
    if (actorResolution < SHADOW_MAP_MIN_RESOLUTION) {
        actorResolution = SHADOW_MAP_MIN_RESOLUTION;
    } else if (actorResolution > resolution) {
        actorResolution = resolution;
    }
    if (!CreateShadowMapPipeline()) {
        return false;
    }
    if (!CreateShadowMapTargets(cascadeCount, resolution, actorResolution)) {
        return false;
    }

    // SOH [Enhancement] Static caster cache (see fast/shadow_map.h). Optional in every sense: a failure
    // here leaves the cache unavailable and the split path falls back to redrawing whole slices, which is
    // what happens today.
    if (mShadowQuality.staticCache) {
        CreateShadowStaticTargets(cascadeCount, resolution);
    } else {
        ShadowStaticRelease();
    }
    return true;
}

// SOH [Enhancement] Cascaded shadow maps: this cascade's rasterizer state, differing from the shared one
// only in the slope-scaled depth bias.
//
// That bias is a multiple of the polygon's depth gradient ACROSS A TEXEL, so its world-space effect scales
// with the texel -- and a texel of the far cascade is several world units, which made this the largest
// single source of shadows detaching from their casters at distance. It cannot be capped in the shader the
// way the normal offset is, because the rasterizer reads it from the pipeline state, not per pixel. So each
// cascade gets its own state with the multiplier reduced to whatever keeps its own texel under the world
// ceiling. Near cascades are nowhere near it and keep the base value untouched.
//
// The texel size comes out of the matrix rather than being plumbed in: the projection scales the light's
// unit x axis by 1/radius, so that column's length IS 1/radius, and one texel spans 2*radius/resolution.
// The result is quantised before it is compared, so a value drifting by a hair does not rebuild the state;
// combined with the radius hysteresis, rebuilds happen once in a great while rather than per frame.
ID3D11RasterizerState* GfxRenderingAPIDX11::ShadowRasterizerForCascade(int slice, int resolution,
                                                                       const float lightViewProj[16]) {
    if (slice < 0 || slice >= SHADOW_MAP_MAX_SLICES || resolution <= 0) {
        return mShadowRasterizerState.Get();
    }

    // Fixed, not settable. This is the ONE offset the shadow map has: the receiver compares against the
    // stored depth directly, so without a slope-scaled bias here every surface shadows itself and the whole
    // scene stripes. See SHADOW_MAP_SLOPE_BIAS.
    float slope = SHADOW_MAP_SLOPE_BIAS;
    slope = std::floor((slope * 8.0f) + 0.5f) / 8.0f; // eighths, so a hair of drift rebuilds nothing
    if (slope < 0.0f) {
        slope = 0.0f;
    }

    // The ceiling, enforced where D3D actually enforces it.
    //
    // What the rasterizer applies is SlopeScaledDepthBias * MaxDepthSlope, and MaxDepthSlope is how fast
    // the polygon's depth changes per pixel OF THE SHADOW MAP. On a surface facing the light that is nearly
    // nothing, which is the whole point of a slope-scaled term. On one raking away from it the depth crosses
    // a texel in a single step and the slope is enormous -- and the product with it is what gets written
    // into the map.
    //
    // This used to be bounded by reducing the MULTIPLIER, on the reasoning that a texel's worth of depth is
    // what the slope amounts to. That reasoning holds only while the surface is not grazing, which is the
    // one case the bound exists for: capping a multiplier does not cap a product whose other factor is
    // unbounded. So a wall at eighty-five degrees was written into the depth map displaced by whatever its
    // own slope happened to be, and no receiver-side correction can reach a depth that was already wrong
    // when it was stored.
    //
    // DepthBiasClamp is the hardware's bound on exactly that product, and it was left at zero -- which in
    // D3D means "no clamp", not "no bias". Setting it puts the ceiling where it was always meant to be. The
    // value is SHADOW_MAP_MAX_SLOPE_BIAS_WORLD in this cascade's own depth units: the projection scales the
    // light's unit x axis by 1/radius, so sx IS 1/radius, and a cascade's depth range is five radii by
    // construction.
    float depthBiasClamp = 0.0f;
    const float sx = std::sqrt((lightViewProj[0] * lightViewProj[0]) + (lightViewProj[4] * lightViewProj[4]) +
                               (lightViewProj[8] * lightViewProj[8]));
    if (sx > 1e-9f) {
        depthBiasClamp = SHADOW_MAP_MAX_SLOPE_BIAS_WORLD * sx / 5.0f;
        // Quantised for the same reason the slope is: a value drifting by a hair must not rebuild the state.
        depthBiasClamp = std::floor((depthBiasClamp * 4096.0f) + 0.5f) / 4096.0f;
    }

    // Both facings recorded. Front-face culling was tried here and removed: it does remove self-shadowing
    // acne at the source, by storing only the far side of a caster so the lit surface is never in the map to
    // be compared against itself -- but it needs the caster to HAVE a far side, and this game's scenery is
    // largely modelled from one side only. Those surfaces have nothing left once their front is culled and
    // stop casting entirely.
    if (mShadowRasterizerCascade[slice] == nullptr || mShadowRasterizerCascadeSlope[slice] != slope ||
        mShadowRasterizerCascadeClamp[slice] != depthBiasClamp) {
        D3D11_RASTERIZER_DESC rast_desc;
        ZeroMemory(&rast_desc, sizeof(rast_desc));
        rast_desc.FillMode = D3D11_FILL_SOLID;
        rast_desc.CullMode = D3D11_CULL_NONE;
        rast_desc.DepthClipEnable = FALSE;
        rast_desc.DepthBias = 0;
        rast_desc.SlopeScaledDepthBias = slope;
        rast_desc.DepthBiasClamp = depthBiasClamp;
        ComPtr<ID3D11RasterizerState> built;
        if (FAILED(mDevice->CreateRasterizerState(&rast_desc, built.GetAddressOf()))) {
            // Not fatal: the shared state is the same thing with the base slope, so the cascade keeps the
            // bias it had before this refinement existed.
            SPDLOG_ERROR("Shadow map: could not build the rasterizer state for slice {}.", slice);
            return mShadowRasterizerState.Get();
        }
        mShadowRasterizerCascade[slice] = built;
        mShadowRasterizerCascadeSlope[slice] = slope;
        mShadowRasterizerCascadeClamp[slice] = depthBiasClamp;
    }
    return mShadowRasterizerCascade[slice].Get();
}

bool GfxRenderingAPIDX11::ShadowMapBeginCascade(int layer, int cascadeIndex, const float lightViewProj[16],
                                                uint64_t contentKey) {
    if (!mShadowPipelineReady || mShadowMapTexture == nullptr || lightViewProj == nullptr) {
        return false;
    }
    if (cascadeIndex < 0 || cascadeIndex >= mShadowCascadeCount) {
        return false;
    }
    // The actor layer is shorter than the world layer, so a cascade the world has may have no actor slice at
    // all (see SHADOW_MAP_ACTOR_CASCADES). Refusing it here is what keeps the slice index inside the array.
    if (layer == SHADOW_MAP_LAYER_ACTORS && cascadeIndex >= SHADOW_MAP_ACTOR_CASCADES_FOR(mShadowCascadeCount)) {
        return false;
    }
    if (layer < 0 || layer >= SHADOW_MAP_LAYERS) {
        return false;
    }
    // Layer L, cascade C lives in slice L*cascadeCount + C (see fast/shadow_map.h).
    const int slice = layer * mShadowCascadeCount + cascadeIndex;

    // Resolved before the reuse test rather than after: the slope bias this cascade rasterises with is
    // part of what its depth map contains, and it follows a user setting. Building it here is free -- the
    // state objects are cached and this is the same call the draw path was going to make anyway.
    // Keyed and sized by the slice, because the actor layer may be a different resolution and the cap this
    // applies is a texel-size question -- the same cascade index has a different texel in each layer.
    const int sliceResolutionForBias =
        (layer == SHADOW_MAP_LAYER_ACTORS && mShadowActorSplit) ? mShadowActorResolution : mShadowResolution;
    ID3D11RasterizerState* rasterState = ShadowRasterizerForCascade(slice, sliceResolutionForBias, lightViewProj);

    // Nothing is going to be drawn here and nothing was last time either, so the slice is already the map of
    // nothing this call would produce. That reads identically however it is projected -- every receiver
    // compares as lit against the cleared far plane wherever it lands -- so unlike a slice with casters in
    // it, this one survives a MATRIX change too, and the comparison below is deliberately not reached.
    // Without this, a cascade the characters are nowhere near paid a full-resolution clear and a pipeline
    // setup every frame to produce the same empty map it already held.
    if (contentKey == SHADOW_MAP_EMPTY_CONTENT_KEY && mShadowSliceValid[slice] &&
        mShadowSliceKey[slice] == SHADOW_MAP_EMPTY_CONTENT_KEY) {
        mShadowCurrentSlice = -1; // nothing is open, so nothing may be invalidated by a stray submit
        return false;
    }

    // Nothing that decides this slice's contents has moved, so the slice still holds exactly the image
    // this call would redraw. Skip the clear, the state setup and every caster draw behind it.
    if (mShadowSliceValid[slice] && mShadowSliceKey[slice] == contentKey &&
        mShadowSliceRasterState[slice] == rasterState &&
        memcmp(mShadowSliceMatrix[slice], lightViewProj, 16 * sizeof(float)) == 0) {
        mShadowCurrentSlice = -1; // nothing is open, so nothing may be invalidated by a stray submit
        return false;
    }

    if (!mShadowPassActive) {
        // Remember the viewport once for the whole pass, not per cascade.
        mShadowSavedViewportCount = 1;
        mContext->RSGetViewports(&mShadowSavedViewportCount, &mShadowSavedViewport);
        // The cascade array is about to become a depth target, so it must not still be bound for
        // reading from the previous frame's main pass.
        ID3D11ShaderResourceView* null_srv[2] = { nullptr, nullptr };
        mContext->PSSetShaderResources(SHADER_MAX_TEXTURES, 2, null_srv);
        // Forget the previous pass's ACTOR upload. That vector is double-buffered by the interpreter, so it
        // keeps the same two allocations frame to frame while its contents change completely -- a pointer
        // match across passes would wrongly skip the upload and render last frame's characters forever.
        //
        // The WORLD layer is deliberately NOT reset: it is a cache that is only ever replaced by swapping in
        // the separate capture vector, so its data pointer necessarily changes whenever its contents do.
        // Keeping the record alive across passes is the whole point -- an unchanged room mesh then costs one
        // bind and one draw per cascade, with no upload at all.
        mShadowLastCasterPtr[SHADOW_MAP_LAYER_ACTORS * SHADOW_MAP_CASTER_SLOTS + SHADOW_MAP_CASTER_SLOT_MAIN] =
            nullptr;
        mShadowLastCasterCount[SHADOW_MAP_LAYER_ACTORS * SHADOW_MAP_CASTER_SLOTS + SHADOW_MAP_CASTER_SLOT_MAIN] = 0;
        // The world layer's SCENERY slot is double-buffered per frame exactly like the actor layer, so it
        // gets the same treatment. Only that layer's MAIN slot -- the cached room mesh -- keeps its record
        // across passes, which is the whole reason the slots are separate.
        mShadowLastCasterPtr[SHADOW_MAP_LAYER_WORLD * SHADOW_MAP_CASTER_SLOTS + SHADOW_MAP_CASTER_SLOT_SCENERY] =
            nullptr;
        mShadowLastCasterCount[SHADOW_MAP_LAYER_WORLD * SHADOW_MAP_CASTER_SLOTS + SHADOW_MAP_CASTER_SLOT_SCENERY] = 0;
        // The alpha list gets no such exemption: it shares one buffer between the two layers, so whatever it
        // holds is overwritten within the pass anyway, and the actor half is double-buffered exactly like the
        // opaque one. Forget it wholesale rather than reason about which half is safe.
        mShadowAlphaLastPtr = nullptr;
        mShadowAlphaLastCount = 0;
        mShadowPassActive = true;
        ShadowTimerBegin();
    }
    if (mShadowTimerFrameOpen) {
        // Counted only inside a timed frame, and against the frames actually counted. Incrementing it
        // unconditionally made the first report after the timer was switched on dump however many slices had
        // piled up while nothing was draining it -- a figure in the hundreds, "out of 8".
        mShadowSlicesDrawn++;
        // Which of the two reuse conditions failed. Split here, where both comparisons have just been made,
        // and split on the matrix rather than the key: a slice whose matrix is unchanged is one a narrower
        // key could have saved, and a slice whose matrix moved is one no key can save. Everything else --
        // the raster state, which follows a user setting and so changes almost never -- lands in the
        // matrix-held bucket alongside the key, which is where a rounding error belongs.
        if (!mShadowSliceValid[slice]) {
            mShadowRedrawFirst++;
        } else if (memcmp(mShadowSliceMatrix[slice], lightViewProj, 16 * sizeof(float)) != 0) {
            mShadowRedrawMatrix++;
        } else {
            mShadowRedrawContent++;
            if (layer == SHADOW_MAP_LAYER_WORLD) {
                mShadowRedrawContentWorld++;
            } else {
                mShadowRedrawContentActors++;
            }
        }
    }
    mShadowCurrentLayer = layer;
    mShadowCurrentSlice = slice;
    // Claimed before the draws, and dropped again by any of them that cannot complete (see
    // ShadowMapDrawCasters): a half-filled slice must not be mistaken for a finished one next frame.
    mShadowSliceValid[slice] = true;
    mShadowSliceKey[slice] = contentKey;
    mShadowSliceRasterState[slice] = rasterState;
    memcpy(mShadowSliceMatrix[slice], lightViewProj, 16 * sizeof(float));
    // Everything below re-establishes the OPAQUE pipeline, so any alpha binding from the previous cascade is
    // gone by the time this returns.
    mShadowAlphaBound = false;

    // SOH [Enhancement] Static caster cache: the split path aims this same body at the static copy while
    // its half is drawn, and suppresses the clear on the frame where that copy is about to be blitted in.
    // Both flags are off for every other caller, so the ordinary path is unchanged.
    ID3D11DepthStencilView* sliceTarget = (mShadowStaticTargetSlice == slice && mShadowStaticDsv[slice] != nullptr)
                                              ? mShadowStaticDsv[slice].Get()
                                              : mShadowMapDsv[slice].Get();
    mContext->OMSetRenderTargets(0, nullptr, sliceTarget);
    if (!mShadowSkipSliceClear) {
        mContext->ClearDepthStencilView(sliceTarget, D3D11_CLEAR_DEPTH, 1.0f, 0);
    }

    D3D11_VIEWPORT viewport;
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    // Sized for the array this slice actually lives in, not for the world layer.
    const float sliceResolution =
        (float)((layer == SHADOW_MAP_LAYER_ACTORS && mShadowActorSplit) ? mShadowActorResolution : mShadowResolution);
    viewport.Width = sliceResolution;
    viewport.Height = sliceResolution;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    mContext->RSSetViewports(1, &viewport);

    ShadowDepthCB cb;
    memcpy(cb.lightViewProj, lightViewProj, sizeof(cb.lightViewProj));
    D3D11_MAPPED_SUBRESOURCE ms;
    ZeroMemory(&ms, sizeof(ms));
    if (SUCCEEDED(mContext->Map(mShadowDepthCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
        memcpy(ms.pData, &cb, sizeof(cb));
        mContext->Unmap(mShadowDepthCb.Get(), 0);
    }

    mContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    mContext->IASetInputLayout(mShadowDepthLayout.Get());
    mContext->VSSetShader(mShadowDepthVs.Get(), nullptr, 0);
    mContext->VSSetConstantBuffers(0, 1, mShadowDepthCb.GetAddressOf());
    mContext->PSSetShader(nullptr, nullptr, 0); // depth-only: no pixel shader at all
    mContext->RSSetState(rasterState);
    mContext->OMSetDepthStencilState(mShadowDepthStencilState.Get(), 0);
    mContext->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
    return true;
}

// Drop the record of what the open slice holds. Called from the submit paths when they cannot complete, so
// a slice that ends up with only part of its casters is redrawn next frame rather than reused.
void GfxRenderingAPIDX11::ShadowMapInvalidateOpenSlice() {
    if (mShadowCurrentSlice >= 0 && mShadowCurrentSlice < SHADOW_MAP_MAX_SLICES) {
        mShadowSliceValid[mShadowCurrentSlice] = false;
    }
}

void GfxRenderingAPIDX11::ShadowMapDrawCasters(const float* worldXyz, size_t vertexCount, int slot, size_t firstVertex,
                                               size_t drawCount) {
    if (!mShadowPassActive || worldXyz == nullptr || vertexCount < 3) {
        return;
    }
    vertexCount -= vertexCount % 3; // whole triangles only

    // The sub-range to draw out of the uploaded list. Clamped rather than trusted, and rounded to whole
    // triangles the same way the list itself is: a caller that names a range running past the end must lose
    // the overhang, not read whatever the buffer happens to hold past it.
    if (firstVertex >= vertexCount) {
        return;
    }
    const size_t maxDraw = vertexCount - firstVertex;
    if (drawCount == 0 || drawCount > maxDraw) {
        drawCount = maxDraw;
    }
    drawCount -= drawCount % 3;
    if (drawCount < 3) {
        return;
    }

    // The cutout path binds its own pipeline and leaves it bound, on the reasoning that the next
    // ShadowMapBeginCascade puts the opaque one back. That held only while no opaque draw ever followed a
    // cutout draw INSIDE one slice -- an ordering rule that lived in a comment and nowhere else, and that
    // the static caster cache broke the moment it moved the cached half's cutouts above the scenery.
    //
    // What that cost: scenery rasterised through the cutout pipeline, a vertex shader expecting five floats
    // per vertex fed three, and a pixel shader clipping against whatever texture was still bound. The depth
    // written was noise, and the shadow came out stippled.
    //
    // So the opaque path restores what it needs instead of trusting the order. One branch per draw, taken
    // only when a cutout batch actually preceded this one.
    if (mShadowAlphaBound) {
        mContext->IASetInputLayout(mShadowDepthLayout.Get());
        mContext->VSSetShader(mShadowDepthVs.Get(), nullptr, 0);
        mContext->VSSetConstantBuffers(0, 1, mShadowDepthCb.GetAddressOf());
        mContext->PSSetShader(nullptr, nullptr, 0); // depth-only: no pixel shader at all
        mShadowAlphaBound = false;
    }

    const int layerIndex = (mShadowCurrentLayer >= 0 && mShadowCurrentLayer < SHADOW_MAP_LAYERS) ? mShadowCurrentLayer : 0;
    const int slotIndex = (slot >= 0 && slot < SHADOW_MAP_CASTER_SLOTS) ? slot : 0;
    // One buffer per (layer, slot). The slot is what lets the world layer draw its cached room mesh and a
    // per-frame scenery list in the same cascade without either one evicting the other's upload.
    const int layer = layerIndex * SHADOW_MAP_CASTER_SLOTS + slotIndex;
    const UINT stride = 3 * sizeof(float);
    const UINT offset = 0;

    // Every cascade draws the same caster list, and for the world layer the list is usually the same one as
    // last frame too, so the buffer already holds exactly what is needed. Re-uploading it would cost a full
    // copy of the room mesh per cascade per frame for nothing.
    if (mShadowLastCasterPtr[layer] == worldXyz && mShadowLastCasterCount[layer] == vertexCount &&
        mShadowCasterVb[layer] != nullptr) {
        // The other layer may have bound its own buffer since, so rebind -- it is a state change, not a copy.
        mContext->IASetVertexBuffers(0, 1, mShadowCasterVb[layer].GetAddressOf(), &stride, &offset);
        mContext->Draw((UINT)drawCount, (UINT)firstVertex);
        return;
    }

    // Grow this layer's caster buffer to fit the largest batch seen so far; batches are then uploaded whole.
    if (mShadowCasterVb[layer] == nullptr || mShadowCasterVbVertices[layer] < vertexCount) {
        // Start large rather than at a few thousand vertices. Growing means creating a new buffer, which is
        // a driver allocation in the middle of a frame -- and the caster count climbs as the camera turns
        // and more of the scene is submitted, so a small starting size turns every early rotation into a
        // series of stalls. 128k vertices is about 1.5 MB and covers a room mesh plus its actors outright.
        size_t capacity = mShadowCasterVbVertices[layer] ? mShadowCasterVbVertices[layer] : 128u * 1024u;
        while (capacity < vertexCount) {
            capacity *= 2;
        }
        D3D11_BUFFER_DESC vb_desc;
        ZeroMemory(&vb_desc, sizeof(vb_desc));
        vb_desc.Usage = D3D11_USAGE_DYNAMIC;
        vb_desc.ByteWidth = (UINT)(capacity * 3 * sizeof(float));
        vb_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        ComPtr<ID3D11Buffer> grown;
        if (FAILED(mDevice->CreateBuffer(&vb_desc, nullptr, grown.GetAddressOf()))) {
            SPDLOG_ERROR("Shadow map: could not grow the caster buffer to {} vertices.", capacity);
            ShadowMapInvalidateOpenSlice(); // these casters never reached the slice
            return;
        }
        mShadowCasterVb[layer] = grown;
        mShadowCasterVbVertices[layer] = capacity;
        // The new buffer holds nothing yet, so any record of what the old one held is worthless.
        mShadowLastCasterPtr[layer] = nullptr;
        mShadowLastCasterCount[layer] = 0;
    }

    D3D11_MAPPED_SUBRESOURCE ms;
    ZeroMemory(&ms, sizeof(ms));
    if (FAILED(mContext->Map(mShadowCasterVb[layer].Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
        ShadowMapInvalidateOpenSlice();
        return;
    }
    memcpy(ms.pData, worldXyz, vertexCount * 3 * sizeof(float));
    mContext->Unmap(mShadowCasterVb[layer].Get(), 0);

    mContext->IASetVertexBuffers(0, 1, mShadowCasterVb[layer].GetAddressOf(), &stride, &offset);
    mContext->Draw((UINT)drawCount, (UINT)firstVertex);
    // Records the whole uploaded list, not the range drawn: the next call for this slot compares against
    // what the BUFFER holds, and it holds all of it however little of it was just drawn.
    mShadowLastCasterPtr[layer] = worldXyz;
    mShadowLastCasterCount[layer] = vertexCount;
}

bool GfxRenderingAPIDX11::SupportsShadowMapAlphaCasters() {
    return mShadowAlphaPipelineReady;
}

void GfxRenderingAPIDX11::ShadowMapUploadAlphaCasters(const float* xyzUv, size_t vertexCount) {
    if (!mShadowPassActive || !mShadowAlphaPipelineReady || xyzUv == nullptr || vertexCount < 3) {
        return;
    }
    // Same reuse rule as the opaque list: identical pointer and count means the buffer already holds this
    // geometry, whether that is from the previous cascade or (for the cached world layer) the previous
    // frame. Only the world layer's record survives a pass; see ShadowMapBeginCascade.
    if (mShadowAlphaLastPtr == xyzUv && mShadowAlphaLastCount == vertexCount && mShadowAlphaVb != nullptr) {
        return;
    }

    if (mShadowAlphaVb == nullptr || mShadowAlphaVbVertices < vertexCount) {
        size_t capacity = mShadowAlphaVbVertices ? mShadowAlphaVbVertices : 32u * 1024u;
        while (capacity < vertexCount) {
            capacity *= 2;
        }
        D3D11_BUFFER_DESC vb_desc;
        ZeroMemory(&vb_desc, sizeof(vb_desc));
        vb_desc.Usage = D3D11_USAGE_DYNAMIC;
        vb_desc.ByteWidth = (UINT)(capacity * 5 * sizeof(float));
        vb_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        ComPtr<ID3D11Buffer> grown;
        if (FAILED(mDevice->CreateBuffer(&vb_desc, nullptr, grown.GetAddressOf()))) {
            SPDLOG_ERROR("Shadow map: could not grow the alpha caster buffer to {} vertices.", capacity);
            ShadowMapInvalidateOpenSlice();
            return;
        }
        mShadowAlphaVb = grown;
        mShadowAlphaVbVertices = capacity;
        mShadowAlphaLastPtr = nullptr;
        mShadowAlphaLastCount = 0;
    }

    D3D11_MAPPED_SUBRESOURCE ms;
    ZeroMemory(&ms, sizeof(ms));
    if (FAILED(mContext->Map(mShadowAlphaVb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
        ShadowMapInvalidateOpenSlice();
        return;
    }
    memcpy(ms.pData, xyzUv, vertexCount * 5 * sizeof(float));
    mContext->Unmap(mShadowAlphaVb.Get(), 0);
    mShadowAlphaLastPtr = xyzUv;
    mShadowAlphaLastCount = vertexCount;
}

void GfxRenderingAPIDX11::ShadowMapDrawAlphaRange(uint32_t textureId, size_t firstVertex, size_t vertexCount) {
    if (!mShadowPassActive || !mShadowAlphaPipelineReady || mShadowAlphaVb == nullptr || vertexCount < 3) {
        return;
    }
    if (firstVertex + vertexCount > mShadowAlphaLastCount) {
        ShadowMapInvalidateOpenSlice();
        return; // range does not lie inside what was uploaded
    }
    if (textureId >= mTextures.size() || mTextures[textureId].resource_view == nullptr) {
        // The caller resolved this id against the texture cache and cannot see that the backend has since
        // dropped the texture, so the slice's content key does not describe what actually got drawn. Drop
        // the record rather than reuse a slice that is missing a material.
        ShadowMapInvalidateOpenSlice();
        return;
    }
    vertexCount -= vertexCount % 3;

    // Switch the pipeline once and leave it: consecutive ranges differ only by texture and draw offset.
    // ShadowMapBeginCascade puts the opaque pipeline back for the next cascade.
    if (!mShadowAlphaBound) {
        UINT stride = 5 * sizeof(float);
        UINT offset = 0;
        mContext->IASetInputLayout(mShadowAlphaLayout.Get());
        mContext->IASetVertexBuffers(0, 1, mShadowAlphaVb.GetAddressOf(), &stride, &offset);
        mContext->VSSetShader(mShadowAlphaVs.Get(), nullptr, 0);
        mContext->PSSetShader(mShadowAlphaPs.Get(), nullptr, 0);
        mContext->PSSetSamplers(0, 1, mShadowAlphaSampler.GetAddressOf());
        mShadowAlphaBound = true;
        // The main pass tracks which resource views it left bound; this pass binds its own, so that record
        // has to be invalidated or the next draw would skip a rebind it actually needs.
        mLastResourceViews[0] = nullptr;
        // And the same for the sampler beside it, for exactly the same reason. This line was not needed
        // while the main pass re-bound every sampler on every draw and so could not be misled by a stale
        // record; it is needed now that the pass skips the ones it believes are already bound.
        mLastSamplerStates[0] = nullptr;
    }
    mContext->PSSetShaderResources(0, 1, mTextures[textureId].resource_view.GetAddressOf());
    mContext->Draw((UINT)vertexCount, (UINT)firstVertex);
}

void GfxRenderingAPIDX11::SetShadowMapParams(const float* viewProj, const float* splitDistances, int cascadeCount,
                                             float blendFraction, float strength, float debugMode) {
    GfxRenderingAPI::SetShadowMapParams(viewProj, splitDistances, cascadeCount, blendFraction, strength, debugMode);

    ZeroMemory(&mPerShadowCbData, sizeof(mPerShadowCbData));
    const int count = mShadowCascadesActive;
    mPerShadowCbData.shadow_params[0] = (float)count;
    mPerShadowCbData.shadow_params[1] = mShadowBlendFraction;
    mPerShadowCbData.shadow_params[2] = 0.0f;
    mPerShadowCbData.shadow_params[3] = mShadowStrength;
    mPerShadowCbData.shadow_range[0] = mShadowMaxViewDepth;
    mPerShadowCbData.shadow_range[1] = mShadowDebug;
    mPerShadowCbData.shadow_range[2] = 0.0f;
    mPerShadowCbData.shadow_range[3] = 0.0f;
    for (int i = 0; i < 3; i++) {
        mPerShadowCbData.shadow_actor_min[i] = mShadowActorBoundsMin[i];
        mPerShadowCbData.shadow_actor_max[i] = mShadowActorBoundsMax[i];
    }
    mPerShadowCbData.shadow_actor_min[3] = 0.0f;
    mPerShadowCbData.shadow_actor_max[3] = 0.0f;

    for (int c = 0; c < count; c++) {
        const float* m = &mShadowViewProj[c * 16];
        memcpy(mPerShadowCbData.shadow_view_proj[c], m, 16 * sizeof(float));
        mPerShadowCbData.shadow_splits[c] = mShadowSplits[c];

        // Recover the cascade's world extent from its own matrix instead of plumbing it through the API:
        // the projection scales the light's x axis by 1/radius, and that axis is a unit vector, so the
        // length of its column IS 1/radius. One texel then spans 2*radius/resolution.
        const float sx = std::sqrt(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]);
        if (sx > 1e-9f && mShadowResolution > 0) {
            const float texelWorld = 2.0f / (sx * (float)mShadowResolution);
            mPerShadowCbData.shadow_texel_world[c] = texelWorld;
            // Kept where the application can read it too (see ShadowMapCascadeReport): once the automatic
            // ladder is choosing the splits, this is the only place the real texel size exists.
            mShadowTexelWorld[c] = texelWorld;
            mPerShadowCbData.shadow_texel_uv[c] = 1.0f / (float)mShadowResolution;
            mPerShadowCbData.shadow_actor_texel_uv[c] =
                1.0f / (float)(mShadowActorSplit ? mShadowActorResolution : mShadowResolution);
        }
    }

    // SOH [Enhancement] Edge quality (see fast/shadow_map.h). Written after the loop because ZeroMemory
    // above cleared it, and these are frame-global rather than per cascade.
    {
        const ShadowMapQuality& q = mShadowQuality;
        mPerShadowCbData.shadow_edge[0] = q.analyticEdge ? 1.0f : 0.0f;
        mPerShadowCbData.shadow_edge[1] = q.analyticEdgeWidth;
        // Jitter with a zero radius would fetch the same quad N times and average it to itself -- N times
        // the bandwidth for the picture it already had. Reported as off rather than honoured literally.
        const bool jitterOn = q.jitter != 0 && q.jitterRadius > 0.0f && q.jitterTaps > 1;
        mPerShadowCbData.shadow_edge[2] = jitterOn ? 1.0f : 0.0f;
        mPerShadowCbData.shadow_edge[3] = (float)q.jitterTaps;
        mPerShadowCbData.shadow_jitter[0] = q.jitterRadius;
        // y, z and w are dead slots: they carried the per-frame tap rotation and the filterable modes,
        // both removed. Left zeroed rather than repacked, so this removal does not also shift the cbuffer
        // layout -- that is its own change, with its own layout check.
        mPerShadowCbData.shadow_jitter[1] = 0.0f;
        mPerShadowCbData.shadow_jitter[2] = 0.0f;
        mPerShadowCbData.shadow_jitter[3] = 0.0f;

        // SOH [Enhancement] Shadow acne (see fast/shadow_map.h). Each magnitude is zeroed when its own
        // switch is off, so the shader multiplies by it rather than branching on a second flag.
        const ShadowMapAcne& acne = q.acne;
        mPerShadowCbData.shadow_acne0[0] = acne.enabled ? 1.0f : 0.0f;
        mPerShadowCbData.shadow_acne0[1] = acne.normalOffset ? acne.normalTexels : 0.0f;
        // z and w are dead slots: the light-offset and depth-bias methods are gone. Normal offset and
        // slope scaling do the same job, and these two were off.
        mPerShadowCbData.shadow_acne0[2] = 0.0f;
        mPerShadowCbData.shadow_acne0[3] = 0.0f;
        mPerShadowCbData.shadow_acne1[0] = acne.slopeScaled ? 1.0f : 0.0f;
        mPerShadowCbData.shadow_acne1[1] = acne.slopeMax;
        mPerShadowCbData.shadow_acne1[2] = acne.enabled ? 1.0f : 0.0f;
        mPerShadowCbData.shadow_acne1[3] = 0.0f;

        // SOH [Enhancement] Edge hardening (see fast/shadow_map.h).
        mPerShadowCbData.shadow_harden[0] = q.edgeHarden ? 1.0f : 0.0f;
        mPerShadowCbData.shadow_harden[1] = q.edgeHardness;
        mPerShadowCbData.shadow_harden[2] = q.edgeThreshold;
        mPerShadowCbData.shadow_harden[3] = 0.0f; // dead slot; carried bleed reduction

        // SOH [Enhancement] Clipmap layout (see fast/shadow_map.h).
        for (int i = 0; i < 3; i++) {
            mPerShadowCbData.shadow_clip_x[i] = mShadowClipmapX[i];
            mPerShadowCbData.shadow_clip_y[i] = mShadowClipmapY[i];
            mPerShadowCbData.shadow_clip_z[i] = mShadowClipmapZ[i];
        }
        // The camera's coordinate along each axis rides in that axis's w, which is where the shader wants
        // it: every use of an axis is immediately followed by subtracting the camera along it.
        mPerShadowCbData.shadow_clip_x[3] = mShadowClipmapCamera[0];
        mPerShadowCbData.shadow_clip_y[3] = mShadowClipmapCamera[1];
        mPerShadowCbData.shadow_clip_z[3] = mShadowClipmapCamera[2];
        mPerShadowCbData.shadow_clip_p[0] = mShadowClipmapBase;
        mPerShadowCbData.shadow_clip_p[1] = (float)mShadowClipmapLevels;
        mPerShadowCbData.shadow_clip_p[2] = (float)mShadowClipmapResolution;
        mPerShadowCbData.shadow_clip_p[3] = 0.0f;

        mShadowQualityFrame++;
    }

    mShadowCbDirty = true;
}

// SOH [Enhancement] GPU timing. See the members in gfx_direct3d_common.h for why it exists at all.
//
// One disjoint block per frame, with the frame's own interval inside it and the depth pass's interval inside
// that. Two numbers out of one clock: what the whole frame costs the GPU, and how much of it the cascades
// cost to fill. The rest of the shadow feature -- the receiver shaders, which no pass boundary can bracket
// because they run inside ordinary scene draws -- is then had by difference, by reading the frame number
// with the feature off and again with it on.
void GfxRenderingAPIDX11::ShadowTimerFrameBegin() {
    mShadowTimerOpen = false;
    mShadowTimerFrameOpen = false;
    // Either switch arms the timer: the debug view (which has always timed as a side effect) or the
    // profiling flag, which times without touching what the shader draws. See SetShadowMapProfiling.
    if ((!mShadowProfile && mShadowDebug < 0.5f) || mShadowTimerFailed || mDevice == nullptr) {
        return;
    }
    const int i = mShadowTimerSlot;
    if (mShadowTimerDisjoint[i] == nullptr) {
        D3D11_QUERY_DESC qd;
        ZeroMemory(&qd, sizeof(qd));
        qd.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
        HRESULT hr = mDevice->CreateQuery(&qd, mShadowTimerDisjoint[i].GetAddressOf());
        qd.Query = D3D11_QUERY_TIMESTAMP;
        ID3D11Query** stamps[4] = { mShadowTimerFrameStart[i].GetAddressOf(), mShadowTimerFrameEnd[i].GetAddressOf(),
                                    mShadowTimerStart[i].GetAddressOf(), mShadowTimerEnd[i].GetAddressOf() };
        for (int k = 0; SUCCEEDED(hr) && k < 4; k++) {
            hr = mDevice->CreateQuery(&qd, stamps[k]);
        }
        if (FAILED(hr)) {
            // Not fatal and not worth retrying: timing is a diagnostic, and a device that will not give us
            // queries still renders shadows perfectly well.
            SPDLOG_WARN("Shadow map: timestamp queries unavailable; the GPU will not be timed.");
            mShadowTimerFailed = true;
            return;
        }
    }
    mContext->Begin(mShadowTimerDisjoint[i].Get());
    mContext->End(mShadowTimerFrameStart[i].Get());
    mShadowTimerFrameOpen = true;
}

// Returns whether a start timestamp was issued, which is the only thing that makes the matching end
// meaningful: a frame where every slice was reused opens no pass at all and must not be recorded as a
// zero-millisecond one.
bool GfxRenderingAPIDX11::ShadowTimerBegin() {
    if (!mShadowTimerFrameOpen) {
        return false;
    }
    mContext->End(mShadowTimerStart[mShadowTimerSlot].Get());
    mShadowTimerOpen = true;
    return true;
}

void GfxRenderingAPIDX11::ShadowTimerEnd() {
    if (!mShadowTimerOpen) {
        return;
    }
    mContext->End(mShadowTimerEnd[mShadowTimerSlot].Get());
    mShadowTimerOpen = false;
    mShadowTimerPassIssued[mShadowTimerSlot] = true;
}

void GfxRenderingAPIDX11::ShadowTimerFrameEnd() {
    if (!mShadowTimerFrameOpen) {
        return;
    }
    const int i = mShadowTimerSlot;
    mContext->End(mShadowTimerFrameEnd[i].Get());
    mContext->End(mShadowTimerDisjoint[i].Get());
    mShadowTimerPending[i] = true;
    mShadowTimerFrameOpen = false;
    mShadowSlicesFrames++;
    mShadowTimerSlot = (i + 1) % kShadowTimerFrames;
    ShadowTimerCollect();
}

// Reads back whichever frame's queries are old enough to have finished, and reports the average once a
// second. D3D11_ASYNC_GETDATA_DONOTFLUSH throughout: this must never push the GPU along to get an answer.
void GfxRenderingAPIDX11::ShadowTimerCollect() {
    if (!mShadowProfile && mShadowDebug < 0.5f) {
        return;
    }
    // The slot about to be reused is the oldest one outstanding, so that is the one to drain.
    const int i = mShadowTimerSlot;
    if (mShadowTimerPending[i]) {
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
        UINT64 f0 = 0, f1 = 0, t0 = 0, t1 = 0;
        bool ready =
            mContext->GetData(mShadowTimerDisjoint[i].Get(), &dj, sizeof(dj), D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK &&
            mContext->GetData(mShadowTimerFrameStart[i].Get(), &f0, sizeof(f0), D3D11_ASYNC_GETDATA_DONOTFLUSH) ==
                S_OK &&
            mContext->GetData(mShadowTimerFrameEnd[i].Get(), &f1, sizeof(f1), D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK;
        if (ready && mShadowTimerPassIssued[i]) {
            ready =
                mContext->GetData(mShadowTimerStart[i].Get(), &t0, sizeof(t0), D3D11_ASYNC_GETDATA_DONOTFLUSH) ==
                    S_OK &&
                mContext->GetData(mShadowTimerEnd[i].Get(), &t1, sizeof(t1), D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK;
        }
        if (ready) {
            mShadowTimerPending[i] = false;
            // Disjoint means the clock changed rate mid-measurement (power management), so the interval is
            // meaningless and is dropped rather than averaged in.
            if (!dj.Disjoint && dj.Frequency != 0 && f1 >= f0) {
                const double toMs = 1000.0 / (double)dj.Frequency;
                mShadowTimerFrameSumMs += (double)(f1 - f0) * toMs;
                mShadowTimerFrameSamples++;
                if (mShadowTimerPassIssued[i] && t1 >= t0) {
                    mShadowTimerSumMs += (double)(t1 - t0) * toMs;
                    mShadowTimerSamples++;
                }
            }
            mShadowTimerPassIssued[i] = false;
        }
    }
    if (++mShadowTimerReported >= 60) {
        mShadowTimerReported = 0;
        if (mShadowTimerFrameSamples > 0) {
            const double frameMs = mShadowTimerFrameSumMs / mShadowTimerFrameSamples;
            const double passMs = mShadowTimerSamples > 0 ? mShadowTimerSumMs / mShadowTimerSamples : 0.0;
            // The pass sample count is the interesting one now that cascades can be parked: it is how many
            // frames had to submit a depth pass at all, and the gap to the frame count is how many got away
            // with reusing every slice they needed.
            const float perFrame = mShadowSlicesFrames > 0 ? 1.0f / (float)mShadowSlicesFrames : 0.0f;
            SPDLOG_INFO("Shadow map GPU: frame {:.2f} ms, of which the depth pass is {:.2f} ms ({:.0f}%); "
                        "{} of {} timed frames submitted a pass, {:.1f} of {} slices redrawn per frame "
                        "({:.1f} content only = {:.1f} world + {:.1f} actors, {:.1f} cascade moved, "
                        "{:.1f} first fill) "
                        "({} cascades x {} layers at {}px)",
                        frameMs, passMs, frameMs > 0.0 ? (passMs / frameMs * 100.0) : 0.0, mShadowTimerSamples,
                        mShadowTimerFrameSamples, (float)mShadowSlicesDrawn * perFrame,
                        SHADOW_MAP_SLICES_FOR(mShadowCascadeCount), (float)mShadowRedrawContent * perFrame,
                        (float)mShadowRedrawContentWorld * perFrame, (float)mShadowRedrawContentActors * perFrame,
                        (float)mShadowRedrawMatrix * perFrame, (float)mShadowRedrawFirst * perFrame,
                        mShadowCascadeCount, SHADOW_MAP_LAYERS, mShadowResolution);
        } else {
            SPDLOG_INFO("Shadow map GPU: no timing collected in the last 60 frames");
        }
        mShadowTimerSumMs = 0.0;
        mShadowTimerSamples = 0;
        mShadowTimerFrameSumMs = 0.0;
        mShadowTimerFrameSamples = 0;
        mShadowSlicesDrawn = 0;
        mShadowSlicesFrames = 0;
        mShadowRedrawContent = 0;
        mShadowRedrawContentWorld = 0;
        mShadowRedrawContentActors = 0;
        mShadowRedrawMatrix = 0;
        mShadowRedrawFirst = 0;
    }
}

void GfxRenderingAPIDX11::ShadowMapEndPass() {
    // The cascade array has to be handed to the main pass whether or not anything was drawn into it this
    // frame. A frame in which every slice was reused (see ShadowMapBeginCascade) opens no pass at all, and
    // the maps it is reusing are just as valid as freshly drawn ones -- returning early here would leave
    // the shader sampling whatever happened to be bound and the shadows would vanish while nothing moved.
    if (!mShadowPassActive) {
        ShadowMapBindForReading();
        return;
    }
    mShadowPassActive = false;
    ShadowTimerEnd();
    mShadowAlphaBound = false;
    mShadowCurrentSlice = -1;

    // Put back the frame's render target and viewport.
    if (mCurrentFramebuffer >= 0 && (size_t)mCurrentFramebuffer < mFrameBuffers.size()) {
        FramebufferDX11& fb = mFrameBuffers[mCurrentFramebuffer];
        mContext->OMSetRenderTargets(1, fb.render_target_view.GetAddressOf(),
                                     fb.has_depth_buffer ? fb.depth_stencil_view.Get() : nullptr);
    }
    if (mShadowSavedViewportCount > 0) {
        mContext->RSSetViewports(1, &mShadowSavedViewport);
    }
    mContext->RSSetState(mRasterizerState.Get());

    // The depth pass bound its own shader, layout, blend and depth-stencil state behind the per-draw
    // path's back. Clearing these "last state" trackers forces the next ordinary draw to set all of
    // them again -- without this it would compare against a stale cache and keep the depth-only
    // pipeline bound, rendering nothing.
    mLastShaderProgram = nullptr;
    mLastVertexBufferStride = 0;
    mLastBlendState = nullptr;
    mLastStencilMode = -1;
    mLastDepthTest = -1;
    mLastDepthMask = -1;
    mLastZmodeDecal = -1;

    ShadowMapBindForReading();
}

// Hand the cascades to the main pass, past the combiner's own texture slots.
void GfxRenderingAPIDX11::ShadowMapBindForReading() {
    if (mShadowMapSrv != nullptr) {
        mContext->PSSetShaderResources(SHADER_MAX_TEXTURES, 1, mShadowMapSrv.GetAddressOf());
        mContext->PSSetSamplers(SHADER_MAX_TEXTURES, 1, mShadowMapSampler.GetAddressOf());
        // The actor layer's slot. Holds the same view as above while the layers share an array, so this is
        // the unsplit arrangement reaching the shader through two names rather than one.
        if (mShadowActorSrv != nullptr) {
            mContext->PSSetShaderResources(SHADER_MAX_TEXTURES + 1, 1, mShadowActorSrv.GetAddressOf());
            mContext->PSSetSamplers(SHADER_MAX_TEXTURES + 1, 1, mShadowMapSampler.GetAddressOf());
        }
    }
}

// PRISM-HELPERS-BEGIN
// Everything between this marker and PRISM-HELPERS-END is what the shader preprocessor needs in order to
// expand the combiner, and nothing else. tools/shader-validate slices it out of this file and compiles it,
// so that the shader it checks is expanded by THIS code rather than by a copy of it that could drift --
// which would leave the check quietly validating a shader the game does not build. Keep the markers around
// exactly the free functions; the region is compiled on its own, so it must not come to depend on anything
// declared further down.
#define RAND_NOISE "((random(float3(floor(screenSpace.xy * noise_scale), noise_frame)) + 1.0) / 2.0)"

static const char* prism_shader_item_to_str(uint32_t item, bool with_alpha, bool only_alpha, bool inputs_have_alpha,
                                            bool first_cycle, bool hint_single_element) {
    if (!only_alpha) {
        switch (item) {
            default:
            case SHADER_0:
                return with_alpha ? "float4(0.0, 0.0, 0.0, 0.0)" : "float3(0.0, 0.0, 0.0)";
            case SHADER_1:
                return with_alpha ? "float4(1.0, 1.0, 1.0, 1.0)" : "float3(1.0, 1.0, 1.0)";
            case SHADER_INPUT_1:
                return with_alpha || !inputs_have_alpha ? "input.input1" : "input.input1.rgb";
            case SHADER_INPUT_2:
                return with_alpha || !inputs_have_alpha ? "input.input2" : "input.input2.rgb";
            case SHADER_INPUT_3:
                return with_alpha || !inputs_have_alpha ? "input.input3" : "input.input3.rgb";
            case SHADER_INPUT_4:
                return with_alpha || !inputs_have_alpha ? "input.input4" : "input.input4.rgb";
            case SHADER_TEXEL0:
                return first_cycle ? (with_alpha ? "texVal0" : "texVal0.rgb")
                                   : (with_alpha ? "texVal1" : "texVal1.rgb");
            case SHADER_TEXEL0A:
                return first_cycle
                           ? (hint_single_element ? "texVal0.a"
                                                  : (with_alpha ? "float4(texVal0.a, texVal0.a, texVal0.a, texVal0.a)"
                                                                : "float3(texVal0.a, texVal0.a, texVal0.a)"))
                           : (hint_single_element ? "texVal1.a"
                                                  : (with_alpha ? "float4(texVal1.a, texVal1.a, texVal1.a, texVal1.a)"
                                                                : "float3(texVal1.a, texVal1.a, texVal1.a)"));
            case SHADER_TEXEL1A:
                return first_cycle
                           ? (hint_single_element ? "texVal1.a"
                                                  : (with_alpha ? "float4(texVal1.a, texVal1.a, texVal1.a, texVal1.a)"
                                                                : "float3(texVal1.a, texVal1.a, texVal1.a)"))
                           : (hint_single_element ? "texVal0.a"
                                                  : (with_alpha ? "float4(texVal0.a, texVal0.a, texVal0.a, texVal0.a)"
                                                                : "float3(texVal0.a, texVal0.a, texVal0.a)"));
            case SHADER_TEXEL1:
                return first_cycle ? (with_alpha ? "texVal1" : "texVal1.rgb")
                                   : (with_alpha ? "texVal0" : "texVal0.rgb");
            case SHADER_COMBINED:
                return with_alpha ? "texel" : "texel.rgb";
            case SHADER_NOISE:
                return with_alpha ? "float4(" RAND_NOISE ", " RAND_NOISE ", " RAND_NOISE ", " RAND_NOISE ")"
                                  : "float3(" RAND_NOISE ", " RAND_NOISE ", " RAND_NOISE ")";
        }
    } else {
        switch (item) {
            default:
            case SHADER_0:
                return "0.0";
            case SHADER_1:
                return "1.0";
            case SHADER_INPUT_1:
                return "input.input1.a";
            case SHADER_INPUT_2:
                return "input.input2.a";
            case SHADER_INPUT_3:
                return "input.input3.a";
            case SHADER_INPUT_4:
                return "input.input4.a";
            case SHADER_TEXEL0:
                return first_cycle ? "texVal0.a" : "texVal1.a";
            case SHADER_TEXEL0A:
                return first_cycle ? "texVal0.a" : "texVal1.a";
            case SHADER_TEXEL1A:
                return first_cycle ? "texVal1.a" : "texVal0.a";
            case SHADER_TEXEL1:
                return first_cycle ? "texVal1.a" : "texVal0.a";
            case SHADER_COMBINED:
                return "texel.a";
            case SHADER_NOISE:
                return RAND_NOISE;
        }
    }
}

bool prism_get_bool(prism::ContextTypes* value) {
    if (std::holds_alternative<int>(*value)) {
        return std::get<int>(*value) == 1;
    }
    return false;
}

#undef RAND_NOISE

prism::ContextTypes* prism_append_formula(prism::ContextTypes* _, prism::ContextTypes* a_arg,
                                          prism::ContextTypes* a_single, prism::ContextTypes* a_mult,
                                          prism::ContextTypes* a_mix, prism::ContextTypes* a_with_alpha,
                                          prism::ContextTypes* a_only_alpha, prism::ContextTypes* a_alpha,
                                          prism::ContextTypes* a_first_cycle) {
    auto c = std::get<prism::MTDArray<int>>(*a_arg);
    bool do_single = prism_get_bool(a_single);
    bool do_multiply = prism_get_bool(a_mult);
    bool do_mix = prism_get_bool(a_mix);
    bool with_alpha = prism_get_bool(a_with_alpha);
    bool only_alpha = prism_get_bool(a_only_alpha);
    bool opt_alpha = prism_get_bool(a_alpha);
    bool first_cycle = prism_get_bool(a_first_cycle);
    std::string out = "";
    if (do_single) {
        out += prism_shader_item_to_str(c.at(only_alpha, 3), with_alpha, only_alpha, opt_alpha, first_cycle, false);
    } else if (do_multiply) {
        out += prism_shader_item_to_str(c.at(only_alpha, 0), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += " * ";
        out += prism_shader_item_to_str(c.at(only_alpha, 2), with_alpha, only_alpha, opt_alpha, first_cycle, true);
    } else if (do_mix) {
        out += "lerp(";
        out += prism_shader_item_to_str(c.at(only_alpha, 1), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += ", ";
        out += prism_shader_item_to_str(c.at(only_alpha, 0), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += ", ";
        out += prism_shader_item_to_str(c.at(only_alpha, 2), with_alpha, only_alpha, opt_alpha, first_cycle, true);
        out += ")";
    } else {
        out += "(";
        out += prism_shader_item_to_str(c.at(only_alpha, 0), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += " - ";
        out += prism_shader_item_to_str(c.at(only_alpha, 1), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += ") * ";
        out += prism_shader_item_to_str(c.at(only_alpha, 2), with_alpha, only_alpha, opt_alpha, first_cycle, true);
        out += " + ";
        out += prism_shader_item_to_str(c.at(only_alpha, 3), with_alpha, only_alpha, opt_alpha, first_cycle, false);
    }
    return new prism::ContextTypes{ out };
}

static size_t raw_numFloats = 0;

prism::ContextTypes* update_raw_floats(prism::ContextTypes* _, prism::ContextTypes* num) {
    raw_numFloats += std::get<int>(*num);
    return nullptr;
}

// SOH [Enhancement] See the note beside the OpenGL one: a colour carried as four normalised bytes takes one
// float's worth of the vertex, and the input layout describes that slot as R8G8B8A8_UNORM.
prism::ContextTypes* update_packed_color(prism::ContextTypes* _) {
    raw_numFloats += 1;
    return nullptr;
}
// PRISM-HELPERS-END

std::optional<std::string> dx_include_fs(const std::string& path) {
    auto init = std::make_shared<Ship::ResourceInitData>();
    init->Type = (uint32_t)Ship::ResourceType::Shader;
    init->ByteOrder = Ship::Endianness::Native;
    init->Format = RESOURCE_FORMAT_BINARY;
    auto res = static_pointer_cast<Ship::Shader>(
        Ship::Context::GetInstance()->GetResourceManager()->LoadResource(path, true, init));
    if (res == nullptr) {
        return std::nullopt;
    }

    auto inc = static_cast<std::string*>(res->GetRawPointer());
    return *inc;
}

std::string gfx_direct3d_common_build_shader(size_t& numFloats, const CCFeatures& cc_features,
                                             bool include_root_signature, bool three_point_filtering, bool use_srgb) {
    raw_numFloats = 4;

    prism::Processor processor;
    prism::ContextItems mContext = {
        { "SHADER_0", SHADER_0 },
        { "SHADER_INPUT_1", SHADER_INPUT_1 },
        { "SHADER_INPUT_2", SHADER_INPUT_2 },
        { "SHADER_INPUT_3", SHADER_INPUT_3 },
        { "SHADER_INPUT_4", SHADER_INPUT_4 },
        { "SHADER_INPUT_5", SHADER_INPUT_5 },
        { "SHADER_INPUT_6", SHADER_INPUT_6 },
        { "SHADER_INPUT_7", SHADER_INPUT_7 },
        { "SHADER_TEXEL0", SHADER_TEXEL0 },
        { "SHADER_TEXEL0A", SHADER_TEXEL0A },
        { "SHADER_TEXEL1", SHADER_TEXEL1 },
        { "SHADER_TEXEL1A", SHADER_TEXEL1A },
        { "SHADER_1", SHADER_1 },
        { "SHADER_COMBINED", SHADER_COMBINED },
        { "SHADER_NOISE", SHADER_NOISE },
        { "o_c", M_ARRAY(cc_features.c, int, 2, 2, 4) },
        { "o_alpha", cc_features.opt_alpha },
        { "o_fog", cc_features.opt_fog },
        { "o_texture_edge", cc_features.opt_texture_edge },
        { "o_noise", cc_features.opt_noise },
        { "o_2cyc", cc_features.opt_2cyc },
        { "o_alpha_threshold", cc_features.opt_alpha_threshold },
        { "o_invisible", cc_features.opt_invisible },
        { "o_grayscale", cc_features.opt_grayscale },
        { "o_toon", cc_features.opt_toon },
        { "o_shadow_map", cc_features.opt_shadow_map }, // SOH [Enhancement] cascaded shadow maps
        { "o_shadow_max_cascades", SHADOW_MAP_MAX_CASCADES },
        // How many cascades the actor layer has; beyond it there is no slice to read.
        { "o_shadow_actor_cascades", SHADOW_MAP_ACTOR_CASCADES },
        // Whether the shadow kernel may fetch a 2x2 footprint per instruction. A property of the adapter,
        // not of the material, so it is the same for every shader in a session -- it selects a kernel, it
        // does not add a variant. See sShadowGather.
        { "o_shadow_gather", sShadowGather },
        // Spliced in rather than uploaded: it is a fixed policy value, and having it as a literal lets the
        // compiler fold the smoothstep that uses it.
        { "o_textures", M_ARRAY(cc_features.usedTextures, bool, 2) },
        { "o_masks", M_ARRAY(cc_features.used_masks, bool, 2) },
        { "o_blend", M_ARRAY(cc_features.used_blend, bool, 2) },
        { "o_clamp", M_ARRAY(cc_features.clamp, bool, 2, 2) },
        { "o_inputs", cc_features.numInputs },
        { "o_do_mix", M_ARRAY(cc_features.do_mix, bool, 2, 2) },
        { "o_do_single", M_ARRAY(cc_features.do_single, bool, 2, 2) },
        { "o_do_multiply", M_ARRAY(cc_features.do_multiply, bool, 2, 2) },
        { "o_color_alpha_same", M_ARRAY(cc_features.color_alpha_same, bool, 2) },
        { "o_root_signature", include_root_signature },
        { "o_three_point_filtering", three_point_filtering },
        { "srgb_mode", use_srgb },
        { "append_formula", (InvokeFunc)prism_append_formula },
        { "update_floats", (InvokeFunc)update_raw_floats },
        { "update_packed_color", (InvokeFunc)update_packed_color },
    };
    processor.populate(mContext);
    auto init = std::make_shared<Ship::ResourceInitData>();
    init->Type = (uint32_t)Ship::ResourceType::Shader;
    init->ByteOrder = Ship::Endianness::Native;
    init->Format = RESOURCE_FORMAT_BINARY;
    auto res = static_pointer_cast<Ship::Shader>(Ship::Context::GetInstance()->GetResourceManager()->LoadResource(
        "shaders/directx/default.shader.hlsl", true, init));

    if (res == nullptr) {
        SPDLOG_ERROR("Failed to load default directx shader, missing f3d.o2r?");
        abort();
    }

    auto shader = static_cast<std::string*>(res->GetRawPointer());
    processor.load(*shader);
    processor.bind_include_loader(dx_include_fs);
    auto result = processor.process();
    numFloats = raw_numFloats;
    // SPDLOG_INFO("=========== DX11 SHADER ============");
    // SPDLOG_INFO(result);
    // SPDLOG_INFO("====================================");
    return result;
}
} // namespace Fast
#endif
