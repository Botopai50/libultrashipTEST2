#pragma once

#if defined(ENABLE_DX11) || defined(ENABLE_DX12)

#ifdef __cplusplus
#include "../interpreter.h"
#include <cstdint>
#include <string>
#include "gfx_rendering_api.h"
#include "d3d11.h"
#include "d3dcompiler.h"
#include <thread>
#include <atomic>
#include <vector>

namespace Fast {

struct PerFrameCB {
    uint32_t noise_frame;
    float noise_scale;
    uint32_t padding[2]; // constant buffers must be multiples of 16 bytes in size
};

// SOH [Enhancement] Toon lighting: the per-object dominant light + frame-global ramp shape. Kept in
// its own constant buffer (register b2) rather than PerFrameCB so that buffer stays purely frame-global;
// this one is re-uploaded per toon draw. The padding keeps each float3 on a 16-byte boundary so the
// layout matches the HLSL cbuffer packing rules. Total size 64 bytes (multiple of 16).
struct PerToonCB {
    float toon_light_dir[3];
    float toon_ramp_center;
    float toon_light_color[3];
    float toon_ramp_softness;
    float toon_ambient[3];
    float toon_highlight_intensity;
    float toon_shadow_intensity;
    float toon_debug;
    float _toon_pad[2];
};

// SOH [Enhancement] Cascaded shadow maps (register b3). Layout must match the PerShadowCB cbuffer in
// default.shader.hlsl field for field. Everything is float4-shaped because HLSL gives each element of a
// scalar array its own 16-byte register -- packing these as float[4] would not match the shader.
struct PerShadowCB {
    float shadow_view_proj[SHADOW_MAP_MAX_CASCADES][16];
    float shadow_splits[4];      // far distance of each cascade, world units
    float shadow_texel_world[4]; // world size of one texel, per cascade
    float shadow_texel_uv[4];    // one texel in UV terms (1/resolution), per cascade
    float shadow_depth_bias[4];  // constant bias in NDC depth, per cascade (world units / depth range)
    // x = active cascade count (0 = no shadow map), y = blend fraction, z = normal offset, w = strength
    float shadow_params[4];
    // x = PCF kernel radius in texels, y = debug mode, z = edge hardness, w = hardness in the far cascade.
    float shadow_filter[4];
    // x = incidence at which a scenery receiver takes the shadow in full, y = incidence at which the edge
    // hardening reaches full strength, z = the floor that hardening tapers to between the two, w unused.
    float shadow_incidence[4];
    // World-space bounds of the ACTOR caster layer, xyz used and w ignored. An empty layer is sent as an
    // inverted box, which every test against it fails -- so "no characters" needs no separate flag.
    float shadow_actor_min[4];
    float shadow_actor_max[4];
    // One texel of the ACTOR layer in UV terms, per cascade. Separate from shadow_texel_uv because that
    // layer can be sized on its own (see shadow_map.h); equal to it when the two resolutions match.
    float shadow_actor_texel_uv[4];
};

struct PerDrawCB {
    struct Texture {
        uint32_t width;
        uint32_t height;
        uint32_t linear_filtering;
        uint32_t padding;
    } mTextures[SHADER_MAX_TEXTURES];
};

struct Coord {
    int x, y;
};

struct TextureData {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> resource_view;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> sampler_state;
    // The same sampler clamped to the top mip level, for screen-space draws (see SetTextureLodClamp). Built
    // alongside the other one and only when this texture has a chain, so a draw switching between world and
    // interface is a bind rather than a state rebuild.
    Microsoft::WRL::ComPtr<ID3D11SamplerState> sampler_state_lod0;
    uint32_t width;
    uint32_t height;
    bool linear_filtering;
    // Whether this texture was given a mip chain on upload. Read when its sampler is built: a chain that is
    // not there must not be selected for, and one that is there wants a different filter and the LOD bias.
    bool has_mips = false;
};

struct FramebufferDX11 {
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> render_target_view;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> depth_stencil_view;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_stencil_srv;
    uint32_t texture_id;
    bool has_depth_buffer;
    uint32_t msaa_level;
};

struct ShaderProgramD3D11 {
    Microsoft::WRL::ComPtr<ID3D11VertexShader> vertex_shader;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> pixel_shader;
    Microsoft::WRL::ComPtr<ID3D11InputLayout> input_layout;
    Microsoft::WRL::ComPtr<ID3D11BlendState> blend_state;

    uint64_t shader_id0;
    uint32_t shader_id1;
    uint8_t numInputs;
    uint8_t numFloats;
    bool usedTextures[SHADER_MAX_TEXTURES];
    bool opt_toon = false;       // SOH [Enhancement] toon lighting variant
    bool opt_shadow_map = false; // SOH [Enhancement] cascaded shadow-map receiver variant
};

class GfxWindowBackendDXGI;

class GfxRenderingAPIDX11 final : public GfxRenderingAPI {
  public:
    GfxRenderingAPIDX11() = default;
    ~GfxRenderingAPIDX11() override;
    GfxRenderingAPIDX11(GfxWindowBackendDXGI* backend);
    const char* GetName() override;
    int GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;
    void UnloadShader(struct ShaderProgram* oldPrg) override;
    void LoadShader(struct ShaderProgram* newPrg) override;
    struct ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint32_t shaderId1) override;
    struct ShaderProgram* LookupShader(uint64_t shaderId0, uint32_t shaderId1) override;
    void ShaderGetInfo(struct ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;
    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) override;
    bool ApplyFxaa(int fbIdDst, int fbIdSrc) override;
    // Builds the FXAA pipeline on first use and answers whether it is usable. Compiled here rather than with
    // the combiner shaders because it shares nothing with them: no prism options, one variant forever.
    bool EnsureFxaaPipeline();
    // Picks between a texture's full and top-level-only samplers for the current draw.
    const Microsoft::WRL::ComPtr<ID3D11SamplerState>& SamplerFor(uint32_t textureId);
    void SetDepthTestAndMask(bool depth_test, bool z_upd) override;
    void SetZmodeDecal(bool decal) override;
    void SetViewport(int x, int y, int width, int height) override;
    void SetScissor(int x, int y, int width, int height) override;
    void SetUseAlpha(bool useAlpha) override;
    void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) override;
    void Init() override;
    void OnResize() override;
    void StartFrame() override;
    void EndFrame() override;
    void FinishRender() override;
    int CreateFramebuffer() override;
    void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                     bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                     bool can_extract_depth) override;
    void StartDrawToFramebuffer(int fbId, float noiseScale) override;
    void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0,
                         int dstX1, int dstY1) override;
    void ClearFramebuffer(bool color, bool depth) override;
    void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) override;
    void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) override;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) override;
    void* GetFramebufferTextureId(int fbId) override;
    void SelectTextureFb(int fbId) override;
    void DeleteTexture(uint32_t texId) override;
    void SetTextureFilter(FilteringMode mode) override;
    FilteringMode GetTextureFilter() override;
    void SetSrgbMode() override;
    ImTextureID GetTextureById(int id) override;

    // SOH [Enhancement] Cascaded shadow maps (see fast/shadow_map.h). This is the only backend that
    // implements the depth pass.
    void PrewarmShaderVariants(uint32_t extraOptionBits) override;
    bool ShaderPrewarmInProgress() override;
    bool SupportsShadowMap() override;
    bool ShadowMapConfigure(int cascadeCount, int resolution, int actorResolution) override;
    bool ShadowMapBeginCascade(int layer, int cascadeIndex, const float lightViewProj[16],
                               uint64_t contentKey) override;
    void ShadowMapDrawCasters(const float* worldXyz, size_t vertexCount, int slot, size_t firstVertex,
                              size_t drawCount) override;
    bool SupportsShadowMapAlphaCasters() override;
    void ShadowMapUploadAlphaCasters(const float* xyzUv, size_t vertexCount) override;
    void ShadowMapDrawAlphaRange(uint32_t textureId, size_t firstVertex, size_t vertexCount) override;
    void ShadowMapEndPass() override;
    void SetShadowMapParams(const float* viewProj, const float* splitDistances, int cascadeCount, float blendFraction,
                            float normalOffset, float strength, float filterWidth, float debugMode,
                            float edgeHardness, float edgeHardnessFar) override;

    // SOH [Enhancement] FXAA post-process pass. Null until first used; mFxaaFailed latches a compile or
    // creation failure so a broken pipeline is attempted once and not once per frame.
    Microsoft::WRL::ComPtr<ID3D11VertexShader> mFxaaVs;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> mFxaaPs;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> mFxaaSampler;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mFxaaCb;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> mFxaaDepthStencilState;
    Microsoft::WRL::ComPtr<ID3D11BlendState> mFxaaBlendState;
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> mFxaaRasterizerState;
    bool mFxaaFailed = false;

    PFN_D3D11_CREATE_DEVICE mDX11CreateDevice;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> mContext;
    Microsoft::WRL::ComPtr<ID3D11Device> mDevice;
    GfxWindowBackendDXGI* mWindowBackend = nullptr;
    D3D_FEATURE_LEVEL mFeatureLevel;

  private:
    void CreateDepthStencilObjects(uint32_t width, uint32_t height, uint32_t msaa_count, ID3D11DepthStencilView** view,
                                   ID3D11ShaderResourceView** srv);

    // SOH [Enhancement] Cascaded shadow maps: build the depth-only pipeline (shader, layout, states,
    // sampler, constant/vertex buffers) once, and the cascade array whenever its shape changes.
    // Both report failure instead of throwing -- a device that cannot make a shadow map must degrade to
    // "no shadow map" rather than take the whole renderer down.
    bool CreateShadowMapPipeline();
    bool CreateShadowMapTargets(int cascadeCount, int resolution, int actorResolution);
    ID3D11RasterizerState* ShadowRasterizerForCascade(int slice, int resolution, const float lightViewProj[16]);
    void ShadowMapInvalidateOpenSlice();
    void ShadowMapBindForReading();

    // SOH [Enhancement] Cascaded shadow maps. The array is one D16 texture with a depth-stencil view per
    // slice (written one cascade at a time) and a single shader resource view over all slices (read by
    // the main pass). Everything stays null until the application first asks for a shadow map.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> mShadowMapTexture;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> mShadowMapDsv[SHADOW_MAP_MAX_SLICES];
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mShadowMapSrv;
    // The actor layer's own array, built only when its resolution differs from the world layer's. While the
    // two match these stay null and both shader slots are fed from mShadowMapTexture, which is bit for bit
    // the arrangement that existed before the layers could be sized apart.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> mShadowActorTexture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mShadowActorSrv;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> mShadowMapSampler;
    Microsoft::WRL::ComPtr<ID3D11VertexShader> mShadowDepthVs;
    Microsoft::WRL::ComPtr<ID3D11InputLayout> mShadowDepthLayout;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mShadowDepthCb;
    // One caster buffer per layer, not one shared buffer. The two layers have very different lifetimes:
    // the actor layer changes every frame, while the world layer is a cached, static room mesh (see
    // Interpreter::mShadowMapWorldCache) that is usually byte-identical frame after frame. With a single
    // buffer the actor upload would clobber the world contents on every frame, forcing the room mesh --
    // the largest list by far -- to be re-uploaded forever. Kept apart, the world buffer is written once
    // per room and then only bound and drawn.
    Microsoft::WRL::ComPtr<ID3D11Buffer> mShadowCasterVb[SHADOW_MAP_LAYERS * SHADOW_MAP_CASTER_SLOTS];
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> mShadowRasterizerState;
    // Per-cascade copies of that state, differing only in the slope-scaled bias. The rasterizer takes one
    // slope value per state rather than per draw, and the slope's effect is measured in texels -- so the
    // far cascades, whose texels are several world units, need a smaller multiplier to stay under the same
    // world-space ceiling. Rebuilt only when a cascade's required value actually changes, which the radius
    // hysteresis makes rare; null means "use the shared state above".
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> mShadowRasterizerCascade[SHADOW_MAP_MAX_SLICES];
    float mShadowRasterizerCascadeSlope[SHADOW_MAP_MAX_SLICES] = {};
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> mShadowDepthStencilState;
    size_t mShadowCasterVbVertices[SHADOW_MAP_LAYERS * SHADOW_MAP_CASTER_SLOTS] = {}; // capacity of each buffer, in vertices
    // What each caster buffer currently holds, so a list that has not changed is neither re-uploaded for
    // the next cascade nor for the next frame. The interpreter guarantees the pointer identity is
    // meaningful: a layer whose contents change gets a fresh push into a cleared vector, and the world
    // cache is only ever swapped wholesale when it is genuinely rebuilt.
    const float* mShadowLastCasterPtr[SHADOW_MAP_LAYERS * SHADOW_MAP_CASTER_SLOTS] = {};
    size_t mShadowLastCasterCount[SHADOW_MAP_LAYERS * SHADOW_MAP_CASTER_SLOTS] = {};
    // SOH [Enhancement] Background shader prewarm. The threads compile into the on-disk cache and touch
    // nothing else -- no device objects, no shader pool -- so the only thing that has to be got right is that
    // they are finished before this object is. Joined in the destructor.
    std::vector<std::thread> mPrewarmThreads;
    std::vector<std::string> mPrewarmSources;
    std::atomic<size_t> mPrewarmNext{ 0 };
    std::atomic<uint32_t> mPrewarmRemaining{ 0 };
    void JoinPrewarm();

    int mShadowCurrentLayer = 0; // layer named by the most recent ShadowMapBeginCascade
    int mShadowCurrentSlice = -1; // slice it opened, so a failed upload can drop that slice's record
    // What each slice was last filled with, so a slice whose casters and matrix are both unchanged is left
    // alone instead of being cleared and redrawn. A depth map is a pure function of those two plus the
    // rasterizer state (the slope bias differs per cascade and follows a user setting), so the raster state
    // object is part of the record -- a changed setting builds a new state, and the pointer moves with it.
    //
    // Every entry is dropped whenever the cascade array is (re)built: the new texture holds nothing, and a
    // record claiming otherwise would leave a slice permanently empty.
    float mShadowSliceMatrix[SHADOW_MAP_MAX_SLICES][16] = {};
    uint64_t mShadowSliceKey[SHADOW_MAP_MAX_SLICES] = {};
    const void* mShadowSliceRasterState[SHADOW_MAP_MAX_SLICES] = {};
    bool mShadowSliceValid[SHADOW_MAP_MAX_SLICES] = {};
    // SOH [Enhancement] Alpha-cutout caster pipeline: a second vertex shader (position + uv), pixel shader
    // (sample and clip) and layout, used for foliage so the depth map records the leaf instead of the quad.
    // Optional -- if any of it fails to build, mShadowAlphaPipelineReady stays false and foliage simply keeps
    // casting its quad, which is worse looking but is still a scene with shadows.
    Microsoft::WRL::ComPtr<ID3D11VertexShader> mShadowAlphaVs;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> mShadowAlphaPs;
    Microsoft::WRL::ComPtr<ID3D11InputLayout> mShadowAlphaLayout;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> mShadowAlphaSampler;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mShadowAlphaVb;
    size_t mShadowAlphaVbVertices = 0;
    const float* mShadowAlphaLastPtr = nullptr;
    size_t mShadowAlphaLastCount = 0;
    bool mShadowAlphaPipelineReady = false; // every alpha object built successfully
    bool mShadowAlphaBound = false;         // the alpha pipeline is the one currently set on the context
    int mShadowCascadeCount = 0;        // 0 until the cascade array exists
    int mShadowResolution = 0;
    // Resolution of the actor layer, and whether it ended up in an array of its own. Split is false whenever
    // the resolutions match AND whenever building the second array failed, so the fallback is the shared
    // array rather than no shadows.
    int mShadowActorResolution = 0;
    bool mShadowActorSplit = false;
    bool mShadowPipelineReady = false;
    bool mShadowPipelineFailed = false; // creation already failed once; do not retry every frame
    bool mShadowPassActive = false;     // between BeginCascade and EndPass
    // Viewport to put back when the depth pass ends: the pass overwrites it with the cascade's square
    // one, and the interpreter does not necessarily re-issue SetViewport before the next draw.
    D3D11_VIEWPORT mShadowSavedViewport = {};
    UINT mShadowSavedViewportCount = 0;

    // SOH [Enhancement] How long the depth pass actually takes on the GPU, in milliseconds.
    //
    // Everything the shadow map costs divides into two halves that are invisible from a frame rate: filling
    // the cascades (this) and sampling them (the receiver shaders). They respond to completely different
    // work, so tuning without knowing which one dominates is guesswork. This measures the first directly,
    // which by subtraction bounds the second.
    //
    // Timestamps rather than a CPU clock, because the CPU only records that it submitted the pass, not that
    // the GPU ran it. Results are collected KFrames later so the query is complete by the time it is read --
    // reading it in the frame that issued it would block the CPU on the GPU and change the thing being
    // measured. Built and issued only while a shadow debug mode is on; off, none of it exists.
    static constexpr int kShadowTimerFrames = 4;
    Microsoft::WRL::ComPtr<ID3D11Query> mShadowTimerDisjoint[kShadowTimerFrames];
    Microsoft::WRL::ComPtr<ID3D11Query> mShadowTimerFrameStart[kShadowTimerFrames];
    Microsoft::WRL::ComPtr<ID3D11Query> mShadowTimerFrameEnd[kShadowTimerFrames];
    Microsoft::WRL::ComPtr<ID3D11Query> mShadowTimerStart[kShadowTimerFrames];
    Microsoft::WRL::ComPtr<ID3D11Query> mShadowTimerEnd[kShadowTimerFrames];
    bool mShadowTimerPending[kShadowTimerFrames] = {};
    // Whether that slot's frame opened a depth pass at all. A frame that reused every slice submits no
    // pass, so its pass timestamps were never issued and must not be read.
    bool mShadowTimerPassIssued[kShadowTimerFrames] = {};
    int mShadowTimerSlot = 0;
    bool mShadowTimerFrameOpen = false; // this frame is being timed
    bool mShadowTimerOpen = false;      // and its depth pass issued a start timestamp
    bool mShadowTimerFailed = false;    // query creation failed once; do not retry every frame
    double mShadowTimerSumMs = 0.0;
    int mShadowTimerSamples = 0;
    double mShadowTimerFrameSumMs = 0.0;
    int mShadowTimerFrameSamples = 0;
    // Slices cleared and redrawn since the last log line. The one number that says whether the cascades are
    // staying parked: eight means every slice is rebuilt every frame, which is what this all exists to stop.
    uint32_t mShadowSlicesDrawn = 0;
    uint32_t mShadowSlicesFrames = 0; // timed frames those slices were spread over
    int mShadowTimerReported = 0; // frames since the last log line
    void ShadowTimerFrameBegin();
    bool ShadowTimerBegin();
    void ShadowTimerEnd();
    void ShadowTimerFrameEnd();
    void ShadowTimerCollect();

    HMODULE mDX11Module;

    HMODULE mCompilerModule;
    pD3DCompile mD3dCompile;

    uint32_t mMsaaNumQualityLevels[D3D11_MAX_MULTISAMPLE_SAMPLE_COUNT];

    Microsoft::WRL::ComPtr<ID3D11RasterizerState> mRasterizerState;
    // SOH [Enhancement] Depth-stencil states, created lazily and cached for the device's lifetime (the
    // stencil features flip the mode many times per frame, and each flip used to re-run
    // CreateDepthStencilState). Key: depthTest | depthMask<<1 | zmodeDecal<<2 | stencilMode<<3 — same
    // scheme as the Metal backend's cache.
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> mDepthStencilStates[64];
    int mLastStencilMode = -1; // SOH [Enhancement] world light casting: cache tracker for mStencilMode
    Microsoft::WRL::ComPtr<ID3D11Buffer> mVertexBuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mPerFrameCb;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mPerDrawCb;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mPerToonCb;   // SOH [Enhancement] toon lighting (register b2)
    Microsoft::WRL::ComPtr<ID3D11Buffer> mPerShadowCb; // SOH [Enhancement] shadow cascades (register b3)
    Microsoft::WRL::ComPtr<ID3D11Buffer> mCoordBuffer;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mCoordBufferSrv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mDepthValueOutputBuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mDepthValueOutputBufferCopy;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> mDepthValueOutputUav;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> mComputeShader;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> mComputeShaderMsaa;
    Microsoft::WRL::ComPtr<ID3DBlob> mComputeShaderMsaaBlob;
    size_t mCoordBufferSize;

#if DEBUG_D3D
    Microsoft::WRL::ComPtr<ID3D11Debug> debug;
#endif

    PerFrameCB mPerFrameCbData;
    PerDrawCB mPerDrawCbData;
    PerToonCB mPerToonCbData;     // SOH [Enhancement] toon lighting
    PerShadowCB mPerShadowCbData; // SOH [Enhancement] cascaded shadow maps
    bool mShadowCbDirty = true;   // re-upload the cascade CB only when the frame's values changed

    std::map<std::pair<uint64_t, uint32_t>, struct ShaderProgramD3D11> mShaderProgramPool;

    std::vector<struct TextureData> mTextures;
    int mCurrentTile;
    uint32_t mCurrentTextureIds[SHADER_MAX_TEXTURES];

    std::vector<FramebufferDX11> mFrameBuffers;

    // Current state

    struct ShaderProgramD3D11* mShaderProgram;

    int32_t mRenderTargetHeight;
    int mCurrentFramebuffer;
    FilteringMode mCurrentFilterMode = FILTER_NONE;

    // Previous states (to prevent setting states needlessly)

    struct ShaderProgramD3D11* mLastShaderProgram = nullptr;
    uint32_t mLastVertexBufferStride = 0;
    Microsoft::WRL::ComPtr<ID3D11BlendState> mLastBlendState = nullptr;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mLastResourceViews[SHADER_MAX_TEXTURES] = { nullptr, nullptr };
    Microsoft::WRL::ComPtr<ID3D11SamplerState> mLastSamplerStates[SHADER_MAX_TEXTURES] = { nullptr, nullptr };

    D3D_PRIMITIVE_TOPOLOGY mLastPrimitaveTopology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
};

std::string gfx_direct3d_common_build_shader(size_t& numFloats, const CCFeatures& cc_features,
                                             bool include_root_signature, bool three_point_filtering, bool use_srgb);
} // namespace Fast
#endif
#endif
