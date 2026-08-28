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

// SOH [Enhancement] Size of the dynamic vertex ring DrawTriangles appends into, in bytes.
//
// Eight worst-case batches. The ring is written forwards and only renamed (MAP_WRITE_DISCARD) when the next
// flush will not fit, so this size decides how OFTEN that rename happens -- and the rename is the expensive
// part, since it costs a fresh block of the whole buffer out of the driver's pool. Eight puts it at a
// handful of times a frame rather than once per draw, which is what it used to be.
//
// Worst case rather than typical: most draws use about a third of the vertex layout, but the ring must be
// able to take the largest single flush whole.
constexpr uint32_t kVertexRingBytes = 8u * MAX_TRI_BUFFER * VBO_MAX_FLOATS_PER_VERTEX * 3u * sizeof(float);

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
    // x = active cascade count (0 = no shadow map), y = blend fraction, z unused, w = strength
    float shadow_params[4];
    // x = furthest view depth any cascade's footprint reaches, y = debug mode, z and w unused.
    float shadow_range[4];
    // World-space bounds of the ACTOR caster layer, xyz used and w ignored. An empty layer is sent as an
    // inverted box, which every test against it fails -- so "no characters" needs no separate flag.
    float shadow_actor_min[4];
    float shadow_actor_max[4];
    // One texel of the ACTOR layer in UV terms, per cascade. Separate from shadow_texel_uv because that
    // layer can be sized on its own (see shadow_map.h); equal to it when the two resolutions match.
    float shadow_actor_texel_uv[4];
    // SOH [Enhancement] Edge quality (see shadow_map.h). Packing matches the shader's comment exactly:
    //   shadow_edge:   x analytic on/off, y ramp width (texels), z jitter on/off, w jitter taps
    //   shadow_jitter: x jitter radius (texels), y per-frame rotation, z filter mode, w ESM exponent
    float shadow_edge[4];
    float shadow_jitter[4];
    // SOH [Enhancement] Shadow acne (see shadow_map.h). Magnitudes arrive already zeroed when their switch
    // is off, so the shaders multiply rather than branch.
    //   acne0: x enabled, y normal offset (texels), z light offset (world), w depth bias (world)
    //   acne1: x slope-scaled, y slope ceiling, z apply in the ordinary receiver, w unused
    float shadow_acne0[4];
    float shadow_acne1[4];
    // SOH [Enhancement] Edge hardening (see shadow_map.h). x on, y hardness, z threshold, w unused.
    float shadow_harden[4];
    // SOH [Enhancement] Clipmap layout (see shadow_map.h). The light's axes with the camera's coordinate
    // along each in w, then the ladder: base half-extent, level count, resolution. No per-level matrix --
    // levels differ by a power of two and a snapped centre, both computed in the shader.
    //   clip_x: light X axis, w = camera along it
    //   clip_y: light Y axis, w = camera along it
    //   clip_z: light Z axis, w = camera along it
    //   clip_p: x base half-extent, y level count (0 = cascade path), z resolution, w unused
    float shadow_clip_x[4];
    float shadow_clip_y[4];
    float shadow_clip_z[4];
    float shadow_clip_p[4];
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
    size_t GetVertexStrideFloats(struct ShaderProgram* prg) override;
    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) override;
    bool ApplyFxaa(int fbIdDst, int fbIdSrc) override;
    // Builds the FXAA pipeline on first use and answers whether it is usable. Compiled here rather than with
    // the combiner shaders because it shares nothing with them: no prism options, one variant forever.
    bool EnsureFxaaPipeline();
    // SOH [Enhancement] The shadow-map viewer's pipeline and its draw. Both answer false / do nothing when
    // there is no shadow map yet or the pipeline could not be built, so a failure here costs the overlay
    // and never the frame.
    bool EnsureShadowMapViewPipeline();
    void DrawShadowMapView();
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
    int ShadowMapBeginCascadeSplit(int layer, int cascadeIndex, const float lightViewProj[16], uint64_t staticKey,
                                   uint64_t dynamicKey) override;
    void ShadowMapEndStaticCasters() override;
    void SetShadowMapParams(const float* viewProj, const float* splitDistances, int cascadeCount, float blendFraction,
                            float strength, float debugMode) override;

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
    // SOH [Enhancement] Filterable shadow maps (technique 1). Compiling the fullscreen resolve/blur
    // pipeline, allocating the moment array for a mode, and running the two over every slice the depth
    // pass redrew. All three are no-ops in SHADOW_MAP_FILTER_DEPTH.
    bool CreateShadowMomentPipeline();
    bool CreateShadowMomentTargets(int mode, int resolution, int sliceCount);
    void ShadowMomentResolveAndBlur();
    void ShadowMomentRelease();
    // SOH [Enhancement] Static caster cache (technique from the Unity-style cached shadow maps).
    bool CreateShadowStaticTargets(int cascadeCount, int resolution);
    void ShadowStaticRelease();
    void ShadowStaticBlit(int slice);

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
    // SOH [Enhancement] The shadow-map viewer: a corner overlay that draws one slice of the depth array to
    // the screen. Its own tiny pipeline, compiled once on first use and never through prism, exactly like
    // the FXAA pass -- it shares nothing with the combiner shaders and has one variant forever.
    //
    // It exists because every other diagnostic in this renderer looks at the RECEIVER. Nine debug views ask
    // what the shading pixel was given; none of them shows what the depth pass actually stored, and a
    // shadow artefact can live in either half. This is the other half.
    Microsoft::WRL::ComPtr<ID3D11VertexShader> mShadowViewVs;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> mShadowViewPs;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mShadowViewCb;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> mShadowViewSampler;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> mShadowViewDepthStencilState;
    Microsoft::WRL::ComPtr<ID3D11BlendState> mShadowViewBlendState;
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> mShadowViewRasterizerState;
    bool mShadowViewFailed = false;
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
    // The depth-bias clamp each cascade's state was built with. Part of the cache key alongside the slope,
    // because it varies per cascade -- it is the world ceiling expressed in that cascade's depth range.
    float mShadowRasterizerCascadeClamp[SHADOW_MAP_MAX_SLICES] = {};
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

    // SOH [Enhancement] Filterable shadow maps (technique 1 -- see fast/shadow_map.h).
    //
    // A second array holding a quantity whose AVERAGE is meaningful, so the map itself can be blurred --
    // which raw depth cannot be, since the mean of two depths is a surface at neither. Written by a resolve
    // pass that reads the depth slice after the depth pass has closed it, then blurred in place through the
    // scratch below, and sampled by the receiver instead of the depth array.
    //
    // Allocated only when a filterable mode is asked for AND it fits the memory ceiling (see
    // SHADOW_MAP_MOMENT_BUDGET_MB). Null means the mode is unavailable this frame and the receiver stays on
    // depth and PCF, which is a fallback rather than a failure.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> mShadowMomentTexture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mShadowMomentSrv;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> mShadowMomentRtv[SHADOW_MAP_MAX_SLICES];
    // One slice's worth of scratch for the separable blur: the horizontal pass writes here and the vertical
    // pass writes back over the slice. An array of one, not a plain 2D texture, so the same shader reads
    // both with the same declaration and only the slice index differs.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> mShadowBlurTexture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mShadowBlurSrv;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> mShadowBlurRtv;
    // Fullscreen resolve/blur pipeline. The vertex shader takes no buffer and no layout -- it builds a
    // covering triangle from SV_VertexID -- so there is nothing here but the three shader objects, their
    // constant buffer and a sampler.
    Microsoft::WRL::ComPtr<ID3D11VertexShader> mShadowMomentVs;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> mShadowMomentResolvePs;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> mShadowMomentBlurPs;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mShadowMomentCb;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> mShadowMomentSampler;
    // Cull NONE, so the covering triangle's winding cannot matter. Worth its own state rather than
    // borrowing the frame's: the viewport transform flips the sign of a triangle's winding, and
    // reasoning about which way a fullscreen triangle faces is exactly the kind of thing that renders
    // nothing at all and gives no error.
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> mShadowMomentRasterState;
    // Which filter mode the arrays above were built for, so a change to the setting rebuilds them. 0 is
    // SHADOW_MAP_FILTER_DEPTH, which means "not allocated".
    int mShadowMomentMode = 0;
    // Resolution the moment array was built at, which tracks the world layer's.
    int mShadowMomentResolution = 0;
    // Whether the moment pipeline compiled. Separate from the mode, because a compile failure must not be
    // retried every frame.
    bool mShadowMomentPipelineReady = false;
    bool mShadowMomentPipelineFailed = false;
    // Slices the depth pass redrew this frame and which therefore need resolving and blurring again. A
    // parked slice keeps the moments it already holds, exactly as it keeps the depths.
    bool mShadowSliceMomentDirty[SHADOW_MAP_MAX_SLICES] = {};

    // SOH [Enhancement] Static caster cache (see fast/shadow_map.h). A second copy of the WORLD layer's
    // slices holding only the casters that do not move, so a frame where scenery moved can blit that in and
    // redraw the movers alone instead of the room mesh as well.
    //
    // World layer only. The actor layer is characters, whose content changes every frame by definition --
    // there is no static half of it to cache.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> mShadowStaticTexture;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> mShadowStaticDsv[SHADOW_MAP_MAX_SLICES];
    // What each static slice was drawn from, and with. Both have to match for the copy to be legitimate:
    // the same casters projected through a different matrix is a different image.
    uint64_t mShadowStaticKey[SHADOW_MAP_MAX_SLICES] = {};
    float mShadowStaticMatrix[SHADOW_MAP_MAX_SLICES][16] = {};
    bool mShadowStaticValid[SHADOW_MAP_MAX_SLICES] = {};
    // The slice a split pass is filling, while its static half is being drawn. -1 once the caller has said
    // the static casters are done, after which the writes are dynamic and must not reach the cache.
    int mShadowStaticOpenSlice = -1;
    // Set around the shared slice-opening body so it targets the STATIC copy, and so it leaves the live
    // slice's contents alone when the static half is about to be blitted over it. Both are cleared again
    // immediately; nothing outside the split path ever sees them set.
    int mShadowStaticTargetSlice = -1;
    bool mShadowSkipSliceClear = false;
    bool mShadowStaticReady = false;
    int mShadowStaticResolution = 0;
    int mShadowStaticSlices = 0;

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
    // SOH [Enhancement] WHY each of those slices had to be redrawn, which the count alone cannot say and
    // which decides what is worth fixing. Two mutually exclusive causes, split on the question that
    // actually separates the available remedies:
    //
    //   mShadowRedrawContent - the cascade had not moved at all; only what goes IN it changed. These are
    //                          the redraws a narrower reuse key could avoid, since the key is currently
    //                          computed per layer and mixes in per-frame scenery from anywhere in the map.
    //   mShadowRedrawMatrix  - the cascade itself moved, because the camera walked out of it or the sun
    //                          swung past the light-direction hysteresis. No reuse key can help these; the
    //                          remedy would be parking margins.
    //
    // Plus the ones that were simply never filled. The three sum to mShadowSlicesDrawn.
    uint32_t mShadowRedrawContent = 0;
    // SOH [Enhancement] ...and split by layer, because the two have very different floors. The ACTOR layer
    // holds animating characters, so its slices genuinely change every frame and nothing can reuse them --
    // it is the irreducible part. Only the WORLD figure is the one with room left in it, and without the
    // split the two are added together and the floor is invisible.
    uint32_t mShadowRedrawContentWorld = 0;
    uint32_t mShadowRedrawContentActors = 0;
    uint32_t mShadowRedrawMatrix = 0;
    uint32_t mShadowRedrawFirst = 0;
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
    // SOH [Enhancement] toon lighting. What the buffer currently HOLDS, not a scratch area: the draw path
    // compares against it to decide whether the upload is worth doing at all. Value-initialised, including
    // the named padding, so that comparison never reads an indeterminate byte.
    PerToonCB mPerToonCbData{};
    bool mPerToonCbValid = false; // false until something has actually been uploaded to compare against
    PerShadowCB mPerShadowCbData; // SOH [Enhancement] cascaded shadow maps
    // SOH [Enhancement] Frames since the backend started, used only to advance the jitter pattern's
    // rotation when temporal jitter is on. Incremented where the shadow constants are written, which
    // happens exactly once per frame.
    uint32_t mShadowQualityFrame = 0;
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
    uint32_t mLastVertexBufferOffset = UINT32_MAX; // no offset bound yet
    // Where the next flush is written in the vertex ring, in bytes. See DrawTriangles: the buffer is filled
    // forwards with MAP_WRITE_NO_OVERWRITE and only renamed with MAP_WRITE_DISCARD when it runs out.
    uint32_t mVertexRingOffset = 0;
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
