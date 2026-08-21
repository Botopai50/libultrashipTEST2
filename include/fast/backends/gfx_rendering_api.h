#pragma once

#include <stdint.h>

#include <unordered_map>
#include <set>
#include "imconfig.h"
#include "fast/toon_shading.h"
#include "fast/shadow_map.h"

namespace Fast {
struct ShaderProgram;

struct GfxClipParameters {
    bool z_is_from_0_to_1;
    bool invertY;
};

enum FilteringMode { FILTER_THREE_POINT, FILTER_LINEAR, FILTER_NONE };

// SOH [Enhancement] Smallest texture, per side, that is given a mip chain.
//
// The threshold is the whole reason this is close to free on stock assets. A mip chain costs memory, a
// generation pass on every upload, and a second bilinear fetch per sample -- and it repays that only where
// a texture is MINIFIED enough that neighbouring pixels land in far-apart texels and every fetch misses the
// cache. Stock N64 textures cannot get there: TMEM is 4 KB, so nothing native exceeds roughly 64 by 64, and
// a texture that small sits entirely in the GPU's texture cache however it is sampled -- there are no misses
// left to remove. At any modern internal resolution those textures are being magnified anyway, and
// magnification always reads level 0, so the chain would never even be consulted.
//
// What DOES get there is a replacement texture pack, where a 1024-square image stands in for a 32-square
// one. Minified onto a distant wall that is megabytes of traffic per frame, and a mip chain removes nearly
// all of it. 128 sits above everything the stock game can produce and below anything a pack ships.
#define GFX_MIPMAP_MIN_TEXTURE_SIZE 128

// SOH [Enhancement] World light casting / actor shadows: per-draw stencil mode for the Wind Waker-style
// stencil-volume techniques. The interpreter pushes this via SetStencilMode (from a gSPStencil command,
// or directly from RenderShadowVolumes); backends apply the matching stencil state in their per-draw
// path. Off (0) is normal rendering, so ordinary draws are unaffected. These values must match the
// WorldLighting policy module.
enum class StencilMode {
    Off = 0,        // no stencil test/write (normal rendering)
    VolumeIncr = 1, // mask: stencil += 1 where a volume face fails the depth test (z-fail)
    VolumeDecr = 2, // mask: stencil -= 1 where a volume face fails the depth test (z-fail)
    Composite = 3,  // draw where stencil != 0, zeroing it as it goes (self-clearing composite)
    // Single two-sided z-fail pass: one facing increments, the other decrements, both WRAPPING.
    // Wrap + the Composite's nonzero test make the result independent of primitive order AND of which
    // facing gets which op — only "opposite ops per facing" matters — so the volume can be submitted
    // once with culling off instead of as a cull-front/cull-back pass pair.
    VolumeIncrDecr = 4,
};

// A hash function used to hash a: pair<float, float>
struct hash_pair_ff {
    size_t operator()(const std::pair<float, float>& p) const {
        const auto hash1 = std::hash<float>{}(p.first);
        const auto hash2 = std::hash<float>{}(p.second);

        // If hash1 == hash2, their XOR is zero.
        return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
    }
};

class GfxRenderingAPI {
  public:
    virtual ~GfxRenderingAPI() = default;
    virtual const char* GetName() = 0;
    virtual int GetMaxTextureSize() = 0;
    virtual GfxClipParameters GetClipParameters() = 0;
    virtual void UnloadShader(ShaderProgram* oldPrg) = 0;
    virtual void LoadShader(ShaderProgram* newPrg) = 0;
    virtual ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint32_t shaderId1) = 0;
    virtual ShaderProgram* LookupShader(uint64_t shaderId0, uint32_t shaderId1) = 0;
    virtual void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) = 0;
    // SOH [Enhancement] How many floats per vertex this program's vertex layout expects.
    //
    // The interpreter packs a vertex attribute by attribute and the backend reads it back with a stride
    // taken from the shader -- and the two agree only because they were written to agree. Nothing checks
    // them, and the stride is not even a constant: it is accumulated at runtime by the template engine as
    // it walks the shader source (see the update_floats calls in default.shader.*), so the number lives in
    // an asset rather than in code. A mismatch produces a vertex buffer the shader reads at the wrong
    // offsets, which is geometry flying apart, and it produces it silently.
    //
    // Answering this lets the interpreter check the agreement rather than assume it. Zero means "this
    // backend cannot say", and the check is skipped -- the default, so a backend that has no answer needs
    // no change.
    virtual size_t GetVertexStrideFloats(ShaderProgram* prg) {
        (void)prg;
        return 0;
    }
    virtual uint32_t NewTexture() = 0;
    virtual void SelectTexture(int tile, uint32_t textureId) = 0;
    virtual void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) = 0;
    virtual void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) = 0;
    virtual void SetDepthTestAndMask(bool depth_test, bool z_upd) = 0;
    virtual void SetZmodeDecal(bool decal) = 0;
    virtual void SetViewport(int x, int y, int width, int height) = 0;
    virtual void SetScissor(int x, int y, int width, int height) = 0;
    virtual void SetUseAlpha(bool useAlpha) = 0;
    virtual void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) = 0;
    virtual void Init() = 0;
    virtual void OnResize() = 0;
    virtual void StartFrame() = 0;
    virtual void EndFrame() = 0;
    virtual void FinishRender() = 0;
    virtual int CreateFramebuffer() = 0;
    virtual void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                             bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                             bool can_extract_depth) = 0;
    virtual void StartDrawToFramebuffer(int fbId, float noiseScale) = 0;
    virtual void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0,
                                 int dstY0, int dstX1, int dstY1) = 0;
    virtual void ClearFramebuffer(bool color, bool depth) = 0;
    virtual void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) = 0;
    virtual void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) = 0;
    // SOH [Enhancement] Run FXAA over fbIdSrc into fbIdDst. Answers false when the backend has no such pass
    // (every backend but D3D11) or when its pipeline could not be built, and the caller then presents the
    // source unchanged -- so a failure here costs the anti-aliasing and never the frame.
    virtual bool ApplyFxaa(int fbIdDst, int fbIdSrc) {
        return false;
    }
    virtual std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) = 0;
    virtual void* GetFramebufferTextureId(int fbId) = 0;
    virtual void SelectTextureFb(int fbId) = 0;
    virtual void DeleteTexture(uint32_t texId) = 0;
    // SOH [Enhancement] Mip chains for large (replacement-pack) textures. `enabled` only decides whether
    // NEW uploads are given a chain, so a change needs the texture cache cleared to take effect everywhere;
    // the bias and the anisotropy are read when the sampler is built and so apply immediately.
    virtual void SetMipmapParams(bool enabled, float lodBias, int maxAnisotropy) {
        mMipmapEnabled = enabled;
        mMipmapLodBias = lodBias;
        mMipmapAnisotropy = maxAnisotropy < 1 ? 1 : (maxAnisotropy > 16 ? 16 : maxAnisotropy);
    }
    bool MipmapEnabled() const {
        return mMipmapEnabled;
    }

    // SOH [Enhancement] Read this draw's textures at full size, ignoring the mip chain.
    //
    // For the interface. A HUD element is drawn at a fixed size on screen, and if its texture is larger than
    // that -- which with a replacement pack it always is -- the GPU sees minification and picks a smaller
    // level, so the interface would soften as a side effect of a setting meant for the world. The LOD bias
    // makes that worse on purpose: turned up for frames it would blur every button and heart on screen.
    //
    // Clamping to the top level is what the interface had before mipmapping existed, so this is not a
    // correction applied to the interface -- it is the interface being left alone.
    virtual void SetTextureLodClamp(bool clamped) {
        mTextureLodClamped = clamped;
    }

    virtual void SetTextureFilter(FilteringMode mode) = 0;
    virtual FilteringMode GetTextureFilter() = 0;
    virtual void SetSrgbMode() = 0;
    virtual ImTextureID GetTextureById(int id) = 0;

    // SOH [Enhancement] Toon lighting: the interpreter pushes the per-object dominant light here
    // before each batch; backends read the mToon* members in their per-draw uniform paths.
    virtual void SetToonLighting(const float dir[3], const float color[3], const float ambient[3]) {
        for (int i = 0; i < 3; i++) {
            mToonLightDir[i] = dir[i];
            mToonLightColor[i] = color[i];
            mToonAmbient[i] = ambient[i];
        }
    }

    // SOH [Enhancement] Toon lighting: the application pushes the frame-global ramp shape here (the
    // values are app-side tuning, so the framework never reaches into the app's config to read them).
    // Backends read the mToonRamp* members in their per-draw uniform paths; they keep their default
    // (a plain two-tone ramp) until the application overrides them.
    // debug != 0 switches the toon variant to a diagnostic view: each relit object is drawn as flat
    // white on the lit side of the ramp and flat black on the shadow side (albedo discarded), so it is
    // obvious at a glance which draws actually receive toon lighting.
    virtual void SetToonRamp(float center, float softness, float highlight, float shadow, float debug) {
        mToonRampCenter = center;
        mToonRampSoftness = softness;
        mToonHighlightIntensity = highlight;
        mToonShadowIntensity = shadow;
        mToonDebug = debug;
    }

    // SOH [Enhancement] World light casting / actor shadows: the interpreter pushes the current stencil
    // mode here when a gSPStencil command is seen, or directly from RenderShadowVolumes; backends read
    // mStencilMode in their per-draw path. Off (0) is normal rendering, so ordinary draws are unaffected.
    virtual void SetStencilMode(int mode) {
        mStencilMode = mode;
    }

    // SOH [Enhancement] Cascaded shadow maps (see fast/shadow_map.h). Everything below is a no-op by
    // default: only Direct3D 11 renders the depth pass, and a backend that does not override these must
    // behave exactly as it did before. The application MUST consult SupportsShadowMap() before hiding
    // whatever shadows it was drawing, or an unsupported backend ends up with no shadows at all.

    // SOH [Enhancement] Compile the shader variants that turning an option ON will need, ahead of the draws
    // that need them, and in parallel. Shaders are built per material and compiled the first time a draw
    // asks for one, synchronously, inside the frame -- so switching the shadow map on makes every receiver
    // in the scene queue up behind a compiler, one at a time. Given the bit that was added to their ids, a
    // backend can derive the whole set from the variants it already holds and get ahead of it.
    //
    // Best effort by definition: it warms a cache, it does not create anything the renderer then depends on,
    // and a variant it fails or declines to prepare is simply compiled on demand exactly as before.
    virtual void PrewarmShaderVariants(uint32_t extraOptionBits) {
    }

    // Whether a PrewarmShaderVariants call is still working. The application uses it to hold the option OFF
    // until its shaders exist, so that turning one on costs a moment of the feature arriving late instead of
    // a frozen frame per material. Always false where prewarming is synchronous or absent, which keeps the
    // old behaviour exactly.
    virtual bool ShaderPrewarmInProgress() {
        return false;
    }

    // Whether this backend can render the depth pass. Checked once per frame by the application.
    virtual bool SupportsShadowMap() {
        return false;
    }

    // Allocate (or resize) the cascade depth array. Safe to call every frame: implementations
    // reallocate only when the count or resolution actually changes. Returns false if the resources
    // could not be created, in which case the caller must treat shadow maps as unavailable this frame.
    // cascadeCount is clamped to [1, SHADOW_MAP_MAX_CASCADES], both resolutions to the
    // SHADOW_MAP_*_RESOLUTION bounds.
    //
    // actorResolution sizes the actor layer independently of the world layer (see shadow_map.h). Passing the
    // same value for both is the unsplit arrangement and must behave exactly as it did before the parameter
    // existed; an implementation that does not support the split is free to ignore it entirely.
    virtual bool ShadowMapConfigure(int cascadeCount, int resolution, int actorResolution) {
        return false;
    }

    // Begin depth-only rendering into one cascade of one layer (SHADOW_MAP_LAYER_WORLD or _ACTORS -- see
    // fast/shadow_map.h for why casters are split). lightViewProj is row-major, the same convention the
    // interpreter already uses for its own matrices. Clears that slice's depth to far before drawing.
    //
    // `contentKey` identifies the caster geometry the caller is about to submit into this slice. A depth
    // map is a pure function of that geometry and the matrix, so when both match what the slice was last
    // filled with, the slice ALREADY holds the exact image this frame would redraw -- and the answer is
    // false, meaning "leave it alone and skip the draws". The caller must then submit nothing for this
    // slice. True means the slice was cleared and the pipeline set, exactly as before.
    //
    // The saving is the whole slice: at the default 4096 square that is a 32 MB depth clear plus every
    // caster re-rasterised, per cascade, and the world layer's geometry is static for long stretches.
    virtual bool ShadowMapBeginCascade(int layer, int cascadeIndex, const float lightViewProj[16],
                                       uint64_t contentKey) {
        return false; // no depth pass in this backend, so there is nothing to draw into
    }

    // Submit caster geometry into the cascade opened by ShadowMapBeginCascade: `vertexCount` vertices of
    // 3 floats each (world-space xyz), as a plain triangle list. Depth-only -- no textures, no combiner,
    // no lighting -- which is why casters can be fed in raw rather than through the normal draw path.
    // `slot` selects which of the layer's caster lists this is (see SHADOW_MAP_CASTER_SLOTS). Each slot
    // keeps its own buffer, so the world layer's cached room mesh is not disturbed by the per-frame scenery
    // list drawn beside it.
    //
    // `firstVertex`/`drawCount` name a sub-range of that list to draw; drawCount 0 means "to the end", so
    // the two-argument form still draws everything. The list uploaded is always the WHOLE list -- the range
    // selects what is drawn from it, never what is uploaded. That split is the point: a caller can hand over
    // one big list once and then, per cascade, draw only the parts of it that cascade can actually see,
    // without giving up the upload caching that keeps a static room mesh resident on the GPU.
    virtual void ShadowMapDrawCasters(const float* worldXyz, size_t vertexCount, int slot = 0, size_t firstVertex = 0,
                                      size_t drawCount = 0) {
    }

    // Alpha-cutout casters, in two parts because the geometry is uploaded once but drawn once per material.
    // `xyzUv` is `vertexCount` vertices of 5 floats (world xyz + normalised uv), laid out so that every
    // range handed to ShadowMapDrawAlphaRange is contiguous within it. Upload is safe to call per cascade:
    // implementations skip a buffer they already hold, exactly as for the opaque list.
    // Whether the cutout caster pipeline above actually exists. Separate from SupportsShadowMap because it
    // is built lazily and is allowed to fail on its own: the caller must be able to keep cutout geometry on
    // the opaque list when it does, or foliage casts nothing at all instead of casting its quad.
    virtual bool SupportsShadowMapAlphaCasters() {
        return false;
    }

    virtual void ShadowMapUploadAlphaCasters(const float* xyzUv, size_t vertexCount) {
    }

    // Draw one material's slice of the uploaded alpha casters, clipping against that texture's alpha so the
    // depth map records the leaf rather than the quad holding it. `textureId` is a texture the backend
    // already holds from the main pass.
    virtual void ShadowMapDrawAlphaRange(uint32_t textureId, size_t firstVertex, size_t vertexCount) {
    }

    // Close the depth pass and restore the render target the frame was drawing to. After this the
    // cascade array is readable by the main pass.
    virtual void ShadowMapEndPass() {
    }

    // Per-frame values the main pass needs to sample the cascades. `viewProj` holds cascadeCount
    // row-major light matrices back to back; `splitDistances` the far distance of each cascade in world
    // units (used to pick a cascade and to size its blend band). Stored here rather than in each
    // backend so the members are available to every per-draw uniform path, exactly like the toon ones.
    virtual void SetShadowMapParams(const float* viewProj, const float* splitDistances, int cascadeCount,
                                    float blendFraction, float strength, float debugMode) {
        mShadowCascadesActive = cascadeCount < 0 ? 0
                                : cascadeCount > SHADOW_MAP_MAX_CASCADES ? SHADOW_MAP_MAX_CASCADES
                                                                         : cascadeCount;
        if (viewProj != nullptr) {
            for (int i = 0; i < mShadowCascadesActive * 16; i++) {
                mShadowViewProj[i] = viewProj[i];
            }
        }
        if (splitDistances != nullptr) {
            for (int i = 0; i < mShadowCascadesActive; i++) {
                mShadowSplits[i] = splitDistances[i];
            }
        }
        mShadowBlendFraction = blendFraction;
        mShadowStrength = strength;
        mShadowDebug = debugMode;
    }

    // SOH [Enhancement] Turns the shadow map's GPU timing on WITHOUT the cascade-bounds debug view.
    //
    // The two used to be the same switch, which made the pass impossible to measure honestly: the debug
    // value the timer was gated on is the same value the receiver shader reads, and at anything above zero
    // it paints every point outside a cascade's footprint as fully occluded. So the only way to learn what
    // the depth pass costs was to blacken most of the screen -- which changes the very frame being timed,
    // and tells you nothing about what the pass costs while playing normally.
    //
    // Separate flag, no shader involvement. The debug view still enables timing on its own, so nothing that
    // worked before stops working; this only adds a way to get the numbers without the picture.
    void SetShadowMapProfiling(bool enabled) {
        mShadowProfile = enabled;
    }

    // SOH [Enhancement] Which slice of the cascade array to draw as a corner overlay; 0 is off.
    //
    // Separate from the receiver's debug views, and not because it is tidier: those nine ask what the
    // SHADING pixel was given, and this one shows what the depth pass STORED. An artefact can live in
    // either half, and until this existed only one half could be looked at.
    //
    // 1..N are the world layer's cascades in order and the rest are the actor layer's; the backend knows
    // the split, since the actor layer can be shorter. Backends without a depth pass ignore it.
    void SetShadowMapViewSlice(int slice) {
        mShadowViewSlice = slice < 0 ? 0 : slice;
    }

    // Readable so the interpreter can put its own once-a-second shadow census behind the same switch. The
    // backend times the pass; only the interpreter knows what went into it, and the two answer halves of the
    // same question.
    bool ShadowMapProfiling() const {
        return mShadowProfile;
    }

    // SOH [Enhancement] World bounds of the ACTOR caster layer, pushed once a frame. Everything that layer
    // can shadow with lies inside this box, and a receiver with none of it behind them along the light
    // cannot be shadowed by it -- which is most of a map, most of the time, since the box holds a handful of
    // characters. The shader uses it to skip that layer's whole kernel.
    // SOH [Enhancement] The furthest view depth any cascade's footprint can reach. Past it no cascade covers
    // anything, so the receiver shader can answer "lit" without projecting at all -- which is otherwise
    // impossible to know cheaply, since a cascade's box overshoots its own band by a long way and cutting at
    // the split would take real shadows with it.
    virtual void SetShadowMapReach(float maxViewDepth) {
        mShadowMaxViewDepth = maxViewDepth;
    }

    virtual void SetShadowMapActorBounds(const float boundsMin[3], const float boundsMax[3]) {
        for (int i = 0; i < 3; i++) {
            mShadowActorBoundsMin[i] = boundsMin[i];
            mShadowActorBoundsMax[i] = boundsMax[i];
        }
    }

  protected:
    // SOH [Enhancement] Mipmapping (see GFX_MIPMAP_MIN_TEXTURE_SIZE). Off until the application asks, so a
    // host that never pushes these keeps exactly the single-level uploads it had before.
    bool mMipmapEnabled = false;
    bool mTextureLodClamped = false;
    float mMipmapLodBias = 0.0f;
    int mMipmapAnisotropy = 1;
    float mToonLightDir[3] = { 0.0f, 0.0f, 1.0f };
    float mToonLightColor[3] = { 1.0f, 1.0f, 1.0f };
    float mToonAmbient[3] = { 0.0f, 0.0f, 0.0f };
    float mToonRampCenter = TOON_SHADING_DEFAULT_RAMP_CENTER;
    float mToonRampSoftness = TOON_SHADING_DEFAULT_RAMP_SOFTNESS;
    float mToonHighlightIntensity = TOON_SHADING_DEFAULT_HIGHLIGHT;
    float mToonShadowIntensity = TOON_SHADING_DEFAULT_SHADOW;
    float mToonDebug = 0.0f;
    int mStencilMode = 0; // SOH [Enhancement] world light casting / actor shadows (see StencilMode)
    // SOH [Enhancement] Cascaded shadow maps: the frame's cascade transforms and tuning, pushed by
    // SetShadowMapParams. mShadowCascadesActive == 0 means "no shadow map this frame", which is the
    // state every backend that does not implement the depth pass stays in forever.
    float mShadowViewProj[SHADOW_MAP_MAX_CASCADES * 16] = {};
    float mShadowSplits[SHADOW_MAP_MAX_CASCADES] = {};
    int mShadowCascadesActive = 0;
    float mShadowBlendFraction = SHADOW_MAP_DEFAULT_BLEND_FRACTION;
    float mShadowStrength = SHADOW_MAP_DEFAULT_STRENGTH;
    float mShadowDebug = 0.0f;
    // Timing only -- read by the backend's timer, never by a shader. See SetShadowMapProfiling.
    bool mShadowProfile = false;
    // Inverted until a frame says otherwise, so every test against it fails and the actor layer is skipped.
    // Zero until a frame says otherwise, which reads as "no cascade reaches anywhere" and costs nothing.
    float mShadowMaxViewDepth = 0.0f;
    float mShadowActorBoundsMin[3] = { 1e30f, 1e30f, 1e30f };
    float mShadowActorBoundsMax[3] = { -1e30f, -1e30f, -1e30f };
    int mShadowViewSlice = 0;
    int8_t mCurrentDepthTest = 0;
    int8_t mCurrentDepthMask = 0;
    int8_t mCurrentZmodeDecal = 0;
    int8_t mLastDepthTest = -1;
    int8_t mLastDepthMask = -1;
    int8_t mLastZmodeDecal = -1;
    bool mSrgbMode = false;
};
} // namespace Fast
