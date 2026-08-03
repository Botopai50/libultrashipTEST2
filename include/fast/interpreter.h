#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <unordered_map>
#include <map>
#include <list>
#include <cstddef>
#include <vector>
#include <stack>
#include <string>

#include "fast/lus_gbi.h"
#include "fast/types.h"
#include "fast/ucodehandlers.h"
#include "backends/gfx_rendering_api.h"

#include "fast/resource/type/Texture.h"
#include "ship/resource/Resource.h"

// TODO figure out why changing these to 640x480 makes the game only render in a quarter of the window
#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 240

// SOH [Enhancement] Max floats packed per vertex into the Fast3D VBO, sized for the largest layout
// (including toon lighting's world-space normal attribute and the shadow-map receiver's world position).
// The interpreter packs from this and every backend sizes its vertex buffers from it, so they stay in
// lockstep — change it in one place only.
//
// Raised 40 -> 44 for the shadow-map world position (3 floats) plus the receiver kind packed beside it. The headroom is deliberate rather than
// tight: summing every optional attribute at its maximum (4 position + 4 per texcoord with both clamps +
// 4 fog + 4 grayscale + 3 normal + 3 world position + 4 per input up to seven inputs) already exceeds
// this, so the ceiling is set by which combinations actually co-occur -- something not provable from the
// combiner alone. Over-allocating a few floats per vertex costs a little memory; under-allocating
// overruns the buffer.
#define VBO_MAX_FLOATS_PER_VERTEX 44

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <compare>
#endif

/*enum {
    CC_0,
    CC_TEXEL0,
    CC_TEXEL1,
    CC_PRIM,
    CC_SHADE,
    CC_ENV,
    CC_TEXEL0A,
    CC_LOD
};*/

enum {
    SHADER_0,
    SHADER_INPUT_1,
    SHADER_INPUT_2,
    SHADER_INPUT_3,
    SHADER_INPUT_4,
    SHADER_INPUT_5,
    SHADER_INPUT_6,
    SHADER_INPUT_7,
    SHADER_TEXEL0,
    SHADER_TEXEL0A,
    SHADER_TEXEL1,
    SHADER_TEXEL1A,
    SHADER_1,
    SHADER_COMBINED,
    SHADER_NOISE
};

#ifdef __cplusplus
enum class ShaderOpts {
    ALPHA,
    FOG,
    TEXTURE_EDGE,
    NOISE,
    _2CYC,
    ALPHA_THRESHOLD,
    INVISIBLE,
    GRAYSCALE,
    TEXEL0_CLAMP_S,
    TEXEL0_CLAMP_T,
    TEXEL1_CLAMP_S,
    TEXEL1_CLAMP_T,
    TEXEL0_MASK,
    TEXEL1_MASK,
    TEXEL0_BLEND,
    TEXEL1_BLEND,
    USE_SHADER,
    TOON,       // SOH [Enhancement] toon-lighting variant. Bit 17.
    SHADOW_MAP, // SOH [Enhancement] cascaded shadow-map receiver variant. Bit 18; the loaded-shader id
                // packs ABOVE it (interpreter.cpp shifts shader.id by 19). Adding an opt here without
                // bumping that shift would overlap the id and corrupt shader selection for every draw.
                // shader_id1 is 32 bits, so the id keeps the 13 bits from 19 up -- far more than the
                // handful of loaded shaders that exist, but the ceiling to watch as opts grow.
                //
                // Whether a receiver also takes the ACTOR caster layer deliberately does NOT live here.
                // It rides in the world-position attribute's w instead, because every option bit multiplies
                // the number of shader variants, and each new variant is a shader compiled in the middle of
                // a frame -- which is felt as the game hitching the first time a shadow appears.
    MAX
};

#define SHADER_OPT(opt) ((uint64_t)(1 << static_cast<int>(ShaderOpts::opt)))
#endif

struct ColorCombinerKey {
    uint64_t combine_mode;
    uint64_t options;

#ifdef __cplusplus
    auto operator<=>(const ColorCombinerKey&) const = default;
#endif
};

#define SHADER_MAX_TEXTURES 6
#define SHADER_FIRST_TEXTURE 0
#define SHADER_FIRST_MASK_TEXTURE 2
#define SHADER_FIRST_REPLACEMENT_TEXTURE 4

struct CCFeatures {
    int c[2][2][4];
    bool opt_alpha;
    bool opt_fog;
    bool opt_texture_edge;
    bool opt_noise;
    bool opt_2cyc;
    bool opt_alpha_threshold;
    bool opt_invisible;
    bool opt_grayscale;
    bool opt_toon;       // SOH [Enhancement] toon lighting
    bool opt_shadow_map; // SOH [Enhancement] cascaded shadow maps: this draw receives shadow
    bool usedTextures[2];
    bool used_masks[2];
    bool used_blend[2];
    bool clamp[2][2];
    int numInputs;
    bool do_single[2][2];
    bool do_multiply[2][2];
    bool do_mix[2][2];
    bool color_alpha_same[2];
    int16_t shader_id;
};

void gfx_cc_get_features(uint64_t shader_id0, uint32_t shader_id1, struct CCFeatures* cc_features);

union Gfx;

namespace Fast {

class GfxRenderingAPI;
class GfxWindowBackend;

constexpr size_t MAX_SEGMENT_POINTERS = 16;

struct GfxExecStack {
    // This is a dlist stack used to handle dlist calls.
    std::stack<F3DGfx*> cmd_stack = {};
    // This is also a dlist stack but a std::vector is used to make it possible
    // to iterate on the elements.
    // The purpose of this is to identify an instruction at a poin in time
    // which would not be possible with just a F3DGfx* because a dlist can be called multiple times
    // what we do instead is store the call path that leads to the instruction (including branches)
    std::vector<const F3DGfx*> gfx_path = {};
    struct CodeDisp {
        const char* file;
        int line;
    };
    // stack for OpenDisp/CloseDisps
    std::vector<CodeDisp> disp_stack{};

    void start(F3DGfx* dlist);
    void stop();
    F3DGfx*& currCmd();
    void openDisp(const char* file, int line);
    void closeDisp();
    const std::vector<CodeDisp>& getDisp() const;
    void branch(F3DGfx* caller);
    void call(F3DGfx* caller, F3DGfx* callee);
    F3DGfx* ret();
};

struct XYWidthHeight {
    int16_t x, y;
    uint32_t width, height;
};

struct GfxDimensions {
    float internal_mul;
    uint32_t width, height;
    float aspect_ratio;
};

struct TextureCacheKey {
    const uint8_t* texture_addr;
    const uint8_t* palette_addrs[2];
    uint8_t fmt, siz;
    uint8_t palette_index;
    uint32_t size_bytes;

    bool operator==(const TextureCacheKey&) const noexcept = default;

    struct Hasher {
        size_t operator()(const TextureCacheKey& key) const noexcept {
            uintptr_t addr = (uintptr_t)key.texture_addr;
            return (size_t)(addr ^ (addr >> 5));
        }
    };
};

typedef std::unordered_map<TextureCacheKey, struct TextureCacheValue, TextureCacheKey::Hasher> TextureCacheMap;
typedef std::pair<const TextureCacheKey, struct TextureCacheValue> TextureCacheNode;

struct TextureCacheValue {
    uint32_t texture_id;
    uint8_t cms, cmt;
    bool linear_filter;

    std::list<struct TextureCacheMapIter>::iterator lru_location;
};

struct TextureCacheMapIter {
    TextureCacheMap::iterator it;
};

struct RGBA {
    uint8_t r, g, b, a;
};

struct LoadedVertex {
    float x, y, z, w;
    float u, v;
    struct RGBA color;
    uint8_t clip_rej;
    // SOH [Enhancement] World-space vertex normal, forwarded to the fragment shader for toon lighting.
    float nx, ny, nz;
    // SOH [Enhancement] World-space vertex position (object x modelview, camera lives in the projection
    // matrix). Captured for the actor-shadow pass so geometry can be flattened onto the ground plane.
    float wx, wy, wz;
};

struct RawTexMetadata {
    uint16_t width, height;
    float h_byte_scale = 1, v_pixel_scale = 1;
    std::shared_ptr<Fast::Texture> resource;
    Fast::TextureType type;
};

struct ShaderMod {
    bool enabled = false;
    int16_t id;
    uint8_t type;
};

#define MAX_LIGHTS 32
#define MAX_VERTICES 64

struct RSP {
    float modelview_matrix_stack[11][4][4];
    uint8_t modelview_matrix_stack_size;

    float MP_matrix[4][4];
    float P_matrix[4][4];

    F3DLight_t lookat[2];
    F3DLight current_lights[MAX_LIGHTS + 1];
    float current_lights_coeffs[MAX_LIGHTS][3];
    float current_lookat_coeffs[2][3]; // lookat_x, lookat_y
    uint8_t current_num_lights;        // includes ambient light
    bool lights_changed;

    // SOH [Enhancement] Toon lighting: the single dominant light chosen for the current object,
    // recomputed when lights change. World-space direction, light color, and ambient color (0..1).
    float toon_light_dir[3];
    float toon_light_color[3];
    float toon_ambient[3];

    // SOH [Enhancement] Toon lighting: a per-object key light supplied by the game (gSPToonKey),
    // world-space direction + color. When valid it overrides the renderer's own light averaging so
    // the game can drive a Wind Waker-style sun/torch key with smooth day-night animation.
    bool toon_key_valid;
    float toon_key_dir[3];
    float toon_key_color[3];

    // SOH [Enhancement] Actor shadow: per-object state supplied by gSPToonShadow, including the eased key
    // direction snapshotted when the object armed. The snapshot — not the live toon_key_dir, which the
    // next object overwrites before this object's deferred shadow flush — keeps the drop shadow on the
    // same light the cel shading uses.
    float toon_shadow_size;   // eased 0..1 drop-shadow size for this object (carried in the arm command's w1)
    float toon_shadow_dir[3]; // key direction captured at arm time, used by the deferred shadow flush
    // Optional per-object feet clamp: when armed, raise the shadow slab's feet UP to this world Y so a model
    // whose geometry dips below the floor (a signpost's buried post) doesn't sink the slab below ground. The
    // flag is false when the object passed TOON_SHADOW_NO_CLAMP (the default — leave the feet at the geometry).
    float toon_shadow_feet_clamp_y;
    bool toon_shadow_clamp_feet;

    uint32_t geometry_mode;
    int16_t fog_mul, fog_offset;

    uint32_t extra_geometry_mode;

    struct {
        // U0.16
        uint16_t s, t;
    } texture_scaling_factor;

    struct LoadedVertex loaded_vertices[MAX_VERTICES + 4];
    ShaderMod current_shader;
};

struct RDP {
    const uint8_t* palettes[2];
    struct {
        const uint8_t* addr;
        uint8_t siz;
        uint32_t width;
        uint32_t tex_flags;
        struct RawTexMetadata raw_tex_metadata;
    } texture_to_load;
    struct {
        const uint8_t* addr;
        uint32_t orig_size_bytes;
        uint32_t size_bytes;
        uint32_t full_image_line_size_bytes;
        uint32_t line_size_bytes;
        uint32_t tex_flags;
        struct RawTexMetadata raw_tex_metadata;
        bool masked;
        bool blended;
    } loaded_texture[2];
    struct {
        uint8_t fmt;
        uint8_t siz;
        uint8_t cms, cmt;
        uint8_t shifts, shiftt;
        float uls, ult, lrs, lrt;
        uint16_t tmem; // 0-511, in 64-bit word units
        uint32_t line_size_bytes;
        uint8_t palette;
        uint8_t tmem_index; // 0 or 1 for offset 0 kB or offset 2 kB, respectively
    } texture_tile[8];
    bool textures_changed[2];

    uint8_t first_tile_index;

    uint32_t other_mode_l, other_mode_h;
    uint64_t combine_mode;
    bool grayscale;
    bool toon;        // SOH [Enhancement] toon lighting active for the current draw (set by gSPToon)
    bool toon_shadow; // SOH [Enhancement] actor shadow armed for the current object (set by gSPToonShadow)
    // SOH [Enhancement] Cascaded shadow maps: the current draws are world geometry, so capture them as
    // casters. Deliberately separate from toon_shadow, which also means "do not receive": world geometry
    // must cast AND receive, since scenery shadowing scenery is the whole point.
    bool shadow_world_caster;
    // SOH [Enhancement] Cascaded shadow maps: the current draws must not be shadowed at all (sky, sun,
    // moon). Receiving is otherwise implicit for anything with a world position.
    bool shadow_no_receive;
    // SOH [Enhancement] Cascaded shadow maps: the current draws belong to an actor that is really scenery --
    // a tree. It is set alongside shadow_world_caster, not instead of it, and carries the one thing that
    // flag does not: inside this bracket translucent geometry is taken as a CUTOUT rather than excluded,
    // because a canopy declares itself blended in order to fade and the alpha compare is often off entirely.
    // Nothing infers that from render state; the game names the actors it holds for. Kept separate from
    // shadow_world_caster so the room's own bracket keeps excluding its water.
    bool shadow_scenery_caster;
    // SOH [Enhancement] Cascaded shadow maps: capture is FORBIDDEN for these draws, whatever else is armed.
    // Every other flag here grants capture and lets the renderer judge the render state; this one overrules
    // them, for geometry whose exclusion is not a render-state question -- effects, which are light drawn as
    // polygons and declare whatever mode suited the artist.
    bool shadow_no_cast;
    ShaderMod current_shader;

    uint8_t prim_lod_fraction;
    struct RGBA env_color, prim_color, fog_color, fill_color, grayscale_color;
    struct XYWidthHeight viewport, scissor;
    bool viewport_or_scissor_changed;
    void* z_buf_address;
    void* color_image_address;
};

typedef enum Attribute {
    MTX_PROJECTION,
    MTX_LOAD,
    MTX_PUSH,
    MTX_NOPUSH,
    CULL_FRONT,
    CULL_BACK,
    CULL_BOTH,
    MV_VIEWPORT,
    MV_LIGHT,
} Attribute;

extern GfxExecStack g_exec_stack;

struct GfxTextureCache {
    TextureCacheMap map;
    std::list<TextureCacheMapIter> lru;
    std::vector<uint32_t> free_texture_ids;
};

struct ColorCombiner {
    uint64_t shader_id0;
    uint32_t shader_id1;
    bool usedTextures[2];
    struct ShaderProgram* prg[16];
    uint8_t shader_input_mapping[2][7];
};

struct RenderingState {
    uint8_t depth_test_and_mask; // 1: depth test, 2: depth mask
    bool decal_mode;
    bool alpha_blend;
    struct XYWidthHeight viewport, scissor;
    struct ShaderProgram* mShaderProgram;
    TextureCacheNode* mTextures[SHADER_MAX_TEXTURES];
};

struct FBInfo {
    uint32_t orig_width, orig_height;       // Original shape
    uint32_t applied_width, applied_height; // Up-scaled for the viewport
    uint32_t native_width, native_height;   // Max "native" size of the screen, used for up-scaling
    bool resize;                            // Scale to match the viewport
};

struct MaskedTextureEntry {
    uint8_t* mask;
    uint8_t* replacementData;
};

class Interpreter {
  public:
    Interpreter();
    ~Interpreter();

    void Init(GfxWindowBackend* wapi, class GfxRenderingAPI* rapi, const char* game_name, bool start_in_fullscreen,
              uint32_t width, uint32_t height, uint32_t posX, uint32_t posY);
    void Destroy();
    void GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY);
    GfxRenderingAPI* GetCurrentRenderingAPI();
    // SOH [Enhancement] Actor shadow: global look tuning pushed once per frame by the game. alpha is the
    // core blend strength; minElevation is the floor the key's elevation is remapped into (higher = the
    // light is forced steeper = shorter shadows); slabDepth/slabRise bound the shadow slab below/above
    // the feet; edgeSoftness is the number of penumbra rings (0 = hard edge, max 2); showVolume draws the
    // volumes translucently for debugging.
    void SetToonShadowParams(float alpha, float minElevation, float slabDepth, float slabRise, int edgeSoftness,
                             bool showVolume) {
        mToonShadowAlpha = alpha;
        mToonShadowMinElevation = minElevation;
        mShadowSlabDepth = slabDepth;
        mShadowSlabRise = slabRise;
        mShadowEdgeSoftness = edgeSoftness;
        mShadowShowVolume = showVolume;
    }
    // SOH [Enhancement] Cascaded shadow maps: frame-global policy pushed by the game. `enabled` must
    // already account for the backend's capability -- the interpreter does not second-guess it, it just
    // stops capturing and stops rendering the pass when this is false. lightDir is the world-space
    // direction the light travels (from the sky toward the ground), the same key the cel shading picks.
    void SetShadowMapParams(bool enabled, int cascadeCount, int resolution, const float splits[4],
                            const float lightDir[3], float blendFraction, float normalOffset, float strength,
                            float filterWidth, float minCasterSize, float debugMode,
                            float edgeHardness, float edgeHardnessFar, float minIncidence, float fullIncidence,
                            float minHardnessScale, float depthBiasWorld, float slopeBias) {
        mShadowMapEnabled = enabled;
        mShadowMapCascadeCount = cascadeCount < 1                       ? 1
                                 : cascadeCount > SHADOW_MAP_MAX_CASCADES ? SHADOW_MAP_MAX_CASCADES
                                                                          : cascadeCount;
        mShadowMapResolution = resolution;
        if (splits != nullptr) {
            for (int i = 0; i < SHADOW_MAP_MAX_CASCADES; i++) {
                mShadowMapSplits[i] = splits[i];
            }
        }
        if (lightDir != nullptr) {
            for (int i = 0; i < 3; i++) {
                mShadowMapLightDir[i] = lightDir[i];
            }
        }
        mShadowMapBlendFraction = blendFraction;
        mShadowMapNormalOffset = normalOffset;
        mShadowMapStrength = strength;
        mShadowMapFilterWidth = filterWidth;
        mShadowMapMinCasterSize = minCasterSize;
        mShadowMapDebug = debugMode;
        mShadowMapEdgeHardness = edgeHardness;
        mShadowMapEdgeHardnessFar = edgeHardnessFar;
        // Straight through to the backend rather than held here. Nothing in the cascade fit reads them --
        // they only ever reach the receiver's shader -- so there is no reason for the interpreter to carry
        // a copy that could drift out of step with the one the constant buffer holds.
        if (mRapi != nullptr) {
            mRapi->SetShadowMapIncidence(minIncidence, fullIncidence, minHardnessScale);
            mRapi->SetShadowMapBias(depthBiasWorld, slopeBias);
        }
        if (!enabled) {
            // Drop both buffers so turning the mode off cannot leave a stale frame of casters that would
            // reappear the moment it is turned back on.
            for (int l = 0; l < SHADOW_MAP_LAYERS; l++) {
                mShadowMapCasters[l].clear();
                mShadowMapCastersReady[l].clear();
            }
            // Same reasoning for the cached world layer, plus: force a rebuild on the next enable, since
            // nothing accumulated a signature while the mode was off.
            mShadowMapWorldCache.clear();
            for (int l = 0; l < SHADOW_MAP_LAYERS; l++) {
                mShadowAlphaCasters[l].clear();
                mShadowAlphaReady[l].clear();
            }
            mShadowAlphaWorldCache.clear();
            mShadowSceneryCasters.clear();
            mShadowSceneryReady.clear();
            mShadowAlphaScenery.clear();
            mShadowAlphaSceneryReady.clear();
            mShadowWorldKeyAccum = 0;
            mShadowWorldKeyCached = 0;
            mShadowWorldCapture = true;
        }
    }
    void StartFrame();
    void RunGuiOnly();
    void Run(Gfx* commands, const std::unordered_map<Mtx*, MtxF>& mtx_replacements);
    void EndFrame();
    void HandleWindowEvents();
    bool IsFrameReady();
    bool ViewportMatchesRendererResolution();
    int GetTargetFps();
    void SetTargetFps(int fps);
    void SetMaxFrameLatency(int latency);
    int CreateFrameBuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                          uint8_t resize);
    void SetFrameBuffer(int fb, float noiseScale);
    void CopyFrameBuffer(int fb_dst_id, int fb_src_id, bool copyOnce, bool* hasCopiedPtr);
    void ResetFrameBuffer();
    void AdjustPixelDepthCoordinates(float& x, float& y);
    void GetPixelDepthPrepare(float x, float y);
    uint16_t GetPixelDepth(float x, float y);
    void RegisterBlendedTexture(const char* name, uint8_t* mask, uint8_t* replacement);
    void UnregisterBlendedTexture(const char* name);

    void SetNativeDimensions(float width, float height);
    void SetResolutionMultiplier(float multiplier);
    void SetMsaaLevel(uint32_t level);
    void GetCurDimensions(uint32_t* width, uint32_t* height);

    // private: TODO make these private
    void Flush();
    ShaderProgram* LookupOrCreateShaderProgram(uint64_t id0, uint64_t id1);
    ColorCombiner* LookupOrCreateColorCombiner(const ColorCombinerKey& key);
    void TextureCacheClear();
    bool TextureCacheLookup(int i, const TextureCacheKey& key);
    void TextureCacheDelete(const uint8_t* origAddr);
    void ImportTextureRgba16(int tile, bool importReplacement);
    void ImportTextureRgba32(int tile, bool importReplacement);
    void ImportTextureIA4(int tile, bool importReplacement);
    void ImportTextureIA8(int tile, bool importReplacement);
    void ImportTextureIA16(int tile, bool importReplacement);
    void ImportTextureI4(int tile, bool importReplacement);
    void ImportTextureI8(int tile, bool importReplacement);
    void ImportTextureCi4(int tile, bool importReplacement);
    void ImportTextureCi8(int tile, bool importReplacement);
    void ImportTextureRaw(int tile, bool importReplacement);
    void ImportTextureImg(int tile, bool importReplacement);
    void ImportTexture(int i, int tile, bool importReplacement);
    void ImportTextureMask(int i, int tile);
    void CalculateNormalDir(const F3DLight_t*, float coeffs[3]);
    // SOH [Enhancement] Toon lighting: pick the single dominant light for the current object and
    // cache its world-space direction / color / ambient in the RSP for the fragment shader.
    void SelectToonLight();

    // SOH [Enhancement] Actor shadow: build the current object's shadow-slab volume from its captured
    // world-space triangles (projected along the toon key direction) and accumulate it for the frame —
    // nothing draws here. Called at each per-object boundary; RenderShadowVolumes draws the batch.
    void FlushToonShadow();
    // SOH [Enhancement] Cascaded shadow maps: render last frame's casters into the cascade array and hand
    // the transforms to the backend. Called at the same pre-actor point as RenderShadowVolumes().
    void RenderShadowMap();
    // SOH [Enhancement] Actor shadow: draw all volumes accumulated this frame (batched z-fail stencil +
    // composite), then clear them. Called once per frame at the pre-actor hook so shadows fall only on the
    // environment (no self-shadow / no shadowing other actors).
    void RenderShadowVolumes();

    void GfxSpMatrix(uint8_t params, const int32_t* addr);
    void GfxSpPopMatrix(uint32_t count);
    void GfxSpVertex(size_t numVertices, size_t destIndex, const F3DVtx* vertices);
    void GfxSpModifyVertex(uint16_t vtxIdx, uint8_t where, uint32_t val);
    void GfxSpTri1(uint8_t vtx1Idx, uint8_t vtx2Idx, uint8_t vtx3Idx, bool isRect);
    void GfxSpGeometryMode(uint32_t clear, uint32_t set);
    void GfxSpExtraGeometryMode(uint32_t clear, uint32_t set);
    void GfxSpMovememF3dex2(uint8_t index, uint8_t offset, const void* data);
    void GfxSpMovememF3d(uint8_t index, uint8_t offset, const void* data);
    void GfxSpMovewordF3dex2(uint8_t index, uint16_t offset, uintptr_t data);
    void GfxSpMovewordF3d(uint8_t index, uint16_t offset, uintptr_t data);
    void GfxSpTexture(uint16_t sc, uint16_t tc, uint8_t level, uint8_t tile, uint8_t on);
    void GfxDpSetScissor(uint32_t mode, uint32_t ulx, uint32_t uly, uint32_t lrx, uint32_t lry);
    void GfxDpSetTextureImage(uint32_t format, uint32_t size, uint32_t width, const char* texPath, uint32_t texFlags,
                              RawTexMetadata rawTexMetdata, const void* addr);
    void GfxDpSetTile(uint8_t fmt, uint32_t siz, uint32_t line, uint32_t tmem, uint8_t tile, uint32_t palette,
                      uint32_t cmt, uint32_t maskt, uint32_t shiftt, uint32_t cms, uint32_t masks, uint32_t shifts);
    void GfxDpSetTileSize(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt);
    void GfxDpLoadTlut(uint8_t tile, uint32_t high_index);
    void GfxDpLoadBlock(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t dxt);
    void GfxDpLoadTile(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t lrt);
    void GfxDpSetCombineMode(uint32_t rgb, uint32_t alpha, uint32_t rgb_cyc2, uint32_t alpha_cyc2);
    void GfxDpSetGrayscaleColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void GfxDpSetEnvColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void GfxDpSetPrimColor(uint8_t m, uint8_t r, uint8_t l, uint8_t g, uint8_t b, uint8_t a);
    void GfxDpSetFogColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void GfxDpSetBlendColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void GfxDpSetFillColor(uint32_t pickedColor);
    void GfxDrawRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry);
    void GfxDpTextureRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, uint8_t tile, int16_t uls,
                               int16_t ult, int16_t dsdx, int16_t dtdy, bool flip);
    void GfxDpImageRectangle(int32_t tile, int32_t w, int32_t h, int32_t ulx, int32_t uly, int16_t uls, int16_t ult,
                             int32_t lrx, int32_t lry, int16_t lrs, int16_t lrt);
    void GfxDpFillRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry);
    void GfxDpSetZImage(void* zBufAddr);
    void GfxDpSetColorImage(uint32_t format, uint32_t size, uint32_t width, void* address);
    void GfxSpSetOtherMode(uint32_t shift, uint32_t num_bits, uint64_t mode);
    void GfxDpSetOtherMode(uint32_t h, uint32_t l);

    void Gfxs2dexBgCopy(F3DuObjBg* bg);
    void Gfxs2dexBg1cyc(F3DuObjBg* bg);
    void Gfxs2dexRecyCopy(F3DuObjSprite* spr);

    void AdjustWidthHeightForScale(uint32_t& width, uint32_t& height, uint32_t nativeWidth,
                                   uint32_t nativeHeight) const;
    float AdjXForAspectRatio(float x) const;
    void AdjustVIewportOrScissor(XYWidthHeight* area);
    void CalcAndSetViewport(const F3DVp_t* viewport);
    int16_t CreateShader(const std::string& path);

    void SpReset();
    void* SegAddr(uintptr_t w1);

    static const char* CCMUXtoStr(uint32_t ccmux);
    static const char* ACMUXtoStr(uint32_t acmux);
    static void GenerateCC(ColorCombiner* comb, const ColorCombinerKey& key);
    static std::string GetBaseTexturePath(const std::string& path);
    static void NormalizeVector(float v[3]);
    static void TransposedMatrixMul(float res[3], const float a[3], const float b[4][4]);
    static void MatrixMul(float res[4][4], const float a[4][4], const float b[4][4]);

    RSP* mRsp;
    RDP* mRdp;
    RenderingState mRenderingState{};

    GfxTextureCache mTextureCache{};
    std::map<ColorCombinerKey, ColorCombiner> mColorCombinerPool; // color_combiner_pool;
    std::map<ColorCombinerKey, ColorCombiner>::iterator mPrevCombiner = mColorCombinerPool.end();
    uint8_t* mTexUploadBuffer = nullptr;

    GfxDimensions mGfxCurrentWindowDimensions{}; // gfx_current_window_dimensions;
    int32_t mCurWindowPosX{};
    int32_t mCurWindowPosY{};
    GfxDimensions mCurDimensions{};        // gfx_current_dimensions;
    GfxDimensions mPrvDimensions{};        // gfx_prev_dimensions;
    XYWidthHeight mGameWindowViewport{};   // gfx_current_game_window_viewport;
    XYWidthHeight mNativeDimensions{};     // gfx_native_dimensions;
    XYWidthHeight mPrevNativeDimensions{}; // gfx_prev_native_dimensions;
    uintptr_t mGfxFrameBuffer{};

    unsigned int mMsaaLevel = 1;
    bool mDroppedFrame{};
    float* mBufVbo; // 3 vertices per triangle, VBO_MAX_FLOATS_PER_VERTEX floats per vertex
    size_t mBufVboLen{};
    size_t mBufVboNumTris{};
    // SOH [Enhancement] Actor shadow: world-space positions of the current object's triangles (9 floats
    // per tri), accumulated as the object draws and drained by FlushToonShadow at each object boundary.
    std::vector<float> mShadowVerts;
    // SOH [Enhancement] Actor shadow: opacity bands the soft edge is built from. Band 0 is the full-opacity
    // core; higher bands are the one-cell penumbra rings around the silhouette, composited at stepped-down
    // alpha. Hard edge (EdgeSoftness 0) uses band 0 only.
    static constexpr int kShadowBands = 3;
    // SOH [Enhancement] Actor shadow: all shadow-volume triangles built this frame (9 floats/tri, outward
    // wound), per band, drained by RenderShadowVolumes() at the pre-actor hook so shadows fall only on the
    // environment.
    std::vector<float> mShadowVolumeAccum[kShadowBands];
    // Per accumulated tri: 0 = cap, 1 = wall (only filled for the debug view).
    std::vector<uint8_t> mShadowVolumeKind[kShadowBands];
    // Clip-space transform scratch, reused per band in RenderShadowVolumes so the two stencil passes (and
    // the debug overlay) reuse it instead of re-running the projection per pass.
    std::vector<LoadedVertex> mShadowXform;
    float mToonShadowAlpha = 0.5f;        // core blend strength (set per frame by SetToonShadowParams)
    float mToonShadowMinElevation = 0.6f; // min remapped key height above the floor (bounds shadow length)
    float mShadowSlabDepth = 40.0f;    // stencil-volume: how far below the feet the slab reaches (ground band)
    float mShadowSlabRise = 10.0f;     // stencil-volume: how far ABOVE the feet the slab top reaches (uphill)
    int mShadowEdgeSoftness = 1;       // penumbra rings around the silhouette (0 = hard edge, max 2)
    bool mShadowShowVolume = false;    // debug: draw the translucent shadow volume (black caps, blue walls)

    // SOH [Enhancement] Cascaded shadow maps. Casters are captured in world space exactly where the
    // stencil-volume system captures its silhouettes (same gSPToonShadow arming), but they are kept as
    // plain triangles instead of being flattened onto the ground.
    //
    // The list is double-buffered because of an ordering problem: the depth maps have to exist before
    // anything samples them, yet the casters are only known once the frame has drawn. So each frame
    // renders the cascades from the PREVIOUS frame's casters while accumulating the current ones. That
    // costs one frame of shadow lag -- the same trade the stencil volumes already make, and equally
    // imperceptible for shadows that move at gameplay speed.
    // One list per caster layer (SHADOW_MAP_LAYER_WORLD / _ACTORS), because characters must cast onto the
    // scenery without casting onto each other -- see fast/shadow_map.h.
    std::vector<float> mShadowMapCasters[SHADOW_MAP_LAYERS];      // filling this frame (9 floats per tri)
    std::vector<float> mShadowMapCastersReady[SHADOW_MAP_LAYERS]; // completed last frame; what the pass draws
    // SOH [Enhancement] Cascaded shadow maps: the WORLD layer does not use the double buffer above -- it is
    // cached instead. The room mesh is by far the largest caster set and it is also completely static, so
    // re-walking it into a fresh vector every frame (and re-uploading it) was pure waste; that cost showed up
    // as hitching. The cache is rebuilt only when the geometry actually being drawn changes.
    //
    // "Changes" is detected with a running signature over the vertex batches loaded inside the world-caster
    // bracket (see GfxSpVertex). That is deliberately not a room number: the type-2 room handler distance-culls
    // individual shapes, so the drawn set moves with the player even within one room, and a room-number key
    // would freeze whichever subset happened to be visible at capture time. Hashing the batches that actually
    // run covers room changes, scene changes and per-shape culling with one mechanism.
    // Alpha-cutout casters, kept apart from the opaque list above. Foliage in this game is a billboard with
    // a leaf texture, so a depth-only pass with no pixel shader records the whole quad and a tree casts a
    // rectangle. Getting the leaf shape means sampling the material's own texture and clipping, which needs
    // per-vertex UVs and a texture binding -- neither of which the opaque path has, and both of which cost a
    // draw call per material. Splitting them keeps the overwhelming majority of casters on the one-upload,
    // one-draw path and pays the per-material cost only for the geometry that actually needs it.
    struct ShadowAlphaRange {
        TextureCacheKey key;          // resolved against the texture cache at pass time, not at capture time
        uint32_t textureId;           // filled in by ResolveShadowAlphaTextures; 0 means "could not resolve"
        uint32_t firstVertex;         // into ShadowAlphaCasters::verts
        uint32_t vertexCount;
    };
    struct ShadowAlphaCasters {
        std::vector<float> verts; // 5 floats per vertex: world xyz + uv
        std::vector<ShadowAlphaRange> ranges;
        void clear() {
            verts.clear();
            ranges.clear();
        }
        void swap(ShadowAlphaCasters& o) {
            verts.swap(o.verts);
            ranges.swap(o.ranges);
        }
        size_t VertexCount() const {
            return verts.size() / 5;
        }
    };
    ShadowAlphaCasters mShadowAlphaCasters[SHADOW_MAP_LAYERS];
    // Per-object accumulation for the size gate, fed by every armed triangle whichever list it lands in.
    // Measuring only the opaque half would shrink a mostly-cutout actor under the threshold and drop its
    // whole shadow -- and which half of a skeletal actor is cutout changes with the animation, so the
    // shadow would come and go as it walked.
    float mShadowObjectMin[3] = {};
    float mShadowObjectMax[3] = {};
    bool mShadowObjectHasVerts = false;
    size_t mShadowAlphaObjectMark = 0; // where the current object started in the actor cutout list
    // Whether the backend can actually draw cutout casters. False keeps them on the opaque list, where they
    // cast their quad -- which is what this whole path exists to avoid, but is still a shadow.
    bool mShadowAlphaSupported = false;
    ShadowAlphaCasters mShadowAlphaReady[SHADOW_MAP_LAYERS];
    ShadowAlphaCasters mShadowAlphaWorldCache;
    // Turns each range's texture-cache key into a live GPU texture id, once per frame rather than once per
    // cascade. Ranges whose texture has since been evicted are marked unresolved and skipped -- a missing
    // leaf shadow beats reinstating the solid rectangle this whole path exists to remove.
    void ResolveShadowAlphaTextures(ShadowAlphaCasters& set);
    // Tile geometry and texture coordinates for the caster capture, which runs before the combiner setup
    // that normally derives them (see the definitions for why they are duplicated rather than shared).
    void ShadowCasterTexSize(int tile, float* outWidth, float* outHeight);
    void ShadowCasterTexcoord(int tile, const struct LoadedVertex* v, float texWidth, float texHeight, float* outU,
                              float* outV);
    bool ShadowCasterIsAlphaTested(int tile, TextureCacheKey* outKey);
    bool ShadowCasterExcludedByRenderMode() const;
    // Appends one captured triangle to a layer's alpha list, extending the open range when the material has
    // not changed. Consecutive triangles almost always share a texture, so this keeps the range count near
    // the material count rather than near the triangle count.
    void CaptureShadowAlphaTriangle(int layer, const TextureCacheKey& key, struct LoadedVertex* const v[3],
                                    float texWidth, float texHeight, ShadowAlphaCasters* into = nullptr);

    // SOH [Enhancement] Cascaded shadow maps: scenery that the game spawns as an ACTOR -- a gate, a fence, a
    // tree. It belongs in the WORLD layer, because everything samples that layer and a gate's shadow should
    // fall on the player the way a wall's does. But it cannot live in the cache beside the room mesh: the
    // cache is keyed on which geometry is drawn, not on where it is, and an actor moves. The Hyrule Castle
    // gate slides open; cached, its shadow would stay shut.
    //
    // So it is double-buffered per frame exactly like the character layer, and drawn into the world layer's
    // slices from their own buffer slot (SHADOW_MAP_CASTER_SLOT_SCENERY) so the cached room mesh beside it is
    // never evicted.
    std::vector<float> mShadowSceneryCasters; // filling this frame
    std::vector<float> mShadowSceneryReady;   // what the pass draws
    ShadowAlphaCasters mShadowAlphaScenery;
    ShadowAlphaCasters mShadowAlphaSceneryReady;

    std::vector<float> mShadowMapWorldCache; // world casters, rebuilt only when the signature changes
    uint64_t mShadowWorldKeyAccum = 0;       // signature accumulated this frame (0 = no world casters drawn)
    uint64_t mShadowWorldKeyCached = 0;      // signature the cache was built from
    bool mShadowWorldCapture = true;         // capture the world layer this frame (rebuild pending)
    bool mShadowMapEnabled = false;            // app-pushed: shadow-map mode selected AND backend capable
    int mShadowMapCascadeCount = SHADOW_MAP_DEFAULT_CASCADES;
    int mShadowMapResolution = SHADOW_MAP_DEFAULT_RESOLUTION;
    float mShadowMapSplits[SHADOW_MAP_MAX_CASCADES] = { SHADOW_MAP_DEFAULT_SPLIT_0, SHADOW_MAP_DEFAULT_SPLIT_1,
                                                        SHADOW_MAP_DEFAULT_SPLIT_2, SHADOW_MAP_DEFAULT_SPLIT_3 };
    float mShadowMapLightDir[3] = { 0.0f, -1.0f, 0.0f }; // world-space direction the light travels
    float mShadowMapBlendFraction = SHADOW_MAP_DEFAULT_BLEND_FRACTION;
    float mShadowMapNormalOffset = SHADOW_MAP_DEFAULT_NORMAL_OFFSET;
    float mShadowMapStrength = SHADOW_MAP_DEFAULT_STRENGTH;
    float mShadowMapFilterWidth = SHADOW_MAP_DEFAULT_FILTER_WIDTH;
    float mShadowMapMinCasterSize = SHADOW_MAP_DEFAULT_MIN_CASTER_SIZE;
    float mShadowMapDebug = 0.0f;
    float mShadowMapEdgeHardness = SHADOW_MAP_DEFAULT_EDGE_HARDNESS;
    float mShadowMapEdgeHardnessFar = SHADOW_MAP_DEFAULT_EDGE_HARDNESS_FAR;
    // Per-frame budget on captured casters. A pathological scene must cost a dropped shadow, not an
    // unbounded allocation; 9 floats per triangle makes this a little over a million triangles.
    static constexpr size_t kShadowMapCasterBudgetFloats = 12u * 1024u * 1024u;
    // Radius each cascade is currently holding, kept across frames on purpose. The fitted radius wobbles
    // a little every frame because the game moves its near/far planes, and the texel grid is sized from
    // the radius -- so letting it follow the wobble resizes the grid and the texel snapping stops working,
    // which shows up as shadow edges crawling in steps as the camera moves. Held with hysteresis instead:
    // it only grows to cover a larger fit, or shrinks once the fit is clearly smaller. 0 = not yet fitted.
    float mShadowMapCascadeRadius[SHADOW_MAP_MAX_CASCADES] = {};
    // Light direction the cascades are currently built around, held across frames for the same reason the
    // radius is: the texel snapping that stops shadow edges shimmering is done along the light's axes, so
    // those axes have to hold still. The game's light turns continuously with the time of day. 0 = not set.
    float mShadowMapLightDirHeld[3] = {};
    GfxWindowBackend* mWapi = nullptr;
    GfxRenderingAPI* mRapi = nullptr;

    uintptr_t mSegmentPointers[MAX_SEGMENT_POINTERS]{};

    bool mFbActive{};
    bool mRendersToFb{}; // game_renders_to_framebuffer;
    std::map<int, FBInfo>::iterator mActiveFrameBuffer;
    std::map<int, FBInfo> mFrameBuffers;

    int mGameFb{};             // game_framebuffer;
    int mGameFbMsaaResolved{}; // game_framebuffer_msaa_resolved;

    std::set<std::pair<float, float>> mGetPixelDepthPending; // get_pixel_depth_pending;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> mGetPixelDepthCached; // get_pixel_depth_cached;
    std::map<std::string, MaskedTextureEntry> mMaskedTextures;

    const std::unordered_map<Mtx*, MtxF>* mCurMtxReplacements;
    bool mMarkerOn; // This was originally a debug feature. Now it seems to control s2dex?
    std::vector<std::string> shader_ids;
    int mInterpolationIndex;
    int mInterpolationIndexTarget;
};

void gfx_set_target_ucode(UcodeHandlers ucode);
void gfx_push_current_dir(char* path);
int32_t gfx_check_image_signature(const char* imgData);
const char* GfxGetOpcodeName(int8_t opcode);

} // namespace Fast

extern "C" void gfx_texture_cache_clear();
extern "C" int gfx_create_framebuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                                      uint8_t resize);
