#define NOMINMAX

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#include <any>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <list>
#include <stack>
#include "fast/resource/type/Light.h"

#ifndef _LANGUAGE_C
#define _LANGUAGE_C
#endif
#include "fast/debug/GfxDebugger.h"
#include "fast/types.h"
#include <string>

#include "fast/interpreter.h"
#include "fast/lus_gbi.h"
#include "fast/backends/gfx_window_manager_api.h"
#include "fast/backends/gfx_rendering_api.h"

#include "ship/window/gui/Gui.h"
#include "ship/resource/ResourceManager.h"
#include "ship/utils/Utils.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"

#include "libultraship/libultra/os.h"

#include <spdlog/fmt/fmt.h>

std::stack<std::string> currentDir;

#define SEG_ADDR(seg, addr) (addr | (seg << 24) | 1)
#define SUPPORT_CHECK(x) assert(x)

// SCALE_M_N: upscale/downscale M-bit integer to N-bit
#define SCALE_5_8(VAL_) (((VAL_)*0xFF) / 0x1F)
#define SCALE_8_5(VAL_) ((((VAL_) + 4) * 0x1F) / 0xFF)
#define SCALE_4_8(VAL_) ((VAL_)*0x11)
#define SCALE_8_4(VAL_) ((VAL_) / 0x11)
#define SCALE_3_8(VAL_) ((VAL_)*0x24)
#define SCALE_8_3(VAL_) ((VAL_) / 0x24)

// Based off the current set native dimensions or active framebuffer
#define HALF_SCREEN_WIDTH(activeFb) ((mFbActive ? activeFb->second.orig_width : mNativeDimensions.width) / 2)
#define HALF_SCREEN_HEIGHT(activeFb) ((mFbActive ? activeFb->second.orig_height : mNativeDimensions.height) / 2)

// Ratios for current window dimensions or active framebuffer scaled size
#define RATIO_X(activeFb, dims) \
    ((mFbActive ? activeFb->second.applied_width : dims.width) / (2.0f * HALF_SCREEN_WIDTH(activeFb)))
#define RATIO_Y(activeFb, dims) \
    ((mFbActive ? activeFb->second.applied_height : dims.height) / (2.0f * HALF_SCREEN_HEIGHT(activeFb)))

#define TEXTURE_CACHE_MAX_SIZE 1024

namespace Fast {

static UcodeHandlers ucode_handler_index = ucode_f3dex2;

const static uint32_t f3dex2AttrHandler[] = {
    F3DEX2_G_MTX_PROJECTION, F3DEX2_G_MTX_LOAD,  F3DEX2_G_MTX_PUSH,  F3DEX_G_MTX_NOPUSH,
    F3DEX2_G_CULL_FRONT,     F3DEX2_G_CULL_BACK, F3DEX2_G_CULL_BOTH,
};

const static uint32_t f3dexAttrHandler[] = { F3DEX_G_MTX_PROJECTION, F3DEX_G_MTX_LOAD,   F3DEX_G_MTX_PUSH,
                                             F3DEX_G_MTX_NOPUSH,     F3DEX_G_CULL_FRONT, F3DEX_G_CULL_BACK,
                                             F3DEX_G_CULL_BOTH };

static constexpr std::array ucode_attr_handlers = {
    &f3dexAttrHandler,  // ucode_f3db
    &f3dexAttrHandler,  // ucode_f3d
    &f3dexAttrHandler,  // ucode_f3dex
    &f3dexAttrHandler,  // ucode_f3exb
    &f3dex2AttrHandler, // ucode_f3ex2
    &f3dex2AttrHandler, // ucode_s2dex
};

static uint32_t get_attr(Attribute attr) {
    const auto ucode_map = ucode_attr_handlers[ucode_handler_index];
    // assert(ucode_map->contains(attr) && "Attribute not found in the current ucode handler");
    return (*ucode_map)[attr];
}

static std::string GetPathWithoutFileName(char* filePath) {
    size_t len = strlen(filePath);

    for (size_t i = len - 1; (long)i >= 0; i--) {
        if (filePath[i] == '/' || filePath[i] == '\\') {
            return std::string(filePath).substr(0, i);
        }
    }

    return filePath;
}

constexpr size_t MAX_TRI_BUFFER = 256;

Interpreter::Interpreter() {
    mRsp = new RSP();
    mRdp = new RDP();
    // SOH [Enhancement] Sized by VBO_MAX_FLOATS_PER_VERTEX so there is room for the toon normal attribute.
    mBufVbo = new float[MAX_TRI_BUFFER * (VBO_MAX_FLOATS_PER_VERTEX * 3)];
}

Interpreter::~Interpreter() {
    delete mRsp;
    delete mRdp;
    delete[] mBufVbo;
}

static std::weak_ptr<Interpreter> mInstance;
// Set a cached pointer to the instance so we don't need to go through the window every time
void GfxSetInstance(std::shared_ptr<Interpreter> gfx) {
    mInstance = gfx;
}

void Interpreter::Flush() {
    if (mBufVboLen > 0) {
        // SOH [Enhancement] Push the dominant toon light for this batch. The backend only consumes it
        // when the bound shader is a toon variant, so it is a no-op for ordinary draws.
        mRapi->SetToonLighting(mRsp->toon_light_dir, mRsp->toon_light_color, mRsp->toon_ambient);
        mRapi->DrawTriangles(mBufVbo, mBufVboLen, mBufVboNumTris);
        mBufVboLen = 0;
        mBufVboNumTris = 0;
    }
}

ShaderProgram* Interpreter::LookupOrCreateShaderProgram(uint64_t id0, uint64_t id1) {
    ShaderProgram* prg = mRapi->LookupShader(id0, id1);
    if (prg == nullptr) {
        mRapi->UnloadShader(mRenderingState.mShaderProgram);
        prg = mRapi->CreateAndLoadNewShader(id0, id1);
        mRenderingState.mShaderProgram = prg;
    }
    return prg;
}

const char* Interpreter::CCMUXtoStr(uint32_t ccmux) {
    static constexpr std::array tbl = {
        "G_CCMUX_COMBINED",
        "G_CCMUX_TEXEL0",
        "G_CCMUX_TEXEL1",
        "G_CCMUX_PRIMITIVE",
        "G_CCMUX_SHADE",
        "G_CCMUX_ENVIRONMENT",
        "G_CCMUX_1",
        "G_CCMUX_COMBINED_ALPHA",
        "G_CCMUX_TEXEL0_ALPHA",
        "G_CCMUX_TEXEL1_ALPHA",
        "G_CCMUX_PRIMITIVE_ALPHA",
        "G_CCMUX_SHADE_ALPHA",
        "G_CCMUX_ENV_ALPHA",
        "G_CCMUX_LOD_FRACTION",
        "G_CCMUX_PRIM_LOD_FRAC",
        "G_CCMUX_K5",
    };
    if (ccmux > tbl.size()) {
        return "G_CCMUX_0";
    }
    return tbl[ccmux];
}

// Seems unused
const char* Interpreter::ACMUXtoStr(uint32_t acmux) {
    static constexpr std::array tbl = {
        "G_ACMUX_COMBINED or G_ACMUX_LOD_FRACTION",
        "G_ACMUX_TEXEL0",
        "G_ACMUX_TEXEL1",
        "G_ACMUX_PRIMITIVE",
        "G_ACMUX_SHADE",
        "G_ACMUX_ENVIRONMENT",
        "G_ACMUX_1 or G_ACMUX_PRIM_LOD_FRAC",
        "G_ACMUX_0",
    };
    return tbl[acmux];
}

void Interpreter::GenerateCC(ColorCombiner* comb, const ColorCombinerKey& key) {
    const bool is2Cyc = (key.options & SHADER_OPT(_2CYC)) != 0;

    uint8_t c[2][2][4];
    uint64_t shaderId0 = 0;
    uint32_t shaderId1 = key.options;
    uint8_t shaderInputMapping[2][7] = { { 0 } };
    bool usedTextures[2]{};
    for (uint32_t i = 0; i < 2 && (i == 0 || is2Cyc); i++) {
        uint32_t rgbA = (key.combine_mode >> (i * 28)) & 0xf;
        uint32_t rgbB = (key.combine_mode >> (i * 28 + 4)) & 0xf;
        uint32_t rgbC = (key.combine_mode >> (i * 28 + 8)) & 0x1f;
        uint32_t rgbD = (key.combine_mode >> (i * 28 + 13)) & 7;
        uint32_t alphaA = (key.combine_mode >> (i * 28 + 16)) & 7;
        uint32_t alphaB = (key.combine_mode >> (i * 28 + 16 + 3)) & 7;
        uint32_t alphaC = (key.combine_mode >> (i * 28 + 16 + 6)) & 7;
        uint32_t alphaD = (key.combine_mode >> (i * 28 + 16 + 9)) & 7;

        if (rgbA >= 8) {
            rgbA = G_CCMUX_0;
        }
        if (rgbB >= 8) {
            rgbB = G_CCMUX_0;
        }
        if (rgbC >= 16) {
            rgbC = G_CCMUX_0;
        }
        if (rgbD == 7) {
            rgbD = G_CCMUX_0;
        }

        if (rgbA == rgbB || rgbC == G_CCMUX_0) {
            // Normalize
            rgbA = G_CCMUX_0;
            rgbB = G_CCMUX_0;
            rgbC = G_CCMUX_0;
        }
        if (alphaA == alphaB || alphaC == G_ACMUX_0) {
            // Normalize
            alphaA = G_ACMUX_0;
            alphaB = G_ACMUX_0;
            alphaC = G_ACMUX_0;
        }
        if (i == 1) {
            if (rgbA != G_CCMUX_COMBINED && rgbB != G_CCMUX_COMBINED && rgbC != G_CCMUX_COMBINED &&
                rgbD != G_CCMUX_COMBINED) {
                // First cycle RGB not used, so clear it away
                c[0][0][0] = c[0][0][1] = c[0][0][2] = c[0][0][3] = G_CCMUX_0;
            }
            if (rgbC != G_CCMUX_COMBINED_ALPHA && alphaA != G_ACMUX_COMBINED && alphaB != G_ACMUX_COMBINED &&
                alphaD != G_ACMUX_COMBINED) {
                // First cycle ALPHA not used, so clear it away
                c[0][1][0] = c[0][1][1] = c[0][1][2] = c[0][1][3] = G_ACMUX_0;
            }
        }

        c[i][0][0] = rgbA;
        c[i][0][1] = rgbB;
        c[i][0][2] = rgbC;
        c[i][0][3] = rgbD;
        c[i][1][0] = alphaA;
        c[i][1][1] = alphaB;
        c[i][1][2] = alphaC;
        c[i][1][3] = alphaD;
    }
    if (!is2Cyc) {
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t k = 0; k < 4; k++) {
                c[1][i][k] = i == 0 ? G_CCMUX_0 : G_ACMUX_0;
            }
        }
    }
    {
        uint8_t inputNumber[32] = { 0 };
        uint32_t nextInputNumber = SHADER_INPUT_1;
        for (uint32_t i = 0; i < 2 && (i == 0 || is2Cyc); i++) {
            for (uint32_t j = 0; j < 4; j++) {
                uint32_t val = 0;
                switch (c[i][0][j]) {
                    case G_CCMUX_0:
                        val = SHADER_0;
                        break;
                    case G_CCMUX_1:
                        val = SHADER_1;
                        break;
                    case G_CCMUX_TEXEL0:
                        val = SHADER_TEXEL0;
                        // Set the opposite texture when reading from the second cycle color options
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL1:
                        val = SHADER_TEXEL1;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL0_ALPHA:
                        val = SHADER_TEXEL0A;
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL1_ALPHA:
                        val = SHADER_TEXEL1A;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_CCMUX_NOISE:
                        val = SHADER_NOISE;
                        break;
                    case G_CCMUX_PRIMITIVE:
                    case G_CCMUX_PRIMITIVE_ALPHA:
                    case G_CCMUX_PRIM_LOD_FRAC:
                    case G_CCMUX_SHADE:
                    case G_CCMUX_ENVIRONMENT:
                    case G_CCMUX_ENV_ALPHA:
                    case G_CCMUX_LOD_FRACTION:
                        if (inputNumber[c[i][0][j]] == 0) {
                            shaderInputMapping[0][nextInputNumber - 1] = c[i][0][j];
                            inputNumber[c[i][0][j]] = nextInputNumber++;
                        }
                        val = inputNumber[c[i][0][j]];
                        break;
                    case G_CCMUX_COMBINED:
                        val = SHADER_COMBINED;
                        break;
                    default:
                        fprintf(stderr, "Unsupported ccmux: %d\n", c[i][0][j]);
                        break;
                }
                shaderId0 |= (uint64_t)val << (i * 32 + j * 4);
            }
        }
    }
    {
        uint8_t inputNumber[16] = { 0 };
        uint32_t nextInputNumber = SHADER_INPUT_1;
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                uint32_t val = 0;
                switch (c[i][1][j]) {
                    case G_ACMUX_0:
                        val = SHADER_0;
                        break;
                    case G_ACMUX_TEXEL0:
                        val = SHADER_TEXEL0;
                        // Set the opposite texture when reading from the second cycle color options
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_ACMUX_TEXEL1:
                        val = SHADER_TEXEL1;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_ACMUX_LOD_FRACTION:
                        // case G_ACMUX_COMBINED: same numerical value
                        if (j != 2) {
                            val = SHADER_COMBINED;
                            break;
                        }
                        c[i][1][j] = G_CCMUX_LOD_FRACTION;
                        [[fallthrough]]; // for G_ACMUX_LOD_FRACTION
                    case G_ACMUX_1:
                        // case G_ACMUX_PRIM_LOD_FRAC: same numerical value
                        if (j != 2) {
                            val = SHADER_1;
                            break;
                        }
                        [[fallthrough]]; // for G_ACMUX_PRIM_LOD_FRAC
                    case G_ACMUX_PRIMITIVE:
                    case G_ACMUX_SHADE:
                    case G_ACMUX_ENVIRONMENT:
                        if (inputNumber[c[i][1][j]] == 0) {
                            shaderInputMapping[1][nextInputNumber - 1] = c[i][1][j];
                            inputNumber[c[i][1][j]] = nextInputNumber++;
                        }
                        val = inputNumber[c[i][1][j]];
                        break;
                }
                shaderId0 |= (uint64_t)val << (i * 32 + 16 + j * 4);
            }
        }
    }
    comb->shader_id0 = shaderId0;
    comb->shader_id1 = shaderId1;
    comb->usedTextures[0] = usedTextures[0];
    comb->usedTextures[1] = usedTextures[1];
    // comb->prg = gfx_lookup_or_create_mShaderProgram(shader_id0, shader_id1);
    memcpy(comb->shader_input_mapping, shaderInputMapping, sizeof(shaderInputMapping));
}

ColorCombiner* Interpreter::LookupOrCreateColorCombiner(const ColorCombinerKey& key) {
    if (mPrevCombiner != mColorCombinerPool.end() && mPrevCombiner->first == key) {
        return &mPrevCombiner->second;
    }
    mPrevCombiner = mColorCombinerPool.find(key);
    if (mPrevCombiner != mColorCombinerPool.end()) {
        return &mPrevCombiner->second;
    }
    Flush();
    mPrevCombiner = mColorCombinerPool.insert(std::make_pair(key, ColorCombiner())).first;
    GenerateCC(&mPrevCombiner->second, key);
    return &mPrevCombiner->second;
}

void Interpreter::TextureCacheClear() {
    for (const auto& entry : mTextureCache.map) {
        mTextureCache.free_texture_ids.push_back(entry.second.texture_id);
    }
    mTextureCache.map.clear();
    mTextureCache.lru.clear();
}

bool Interpreter::TextureCacheLookup(int i, const TextureCacheKey& key) {
    TextureCacheMap::iterator it = mTextureCache.map.find(key);
    TextureCacheNode** n = &mRenderingState.mTextures[i];

    if (it != mTextureCache.map.end()) {
        mRapi->SelectTexture(i, it->second.texture_id);
        *n = &*it;
        mTextureCache.lru.splice(mTextureCache.lru.end(), mTextureCache.lru,
                                 it->second.lru_location); // move to back
        return true;
    }

    if (mTextureCache.map.size() >= TEXTURE_CACHE_MAX_SIZE) {
        // Remove the texture that was least recently used
        it = mTextureCache.lru.front().it;
        mTextureCache.free_texture_ids.push_back(it->second.texture_id);
        mTextureCache.map.erase(it);
        mTextureCache.lru.pop_front();
    }

    uint32_t texture_id;
    if (!mTextureCache.free_texture_ids.empty()) {
        texture_id = mTextureCache.free_texture_ids.back();
        mTextureCache.free_texture_ids.pop_back();
    } else {
        texture_id = mRapi->NewTexture();
    }

    it = mTextureCache.map.insert(std::make_pair(key, TextureCacheValue())).first;
    TextureCacheNode* node = &*it;
    node->second.texture_id = texture_id;
    node->second.lru_location = mTextureCache.lru.insert(mTextureCache.lru.end(), { it });

    mRapi->SelectTexture(i, texture_id);
    mRapi->SetSamplerParameters(i, false, 0, 0);
    *n = node;
    return false;
}

std::string Interpreter::GetBaseTexturePath(const std::string& path) {
    if (path.starts_with(Ship::IResource::gAltAssetPrefix)) {
        return path.substr(Ship::IResource::gAltAssetPrefix.length());
    }

    return path;
}

void Interpreter::TextureCacheDelete(const uint8_t* origAddr) {
    while (mTextureCache.map.bucket_count() > 0) {
        TextureCacheKey key = { origAddr, { 0 }, 0, 0, 0 }; // bucket index only depends on the address
        size_t bucket = mTextureCache.map.bucket(key);
        bool again = false;
        for (auto it = mTextureCache.map.begin(bucket); it != mTextureCache.map.end(bucket); ++it) {
            if (it->first.texture_addr == origAddr) {
                mTextureCache.lru.erase(it->second.lru_location);
                mTextureCache.free_texture_ids.push_back(it->second.texture_id);
                mTextureCache.map.erase(it->first);
                again = true;
                break;
            }
        }
        if (!again) {
            break;
        }
    }
}

void Interpreter::ImportTextureRgba16(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRgba16: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes / 2;
    uint32_t height = sizeBytes / mRdp->texture_tile[tile].line_size_bytes;

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width * 2;
    }

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (fullImageLineSizeBytes / 2)) + (x);

            uint16_t col16 = (addr[2 * clrIdx] << 8) | addr[2 * clrIdx + 1];
            uint8_t a = col16 & 1;
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;

            i++;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureRgba32(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRgba32: null texture address for tile {}", tile);
        return;
    }

    uint32_t size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t full_image_line_size_bytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;
    SUPPORT_CHECK(full_image_line_size_bytes == line_size_bytes);

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes / 2;
    uint32_t height = (size_bytes / 2) / mRdp->texture_tile[tile].line_size_bytes;
    mRapi->UploadTexture(addr, width, height);
}

void Interpreter::ImportTextureIA4(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;
    SUPPORT_CHECK(fullImageLineSizeBytes == lineSizeBytes);

    for (uint32_t i = 0; i < sizeBytes * 2; i++) {
        uint8_t byte = addr[i / 2];
        uint8_t part = (byte >> (4 - (i % 2) * 4)) & 0xf;
        uint8_t intensity = part >> 1;
        uint8_t alpha = part & 1;
        uint8_t r = intensity;
        uint8_t g = intensity;
        uint8_t b = intensity;
        mTexUploadBuffer[4 * i + 0] = SCALE_3_8(r);
        mTexUploadBuffer[4 * i + 1] = SCALE_3_8(g);
        mTexUploadBuffer[4 * i + 2] = SCALE_3_8(b);
        mTexUploadBuffer[4 * i + 3] = alpha ? 255 : 0;
    }

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes * 2;
    uint32_t height = sizeBytes / mRdp->texture_tile[tile].line_size_bytes;

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureIA8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;
    SUPPORT_CHECK(fullImageLineSizeBytes == lineSizeBytes);

    for (uint32_t i = 0; i < sizeBytes; i++) {
        uint8_t intensity = addr[i] >> 4;
        uint8_t alpha = addr[i] & 0xf;
        uint8_t r = intensity;
        uint8_t g = intensity;
        uint8_t b = intensity;
        mTexUploadBuffer[4 * i + 0] = SCALE_4_8(r);
        mTexUploadBuffer[4 * i + 1] = SCALE_4_8(g);
        mTexUploadBuffer[4 * i + 2] = SCALE_4_8(b);
        mTexUploadBuffer[4 * i + 3] = SCALE_4_8(alpha);
    }

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes;
    uint32_t height = sizeBytes / mRdp->texture_tile[tile].line_size_bytes;

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureIA16(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA16: null texture address for tile {}", tile);
        return;
    }

    uint32_t size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t full_image_line_size_bytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes / 2;
    uint32_t height = size_bytes / mRdp->texture_tile[tile].line_size_bytes;

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (full_image_line_size_bytes == size_bytes) {
        full_image_line_size_bytes = width * 2;
    }

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (full_image_line_size_bytes / 2)) + (x);

            uint8_t intensity = addr[2 * clrIdx];
            uint8_t alpha = addr[2 * clrIdx + 1];
            uint8_t r = intensity;
            uint8_t g = intensity;
            uint8_t b = intensity;
            mTexUploadBuffer[4 * i + 0] = r;
            mTexUploadBuffer[4 * i + 1] = g;
            mTexUploadBuffer[4 * i + 2] = b;
            mTexUploadBuffer[4 * i + 3] = alpha;

            i++;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureI4(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureI4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes * 2;
    uint32_t height = sizeBytes / mRdp->texture_tile[tile].line_size_bytes;

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width / 2;
    }

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (fullImageLineSizeBytes * 2)) + (x);

            uint8_t byte = addr[clrIdx / 2];
            uint8_t part = (byte >> (4 - (clrIdx % 2) * 4)) & 0xf;
            uint8_t intensity = part;
            uint8_t r = intensity;
            uint8_t g = intensity;
            uint8_t b = intensity;
            uint8_t a = intensity;
            mTexUploadBuffer[4 * i + 0] = SCALE_4_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_4_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_4_8(b);
            mTexUploadBuffer[4 * i + 3] = SCALE_4_8(a);

            i++;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureI8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureI8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t full_image_line_size_bytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    for (uint32_t i = 0; i < sizeBytes; i++) {
        uint8_t intensity = addr[i];
        mTexUploadBuffer[4 * i + 0] = intensity;
        mTexUploadBuffer[4 * i + 1] = intensity;
        mTexUploadBuffer[4 * i + 2] = intensity;
        mTexUploadBuffer[4 * i + 3] = intensity;
    }

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes;
    uint32_t height = sizeBytes / mRdp->texture_tile[tile].line_size_bytes;

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureCi4(int tile, bool importReplacement) {
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureCi4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;
    uint32_t palIdx = mRdp->texture_tile[tile].palette; // 0-15

    const uint8_t* palette;

    if (palIdx > 7)
        palette = mRdp->palettes[palIdx / 8]; // 16 pixel entries, 16 bits each
    else
        palette = mRdp->palettes[palIdx / 8] + (palIdx % 8) * 16 * 2;

    SUPPORT_CHECK(fullImageLineSizeBytes == lineSizeBytes);

    for (uint32_t i = 0; i < sizeBytes * 2; i++) {
        uint8_t byte = addr[i / 2];
        uint8_t idx = (byte >> (4 - (i % 2) * 4)) & 0xf;
        uint16_t col16 = (palette[idx * 2] << 8) | palette[idx * 2 + 1]; // Big endian load
        uint8_t a = col16 & 1;
        uint8_t r = col16 >> 11;
        uint8_t g = (col16 >> 6) & 0x1f;
        uint8_t b = (col16 >> 1) & 0x1f;
        mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
        mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
        mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
        mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;
    }

    uint32_t resultLineSizeBytes = mRdp->texture_tile[tile].line_size_bytes;
    if (metadata->h_byte_scale != 1) {
        resultLineSizeBytes *= metadata->h_byte_scale;
    }

    uint32_t width = resultLineSizeBytes * 2;
    uint32_t height = sizeBytes / resultLineSizeBytes;

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureCi8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureCi8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    for (uint32_t i = 0, j = 0; i < sizeBytes; j += fullImageLineSizeBytes - lineSizeBytes) {
        for (uint32_t k = 0; k < lineSizeBytes; i++, k++, j++) {
            uint8_t idx = addr[j];
            uint16_t col16 = (mRdp->palettes[idx / 128][(idx % 128) * 2] << 8) |
                             mRdp->palettes[idx / 128][(idx % 128) * 2 + 1]; // Big endian load
            uint8_t a = col16 & 1;
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;
        }
    }

    uint32_t resultLineSizeBytes = mRdp->texture_tile[tile].line_size_bytes;
    if (metadata->h_byte_scale != 1) {
        resultLineSizeBytes *= metadata->h_byte_scale;
    }

    uint32_t width = resultLineSizeBytes;
    uint32_t height = sizeBytes / resultLineSizeBytes;

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureImg(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureImg: null texture address for tile {}", tile);
        return;
    }

    uint16_t width = metadata->width;
    uint16_t height = metadata->height;
    mRapi->UploadTexture(addr, width, height);
}

void Interpreter::ImportTextureRaw(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRaw: null texture address for tile {}", tile);
        return;
    }

    uint16_t width = metadata->width;
    uint16_t height = metadata->height;
    Fast::TextureType type = metadata->type;
    std::shared_ptr<Fast::Texture> resource = metadata->resource;

    // if texture type is CI4 or CI8 we need to apply tlut to it
    switch (type) {
        case Fast::TextureType::Palette4bpp:
            ImportTextureCi4(tile, importReplacement);
            return;
        case Fast::TextureType::Palette8bpp:
            ImportTextureCi8(tile, importReplacement);
            return;
        default:
            break;
    }

    uint32_t numLoadedBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t numOriginallyLoadedBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;

    uint32_t resultOrigLineSize = mRdp->texture_tile[tile].line_size_bytes;
    switch (mRdp->texture_tile[tile].siz) {
        case G_IM_SIZ_32b:
            resultOrigLineSize *= 2;
            break;
    }
    uint32_t resultOrigHeight = numOriginallyLoadedBytes / resultOrigLineSize;
    uint32_t resultNewLineSize = resultOrigLineSize * metadata->h_byte_scale;
    uint32_t resultNewHeight = resultOrigHeight * metadata->v_pixel_scale;

    if (resultNewLineSize == 4 * width && resultNewHeight == height) {
        // Can use the texture directly since it has the correct dimensions
        mRapi->UploadTexture(addr, width, height);
        return;
    }

    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    // Get the resource's true image size
    uint32_t resourceImageSizeBytes = resource->ImageDataSize;
    uint32_t safeFullImageLineSizeBytes = fullImageLineSizeBytes;
    uint32_t safeLineSizeBytes = line_size_bytes;
    uint32_t safeLoadedBytes = numLoadedBytes;

    // Sometimes the texture load commands will specify a size larger than the authentic texture
    // Normally the OOB info is read as garbage, but will cause a crash on some platforms
    // Restrict the bytes to a safe amount
    if (numLoadedBytes > resourceImageSizeBytes) {
        safeLoadedBytes = resourceImageSizeBytes;
        safeLineSizeBytes = resourceImageSizeBytes;
        safeFullImageLineSizeBytes = resourceImageSizeBytes;
    }

    // Safely only copy the amount of bytes the resource can allow
    for (uint32_t i = 0, j = 0; i < safeLoadedBytes; i += safeLineSizeBytes, j += safeFullImageLineSizeBytes) {
        memcpy(mTexUploadBuffer + i, addr + j, safeLineSizeBytes);
    }

    // Set the remaining bytes to load as 0
    if (numLoadedBytes > resourceImageSizeBytes) {
        memset(mTexUploadBuffer + resourceImageSizeBytes, 0, numLoadedBytes - resourceImageSizeBytes);
    }

    mRapi->UploadTexture(mTexUploadBuffer, resultNewLineSize / 4, resultNewHeight);
}

void Interpreter::ImportTexture(int i, int tile, bool importReplacement) {
    uint8_t fmt = mRdp->texture_tile[tile].fmt;
    uint8_t siz = mRdp->texture_tile[tile].siz;
    uint32_t texFlags = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].tex_flags;
    uint32_t tmemIdex = mRdp->texture_tile[tile].tmem_index;
    uint8_t paletteIndex = mRdp->texture_tile[tile].palette;
    uint32_t origSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;

    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* origAddr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[tmemIdex].addr;

    if (origAddr == nullptr) {
        SPDLOG_ERROR("ImportTexture: null texture address for tile {}", tile);
        return;
    }

    TextureCacheKey key;
    if (fmt == G_IM_FMT_CI) {
        key = { origAddr, { mRdp->palettes[0], mRdp->palettes[1] }, fmt, siz, paletteIndex, origSizeBytes };
    } else {
        key = { origAddr, {}, fmt, siz, paletteIndex, origSizeBytes };
    }

    if (TextureCacheLookup(i, key)) {
        return;
    }

    if ((texFlags & TEX_FLAG_LOAD_AS_IMG) != 0) {
        ImportTextureImg(tile, importReplacement);
        return;
    }

    // if load as raw is set then we load_raw();
    if ((texFlags & TEX_FLAG_LOAD_AS_RAW) != 0) {
        ImportTextureRaw(tile, importReplacement);
        return;
    }

    switch (fmt) {
        case G_IM_FMT_RGBA:
            if (siz == G_IM_SIZ_16b) {
                ImportTextureRgba16(tile, importReplacement);
            } else if (siz == G_IM_SIZ_32b) {
                ImportTextureRgba32(tile, importReplacement);
            } else {
                SPDLOG_ERROR("RGBA Texture that isn't 16 or 32 bit. Size = {}", siz);
                // OTRTODO: Sometimes, seemingly randomly, we end up here. Could be a bad dlist, could be
                // something F3D does not have supported. Further investigation is needed.
            }
            break;
        case G_IM_FMT_IA:
            if (siz == G_IM_SIZ_4b) {
                ImportTextureIA4(tile, importReplacement);
            } else if (siz == G_IM_SIZ_8b) {
                ImportTextureIA8(tile, importReplacement);
            } else if (siz == G_IM_SIZ_16b) {
                ImportTextureIA16(tile, importReplacement);
            } else {
                SPDLOG_ERROR("IA Texture that isn't 4, 8, or 16 bit. Size = {}", siz);
                ;
            }
            break;
        case G_IM_FMT_CI:
            if (siz == G_IM_SIZ_4b) {
                ImportTextureCi4(tile, importReplacement);
            } else if (siz == G_IM_SIZ_8b) {
                ImportTextureCi8(tile, importReplacement);
            } else {
                SPDLOG_ERROR("CI Texture that isn't 4 or 8 bit. Size = {}", siz);
            }
            break;
        case G_IM_FMT_I:
            if (siz == G_IM_SIZ_4b) {
                ImportTextureI4(tile, importReplacement);
            } else if (siz == G_IM_SIZ_8b) {
                ImportTextureI8(tile, importReplacement);
            } else {
                SPDLOG_ERROR("I Texture that isn't 4 or 8 bit. Size = {}", siz);
            }
            break;
        case G_IM_FMT_YUV:
            SPDLOG_ERROR("YUV Textures not supported");
            break;
        default:
            SPDLOG_ERROR("Invalid texture format. Fmt = {}", fmt);
            break;
    }
}

void Interpreter::ImportTextureMask(int i, int tile) {
    uint32_t tmemIndex = mRdp->texture_tile[tile].tmem_index;
    RawTexMetadata metadata = mRdp->loaded_texture[tmemIndex].raw_tex_metadata;

    if (metadata.resource == nullptr) {
        return;
    }

    auto maskIter = mMaskedTextures.find(GetBaseTexturePath(metadata.resource->GetInitData()->Path));
    if (maskIter == mMaskedTextures.end()) {
        return;
    }

    const uint8_t* orig_addr = maskIter->second.mask;

    if (orig_addr == nullptr) {
        return;
    }

    TextureCacheKey key = { orig_addr, {}, 0, 0, 0, 0 };

    if (TextureCacheLookup(i, key)) {
        return;
    }

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes;
    uint32_t height = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes /
                      mRdp->texture_tile[tile].line_size_bytes;
    switch (mRdp->texture_tile[tile].siz) {
        case G_IM_SIZ_4b:
            width *= 2;
            break;
        case G_IM_SIZ_8b:
        default:
            break;
        case G_IM_SIZ_16b:
            width /= 2;
            break;
        case G_IM_SIZ_32b:
            width /= 4;
            break;
    }

    for (uint32_t texIndex = 0; texIndex < width * height; texIndex++) {
        uint8_t masked = orig_addr[texIndex];
        if (masked) {
            mTexUploadBuffer[4 * texIndex + 0] = 0;
            mTexUploadBuffer[4 * texIndex + 1] = 0;
            mTexUploadBuffer[4 * texIndex + 2] = 0;
            mTexUploadBuffer[4 * texIndex + 3] = 0xFF;
        } else {
            mTexUploadBuffer[4 * texIndex + 0] = 0;
            mTexUploadBuffer[4 * texIndex + 1] = 0;
            mTexUploadBuffer[4 * texIndex + 2] = 0;
            mTexUploadBuffer[4 * texIndex + 3] = 0;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::NormalizeVector(float v[3]) {
    float s = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= s;
    v[1] /= s;
    v[2] /= s;
}

void Interpreter::TransposedMatrixMul(float res[3], const float a[3], const float b[4][4]) {
    res[0] = a[0] * b[0][0] + a[1] * b[0][1] + a[2] * b[0][2];
    res[1] = a[0] * b[1][0] + a[1] * b[1][1] + a[2] * b[1][2];
    res[2] = a[0] * b[2][0] + a[1] * b[2][1] + a[2] * b[2][2];
}

void Interpreter::MatrixMul(float res[4][4], const float a[4][4], const float b[4][4]) {
    float tmp[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tmp[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    memcpy(res, tmp, sizeof(tmp));
}

void Interpreter::CalculateNormalDir(const F3DLight_t* light, float coeffs[3]) {
    float light_dir[3] = { light->dir[0] / 127.0f, light->dir[1] / 127.0f, light->dir[2] / 127.0f };

    Interpreter::TransposedMatrixMul(coeffs, light_dir,
                                     mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);
    Interpreter::NormalizeVector(coeffs);
}

// SOH [Enhancement] Resolve the single effective light for the current object's toon shading:
// ambient from the binding's ambient light, and the key direction/colour from the application-supplied
// gSPToonKey (defaulting to a straight-on white key if none was provided this batch). The application
// chooses and eases the key; the renderer just consumes it.
void Interpreter::SelectToonLight() {
    int amb_idx = mRsp->current_num_lights - 1;
    if (amb_idx < 0) {
        amb_idx = 0;
    }
    mRsp->toon_ambient[0] = mRsp->current_lights[amb_idx].l.col[0] / 255.0f;
    mRsp->toon_ambient[1] = mRsp->current_lights[amb_idx].l.col[1] / 255.0f;
    mRsp->toon_ambient[2] = mRsp->current_lights[amb_idx].l.col[2] / 255.0f;

    // SOH [Enhancement] The game supplies one world-space key light per object via gSPToonKey. The
    // forwarded normals are world-space too (see GfxSpVertex), so the key is used as-is — no
    // object-space transform (which would only be correct for one limb of a batched skeletal actor,
    // but they share a single light uniform). The game drives a single Wind Waker-style key (sun by
    // day, animating to torches at night).
    if (!mRsp->toon_key_valid) {
        // A toon batch reached here with no key supplied. Not expected in normal use — the game emits
        // gSPToonKey before every object's geometry — so default to a straight-on white key that at
        // least lights the object evenly rather than leaving the shade undefined.
        mRsp->toon_light_dir[0] = 0.0f;
        mRsp->toon_light_dir[1] = 0.0f;
        mRsp->toon_light_dir[2] = 1.0f;
        mRsp->toon_light_color[0] = 1.0f;
        mRsp->toon_light_color[1] = 1.0f;
        mRsp->toon_light_color[2] = 1.0f;
        return;
    }

    mRsp->toon_light_dir[0] = mRsp->toon_key_dir[0];
    mRsp->toon_light_dir[1] = mRsp->toon_key_dir[1];
    mRsp->toon_light_dir[2] = mRsp->toon_key_dir[2];
    // Guard a zero-length key (s8 quantization can collapse a small direction) so NormalizeVector
    // doesn't divide by zero and produce NaNs — fall back to straight-on.
    float key_len2 = mRsp->toon_light_dir[0] * mRsp->toon_light_dir[0] +
                     mRsp->toon_light_dir[1] * mRsp->toon_light_dir[1] +
                     mRsp->toon_light_dir[2] * mRsp->toon_light_dir[2];
    if (key_len2 < 1e-8f) {
        mRsp->toon_light_dir[0] = 0.0f;
        mRsp->toon_light_dir[1] = 0.0f;
        mRsp->toon_light_dir[2] = 1.0f;
    } else {
        NormalizeVector(mRsp->toon_light_dir);
    }
    mRsp->toon_light_color[0] = mRsp->toon_key_color[0];
    mRsp->toon_light_color[1] = mRsp->toon_key_color[1];
    mRsp->toon_light_color[2] = mRsp->toon_key_color[2];
}

void Interpreter::GfxSpMatrix(uint8_t parameters, const int32_t* addr) {
    float matrix[4][4];

    if (auto it = mCurMtxReplacements->find((Mtx*)addr); it != mCurMtxReplacements->end()) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float v = it->second.mf[i][j];
                int as_int = (int)(v * 65536.0f);
                matrix[i][j] = as_int * (1.0f / 65536.0f);
            }
        }
    } else {
#ifndef GBI_FLOATS
        // Original GBI where fixed point matrices are used
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j += 2) {
                int32_t int_part = addr[i * 2 + j / 2];
                uint32_t frac_part = addr[8 + i * 2 + j / 2];
                matrix[i][j] = (int32_t)((int_part & 0xffff0000) | (frac_part >> 16)) / 65536.0f;
                matrix[i][j + 1] = (int32_t)((int_part << 16) | (frac_part & 0xffff)) / 65536.0f;
            }
        }
#else
        // For a modified GBI where fixed point values are replaced with floats
        memcpy(matrix, addr, sizeof(matrix));
#endif
    }

    const int8_t mtx_projection = get_attr(MTX_PROJECTION);
    const int8_t mtx_load = get_attr(MTX_LOAD);
    const int8_t mtx_push = get_attr(MTX_PUSH);

    if (parameters & mtx_projection) {
        if (parameters & mtx_load) {
            memcpy(mRsp->P_matrix, matrix, sizeof(matrix));
        } else {
            MatrixMul(mRsp->P_matrix, matrix, mRsp->P_matrix);
        }
    } else { // G_MTX_MODELVIEW
        if ((parameters & mtx_push) && mRsp->modelview_matrix_stack_size < 11) {
            ++mRsp->modelview_matrix_stack_size;
            memcpy(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1],
                   mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 2], sizeof(matrix));
        }
        if (parameters & mtx_load) {
            if (mRsp->modelview_matrix_stack_size == 0)
                ++mRsp->modelview_matrix_stack_size;
            memcpy(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], matrix, sizeof(matrix));
        } else {
            MatrixMul(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], matrix,
                      mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);
        }
        mRsp->lights_changed = 1;
    }
    MatrixMul(mRsp->MP_matrix, mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], mRsp->P_matrix);
}

void Interpreter::GfxSpPopMatrix(uint32_t count) {
    while (count--) {
        if (mRsp->modelview_matrix_stack_size > 0) {
            --mRsp->modelview_matrix_stack_size;
            if (mRsp->modelview_matrix_stack_size > 0) {
                MatrixMul(mRsp->MP_matrix, mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1],
                          mRsp->P_matrix);
            }
        }
    }
    mRsp->lights_changed = true;
}

float Interpreter::AdjXForAspectRatio(float x) const {
    if (mFbActive) {
        return x;
    } else {
        return x * (4.0f / 3.0f) / ((float)mCurDimensions.width / (float)mCurDimensions.height);
    }
}

// Scale the width and height value based on the ratio of the viewport to the native size
void Interpreter::AdjustWidthHeightForScale(uint32_t& width, uint32_t& height, uint32_t nativeWidth,
                                            uint32_t nativeHeight) const {
    width = round(width * (mCurDimensions.width / (2.0f * (nativeWidth / 2))));
    height = round(height * (mCurDimensions.height / (2.0f * (nativeHeight / 2))));

    if (width == 0) {
        width = 1;
    }
    if (height == 0) {
        height = 1;
    }
}

// SOH [Enhancement] Cascaded shadow maps: geometry that must never be recorded as a caster, whatever else
// it is. Only consulted when the shadow map is on, so the stencil volumes keep their previous behaviour
// exactly.
//
// This became necessary the moment the actor capture stopped requiring G_LIGHTING. That change was right --
// unlit geometry blocks light -- but "unlit" is also what decals and translucent overlays are, and those
// had been excluded by accident rather than by intent. A decal is glued coplanar to the surface under it,
// so recording one makes that surface shadow itself: a hard-edged blotch appearing and disappearing with
// sub-texel depth noise, which is not a shadow of anything. Translucent geometry is see-through, and a
// see-through surface casting a solid shadow looks worse than casting none.
bool Interpreter::ShadowCasterExcludedByRenderMode() const {
    const uint32_t zmode = mRdp->other_mode_l & ZMODE_DEC; // ZMODE_DEC is the full two-bit field mask
    if (zmode == ZMODE_DEC) {
        return true;
    }
    // Translucent geometry is see-through and a see-through surface casting a solid shadow looks worse than
    // casting none -- unless the game has said otherwise for these particular draws. A tree canopy is drawn
    // translucent so it can fade with distance, but what the texture holds is a leaf silhouette, not a sheet
    // of glass. Inside the bracket the exclusion lifts; the cutout requirement below it does not, so the
    // canopy casts leaves through its alpha or it casts nothing.
    if (zmode == ZMODE_XLU && !mRdp->shadow_scenery_caster) {
        return true;
    }
    // The zmode field alone, and nothing else. FORCE_BL was tested here too, on the reasoning that it marks
    // a blend as unconditional rather than coverage-driven -- but plenty of visually solid geometry sets it
    // while staying in ZMODE_OPA, and excluding those punched holes in the shadows of solid walls. Every
    // genuinely translucent surface mode carries ZMODE_XLU anyway, so the narrower test loses nothing and
    // cannot take an opaque caster with it.
    return false;
}

// SOH [Enhancement] Cascaded shadow maps: the tile's texture dimensions, split out of GfxSpTri1's combiner
// setup so the caster capture can compute UVs without waiting for it. The capture has to run BEFORE the
// trivial clip rejection -- a tree behind the camera still casts into the view, and culling casters by the
// view frustum is what made shadows blink out when the camera turned -- but the combiner setup that
// normally derives these runs long after. Same arithmetic, deliberately duplicated rather than shared,
// because the original is interleaved with texture importing and render-state flushing that must NOT
// happen during a capture.
void Interpreter::ShadowCasterTexSize(int tile, float* outWidth, float* outHeight) {
    uint32_t tex_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;
    uint32_t line_size = mRdp->texture_tile[tile].line_size_bytes;
    if (line_size == 0) {
        line_size = 1;
    }
    uint32_t height = tex_size_bytes / line_size;
    switch (mRdp->texture_tile[tile].siz) {
        case G_IM_SIZ_4b:
            line_size <<= 1;
            break;
        case G_IM_SIZ_8b:
            break;
        case G_IM_SIZ_16b:
            line_size /= G_IM_SIZ_16b_LINE_BYTES;
            break;
        case G_IM_SIZ_32b:
            line_size /= G_IM_SIZ_32b_LINE_BYTES;
            height /= 2;
            break;
    }
    *outWidth = (float)(line_size == 0 ? 1 : line_size);
    *outHeight = (float)(height == 0 ? 1 : height);
}

// One vertex's normalised texture coordinate for tile 0, mirroring the packing loop in GfxSpTri1. The
// half-texel that linear filtering adds is included: the shadow samples the same texels the main pass
// does, so a mismatch here would offset the cutout against the visible leaf.
void Interpreter::ShadowCasterTexcoord(int tile, const struct LoadedVertex* v, float texWidth, float texHeight,
                                       float* outU, float* outV) {
    float u = v->u / 32.0f;
    float w = v->v / 32.0f;
    const int shifts = mRdp->texture_tile[tile].shifts;
    const int shiftt = mRdp->texture_tile[tile].shiftt;
    if (shifts != 0) {
        if (shifts <= 10) {
            u /= 1 << shifts;
        } else {
            u *= 1 << (16 - shifts);
        }
    }
    if (shiftt != 0) {
        if (shiftt <= 10) {
            w /= 1 << shiftt;
        } else {
            w *= 1 << (16 - shiftt);
        }
    }
    u -= mRdp->texture_tile[tile].uls / 4.0f;
    w -= mRdp->texture_tile[tile].ult / 4.0f;
    if ((mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT) {
        u += 0.5f;
        w += 0.5f;
    }
    *outU = u / texWidth;
    *outV = w / texHeight;
}

// True when this draw punches its own silhouette out of the texture's alpha rather than filling its
// polygon -- the two render-mode bits GfxSpTri1 turns into the shader's texture_edge / alpha_threshold
// options. Alpha BLENDING is deliberately not included: a blended surface is see-through, and a
// see-through surface casting a hard shadow looks worse than casting none.
bool Interpreter::ShadowCasterIsAlphaTested(int tile, TextureCacheKey* outKey) {
    const bool texture_edge = (mRdp->other_mode_l & CVG_X_ALPHA) == CVG_X_ALPHA;
    const bool alpha_threshold = (mRdp->other_mode_l & (3U << G_MDSFT_ALPHACOMPARE)) == G_AC_THRESHOLD;
    // Inside a scenery bracket the render state cannot be the witness for TRANSLUCENT geometry: the draw
    // declares itself blended precisely so it can fade, which is the same state a sheet of glass sets. The
    // bracket is the witness instead -- the game named this actor's canopy -- so the only remaining question
    // is whether there is a texture whose alpha can be cut against, which the rest of this function answers.
    //
    // Only for the translucent half of it. The same tree's trunk is ordinary opaque geometry and has no
    // business being diverted through the cutout pipeline: it would pay a pipeline switch and five floats a
    // vertex to clip against an alpha channel that is solid everywhere.
    const bool sceneryCutout =
        mRdp->shadow_scenery_caster && ((mRdp->other_mode_l & ZMODE_DEC) == ZMODE_XLU);
    if (!texture_edge && !alpha_threshold && !sceneryCutout) {
        return false;
    }
    const uint32_t tmemIndex = mRdp->texture_tile[tile].tmem_index;
    const uint8_t* addr = mRdp->loaded_texture[tmemIndex].addr;
    if (addr == nullptr) {
        return false;
    }
    // Intensity textures carry no opacity: the importer expands I4/I8 by copying the intensity into all
    // four channels, so alpha there is a brightness map. Clipping against it punches the texture's own dark
    // areas out of the depth map, which on a solid stone wall means a shadow full of holes and no
    // transparency anywhere in sight. Those materials cast as plain opaque geometry.
    if (mRdp->texture_tile[tile].fmt == G_IM_FMT_I) {
        return false;
    }
    // Same key ImportTexture builds, so the lookup at pass time finds the texture the main pass uploaded.
    // Built here rather than resolved here: the texture may not have been imported yet at capture time.
    const uint8_t fmt = mRdp->texture_tile[tile].fmt;
    const uint8_t siz = mRdp->texture_tile[tile].siz;
    const uint8_t paletteIndex = mRdp->texture_tile[tile].palette;
    const uint32_t origSizeBytes = mRdp->loaded_texture[tmemIndex].orig_size_bytes;
    if (fmt == G_IM_FMT_CI) {
        *outKey = { addr, { mRdp->palettes[0], mRdp->palettes[1] }, fmt, siz, paletteIndex, origSizeBytes };
    } else {
        *outKey = { addr, {}, fmt, siz, paletteIndex, origSizeBytes };
    }
    return true;
}

void Interpreter::CaptureShadowAlphaTriangle(int layer, const TextureCacheKey& key, struct LoadedVertex* const v[3],
                                             float texWidth, float texHeight, ShadowAlphaCasters* into) {
    // `into` overrides the per-layer list. Only the scenery bucket uses it: that geometry is drawn into the
    // world layer's slices but kept in its own per-frame list, so it cannot share the layer's.
    ShadowAlphaCasters& dst = (into != nullptr) ? *into : mShadowAlphaCasters[layer];
    if (dst.verts.size() >= kShadowMapCasterBudgetFloats) {
        return;
    }
    // Extend the open range when the material has not changed. Draws arrive grouped by material, so this
    // keeps the range count near the number of materials rather than near the number of triangles -- which
    // matters because every range is its own draw call in every cascade.
    if (dst.ranges.empty() || !(dst.ranges.back().key == key)) {
        const float inf = std::numeric_limits<float>::max();
        dst.ranges.push_back(
            { key, 0u, (uint32_t)(dst.verts.size() / 5), 0u, { inf, inf, inf }, { -inf, -inf, -inf } });
    }
    // Built into a local and inserted once, for the same reason the opaque path does (see
    // ShadowAppendTriangle): fifteen push_backs per triangle is fifteen capacity checks.
    float tri[15];
    for (int si = 0; si < 3; si++) {
        float u, w;
        ShadowCasterTexcoord(mRdp->first_tile_index, v[si], texWidth, texHeight, &u, &w);
        float* o = &tri[si * 5];
        o[0] = v[si]->wx;
        o[1] = v[si]->wy;
        o[2] = v[si]->wz;
        o[3] = u;
        o[4] = w;
    }
    dst.verts.insert(dst.verts.end(), tri, tri + 15);
    ShadowAlphaRange& range = dst.ranges.back();
    range.vertexCount += 3;
    // Grow the range's box with this triangle. Done here rather than in a second pass because the vertices
    // are already in hand and in cache; walking the list again later to measure it would cost more than the
    // culling saves on a small range.
    for (int si = 0; si < 3; si++) {
        const float* o = &tri[si * 5];
        for (int a = 0; a < 3; a++) {
            range.min[a] = std::min(range.min[a], o[a]);
            range.max[a] = std::max(range.max[a], o[a]);
        }
    }
}

// 64-bit FNV-1a, eight bytes at a time. Only ever asked one question -- "is this the same data as last
// frame?" -- so speed matters and cryptographic strength does not; a 64-bit digest makes a false match
// vanishingly unlikely, and the cost of one would be a single frame of stale shadow map.
uint64_t Interpreter::ShadowHashBytes(uint64_t seed, const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = seed;
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t chunk;
        memcpy(&chunk, p + i, sizeof(chunk)); // the lists are float/struct arrays; no alignment assumption
        h = (h ^ chunk) * 0x100000001B3ull;
        h ^= h >> 29;
    }
    for (; i < bytes; i++) {
        h = (h ^ (uint64_t)p[i]) * 0x100000001B3ull;
    }
    return h;
}

// Box the cached world list, one span at a time. Runs once per cache rebuild -- which is once per room, not
// once per frame -- so a single walk of the list here buys a cull test in every cascade of every frame until
// the room changes.
void Interpreter::BuildShadowWorldChunks() {
    mShadowWorldChunks.clear();
    const std::vector<float>& v = mShadowMapWorldCache;
    const size_t total = (v.size() / 9) * 9; // whole triangles only, same rule the draw path uses
    if (total == 0) {
        return;
    }
    const size_t chunkFloats = kShadowChunkTriangles * 9;
    mShadowWorldChunks.reserve((total + chunkFloats - 1) / chunkFloats);
    for (size_t base = 0; base < total; base += chunkFloats) {
        const size_t end = std::min(base + chunkFloats, total);
        ShadowCasterChunk chunk;
        chunk.firstVertex = (uint32_t)(base / 3);
        chunk.vertexCount = (uint32_t)((end - base) / 3);
        for (int a = 0; a < 3; a++) {
            chunk.min[a] = std::numeric_limits<float>::max();
            chunk.max[a] = -std::numeric_limits<float>::max();
        }
        for (size_t i = base; i < end; i += 3) {
            for (int a = 0; a < 3; a++) {
                const float p = v[i + a];
                chunk.min[a] = std::min(chunk.min[a], p);
                chunk.max[a] = std::max(chunk.max[a], p);
            }
        }
        mShadowWorldChunks.push_back(chunk);
    }
}

// What this layer is about to submit, as one value. Two frames that produce the same key would rasterise
// the same depth map, so the backend can leave the slice it already has (see ShadowMapBeginCascade).
//
// Everything the depth pass reads has to be in here. That is: the opaque caster positions, the cutout
// caster positions AND their uvs, and each cutout range's RESOLVED texture id -- the id is what the pass
// binds, and a texture evicted and re-imported between frames changes the picture without moving a single
// vertex. The matrix is compared separately by the backend, which is also what covers the camera moving.
uint64_t Interpreter::ShadowMapLayerContentKey(int layer) const {
    uint64_t h = 0xCBF29CE484222325ull ^ (uint64_t)layer;

    auto mixFloats = [&h](const std::vector<float>& v) {
        const uint64_t n = (uint64_t)v.size();
        h = ShadowHashBytes(h, &n, sizeof(n)); // length first: two lists cannot alias by being prefixes
        if (!v.empty()) {
            h = ShadowHashBytes(h, v.data(), v.size() * sizeof(float));
        }
    };
    auto mixAlphaRanges = [&h](const ShadowAlphaCasters& a) {
        const uint64_t n = (uint64_t)a.ranges.size();
        h = ShadowHashBytes(h, &n, sizeof(n));
        for (const ShadowAlphaRange& r : a.ranges) {
            const uint32_t fields[3] = { r.textureId, r.firstVertex, r.vertexCount };
            h = ShadowHashBytes(h, fields, sizeof(fields));
        }
    };

    if (layer == SHADOW_MAP_LAYER_WORLD) {
        // The room mesh and its cutout half are the cache, and the cache is only ever replaced wholesale --
        // so the generation counter identifies both without walking either. This is the point of the
        // counter: these are by far the largest lists in the frame.
        h = ShadowHashBytes(h, &mShadowWorldCacheGeneration, sizeof(mShadowWorldCacheGeneration));
        mixAlphaRanges(mShadowAlphaWorldCache); // resolved ids still move under a stable cache
        // Scenery actors are rebuilt every frame because they can move, so they are hashed for real.
        mixFloats(mShadowSceneryReady);
        mixFloats(mShadowAlphaSceneryReady.verts);
        mixAlphaRanges(mShadowAlphaSceneryReady);
    } else {
        mixFloats(mShadowMapCastersReady[SHADOW_MAP_LAYER_ACTORS]);
        mixFloats(mShadowAlphaReady[SHADOW_MAP_LAYER_ACTORS].verts);
        mixAlphaRanges(mShadowAlphaReady[SHADOW_MAP_LAYER_ACTORS]);
    }
    return h;
}

void Interpreter::ResolveShadowAlphaTextures(ShadowAlphaCasters& set) {
    for (ShadowAlphaRange& r : set.ranges) {
        TextureCacheMap::iterator it = mTextureCache.map.find(r.key);
        // Unresolved ranges are skipped at draw time rather than drawn untextured. The whole point of this
        // path is to stop foliage casting its bounding quad, so falling back to the opaque draw would
        // reinstate exactly the artefact it exists to remove.
        r.textureId = (it != mTextureCache.map.end()) ? it->second.texture_id : UINT32_MAX;
    }
}

void Interpreter::GfxSpVertex(size_t n_vertices, size_t dest_index, const F3DVtx* vertices) {
    // SOH [Enhancement] Cascaded shadow maps: signature of the world-caster geometry drawn this frame, used to
    // decide whether the cached caster list is still valid (see mShadowMapWorldCache). Every batch that runs
    // inside the bracket folds its source address and size in, order-sensitively, so a different room, a
    // different scene, or a different distance-culled subset all produce a different value. This runs on the
    // vertex-batch path rather than the triangle path on purpose: there are one or two orders of magnitude
    // fewer batches than triangles, and it costs nothing at all outside the bracket.
    if (mShadowMapEnabled && mRdp->shadow_world_caster) {
        uint64_t h = mShadowWorldKeyAccum ^ ((uint64_t)(uintptr_t)vertices + (uint64_t)n_vertices * 0x9E3779B9u);
        h *= 0xFF51AFD7ED558CCDull;
        h ^= h >> 33;
        // Never let the running value land on 0: that is the "no world casters drawn at all this frame"
        // marker, and mistaking a real room for one would drop the cache.
        mShadowWorldKeyAccum = h | 1ull;
    }

    for (size_t i = 0; i < n_vertices; i++, dest_index++) {
        const F3DVtx_t* v = &vertices[i].v;
        const F3DVtx_tn* vn = &vertices[i].n;
        struct LoadedVertex* d = &mRsp->loaded_vertices[dest_index];

        if (v == nullptr) {
            return;
        }

        float x = v->ob[0] * mRsp->MP_matrix[0][0] + v->ob[1] * mRsp->MP_matrix[1][0] +
                  v->ob[2] * mRsp->MP_matrix[2][0] + mRsp->MP_matrix[3][0];
        float y = v->ob[0] * mRsp->MP_matrix[0][1] + v->ob[1] * mRsp->MP_matrix[1][1] +
                  v->ob[2] * mRsp->MP_matrix[2][1] + mRsp->MP_matrix[3][1];
        float z = v->ob[0] * mRsp->MP_matrix[0][2] + v->ob[1] * mRsp->MP_matrix[1][2] +
                  v->ob[2] * mRsp->MP_matrix[2][2] + mRsp->MP_matrix[3][2];
        float w = v->ob[0] * mRsp->MP_matrix[0][3] + v->ob[1] * mRsp->MP_matrix[1][3] +
                  v->ob[2] * mRsp->MP_matrix[2][3] + mRsp->MP_matrix[3][3];

        float world_pos[3] = { 0.0 };
        if (mRsp->geometry_mode & G_LIGHTING_POSITIONAL) {
            float(*mtx)[4] = mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1];
            world_pos[0] = v->ob[0] * mtx[0][0] + v->ob[1] * mtx[1][0] + v->ob[2] * mtx[2][0] + mtx[3][0];
            world_pos[1] = v->ob[0] * mtx[0][1] + v->ob[1] * mtx[1][1] + v->ob[2] * mtx[2][1] + mtx[3][1];
            world_pos[2] = v->ob[0] * mtx[0][2] + v->ob[1] * mtx[1][2] + v->ob[2] * mtx[2][2] + mtx[3][2];
        }

        x = AdjXForAspectRatio(x);

        // SOH [Enhancement] World-space vertex position (object x modelview; the camera lives in the
        // projection matrix, so world pos x P_matrix later yields clip space). Used by the actor-shadow
        // pass, by shadow-map casters, and by shadow-map receivers.
        //
        // OUTSIDE the G_LIGHTING branch below, deliberately. It used to live inside it, which meant unlit
        // geometry kept whatever world position the last LIT object had left in that loaded-vertex slot.
        // A caster captured from a stale slot is recorded wherever some other object happened to be, and
        // which objects fill those slots changes as actors enter and leave the view -- so an unlit caster's
        // shadow moved, appeared and vanished as the camera turned, with nothing in the scene moving.
        // Unlit geometry is not rare here either: tree canopies, billboards and much of the room mesh draw
        // with lighting off, and they all cast.
        if (mRdp->toon || mRdp->toon_shadow || mShadowMapEnabled) {
            float(*mv)[4] = mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1];
            d->wx = v->ob[0] * mv[0][0] + v->ob[1] * mv[1][0] + v->ob[2] * mv[2][0] + mv[3][0];
            d->wy = v->ob[0] * mv[0][1] + v->ob[1] * mv[1][1] + v->ob[2] * mv[2][1] + mv[3][1];
            d->wz = v->ob[0] * mv[0][2] + v->ob[1] * mv[1][2] + v->ob[2] * mv[2][2] + mv[3][2];
        }

        short U = v->tc[0] * mRsp->texture_scaling_factor.s >> 16;
        short V = v->tc[1] * mRsp->texture_scaling_factor.t >> 16;

        if (mRsp->geometry_mode & G_LIGHTING) {
            if (mRsp->lights_changed) {
                for (int i = 0; i < mRsp->current_num_lights - 1; i++) {
                    CalculateNormalDir(&mRsp->current_lights[i].l, mRsp->current_lights_coeffs[i]);
                }
                /*static const Light_t lookat_x = {{0, 0, 0}, 0, {0, 0, 0}, 0, {127, 0, 0}, 0};
                static const Light_t lookat_y = {{0, 0, 0}, 0, {0, 0, 0}, 0, {0, 127, 0}, 0};*/
                CalculateNormalDir(&mRsp->lookat[0], mRsp->current_lookat_coeffs[0]);
                CalculateNormalDir(&mRsp->lookat[1], mRsp->current_lookat_coeffs[1]);
                if (mRdp->toon) { // SOH [Enhancement] toon lighting: cache the dominant light
                    SelectToonLight();
                }
                mRsp->lights_changed = false;
            }

            int r = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[0];
            int g = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[1];
            int b = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[2];

            for (int i = 0; i < mRsp->current_num_lights - 1; i++) {
                float intensity = 0;
                if ((mRsp->geometry_mode & G_LIGHTING_POSITIONAL) && (mRsp->current_lights[i].p.unk3 != 0)) {
                    // Calculate distance from the light to the vertex
                    float dist_vec[3] = { mRsp->current_lights[i].p.pos[0] - world_pos[0],
                                          mRsp->current_lights[i].p.pos[1] - world_pos[1],
                                          mRsp->current_lights[i].p.pos[2] - world_pos[2] };
                    float dist_sq =
                        dist_vec[0] * dist_vec[0] + dist_vec[1] * dist_vec[1] +
                        dist_vec[2] * dist_vec[2] * 2; // The *2 comes from GLideN64, unsure of why it does it
                    float dist = sqrt(dist_sq);

                    // Transform distance vector (which acts as a direction light vector) into model's space
                    float light_model[3];
                    TransposedMatrixMul(light_model, dist_vec,
                                        mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);

                    // Calculate intensity for each axis using standard formula for intensity
                    float light_intensity[3];
                    for (int light_i = 0; light_i < 3; light_i++) {
                        light_intensity[light_i] = 4.0f * light_model[light_i] / dist_sq;
                        light_intensity[light_i] = std::clamp(light_intensity[light_i], -1.0f, 1.0f);
                    }

                    // Adjust intensity based on surface normal and sum up total
                    float total_intensity =
                        light_intensity[0] * vn->n[0] + light_intensity[1] * vn->n[1] + light_intensity[2] * vn->n[2];
                    total_intensity = std::clamp(total_intensity, -1.0f, 1.0f);

                    // Attenuate intensity based on attenuation values.
                    // Example formula found at https://ogldev.org/www/tutorial20/tutorial20.html
                    // Specific coefficients for MM's microcode sourced from GLideN64
                    // https://github.com/gonetz/GLideN64/blob/3b43a13a80dfc2eb6357673440b335e54eaa3896/src/gSP.cpp#L636
                    float distf = floorf(dist);
                    float attenuation = (distf * mRsp->current_lights[i].p.unk7 * 2.0f +
                                         distf * distf * mRsp->current_lights[i].p.unkE / 8.0f) /
                                            (float)0xFFFF +
                                        1.0f;
                    intensity = total_intensity / attenuation;
                } else {
                    intensity += vn->n[0] * mRsp->current_lights_coeffs[i][0];
                    intensity += vn->n[1] * mRsp->current_lights_coeffs[i][1];
                    intensity += vn->n[2] * mRsp->current_lights_coeffs[i][2];
                    intensity /= 127.0f;
                }
                if (intensity > 0.0f) {
                    r += intensity * mRsp->current_lights[i].l.col[0];
                    g += intensity * mRsp->current_lights[i].l.col[1];
                    b += intensity * mRsp->current_lights[i].l.col[2];
                }
            }

            d->color.r = r > 255 ? 255 : r;
            d->color.g = g > 255 ? 255 : g;
            d->color.b = b > 255 ? 255 : b;

            // SOH [Enhancement] Toon lighting: forward the WORLD-space normal to the fragment shader
            // and neutralize the vertex shade so the combiner emits pure albedo. The fragment shader
            // then re-lights it with the single dominant light (also world-space, see SelectToonLight)
            // through the toon ramp.
            //
            // The normal must be transformed object->world here, NOT left in object space: a skeletal
            // actor (Link, NPCs) draws every limb under its own modelview matrix but batches them into
            // a single draw call, while the light direction is one per-batch uniform. Object-space
            // normals would each be in a different limb's space yet share that one uniform, so only one
            // limb could ever be lit correctly. Transforming into world space puts every limb's normal
            // in the same frame as the world-space key, so the single uniform is correct for all limbs.
            // (object->world uses the same row-vector convention as the position transform above; the
            // shader renormalizes, so uniform limb scale is harmless.)
            // SOH [Enhancement] Shadow maps need the world position on RECEIVERS too, and those are the
            // draws with no toon marker at all (the room, the terrain). Without mShadowMapEnabled here the
            // receiver variant would sample the cascades using whatever wx/wy/wz happened to be left in
            // the vertex from an earlier object.
            // Computed for the shadow map too, not only for the relight. The map's normal-offset bias
            // needs a surface normal, and with the cel shading turned off this was never written -- so the
            // shader fell back to recovering one from the world position's screen derivatives. That works
            // on the room mesh, which is what it was written for: large flat triangles, where a pixel quad
            // sits inside one face. On a character it does not. Link's mesh is dense enough that most
            // quads straddle a triangle edge, and the recovered normal there is not a normal at all, so
            // the bias pushed the sample in a different direction every few pixels and the comparison
            // flipped with it -- mottled grey speckle across his skin whenever he stood in shadow.
            if (mRdp->toon || mShadowMapEnabled) {
                float(*mv)[4] = mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1];
                d->nx = vn->n[0] * mv[0][0] + vn->n[1] * mv[1][0] + vn->n[2] * mv[2][0];
                d->ny = vn->n[0] * mv[0][1] + vn->n[1] * mv[1][1] + vn->n[2] * mv[2][1];
                d->nz = vn->n[0] * mv[0][2] + vn->n[1] * mv[1][2] + vn->n[2] * mv[2][2];
            }
            if (mRdp->toon) {
                d->color.r = 255;
                d->color.g = 255;
                d->color.b = 255;
            }

            if (mRsp->geometry_mode & G_TEXTURE_GEN) {
                float dotx = 0, doty = 0;
                dotx += vn->n[0] * mRsp->current_lookat_coeffs[0][0];
                dotx += vn->n[1] * mRsp->current_lookat_coeffs[0][1];
                dotx += vn->n[2] * mRsp->current_lookat_coeffs[0][2];
                doty += vn->n[0] * mRsp->current_lookat_coeffs[1][0];
                doty += vn->n[1] * mRsp->current_lookat_coeffs[1][1];
                doty += vn->n[2] * mRsp->current_lookat_coeffs[1][2];

                dotx /= 127.0f;
                doty /= 127.0f;

                dotx = Ship::Math::clamp(dotx, -1.0f, 1.0f);
                doty = Ship::Math::clamp(doty, -1.0f, 1.0f);

                if (mRsp->geometry_mode & G_TEXTURE_GEN_LINEAR) {
                    // Not sure exactly what formula we should use to get accurate values
                    /*dotx = (2.906921f * dotx * dotx + 1.36114f) * dotx;
                    doty = (2.906921f * doty * doty + 1.36114f) * doty;
                    dotx = (dotx + 1.0f) / 4.0f;
                    doty = (doty + 1.0f) / 4.0f;*/
                    dotx = acosf(-dotx) /* M_PI */ * 0.159155f;
                    doty = acosf(-doty) /* M_PI */ * 0.159155f;
                } else {
                    dotx = (dotx + 1.0f) / 4.0f;
                    doty = (doty + 1.0f) / 4.0f;
                }

                U = (int32_t)(dotx * mRsp->texture_scaling_factor.s);
                V = (int32_t)(doty * mRsp->texture_scaling_factor.t);
            }
        } else {
            d->color.r = v->cn[0];
            d->color.g = v->cn[1];
            d->color.b = v->cn[2];
        }

        d->u = U;
        d->v = V;

        // trivial clip rejection
        d->clip_rej = 0;
        if (x < -w) {
            d->clip_rej |= 1; // CLIP_LEFT
        }
        if (x > w) {
            d->clip_rej |= 2; // CLIP_RIGHT
        }
        if (y < -w) {
            d->clip_rej |= 4; // CLIP_BOTTOM
        }
        if (y > w) {
            d->clip_rej |= 8; // CLIP_TOP
        }
        // if (z < -w) d->clip_rej |= 16; // CLIP_NEAR
        if (z > w) {
            d->clip_rej |= 32; // CLIP_FAR
        }

        d->x = x;
        d->y = y;
        d->z = z;
        d->w = w;

        if (mRsp->geometry_mode & G_FOG) {
            if (fabsf(w) < 0.001f) {
                // To avoid division by zero
                w = 0.001f;
            }

            float winv = 1.0f / w;
            if (winv < 0.0f) {
                winv = std::numeric_limits<int16_t>::max();
            }

            float fog_z = z * winv * mRsp->fog_mul + mRsp->fog_offset;
            fog_z = Ship::Math::clamp(fog_z, 0.0f, 255.0f);
            d->color.a = fog_z; // Use alpha variable to store fog factor
        } else {
            d->color.a = v->cn[3];
        }
    }
}

void Interpreter::GfxSpModifyVertex(uint16_t vtx_idx, uint8_t where, uint32_t val) {
    SUPPORT_CHECK(where == G_MWO_POINT_ST);

    int16_t s = (int16_t)(val >> 16);
    int16_t t = (int16_t)val;

    LoadedVertex* v = &mRsp->loaded_vertices[vtx_idx];
    v->u = s;
    v->v = t;
}

// One captured triangle, nine floats, appended in a single go. Written as a bulk insert rather than nine
// push_backs because this runs per triangle on every caster in the scene: push_back re-checks the capacity
// and re-reads the end pointer each time, and the room mesh alone is tens of thousands of triangles per
// rebuild. Same bytes in the same order either way.
static inline void ShadowAppendTriangle(std::vector<float>& dst, struct LoadedVertex* const v[3]) {
    const float tri[9] = { v[0]->wx, v[0]->wy, v[0]->wz, v[1]->wx, v[1]->wy, v[1]->wz, v[2]->wx, v[2]->wy, v[2]->wz };
    dst.insert(dst.end(), tri, tri + 9);
}

void Interpreter::GfxSpTri1(uint8_t vtx1_idx, uint8_t vtx2_idx, uint8_t vtx3_idx, bool is_rect) {
    struct LoadedVertex* v1 = &mRsp->loaded_vertices[vtx1_idx];
    struct LoadedVertex* v2 = &mRsp->loaded_vertices[vtx2_idx];
    struct LoadedVertex* v3 = &mRsp->loaded_vertices[vtx3_idx];
    struct LoadedVertex* v_arr[3] = { v1, v2, v3 };

    // SOH [Enhancement] Actor shadow: while the shadow pass is armed for this object, record its world-space
    // triangles here (before any culling, so the whole silhouette is captured). FlushToonShadow drains them
    // at the object boundary. is_rect screen-space quads (UI) have no world position, so skip them. The
    // replayed shadow geometry itself runs with toon_shadow cleared, so it is never re-captured. NOTE: this
    // is gated on toon_shadow only (NOT mRdp->toon), so shadows work even when the cel relight is disabled.
    // SOH [Enhancement] Cascaded shadow maps: alpha-cutout materials take a separate path in both layers.
    // Resolved here, before either capture, because the answer and the texture key come from RDP state that
    // the combiner setup further down would have moved on from.
    // One test in front of the whole capture block, because this is the hottest function in the renderer and
    // every triangle in the game passes through it -- including the overwhelming majority of frames and
    // draws where no shadow system is armed at all. Nothing below can do anything without one of these
    // three: the stencil path needs toon_shadow, and all three shadow-map paths need the shadow map on plus
    // one of its brackets open. Everything the block computes is derived state with no side effects, so
    // skipping it is exactly equivalent to running it and reaching no branch.
    const bool shadowCaptureActive =
        mRdp->toon_shadow || (mShadowMapEnabled && (mRdp->shadow_world_caster || mRdp->shadow_scenery_caster));
    if (shadowCaptureActive) {
        TextureCacheKey shadowAlphaKey{};
        bool shadowAlphaCaster = false;
        float shadowTexW = 1.0f, shadowTexH = 1.0f;
        // Geometry the shadow map must never record at all (decals, translucent overlays). Checked before the
        // cutout question, because a decal that happens to be alpha-tested is still a decal.
        const bool shadowArmed =
            !mRdp->shadow_no_cast && (mRdp->toon_shadow || mRdp->shadow_world_caster || mRdp->shadow_scenery_caster);
        const bool shadowCasterExcluded = mShadowMapEnabled && shadowArmed && ShadowCasterExcludedByRenderMode();
        if (mShadowMapEnabled && mShadowAlphaSupported && !shadowCasterExcluded && !is_rect && shadowArmed) {
            // mShadowAlphaSupported: when the backend could not build its cutout pipeline there is nowhere for
            // this geometry to go, and diverting it there anyway would mean foliage casts NOTHING rather than
            // casting its quad. Falling back to the opaque list is worse looking and strictly better than a
            // missing shadow.
            shadowAlphaCaster = ShadowCasterIsAlphaTested(mRdp->first_tile_index, &shadowAlphaKey);
            if (shadowAlphaCaster) {
                ShadowCasterTexSize(mRdp->first_tile_index, &shadowTexW, &shadowTexH);
            }
        }

        // The stencil volumes need a lit surface (they cast along a per-vertex-normal key), but a shadow MAP
        // only needs the geometry: unlit geometry blocks light exactly as well as lit geometry does. Tree
        // canopies, billboards and most scenery props draw with lighting off, and requiring it here is why a
        // tree cast from its trunk and not from its leaves. The requirement stays exactly as it was whenever
        // the shadow map is off, so the stencil mode is untouched.
        const bool armedCaster = mRdp->toon_shadow && !is_rect;
        const bool casterLit = (mRsp->geometry_mode & G_LIGHTING) != 0;
        if (armedCaster && (mShadowMapEnabled ? !shadowCasterExcluded : casterLit)) {
            // Every armed triangle grows the object's bounding box, whichever list it lands in. The box is what
            // the size gate in FlushToonShadow judges the object by, so measuring only the opaque half would
            // shrink a mostly-cutout actor below the threshold and drop its whole shadow -- and which half of a
            // skeletal actor is cutout changes with the animation, which is a shadow that flickers as it walks.
            for (int si = 0; si < 3; si++) {
                const float p[3] = { v_arr[si]->wx, v_arr[si]->wy, v_arr[si]->wz };
                for (int a = 0; a < 3; a++) {
                    if (!mShadowObjectHasVerts || p[a] < mShadowObjectMin[a]) {
                        mShadowObjectMin[a] = p[a];
                    }
                    if (!mShadowObjectHasVerts || p[a] > mShadowObjectMax[a]) {
                        mShadowObjectMax[a] = p[a];
                    }
                }
                mShadowObjectHasVerts = true;
            }
            if (shadowAlphaCaster) {
                CaptureShadowAlphaTriangle(SHADOW_MAP_LAYER_ACTORS, shadowAlphaKey, v_arr, shadowTexW, shadowTexH);
            } else if (shadowCasterExcluded) {
                // Nothing: not a caster, and the stencil path is not running (see ShadowCasterExcludedByRenderMode).
            } else if (mShadowMapEnabled || casterLit) {
                // Staging for whichever shadow system is on. Both consume it at the object boundary rather than
                // here: the stencil volumes need the whole silhouette before they can build one, and the shadow
                // map needs the object's bounding box before it can decide the object is worth casting at all
                // (see FlushToonShadow).
                ShadowAppendTriangle(mShadowVerts, v_arr);
            }
        } else if (mShadowMapEnabled && mRdp->shadow_scenery_caster && !is_rect && !shadowCasterExcluded) {
            // Scenery the game spawns as an actor: a gate, a fence, a tree. It is drawn into the WORLD layer,
            // because everything samples that layer and a gate's shadow belongs on the player the way a wall's
            // does -- but into its own per-frame list rather than the cache beside the room mesh, because the
            // cache notices when geometry CHANGES and not when it MOVES, and the castle gate slides open.
            //
            // Ungated by mShadowWorldCapture for the same reason: that flag exists to skip re-walking a cached
            // list, and there is no cache here to skip.
            if (shadowAlphaCaster) {
                CaptureShadowAlphaTriangle(SHADOW_MAP_LAYER_WORLD, shadowAlphaKey, v_arr, shadowTexW, shadowTexH,
                                           &mShadowAlphaScenery);
            } else if (mShadowSceneryCasters.size() < kShadowMapCasterBudgetFloats) {
                ShadowAppendTriangle(mShadowSceneryCasters, v_arr);
            }
        } else if (mShadowMapEnabled && mRdp->shadow_world_caster && mShadowWorldCapture && !is_rect &&
                   !shadowCasterExcluded && shadowAlphaCaster) {
            CaptureShadowAlphaTriangle(SHADOW_MAP_LAYER_WORLD, shadowAlphaKey, v_arr, shadowTexW, shadowTexH);
        } else if (mShadowMapEnabled && mRdp->shadow_world_caster && mShadowWorldCapture && !is_rect &&
                   !shadowCasterExcluded &&
                   mShadowMapCasters[SHADOW_MAP_LAYER_WORLD].size() < kShadowMapCasterBudgetFloats) {
            // SOH [Enhancement] Cascaded shadow maps: world geometry inside a gSPShadowMapWorldCaster bracket.
            // This is what lets the scene shadow itself. No G_LIGHTING requirement, unlike the stencil path --
            // the room mesh is often drawn unlit, and a wall still blocks light whether or not it is being shaded.
            // mShadowWorldCapture gates this to the frames that actually rebuild the cache; on every other frame
            // the room mesh costs nothing here and the cached list is reused as-is.
            //
            // Scenery actors arrive here too, through the same bracket (gSPShadowMapSceneryCasterBegin). A tree
            // belongs in this layer and not the actor one: everything samples this layer, so its shadow lands on
            // the player standing under it, which the actor layer -- the one characters skip so they cannot
            // shadow each other -- can never do. Caching them alongside the room mesh is right for the same
            // reason it is right for the room: they do not move.
            ShadowAppendTriangle(mShadowMapCasters[SHADOW_MAP_LAYER_WORLD], v_arr);
        }
    }

    // if (rand()%2) return;

    if (v1->clip_rej & v2->clip_rej & v3->clip_rej) {
        // The whole triangle lies outside the visible area
        return;
    }

    const uint32_t cull_both = get_attr(CULL_BOTH);
    const uint32_t cull_front = get_attr(CULL_FRONT);
    const uint32_t cull_back = get_attr(CULL_BACK);

    if ((mRsp->geometry_mode & cull_both) != 0) {
        float dx1 = v1->x / (v1->w) - v2->x / (v2->w);
        float dy1 = v1->y / (v1->w) - v2->y / (v2->w);
        float dx2 = v3->x / (v3->w) - v2->x / (v2->w);
        float dy2 = v3->y / (v3->w) - v2->y / (v2->w);
        float cross = dx1 * dy2 - dy1 * dx2;

        if ((v1->w < 0) ^ (v2->w < 0) ^ (v3->w < 0)) {
            // If one vertex lies behind the eye, negating cross will give the correct result.
            // If all vertices lie behind the eye, the triangle will be rejected anyway.
            cross = -cross;
        }

        // If inverted culling is requested, negate the cross
        if (ucode_handler_index == UcodeHandlers::ucode_f3dex2 &&
            (mRsp->extra_geometry_mode & G_EX_INVERT_CULLING) == 1) {
            cross = -cross;
        }

        auto cull_type = mRsp->geometry_mode & cull_both;

        if (cull_type == cull_front) {
            if (cross <= 0) {
                return;
            }
        } else if (cull_type == cull_back) {
            if (cross >= 0) {
                return;
            }
        } else if (cull_type == cull_both) {
            // Why is this even an option?
            return;
        }
    }

    bool depth_test = (mRsp->geometry_mode & G_ZBUFFER) == G_ZBUFFER;
    bool depth_mask = (mRdp->other_mode_l & Z_UPD) == Z_UPD;
    uint8_t depth_test_and_mask = (depth_test ? 1 : 0) | (depth_mask ? 2 : 0);
    if (depth_test_and_mask != mRenderingState.depth_test_and_mask) {
        Flush();
        mRapi->SetDepthTestAndMask(depth_test, depth_mask);
        mRenderingState.depth_test_and_mask = depth_test_and_mask;
    }

    bool zmode_decal = (mRdp->other_mode_l & ZMODE_DEC) == ZMODE_DEC;
    if (zmode_decal != mRenderingState.decal_mode) {
        Flush();
        mRapi->SetZmodeDecal(zmode_decal);
        mRenderingState.decal_mode = zmode_decal;
    }

    if (mRdp->viewport_or_scissor_changed) {
        if (memcmp(&mRdp->viewport, &mRenderingState.viewport, sizeof(mRdp->viewport)) != 0) {
            Flush();
            mRapi->SetViewport(mRdp->viewport.x, mRdp->viewport.y, mRdp->viewport.width, mRdp->viewport.height);
            mRenderingState.viewport = mRdp->viewport;
        }
        if (memcmp(&mRdp->scissor, &mRenderingState.scissor, sizeof(mRdp->scissor)) != 0) {
            Flush();
            mRapi->SetScissor(mRdp->scissor.x, mRdp->scissor.y, mRdp->scissor.width, mRdp->scissor.height);
            mRenderingState.scissor = mRdp->scissor;
        }
        mRdp->viewport_or_scissor_changed = false;
    }

    uint64_t cc_id = mRdp->combine_mode;
    uint64_t cc_options = 0;
    bool use_alpha = ((mRdp->other_mode_l & (3 << 20)) == (G_BL_CLR_MEM << 20) &&
                      (mRdp->other_mode_l & (3 << 16)) == (G_BL_1MA << 16)) ||
                     ((mRdp->other_mode_l & (3 << 22)) == (G_BL_CLR_MEM << 22) &&
                      (mRdp->other_mode_l & (3 << 18)) == (G_BL_1MA << 18));
    bool use_fog = (mRdp->other_mode_l >> 30) == G_BL_CLR_FOG;
    bool texture_edge = (mRdp->other_mode_l & CVG_X_ALPHA) == CVG_X_ALPHA;
    bool use_noise = (mRdp->other_mode_l & (3U << G_MDSFT_ALPHACOMPARE)) == G_AC_DITHER;
    bool use_2cyc = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_2CYCLE;
    bool alpha_threshold = (mRdp->other_mode_l & (3U << G_MDSFT_ALPHACOMPARE)) == G_AC_THRESHOLD;
    bool invisible =
        (mRdp->other_mode_l & (3 << 24)) == (G_BL_0 << 24) && (mRdp->other_mode_l & (3 << 20)) == (G_BL_CLR_MEM << 20);
    bool use_grayscale = mRdp->grayscale;
    // SOH [Enhancement] Toon lighting only applies to lit geometry (where vertex normals exist).
    bool use_toon = mRdp->toon && (mRsp->geometry_mode & G_LIGHTING);
    // SOH [Enhancement] Cascaded shadow maps: which draws RECEIVE shadow. Everything with a world position
    // does -- including the actors that also cast, so characters are shadowed by the scenery around them.
    //
    // Actors used to be excluded here to keep them from sampling the very depth map they drew into. That
    // did prevent self-shadowing, but it also meant a character standing in a building's shadow stayed
    // lit, which is worse: being shadowed by the world is the more visible of the two behaviours by far.
    // The proper fix is two sets of maps -- world-only for actors to sample, world-plus-actors for the
    // world -- which is a bigger change than one flag. Until then actors do self-shadow, which is correct
    // in principle (an arm should shade the torso) and relies on the normal-offset bias to stay clean;
    // actors carry vertex normals, so that bias actually applies to them, unlike the room mesh.
    //
    // Screen-space geometry (UI, backgrounds) stays out: it has no world position to look up with.
    //
    // is_rect alone did not cover it. That flag is G_TEXRECT, and most of this game's interface is not drawn
    // as texrects -- the file-select boxes, the item buttons, the hearts are ordinary quads under a 2D
    // projection. They were being handed the receiver variant, and the "world position" they carried was the
    // vertex times a modelview that maps into SCREEN space: a button at (100, 80) on screen claims to be at
    // world (100, 80, 0). Usually that lands outside every cascade and reads as lit, which costs a
    // projection per pixel and shows nothing -- but it is not always outside. Stand where a cascade covers
    // small world coordinates and the interface starts sampling the shadow map and darkening with it.
    //
    // The test is whether w depends on z, which is what a perspective divide IS. The 3D scene is drawn
    // through guPerspective, where that term is the whole point; the interface is drawn through guOrtho,
    // where it is exactly zero. Reading it off the projection catches every screen-space draw at once
    // instead of naming them one at a time, and it cannot mistake a scene draw for one: a projection with no
    // perspective term has no perspective to lose.
    const bool screenSpaceProjection = std::fabs(mRsp->P_matrix[2][3]) < 1e-6f;
    bool use_shadow_map = mShadowMapEnabled && !is_rect && !screenSpaceProjection && !mRdp->shadow_no_receive;
    // Scenery samples both caster layers; a character samples only the world layer, which is what keeps
    // characters from shadowing each other (or themselves) while still being shadowed by the world.
    bool use_shadow_map_actors = use_shadow_map && !mRdp->toon_shadow;
    auto shader = mRdp->current_shader;

    if (texture_edge) {
        if (use_alpha) {
            alpha_threshold = true;
            texture_edge = false;
        }
        use_alpha = true;
    }

    if (use_alpha) {
        cc_options |= SHADER_OPT(ALPHA);
    }
    if (use_fog) {
        cc_options |= SHADER_OPT(FOG);
    }
    if (texture_edge) {
        cc_options |= SHADER_OPT(TEXTURE_EDGE);
    }
    if (use_noise) {
        cc_options |= SHADER_OPT(NOISE);
    }
    if (use_2cyc) {
        cc_options |= SHADER_OPT(_2CYC);
    }
    if (alpha_threshold) {
        cc_options |= SHADER_OPT(ALPHA_THRESHOLD);
    }
    if (invisible) {
        cc_options |= SHADER_OPT(INVISIBLE);
    }
    if (use_grayscale) {
        cc_options |= SHADER_OPT(GRAYSCALE);
    }
    if (use_toon) {
        cc_options |= SHADER_OPT(TOON);
    }
    if (use_shadow_map) {
        cc_options |= SHADER_OPT(SHADOW_MAP);
    }
    if (mRdp->loaded_texture[0].masked) {
        cc_options |= SHADER_OPT(TEXEL0_MASK);
    }
    if (mRdp->loaded_texture[1].masked) {
        cc_options |= SHADER_OPT(TEXEL1_MASK);
    }
    if (mRdp->loaded_texture[0].blended) {
        cc_options |= SHADER_OPT(TEXEL0_BLEND);
    }
    if (mRdp->loaded_texture[1].blended) {
        cc_options |= SHADER_OPT(TEXEL1_BLEND);
    }
    if (shader.enabled) {
        cc_options |= SHADER_OPT(USE_SHADER);
        // SOH [Enhancement] shader.id packs above the option bits; shifted 17->18 for TOON, 18->19 for
        // SHADOW_MAP. Keep in lockstep with the decode in gfx_cc_get_features -- a mismatch selects the
        // wrong shader for every draw in the game.
        cc_options |= (shader.id << 19);
    }

    ColorCombinerKey key;
    key.combine_mode = mRdp->combine_mode;
    key.options = cc_options;

    // If we are not using alpha, clear the alpha components of the combiner as they have no effect
    if (!use_alpha && !shader.enabled) {
        key.combine_mode &= ~((0xfff << 16) | ((uint64_t)0xfff << 44));
    }

    ColorCombiner* comb = LookupOrCreateColorCombiner(key);

    uint32_t tm = 0;
    uint32_t tex_width[2], tex_height[2], tex_width2[2], tex_height2[2];

    for (int i = 0; i < 2; i++) {
        uint32_t tile = mRdp->first_tile_index + i;
        if (comb->usedTextures[i]) {
            if (mRdp->textures_changed[i]) {
                Flush();
                ImportTexture(i, tile, false);
                if (mRdp->loaded_texture[i].masked) {
                    ImportTextureMask(SHADER_FIRST_MASK_TEXTURE + i, tile);
                }
                if (mRdp->loaded_texture[i].blended) {
                    ImportTexture(SHADER_FIRST_REPLACEMENT_TEXTURE + i, tile, true);
                }
                mRdp->textures_changed[i] = false;
            }

            uint8_t cms = mRdp->texture_tile[tile].cms;
            uint8_t cmt = mRdp->texture_tile[tile].cmt;

            uint32_t tex_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;
            uint32_t line_size = mRdp->texture_tile[tile].line_size_bytes;

            if (line_size == 0) {
                line_size = 1;
            }

            tex_height[i] = tex_size_bytes / line_size;
            switch (mRdp->texture_tile[tile].siz) {
                case G_IM_SIZ_4b:
                    line_size <<= 1;
                    break;
                case G_IM_SIZ_8b:
                    break;
                case G_IM_SIZ_16b:
                    line_size /= G_IM_SIZ_16b_LINE_BYTES;
                    break;
                case G_IM_SIZ_32b:
                    line_size /= G_IM_SIZ_32b_LINE_BYTES; // this is 2!
                    tex_height[i] /= 2;
                    break;
            }
            tex_width[i] = line_size;

            tex_width2[i] = (mRdp->texture_tile[tile].lrs - mRdp->texture_tile[tile].uls + 4) / 4;
            tex_height2[i] = (mRdp->texture_tile[tile].lrt - mRdp->texture_tile[tile].ult + 4) / 4;

            uint32_t tex_width1 = tex_width[i] << (cms & G_TX_MIRROR);
            uint32_t tex_height1 = tex_height[i] << (cmt & G_TX_MIRROR);

            if ((cms & G_TX_CLAMP) && ((cms & G_TX_MIRROR) || tex_width1 != tex_width2[i])) {
                tm |= 1 << 2 * i;
                cms &= ~G_TX_CLAMP;
            }
            if ((cmt & G_TX_CLAMP) && ((cmt & G_TX_MIRROR) || tex_height1 != tex_height2[i])) {
                tm |= 1 << 2 * i + 1;
                cmt &= ~G_TX_CLAMP;
            }

            if (mRenderingState.mTextures[i] == nullptr) {
                continue;
            }

            bool linear_filter = (mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT;
            if (linear_filter != mRenderingState.mTextures[i]->second.linear_filter ||
                cms != mRenderingState.mTextures[i]->second.cms || cmt != mRenderingState.mTextures[i]->second.cmt) {
                Flush();

                // Set the same sampler params on the blended texture. Needed for opengl.
                if (mRdp->loaded_texture[i].blended) {
                    mRapi->SetSamplerParameters(SHADER_FIRST_REPLACEMENT_TEXTURE + i, linear_filter, cms, cmt);
                }

                mRapi->SetSamplerParameters(i, linear_filter, cms, cmt);
                mRenderingState.mTextures[i]->second.linear_filter = linear_filter;
                mRenderingState.mTextures[i]->second.cms = cms;
                mRenderingState.mTextures[i]->second.cmt = cmt;
            }
        }
    }

    struct ShaderProgram* prg = comb->prg[tm];
    if (prg == NULL) {
        comb->prg[tm] = prg =
            LookupOrCreateShaderProgram(comb->shader_id0, comb->shader_id1 | tm * SHADER_OPT(TEXEL0_CLAMP_S));
    }
    if (prg != mRenderingState.mShaderProgram) {
        Flush();
        mRapi->UnloadShader(mRenderingState.mShaderProgram);
        mRapi->LoadShader(prg);
        mRenderingState.mShaderProgram = prg;
    }
    if (use_alpha != mRenderingState.alpha_blend) {
        Flush();
        mRapi->SetUseAlpha(use_alpha);
        mRenderingState.alpha_blend = use_alpha;
    }
    uint8_t numInputs;
    bool usedTextures[2];

    mRapi->ShaderGetInfo(prg, &numInputs, usedTextures);

    struct GfxClipParameters clip_parameters = mRapi->GetClipParameters();

    for (int i = 0; i < 3; i++) {
        float z = v_arr[i]->z, w = v_arr[i]->w;
        if (clip_parameters.z_is_from_0_to_1) {
            z = (z + w) / 2.0f;
        }

        mBufVbo[mBufVboLen++] = v_arr[i]->x;
        mBufVbo[mBufVboLen++] = clip_parameters.invertY ? -v_arr[i]->y : v_arr[i]->y;
        mBufVbo[mBufVboLen++] = z;
        mBufVbo[mBufVboLen++] = w;

        for (int t = 0; t < 2; t++) {
            if (!usedTextures[t]) {
                continue;
            }
            float u = v_arr[i]->u / 32.0f;
            float v = v_arr[i]->v / 32.0f;

            int shifts = mRdp->texture_tile[mRdp->first_tile_index + t].shifts;
            int shiftt = mRdp->texture_tile[mRdp->first_tile_index + t].shiftt;
            if (shifts != 0) {
                if (shifts <= 10) {
                    u /= 1 << shifts;
                } else {
                    u *= 1 << (16 - shifts);
                }
            }
            if (shiftt != 0) {
                if (shiftt <= 10) {
                    v /= 1 << shiftt;
                } else {
                    v *= 1 << (16 - shiftt);
                }
            }

            u -= mRdp->texture_tile[mRdp->first_tile_index + t].uls / 4.0f;
            v -= mRdp->texture_tile[mRdp->first_tile_index + t].ult / 4.0f;

            if ((mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT) {
                // Linear filter adds 0.5f to the coordinates
                if (!is_rect) {
                    u += 0.5f;
                    v += 0.5f;
                }
            }

            mBufVbo[mBufVboLen++] = u / tex_width[t];
            mBufVbo[mBufVboLen++] = v / tex_height[t];

            bool clampS = tm & (1 << 2 * t);
            bool clampT = tm & (1 << 2 * t + 1);

            if (clampS) {
                mBufVbo[mBufVboLen++] = (tex_width2[t] - 0.5f) / tex_width[t];
            }

            if (clampT) {
                mBufVbo[mBufVboLen++] = (tex_height2[t] - 0.5f) / tex_height[t];
            }
        }

        if (use_fog) {
            mBufVbo[mBufVboLen++] = mRdp->fog_color.r / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->fog_color.g / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->fog_color.b / 255.0f;
            mBufVbo[mBufVboLen++] = v_arr[i]->color.a / 255.0f; // fog factor (not alpha)
        }

        if (use_grayscale) {
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.r / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.g / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.b / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.a / 255.0f; // lerp interpolation factor (not alpha)
        }

        // SOH [Enhancement] Toon lighting: world-space normal (aNormal). The dominant light/ambient
        // are sent as uniforms (per draw), not per-vertex, to stay within the vertex-attribute limit.
        // SOH [Enhancement] Cascaded shadow maps want it too, for the normal-offset bias, and want it
        // whether or not the cel relight is on -- so the attribute rides both options. It costs no extra
        // headroom: a draw with both on already carried it, so the widest stride is unchanged.
        //
        // Sent as ZERO when the draw carries no vertex normal, which is the room mesh and most scenery:
        // that is not a normal pointing nowhere, it is the shader's signal to recover one from the world
        // position's screen derivatives instead. Zeroing here rather than trusting the vertex is the whole
        // point -- nx/ny/nz are only written on lit geometry, so an unlit draw would otherwise ship
        // whatever the last lit object happened to leave in the vertex slot.
        if (use_toon || use_shadow_map) {
            const bool haveNormal = (mRsp->geometry_mode & G_LIGHTING) != 0;
            mBufVbo[mBufVboLen++] = haveNormal ? v_arr[i]->nx : 0.0f;
            mBufVbo[mBufVboLen++] = haveNormal ? v_arr[i]->ny : 0.0f;
            mBufVbo[mBufVboLen++] = haveNormal ? v_arr[i]->nz : 0.0f;
        }

        // SOH [Enhancement] Cascaded shadow maps: world position (aWorldPos), so the pixel shader can
        // project into each cascade. The cascade matrices are uniforms, not per-vertex. Order here must
        // match the input-layout element order in every backend and the PSInput field order in the shader
        // -- these three are one implicit contract, and a mismatch shifts every attribute after it.
        if (use_shadow_map) {
            mBufVbo[mBufVboLen++] = v_arr[i]->wx;
            mBufVbo[mBufVboLen++] = v_arr[i]->wy;
            mBufVbo[mBufVboLen++] = v_arr[i]->wz;
            // Receiver kind rides in w: 1 = scenery, which also samples the actor caster layer, 0 = a
            // character, which must not. Carried per vertex rather than as a shader option because each
            // option doubles the shader variants, and every new variant is compiled mid-frame -- felt as
            // the game hitching the first time shadows appear.
            mBufVbo[mBufVboLen++] = use_shadow_map_actors ? 1.0f : 0.0f;
        }

        for (int j = 0; j < numInputs; j++) {
            RGBA* color;
            RGBA tmp;
            for (int k = 0; k < 1 + (use_alpha ? 1 : 0); k++) {
                switch (comb->shader_input_mapping[k][j]) {
                        // Note: CCMUX constants and ACMUX constants used here have same value, which is why this works
                        // (except LOD fraction).
                    case G_CCMUX_PRIMITIVE:
                        color = &mRdp->prim_color;
                        break;
                    case G_CCMUX_SHADE:
                        color = &v_arr[i]->color;
                        break;
                    case G_CCMUX_ENVIRONMENT:
                        color = &mRdp->env_color;
                        break;
                    case G_CCMUX_PRIMITIVE_ALPHA: {
                        tmp.r = tmp.g = tmp.b = mRdp->prim_color.a;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_ENV_ALPHA: {
                        tmp.r = tmp.g = tmp.b = mRdp->env_color.a;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_PRIM_LOD_FRAC: {
                        tmp.r = tmp.g = tmp.b = mRdp->prim_lod_fraction;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_LOD_FRACTION: {
                        if (mRdp->other_mode_l & G_TL_LOD) {
                            // "Hack" that works for Bowser - Peach painting
                            float distance_frac = (v1->w - 3000.0f) / 3000.0f;
                            if (distance_frac < 0.0f) {
                                distance_frac = 0.0f;
                            }
                            if (distance_frac > 1.0f) {
                                distance_frac = 1.0f;
                            }
                            tmp.r = tmp.g = tmp.b = tmp.a = distance_frac * 255.0f;
                        } else {
                            tmp.r = tmp.g = tmp.b = tmp.a = 255.0f;
                        }
                        color = &tmp;
                        break;
                    }
                    case G_ACMUX_PRIM_LOD_FRAC:
                        tmp.a = mRdp->prim_lod_fraction;
                        color = &tmp;
                        break;
                    default:
                        memset(&tmp, 0, sizeof(tmp));
                        color = &tmp;
                        break;
                }
                if (k == 0) {
                    mBufVbo[mBufVboLen++] = color->r / 255.0f;
                    mBufVbo[mBufVboLen++] = color->g / 255.0f;
                    mBufVbo[mBufVboLen++] = color->b / 255.0f;
                } else {
                    if (use_fog && color == &v_arr[i]->color) {
                        // Shade alpha is 100% for fog
                        mBufVbo[mBufVboLen++] = 1.0f;
                    } else {
                        mBufVbo[mBufVboLen++] = color->a / 255.0f;
                    }
                }
            }
        }

        // struct RGBA *color = &v_arr[i]->color;
        // mBufVbo[mBufVboLen++] = color->r / 255.0f;
        // mBufVbo[mBufVboLen++] = color->g / 255.0f;
        // mBufVbo[mBufVboLen++] = color->b / 255.0f;
        // mBufVbo[mBufVboLen++] = color->a / 255.0f;
    }

    if (++mBufVboNumTris == MAX_TRI_BUFFER) {
        // if (++mBufVbo_num_tris == 1) {
        Flush();
    }
}

void Interpreter::GfxSpGeometryMode(uint32_t clear, uint32_t set) {
    mRsp->geometry_mode &= ~clear;
    mRsp->geometry_mode |= set;
}

void Interpreter::GfxSpExtraGeometryMode(uint32_t clear, uint32_t set) {
    mRsp->extra_geometry_mode &= ~clear;
    mRsp->extra_geometry_mode |= set;
}

void Interpreter::AdjustVIewportOrScissor(XYWidthHeight* area) {
    if (!mFbActive) {
        // Adjust the y origin based on the y-inversion for the active framebuffer
        GfxClipParameters clipParameters = mRapi->GetClipParameters();
        if (clipParameters.invertY) {
            area->y -= area->height;
        } else {
            area->y = mNativeDimensions.height - area->y;
        }

        area->width *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
        area->height *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
        area->x *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
        area->y *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);

        if (!mRendersToFb || (mMsaaLevel > 1 && mCurDimensions.width == mGameWindowViewport.width &&
                              mCurDimensions.height == mGameWindowViewport.height)) {
            area->x += mGameWindowViewport.x;
            area->y += mGfxCurrentWindowDimensions.height - (mGameWindowViewport.y + mGameWindowViewport.height);
        }
    } else {
        area->y = mActiveFrameBuffer->second.orig_height - area->y;

        if (mActiveFrameBuffer->second.resize) {
            area->width *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
            area->height *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
            area->x *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
            area->y *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
        }
    }
}

void Interpreter::CalcAndSetViewport(const F3DVp_t* viewport) {
    // 2 bits fraction
    float width = 2.0f * viewport->vscale[0] / 4.0f;
    float height = 2.0f * viewport->vscale[1] / 4.0f;
    float x = (viewport->vtrans[0] / 4.0f) - width / 2.0f;
    float y = ((viewport->vtrans[1] / 4.0f) + height / 2.0f);

    mRdp->viewport.x = x;
    mRdp->viewport.y = y;
    mRdp->viewport.width = width;
    mRdp->viewport.height = height;

    AdjustVIewportOrScissor(&mRdp->viewport);

    mRdp->viewport_or_scissor_changed = true;
}

void Interpreter::GfxSpMovememF3dex2(uint8_t index, uint8_t offset, const void* data) {
    switch (index) {
        case F3DEX2_G_MV_VIEWPORT:
            CalcAndSetViewport((const F3DVp_t*)data);
            break;
        case F3DEX2_G_MV_LIGHT: {
            int lightidx = offset / 24 - 2;
            if (lightidx >= 0 && lightidx <= MAX_LIGHTS) { // skip lookat
                // NOTE: reads out of bounds if it is an ambient light
                memcpy(mRsp->current_lights + lightidx, data, sizeof(F3DLight));
            } else if (lightidx < 0) {
                memcpy(mRsp->lookat + offset / 24, data, sizeof(F3DLight_t)); // TODO Light?
            }
            break;
        }
    }
}

void Interpreter::GfxSpMovememF3d(uint8_t index, uint8_t offset, const void* data) {
    switch (index) {
        case F3DEX_G_MV_VIEWPORT:
            CalcAndSetViewport((const F3DVp_t*)data);
            break;
        case F3DEX_G_MV_LOOKATY:
        case F3DEX_G_MV_LOOKATX:
            memcpy(mRsp->lookat + (index - F3DEX_G_MV_LOOKATY) / 2, data, sizeof(F3DLight_t));
            break;
        case F3DEX_G_MV_L0:
        case F3DEX_G_MV_L1:
        case F3DEX_G_MV_L2:
        case F3DEX_G_MV_L3:
        case F3DEX_G_MV_L4:
        case F3DEX_G_MV_L5:
        case F3DEX_G_MV_L6:
        case F3DEX_G_MV_L7:
            // NOTE: reads out of bounds if it is an ambient light
            memcpy(mRsp->current_lights + (index - F3DEX_G_MV_L0) / 2, data, sizeof(F3DLight_t));
            break;
    }
}

void Interpreter::GfxSpMovewordF3dex2(uint8_t index, uint16_t offset, uintptr_t data) {
    switch (index) {
        case G_MW_NUMLIGHT:
            mRsp->current_num_lights = data / 24 + 1; // add ambient light
            mRsp->lights_changed = true;
            break;
        case G_MW_FOG:
            mRsp->fog_mul = (int16_t)(data >> 16);
            mRsp->fog_offset = (int16_t)data;
            break;
        case G_MW_SEGMENT: {
            int segNumber = offset / 4;
            mSegmentPointers[segNumber] = data;
        } break;
        case G_MW_SEGMENT_INTERP: {
            int segNumber = offset % 16;
            int segIndex = offset / 16;

            if (segIndex == mInterpolationIndex)
                mSegmentPointers[segNumber] = data;
        } break;
    }
}

void Interpreter::GfxSpMovewordF3d(uint8_t index, uint16_t offset, uintptr_t data) {
    switch (index) {
        case G_MW_NUMLIGHT:
            // Ambient light is included
            // The 31st bit is a flag that lights should be recalculated
            mRsp->current_num_lights = (data - 0x80000000U) / 32;
            mRsp->lights_changed = true;
            break;
        case G_MW_FOG:
            mRsp->fog_mul = (int16_t)(data >> 16);
            mRsp->fog_offset = (int16_t)data;
            break;
        case G_MW_SEGMENT: {
            int segNumber = offset / 4;
            mSegmentPointers[segNumber] = data;
        } break;
        case G_MW_SEGMENT_INTERP: {
            int segNumber = offset % 16;
            int segIndex = offset / 16;

            if (segIndex == mInterpolationIndex)
                mSegmentPointers[segNumber] = data;
        } break;
    }
}

void Interpreter::GfxSpTexture(uint16_t sc, uint16_t tc, uint8_t level, uint8_t tile, uint8_t on) {
    mRsp->texture_scaling_factor.s = sc;
    mRsp->texture_scaling_factor.t = tc;
    if (mRdp->first_tile_index != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }

    mRdp->first_tile_index = tile;
}

void Interpreter::GfxDpSetScissor(uint32_t mode, uint32_t ulx, uint32_t uly, uint32_t lrx, uint32_t lry) {
    float x = ulx / 4.0f;
    float y = lry / 4.0f;
    float width = (lrx - ulx) / 4.0f;
    float height = (lry - uly) / 4.0f;

    mRdp->scissor.x = x;
    mRdp->scissor.y = y;
    mRdp->scissor.width = width;
    mRdp->scissor.height = height;

    AdjustVIewportOrScissor(&mRdp->scissor);

    mRdp->viewport_or_scissor_changed = true;
}

void Interpreter::GfxDpSetTextureImage(uint32_t format, uint32_t size, uint32_t width, const char* texPath,
                                       uint32_t texFlags, RawTexMetadata rawTexMetdata, const void* addr) {
    // fprintf(stderr, "GfxDpSetTextureImage: %s (width=%d; size=0x%X)\n",
    //         rawTexMetdata.resource ? rawTexMetdata.resource->GetInitData()->Path.c_str() : nullptr, width, size);
    mRdp->texture_to_load.addr = (const uint8_t*)addr;
    mRdp->texture_to_load.siz = size;
    mRdp->texture_to_load.width = width;
    mRdp->texture_to_load.tex_flags = texFlags;
    mRdp->texture_to_load.raw_tex_metadata = rawTexMetdata;
}

void Interpreter::GfxDpSetTile(uint8_t fmt, uint32_t siz, uint32_t line, uint32_t tmem, uint8_t tile, uint32_t palette,
                               uint32_t cmt, uint32_t maskt, uint32_t shiftt, uint32_t cms, uint32_t masks,
                               uint32_t shifts) {
    // OTRTODO:
    // SUPPORT_CHECK(tmem == 0 || tmem == 256);

    if (cms == G_TX_WRAP && masks == G_TX_NOMASK) {
        cms = G_TX_CLAMP;
    }
    if (cmt == G_TX_WRAP && maskt == G_TX_NOMASK) {
        cmt = G_TX_CLAMP;
    }

    mRdp->texture_tile[tile].palette = palette; // palette should set upper 4 bits of color index in 4b mode
    mRdp->texture_tile[tile].fmt = fmt;
    mRdp->texture_tile[tile].siz = siz;
    mRdp->texture_tile[tile].cms = cms;
    mRdp->texture_tile[tile].cmt = cmt;
    mRdp->texture_tile[tile].shifts = shifts;
    mRdp->texture_tile[tile].shiftt = shiftt;
    mRdp->texture_tile[tile].line_size_bytes = line * 8;

    mRdp->texture_tile[tile].tmem = tmem;
    // mRdp->texture_tile[tile].tmem_index = tmem / 256; // tmem is the 64-bit word offset, so 256 words means 2 kB

    mRdp->texture_tile[tile].tmem_index =
        tmem != 0; // assume one texture is loaded at address 0 and another texture at any other address

    mRdp->textures_changed[0] = true;
    mRdp->textures_changed[1] = true;
}

void Interpreter::GfxDpSetTileSize(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt) {
    mRdp->texture_tile[tile].uls = uls;
    mRdp->texture_tile[tile].ult = ult;
    mRdp->texture_tile[tile].lrs = lrs;
    mRdp->texture_tile[tile].lrt = lrt;
    mRdp->textures_changed[0] = true;
    mRdp->textures_changed[1] = true;
}

void Interpreter::GfxDpLoadTlut(uint8_t tile, uint32_t high_index) {
    SUPPORT_CHECK(mRdp->texture_to_load.siz == G_IM_SIZ_16b);

    if (mRdp->texture_tile[tile].tmem == 256) {
        mRdp->palettes[0] = mRdp->texture_to_load.addr;
        if (high_index == 255) {
            mRdp->palettes[1] = mRdp->texture_to_load.addr + 2 * 128;
        }
    } else {
        mRdp->palettes[1] = mRdp->texture_to_load.addr;
    }
}

void Interpreter::GfxDpLoadBlock(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t dxt) {
    SUPPORT_CHECK(uls == 0);
    SUPPORT_CHECK(ult == 0);

    // The lrs field rather seems to be number of pixels to load
    uint32_t word_size_shift = 0;
    switch (mRdp->texture_to_load.siz) {
        case G_IM_SIZ_4b:
            word_size_shift = -1;
            break;
        case G_IM_SIZ_8b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_16b:
            word_size_shift = 1;
            break;
        case G_IM_SIZ_32b:
            word_size_shift = 2;
            break;
    }
    uint32_t orig_size_bytes =
        word_size_shift > 0 ? (lrs + 1) << word_size_shift : (lrs + 1) >> (-(int64_t)word_size_shift);
    uint32_t size_bytes = orig_size_bytes;
    if (mRdp->texture_to_load.raw_tex_metadata.h_byte_scale != 1 ||
        mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale != 1) {
        size_bytes *= mRdp->texture_to_load.raw_tex_metadata.h_byte_scale;
        size_bytes *= mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale;
    }
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes = orig_size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes = size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes = size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes = size_bytes;
    // assert(size_bytes <= 4096 && "bug: too big texture");
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].tex_flags = mRdp->texture_to_load.tex_flags;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata = mRdp->texture_to_load.raw_tex_metadata;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr = mRdp->texture_to_load.addr;
    // fprintf(stderr, "GfxDpLoadBlock: line_size = 0x%x; orig = 0x%x; bpp=%d; lrs=%d\n", size_bytes,
    // orig_size_bytes,
    //         mRdp->texture_to_load.siz, lrs);

    const std::string& texPath =
        mRdp->texture_to_load.raw_tex_metadata.resource != nullptr
            ? GetBaseTexturePath(mRdp->texture_to_load.raw_tex_metadata.resource->GetInitData()->Path)
            : "";
    auto maskedTextureIter = mMaskedTextures.find(texPath);
    if (maskedTextureIter != mMaskedTextures.end()) {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = true;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended =
            maskedTextureIter->second.replacementData != nullptr;
    } else {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = false;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended = false;
    }

    mRdp->textures_changed[mRdp->texture_tile[tile].tmem_index] = true;
}

void Interpreter::GfxDpLoadTile(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t lrt) {
    SUPPORT_CHECK(tile == G_TX_LOADTILE);

    uint32_t word_size_shift = 0;
    switch (mRdp->texture_to_load.siz) {
        case G_IM_SIZ_4b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_8b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_16b:
            word_size_shift = 1;
            break;
        case G_IM_SIZ_32b:
            word_size_shift = 2;
            break;
    }

    uint32_t offset_x = uls >> G_TEXTURE_IMAGE_FRAC;
    uint32_t offset_y = ult >> G_TEXTURE_IMAGE_FRAC;
    uint32_t tile_width = ((lrs - uls) >> G_TEXTURE_IMAGE_FRAC) + 1;
    uint32_t tile_height = ((lrt - ult) >> G_TEXTURE_IMAGE_FRAC) + 1;
    uint32_t full_image_width = mRdp->texture_to_load.width;

    uint32_t offset_x_in_bytes = offset_x << word_size_shift;
    uint32_t tile_line_size_bytes = tile_width << word_size_shift;
    uint32_t full_image_line_size_bytes = full_image_width << word_size_shift;

    uint32_t orig_size_bytes = tile_line_size_bytes * tile_height;
    uint32_t size_bytes = orig_size_bytes;
    uint32_t start_offset_bytes = full_image_line_size_bytes * offset_y + offset_x_in_bytes;

    float h_byte_scale = mRdp->texture_to_load.raw_tex_metadata.h_byte_scale;
    float v_pixel_scale = mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale;

    if (h_byte_scale != 1 || v_pixel_scale != 1) {
        start_offset_bytes = h_byte_scale * (v_pixel_scale * offset_y * full_image_line_size_bytes + offset_x_in_bytes);
        size_bytes *= h_byte_scale * v_pixel_scale;
        full_image_line_size_bytes *= h_byte_scale;
        tile_line_size_bytes *= h_byte_scale;
    }

    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes = orig_size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes = size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes = full_image_line_size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes = tile_line_size_bytes;

    //    assert(size_bytes <= 4096 && "bug: too big texture");
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].tex_flags = mRdp->texture_to_load.tex_flags;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata = mRdp->texture_to_load.raw_tex_metadata;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr = mRdp->texture_to_load.addr + start_offset_bytes;

    const std::string& texPath =
        mRdp->texture_to_load.raw_tex_metadata.resource != nullptr
            ? GetBaseTexturePath(mRdp->texture_to_load.raw_tex_metadata.resource->GetInitData()->Path)
            : "";
    auto maskedTextureIter = mMaskedTextures.find(texPath);
    if (maskedTextureIter != mMaskedTextures.end()) {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = true;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended =
            maskedTextureIter->second.replacementData != nullptr;
    } else {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = false;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended = false;
    }

    mRdp->texture_tile[tile].uls = uls;
    mRdp->texture_tile[tile].ult = ult;
    mRdp->texture_tile[tile].lrs = lrs;
    mRdp->texture_tile[tile].lrt = lrt;

    mRdp->textures_changed[mRdp->texture_tile[tile].tmem_index] = true;
}

/*static uint8_t color_comb_component(uint32_t v) {
    switch (v) {
        case G_CCMUX_TEXEL0:
            return CC_TEXEL0;
        case G_CCMUX_TEXEL1:
            return CC_TEXEL1;
        case G_CCMUX_PRIMITIVE:
            return CC_PRIM;
        case G_CCMUX_SHADE:
            return CC_SHADE;
        case G_CCMUX_ENVIRONMENT:
            return CC_ENV;
        case G_CCMUX_TEXEL0_ALPHA:
            return CC_TEXEL0A;
        case G_CCMUX_LOD_FRACTION:
            return CC_LOD;
        default:
            return CC_0;
    }
}

static inline uint32_t color_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return color_comb_component(a) |
           (color_comb_component(b) << 3) |
           (color_comb_component(c) << 6) |
           (color_comb_component(d) << 9);
}

static void GfxDpSetCombineMode(uint32_t rgb, uint32_t alpha) {
    mRdp->combine_mode = rgb | (alpha << 12);
}*/

void Interpreter::GfxDpSetCombineMode(uint32_t rgb, uint32_t alpha, uint32_t rgb_cyc2, uint32_t alpha_cyc2) {
    mRdp->combine_mode = rgb | (alpha << 16) | ((uint64_t)rgb_cyc2 << 28) | ((uint64_t)alpha_cyc2 << 44);
}

static inline uint32_t color_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a & 0xf) | ((b & 0xf) << 4) | ((c & 0x1f) << 8) | ((d & 7) << 13);
}

static inline uint32_t alpha_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a & 7) | ((b & 7) << 3) | ((c & 7) << 6) | ((d & 7) << 9);
}

// SOH [Enhancement] Actor shadow: footprint-grid tuning (see FlushToonShadow). Boxes are emitted on a
// kShadowGridSize² grid (128 cells resolve a Link-sized footprint to ~0.5 world units); occupancy is
// rasterized at 2× that (kShadowRasterSize), so each output cell carries a 0–4 sub-cell coverage count that
// drives the anti-aliased edge bands. Rows are bitmasks of kShadow*Words uint64_t words.
static constexpr int kShadowGridSize = 128;
static constexpr int kShadowGridWords = kShadowGridSize / 64;
static constexpr int kShadowRasterSize = kShadowGridSize * 2;
static constexpr int kShadowRasterWords = kShadowRasterSize / 64;
static_assert(kShadowGridSize % 64 == 0 && kShadowRasterSize % 64 == 0, "rows are whole uint64_t words");
// Per-frame volume accumulator budget. Once a frame's volumes exceed it, later objects skip their shadow
// (drop the newest, keep what's built) rather than growing unbounded.
static constexpr size_t kShadowAccumBudgetFloats = 8u * 1024u * 1024u;

// SOH [Enhancement] Actor shadow: BUILD this object's shadow volume and accumulate it for the frame. It is
// NOT drawn here — all the frame's volumes are rendered together by RenderShadowVolumes() at the pre-actor
// hook, so the shadow lands only on the environment (the room is in the depth buffer but actors are not yet),
// exactly like the Wind Waker light pools. That means no self-shadow and no shadowing of other actors, at the
// cost of one frame of lag in the shadow's position (imperceptible for a ground shadow).
//
// The volume is a thin SLAB at the feet: the captured silhouette projected along the cel key-light direction
// onto the feet level, then extruded from slabTop (above the feet, catches uphill ground) to slabBottom
// (below the feet, catches downhill ground / cliffs). The stencil z-fail pass conforms it to the real ground.
//
// The slab is built from the FOOTPRINT, not from the triangles: the projected triangles are rasterized into
// an occupancy grid over the footprint bounds, and the volume is one box per run of occupied cells. This
// keeps the stencil counts small by construction. The obvious alternative — one closed prism per projected
// triangle — breaks on anything denser than a vanilla N64 mesh: every prism overlapping a ground pixel
// z-fail-increments the same 8-bit stencil (and at low camera pitch the view ray's underground segment
// crosses hundreds more prism walls), so past 255 the increment clamps, the decrement pass walks it to 0, and
// the shadow shows angle-dependent holes. A ~2k-triangle replacement model is already far past that limit;
// the grid never gets near it — box walls are only emitted along each band region's OUTLINE (interior walls
// between abutting boxes are skipped), so pass-1 increments scale with the silhouette's boundary crossings,
// a handful even on a grazing ray.
//
// The raster runs at twice the output resolution, so every output cell gets a 0–4 coverage count from its
// 2×2 sub-cells. EdgeSoftness turns that into anti-aliased opacity bands: fully-covered cells form the core
// and partially-covered edge cells render at lighter steps (see pass 3), which smooths the staircase the
// grid would otherwise show along the silhouette.
void Interpreter::FlushToonShadow() {
    // SOH [Enhancement] Shadow-map mode reuses the same caster arming as the stencil volumes, because that
    // marker is how the game says "this object casts". The world-space capture in GfxSpTri1 has already
    // taken what the cascades need, so drop the silhouette here instead of building a volume from it:
    // the two systems are mutually exclusive, and building volumes nobody draws would rasterize a
    // footprint grid per object for nothing.
    // Object boundary bookkeeping, run on every exit from the shadow-map branch below.
    auto beginNextShadowObject = [this] {
        mShadowVerts.clear();
        mShadowObjectHasVerts = false;
        mShadowAlphaObjectMark = mShadowAlphaCasters[SHADOW_MAP_LAYER_ACTORS].verts.size();
    };

    if (mShadowMapEnabled) {
        // Size gate. Grass tufts, flowers and other ground clutter are armed as casters like everything
        // else, but their shadow is a smudge a few texels across that reads as dirt on the ground rather
        // than as a shadow -- and every one of them costs a full re-rasterisation in each cascade. Judge the
        // object by the world-space bounding box accumulated over every triangle it captured, opaque and
        // cutout alike, which is the only per-object information this layer has.
        //
        // The largest extent, not the height: a caster can be small in every direction and still matter if
        // it is long (a fence rail), and a flat wide thing casts a real shadow at a low sun.
        const float extent =
            mShadowObjectHasVerts ? std::max({ mShadowObjectMax[0] - mShadowObjectMin[0],
                                               mShadowObjectMax[1] - mShadowObjectMin[1],
                                               mShadowObjectMax[2] - mShadowObjectMin[2] })
                                  : 0.0f;
        if (extent >= mShadowMapMinCasterSize) {
            if (mShadowVerts.size() >= 9 &&
                mShadowMapCasters[SHADOW_MAP_LAYER_ACTORS].size() < kShadowMapCasterBudgetFloats) {
                std::vector<float>& dst = mShadowMapCasters[SHADOW_MAP_LAYER_ACTORS];
                dst.insert(dst.end(), mShadowVerts.begin(), mShadowVerts.end());
            }
        } else {
            // Too small: roll the cutout half back to where this object started too, or the gate would only
            // ever drop half a caster and clutter would keep casting whatever part of it was alpha-tested.
            ShadowAlphaCasters& alpha = mShadowAlphaCasters[SHADOW_MAP_LAYER_ACTORS];
            if (alpha.verts.size() > mShadowAlphaObjectMark) {
                const uint32_t markVertex = (uint32_t)(mShadowAlphaObjectMark / 5);
                alpha.verts.resize(mShadowAlphaObjectMark);
                while (!alpha.ranges.empty() && alpha.ranges.back().firstVertex >= markVertex) {
                    alpha.ranges.pop_back();
                }
                if (!alpha.ranges.empty()) {
                    // The object may have extended a range opened by the previous one; clip it back.
                    ShadowAlphaRange& last = alpha.ranges.back();
                    if (last.firstVertex + last.vertexCount > markVertex) {
                        last.vertexCount = markVertex - last.firstVertex;
                    }
                }
            }
        }
        beginNextShadowObject();
        return;
    }
    const float coreAlpha = std::clamp(mToonShadowAlpha, 0.0f, 1.0f);
    if (mShadowVerts.size() < 9 || coreAlpha <= 0.0f) {
        mShadowVerts.clear();
        return;
    }

    // Eased size scale (0..1) the game pushes per object, so the shadow grows in / shrinks out instead of
    // popping. At ~0 there's nothing to draw.
    const float sizeScale = std::clamp(mRsp->toon_shadow_size, 0.0f, 1.0f);
    if (sizeScale <= 0.01f) {
        mShadowVerts.clear();
        return;
    }

    // Frame budget: once the accumulated volumes exceed it, drop THIS object's shadow (the newest) and keep
    // everything already built — one missing shadow, not a whole-frame blink of all of them.
    size_t accumTotal = 0;
    for (int b = 0; b < kShadowBands; b++) {
        accumTotal += mShadowVolumeAccum[b].size();
    }
    if (accumTotal > kShadowAccumBudgetFloats) {
        mShadowVerts.clear();
        return;
    }

    const size_t floatCount = mShadowVerts.size();

    // Feet level = the lowest captured vertex (the real rendered feet, not the unreliable collision floor); the
    // XZ centroid is the point the footprint shrinks toward when sizeScale < 1.
    float minY = 1e30f, sumX = 0.0f, sumZ = 0.0f;
    size_t vertN = 0;
    for (size_t i = 0; i + 3 <= floatCount; i += 3) {
        sumX += mShadowVerts[i], minY = std::min(minY, mShadowVerts[i + 1]), sumZ += mShadowVerts[i + 2];
        vertN++;
    }
    const float cenX = sumX / (float)vertN, cenZ = sumZ / (float)vertN;
    // Feet level, optionally raised UP to the floor Y the game passed: a model whose geometry dips far below
    // the floor (a signpost's buried post) would otherwise build the whole slab below ground and cast nothing.
    // Clamp only ever LIFTS the feet (max), so an actor standing on/above the floor is unaffected.
    float feetY = minY;
    if (mRsp->toon_shadow_clamp_feet) {
        feetY = std::max(minY, mRsp->toon_shadow_feet_clamp_y);
    }
    const float slabTop = feetY + mShadowSlabRise;                     // above the feet (uphill ground)
    const float slabBottom = feetY - std::max(5.0f, mShadowSlabDepth); // below the feet (downhill / cliffs)

    // Cast direction from the cel key light (toward-light dir snapshotted at arm time), elevation-remapped
    // against world up so a low light still casts a short shadow (Length slider drives minElev).
    float lx = mRsp->toon_shadow_dir[0], ly = mRsp->toon_shadow_dir[1], lz = mRsp->toon_shadow_dir[2];
    const float llen = sqrtf((lx * lx) + (ly * ly) + (lz * lz));
    float dirX, dirY, dirZ;
    const float minElev = std::clamp(mToonShadowMinElevation, 0.05f, 0.99f);
    if (llen < 0.001f) {
        dirX = 0.0f, dirY = -1.0f, dirZ = 0.0f;
    } else {
        lx /= llen, ly /= llen, lz /= llen;
        const float lUp = ly < 0.0f ? 0.0f : ly;
        const float elev = minElev + ((1.0f - minElev) * lUp);
        const float hLen = sqrtf((lx * lx) + (lz * lz));
        if (hLen < 0.001f) {
            dirX = 0.0f, dirY = -1.0f, dirZ = 0.0f;
        } else {
            const float hScale = sqrtf(std::max(0.0f, 1.0f - (elev * elev))) / hLen;
            dirX = -hScale * lx, dirY = -elev, dirZ = -hScale * lz;
        }
    }
    const float descend = -dirY;

    auto projectXZ = [&](float vx, float vy, float vz, float& ox, float& oz) {
        float t = (descend > 0.001f) ? ((vy - slabTop) / descend) : 0.0f;
        if (t < 0.0f) {
            t = 0.0f;
        }
        ox = vx + (dirX * t);
        oz = vz + (dirZ * t);
    };

    // Append one outward-wound world-space triangle to a band's frame accumulator, tagged cap(0) / wall(1).
    auto pushTri = [&](int band, const float* p0, const float* p1, const float* p2, float ccx, float ccy, float ccz,
                       uint8_t kind) {
        const float ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
        const float vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
        const float nX = (uy * vz) - (uz * vy), nY = (uz * vx) - (ux * vz), nZ = (ux * vy) - (uy * vx);
        const float fx = ((p0[0] + p1[0] + p2[0]) / 3.0f) - ccx;
        const float fy = ((p0[1] + p1[1] + p2[1]) / 3.0f) - ccy;
        const float fz = ((p0[2] + p1[2] + p2[2]) / 3.0f) - ccz;
        const bool outward = ((nX * fx) + (nY * fy) + (nZ * fz)) >= 0.0f;
        const float* q1 = outward ? p1 : p2;
        const float* q2 = outward ? p2 : p1;
        std::vector<float>& acc = mShadowVolumeAccum[band];
        acc.push_back(p0[0]), acc.push_back(p0[1]), acc.push_back(p0[2]);
        acc.push_back(q1[0]), acc.push_back(q1[1]), acc.push_back(q1[2]);
        acc.push_back(q2[0]), acc.push_back(q2[1]), acc.push_back(q2[2]);
        if (mShadowShowVolume) { // cap/wall tag is only read by the debug overlay
            mShadowVolumeKind[band].push_back(kind);
        }
    };

    // Rasterize the projected triangles into an occupancy grid over the footprint bounds, then rebuild the
    // volume as one box per horizontal run of occupied cells. Boxes never overlap and abutting boxes share
    // exactly-coincident, oppositely-wound walls (identical float coordinates → identical rasterization), so
    // their z-fail counts cancel and the union reads as one seamless footprint with stencil overlap 1.
    auto shrink = [&](float& x, float& z) {
        if (sizeScale < 1.0f) {
            x = cenX + (x - cenX) * sizeScale;
            z = cenZ + (z - cenZ) * sizeScale;
        }
    };
    // Pass 1: bounds of the projected (shrunk) footprint.
    float minX = 1e30f, maxX = -1e30f, minZ = 1e30f, maxZ = -1e30f;
    for (size_t i = 0; i + 3 <= floatCount; i += 3) {
        float px, pz;
        projectXZ(mShadowVerts[i], mShadowVerts[i + 1], mShadowVerts[i + 2], px, pz);
        shrink(px, pz);
        minX = std::min(minX, px), maxX = std::max(maxX, px);
        minZ = std::min(minZ, pz), maxZ = std::max(maxZ, pz);
    }
    const float cellW = (maxX - minX) / kShadowGridSize;
    const float cellH = (maxZ - minZ) / kShadowGridSize;
    if (cellW < 1e-4f || cellH < 1e-4f) { // footprint degenerate (a line/point) — nothing worth casting
        mShadowVerts.clear();
        return;
    }
    // Pass 2: rasterize, conservatively. Mark each triangle's three vertex cells (so sub-cell triangles
    // still register) plus every cell in its bounding box whose CENTRE lies within half a cell-diagonal
    // of the triangle (signed edge distance ≥ −margin). The margin matters: a centre exactly on the
    // crack between two adjacent triangles can be float-marginally outside BOTH, and an exact
    // centre-in-triangle test then leaves a pinhole. The margin closes cracks by construction, at the
    // cost of dilating the footprint by up to half a cell.
    //
    // Resolution: the 2× raster only exists to give each output cell a 0–4 sub-cell coverage count for
    // the soft-edge bands. EdgeSoftness 0 (hard edge) needs no counts — any hit is core — so it
    // rasterizes at the output resolution directly, a quarter of the cell tests.
    const int softness = std::clamp(mShadowEdgeSoftness, 0, kShadowBands - 1);
    const int rScale = (softness == 0) ? 1 : 2;
    const int rSize = kShadowGridSize * rScale;
    const float rCellW = cellW / (float)rScale, rCellH = cellH / (float)rScale;
    const float margin = 0.5f * sqrtf((rCellW * rCellW) + (rCellH * rCellH));
    auto rCellX = [&](float x) { return std::clamp((int)((x - minX) / rCellW), 0, rSize - 1); };
    auto rCellZ = [&](float z) { return std::clamp((int)((z - minZ) / rCellH), 0, rSize - 1); };
    uint64_t raster[kShadowRasterSize][kShadowRasterWords] = {}; // sized for 2×; hard edge uses a quarter
    auto rasterSet = [&](int gz, int gx) { raster[gz][gx >> 6] |= 1ull << (gx & 63); };
    for (size_t base = 0; base + 9 <= floatCount; base += 9) {
        float ax, az, bx, bz, cx, cz;
        projectXZ(mShadowVerts[base + 0], mShadowVerts[base + 1], mShadowVerts[base + 2], ax, az);
        projectXZ(mShadowVerts[base + 3], mShadowVerts[base + 4], mShadowVerts[base + 5], bx, bz);
        projectXZ(mShadowVerts[base + 6], mShadowVerts[base + 7], mShadowVerts[base + 8], cx, cz);
        shrink(ax, az), shrink(bx, bz), shrink(cx, cz);
        const int axi = rCellX(ax), azi = rCellZ(az);
        const int bxi = rCellX(bx), bzi = rCellZ(bz);
        const int cxi = rCellX(cx), czi = rCellZ(cz);
        rasterSet(azi, axi);
        rasterSet(bzi, bxi);
        rasterSet(czi, cxi);
        const float area2 = ((bx - ax) * (cz - az)) - ((cx - ax) * (bz - az));
        if (fabsf(area2) < 1e-6f) {
            continue; // sliver — the vertex sub-cells above are all it gets
        }
        // Signed edge distances need a consistent orientation and per-edge length scaling (the raw edge
        // function is distance × edge length).
        const float sgn = (area2 >= 0.0f) ? 1.0f : -1.0f;
        const float len0 = sqrtf(((bx - ax) * (bx - ax)) + ((bz - az) * (bz - az)));
        const float len1 = sqrtf(((cx - bx) * (cx - bx)) + ((cz - bz) * (cz - bz)));
        const float len2 = sqrtf(((ax - cx) * (ax - cx)) + ((az - cz) * (az - cz)));
        const float m0 = -margin * len0, m1 = -margin * len1, m2 = -margin * len2;
        const int x0 = std::min({ axi, bxi, cxi }), x1 = std::max({ axi, bxi, cxi });
        const int z0 = std::min({ azi, bzi, czi }), z1 = std::max({ azi, bzi, czi });
        for (int gz = z0; gz <= z1; gz++) {
            const float pz = minZ + ((gz + 0.5f) * rCellH);
            for (int gx = x0; gx <= x1; gx++) {
                const float px = minX + ((gx + 0.5f) * rCellW);
                const float e0 = sgn * (((bx - ax) * (pz - az)) - ((bz - az) * (px - ax)));
                const float e1 = sgn * (((cx - bx) * (pz - bz)) - ((cz - bz) * (px - bx)));
                const float e2 = sgn * (((ax - cx) * (pz - cz)) - ((az - cz) * (px - cx)));
                if (e0 >= m0 && e1 >= m1 && e2 >= m2) {
                    rasterSet(gz, gx);
                }
            }
        }
    }

    // Pass 3: downsample each output cell's 2×2 sub-cells to a coverage count (0–4) and assign disjoint
    // opacity bands (band alphas step down in RenderShadowVolumes; nothing double-darkens):
    //   EdgeSoftness 0: any coverage → full opacity (hard edge) — raster is already at output
    //                   resolution, so its rows ARE the core band, no downsample.
    //   EdgeSoftness 1: full coverage → core; partial coverage → half opacity. The partially-covered cells
    //                   are exactly the silhouette's staircase, so this reads as an anti-aliased edge.
    //   EdgeSoftness 2: full → core; 2–3 sub-cells → 2/3; 1 sub-cell → 1/3, plus a one-cell halo outside
    //                   the footprint at 1/3 — a finer ramp and a slightly wider fringe.
    uint64_t bandRows[kShadowBands][kShadowGridSize][kShadowGridWords] = {};
    uint64_t occupied[kShadowGridSize][kShadowGridWords] = {};
    if (rScale == 1) {
        for (int gz = 0; gz < kShadowGridSize; gz++) {
            for (int i = 0; i < kShadowGridWords; i++) {
                occupied[gz][i] = raster[gz][i];
                bandRows[0][gz][i] = raster[gz][i];
            }
        }
    } else {
        for (int gz = 0; gz < kShadowGridSize; gz++) {
            const uint64_t* r0 = raster[gz * 2];
            const uint64_t* r1 = raster[(gz * 2) + 1];
            for (int gx = 0; gx < kShadowGridSize; gx++) {
                const int sx = gx * 2;
                const int cov = (int)((r0[sx >> 6] >> (sx & 63)) & 1ull) + (int)((r0[(sx + 1) >> 6] >> ((sx + 1) & 63)) & 1ull) +
                                (int)((r1[sx >> 6] >> (sx & 63)) & 1ull) + (int)((r1[(sx + 1) >> 6] >> ((sx + 1) & 63)) & 1ull);
                if (cov == 0) {
                    continue;
                }
                occupied[gz][gx >> 6] |= 1ull << (gx & 63);
                int band = 0;
                if (softness == 1) {
                    band = (cov == 4) ? 0 : 1;
                } else if (softness >= 2) {
                    band = (cov == 4) ? 0 : ((cov >= 2) ? 1 : 2);
                }
                bandRows[band][gz][gx >> 6] |= 1ull << (gx & 63);
            }
        }
    }
    if (softness >= 2) {
        // Halo: one output cell outside the occupied footprint, at the lightest band.
        static constexpr uint64_t kZeroRow[kShadowGridWords] = {};
        for (int gz = 0; gz < kShadowGridSize; gz++) {
            const uint64_t* cur = occupied[gz];
            const uint64_t* up = (gz > 0) ? occupied[gz - 1] : kZeroRow;
            const uint64_t* down = (gz + 1 < kShadowGridSize) ? occupied[gz + 1] : kZeroRow;
            for (int i = 0; i < kShadowGridWords; i++) {
                const uint64_t left = (cur[i] << 1) | ((i > 0) ? (cur[i - 1] >> 63) : 0);
                const uint64_t right = (cur[i] >> 1) | ((i + 1 < kShadowGridWords) ? (cur[i + 1] << 63) : 0);
                bandRows[2][gz][i] |= (cur[i] | left | right | up[i] | down[i]) & ~cur[i];
            }
        }
    }

    // Pass 4: one box (2 caps + 4 wall quads) per maximal RECTANGLE of occupied cells, per band. A
    // rectangle is a maximal horizontal run repeated IDENTICALLY across consecutive rows. Identical-run
    // merging preserves the invariant the x walls rely on (nothing in any merged row abuts the run's
    // ends — an abutting cell would have made that row's maximal run differ, stopping the merge) while
    // cutting the emitted triangles several-fold on typical blob footprints: a silhouette's edge slope
    // is below one cell per row for most rows, so those rows carry identical runs and merge.
    auto bandTest = [&](int band, int gz, int gx) -> bool {
        return ((bandRows[band][gz][gx >> 6] >> (gx & 63)) & 1ull) != 0;
    };
    // Consumption tracker: a rectangle removes each claimed run WHOLE, so `rem` always holds a subset
    // of the original maximal runs, intact — which is what makes the exact-run test below sound. The
    // z-wall neighbour tests keep consulting bandRows (the full occupancy), not rem.
    uint64_t rem[kShadowGridSize][kShadowGridWords];
    for (int band = 0; band < kShadowBands; band++) {
        memcpy(rem, bandRows[band], sizeof(rem));
        auto remTest = [&](int gz, int gx) -> bool { return ((rem[gz][gx >> 6] >> (gx & 63)) & 1ull) != 0; };
        auto remClearRun = [&](int gz, int x0, int x1) {
            for (int gx = x0; gx < x1; gx++) {
                rem[gz][gx >> 6] &= ~(1ull << (gx & 63));
            }
        };
        // Does row gz still contain EXACTLY the maximal run [x0,x1)?
        auto rowRunMatches = [&](int gz, int x0, int x1) -> bool {
            if ((x0 > 0 && remTest(gz, x0 - 1)) || (x1 < kShadowGridSize && remTest(gz, x1))) {
                return false; // longer run here — not identical
            }
            for (int gx = x0; gx < x1; gx++) {
                if (!remTest(gz, gx)) {
                    return false;
                }
            }
            return true;
        };
        for (int gz = 0; gz < kShadowGridSize; gz++) {
            int gx = 0;
            while (gx < kShadowGridSize) {
                if (!remTest(gz, gx)) {
                    gx++;
                    continue;
                }
                const int runStart = gx;
                while (gx < kShadowGridSize && remTest(gz, gx)) {
                    gx++;
                }
                remClearRun(gz, runStart, gx);
                // Grow the rectangle downward while the run repeats exactly, consuming as it goes.
                int zEnd = gz + 1;
                while (zEnd < kShadowGridSize && rowRunMatches(zEnd, runStart, gx)) {
                    remClearRun(zEnd, runStart, gx);
                    zEnd++;
                }
                const float x0 = minX + (runStart * cellW), x1 = minX + (gx * cellW);
                const float z0 = minZ + (gz * cellH), z1 = minZ + (zEnd * cellH);
                const float ccx = (x0 + x1) * 0.5f, ccy = (slabTop + slabBottom) * 0.5f, ccz = (z0 + z1) * 0.5f;
                auto quad = [&](const float* a, const float* b, const float* c, const float* d, uint8_t kind) {
                    pushTri(band, a, b, c, ccx, ccy, ccz, kind), pushTri(band, a, c, d, ccx, ccy, ccz, kind);
                };
                const float t00[3] = { x0, slabTop, z0 }, t10[3] = { x1, slabTop, z0 };
                const float t11[3] = { x1, slabTop, z1 }, t01[3] = { x0, slabTop, z1 };
                const float b00[3] = { x0, slabBottom, z0 }, b10[3] = { x1, slabBottom, z0 };
                const float b11[3] = { x1, slabBottom, z1 }, b01[3] = { x0, slabBottom, z1 };
                quad(t00, t10, t11, t01, 0); // top cap
                quad(b00, b10, b11, b01, 0); // bottom cap
                quad(t01, t00, b00, b01, 1); // wall x0 (nothing abuts the run ends in any merged row)
                quad(t10, t11, b11, b10, 1); // wall x1
                // z walls at the rectangle's first/last row only: the segments where the neighbouring
                // row's cell (in the SAME band) is unoccupied. Abutting geometry's coincident opposite
                // walls would cancel in the final stencil count anyway, but they still pile up in the
                // increment pass before the decrements land — emitting only outline segments keeps
                // pass-1 peaks proportional to the region's OUTLINE crossings.
                auto zWallSegments = [&](int neighborGz, float zw) {
                    const bool haveNeighbor = (neighborGz >= 0) && (neighborGz < kShadowGridSize);
                    int sx = runStart;
                    while (sx < gx) {
                        if (haveNeighbor && bandTest(band, neighborGz, sx)) {
                            sx++;
                            continue;
                        }
                        const int s0 = sx;
                        while (sx < gx && !(haveNeighbor && bandTest(band, neighborGz, sx))) {
                            sx++;
                        }
                        const float wx0 = minX + (s0 * cellW), wx1 = minX + (sx * cellW);
                        const float ta[3] = { wx0, slabTop, zw }, tb[3] = { wx1, slabTop, zw };
                        const float ba[3] = { wx0, slabBottom, zw }, bb[3] = { wx1, slabBottom, zw };
                        quad(ta, tb, bb, ba, 1);
                    }
                };
                zWallSegments(gz - 1, z0);
                zWallSegments(zEnd, z1);
            }
        }
    }
    mShadowVerts.clear();
}

// SOH [Enhancement] Actor shadow: render every shadow volume accumulated since the last call, then clear the
// accumulators. Each opacity band (the core, then the penumbra rings — disjoint footprint regions built in
// FlushToonShadow) gets its own batched z-fail stencil pass pair + one self-clearing composite at the band's
// stepped-down alpha; the steps read as a small soft edge. Called once per frame at the pre-actor hook
// (after the room is drawn) so the shadows fall only on the environment.
// SOH [Enhancement] Cascaded shadow maps: invert a row-vector 4x4 (general case, via cofactors). Used to
// recover the view axis from the combined view-projection so the cascades can be centred on the camera --
// the interpreter never sees a separate view matrix, only the product the game hands it.
// Returns false for a singular matrix, in which case the caller skips the shadow pass this frame.
static bool ShadowInvertMatrix(const float m[4][4], float out[4][4]) {
    const float* a = &m[0][0];
    float inv[16];

    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] +
             a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] -
             a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] +
             a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] -
              a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] -
             a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] +
             a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] -
             a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] +
              a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] +
             a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] -
             a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] +
              a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] -
              a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] -
             a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] +
             a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] -
              a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] +
              a[8] * a[1] * a[6] - a[8] * a[2] * a[5];

    float det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if (std::fabs(det) < 1e-12f) {
        return false;
    }
    det = 1.0f / det;
    for (int i = 0; i < 16; i++) {
        (&out[0][0])[i] = inv[i] * det;
    }
    return true;
}

// Transform an NDC point through an inverse view-projection, row-vector convention, with the perspective
// divide. Used only to walk the view axis, so w == 0 means "no usable point" rather than an error.
static bool ShadowUnproject(const float invVp[4][4], float x, float y, float z, float out[3]) {
    const float w = x * invVp[0][3] + y * invVp[1][3] + z * invVp[2][3] + invVp[3][3];
    if (std::fabs(w) < 1e-9f) {
        return false;
    }
    for (int i = 0; i < 3; i++) {
        out[i] = (x * invVp[0][i] + y * invVp[1][i] + z * invVp[2][i] + invVp[3][i]) / w;
    }
    return true;
}

// SOH [Enhancement] Cascaded shadow maps: render last frame's casters into the cascade array.
//
// Each cascade is an orthographic box centred on a point along the view axis, sized by that cascade's
// split distance. The box is fitted to a SPHERE rather than to the view frustum's corners: a sphere's
// radius does not change as the camera turns, so the projection stays the same size frame to frame. A
// frustum-corner fit would resize it constantly and every shadow edge would crawl.
//
// The centre is then snapped to whole texels in light space. Without that, sub-texel movement of the
// centre reshuffles which texel each surface lands in and the edges shimmer as the camera walks
// ("shadow swimming") -- the artefact the design calls out first.
void Interpreter::RenderShadowMap() {
    // Roll both caster layers forward for this frame. Done up front, before anything can return early, so
    // there is exactly one place that owns the buffers and no exit path can leak a frame's captures.
    //
    // This hook fires after the room has drawn but before the actors do, which is why the two layers are
    // handled differently: the world captures are already complete and can be consumed immediately, while
    // the actor captures sitting in the buffer are the previous frame's -- hence the swap and the documented
    // one frame of lag on character shadows.
    {
        if (mShadowWorldKeyAccum != 0) {
            if (mShadowWorldCapture) {
                // A rebuild was pending and this frame captured it: adopt it and stop capturing.
                mShadowMapWorldCache.swap(mShadowMapCasters[SHADOW_MAP_LAYER_WORLD]);
                mShadowAlphaWorldCache.swap(mShadowAlphaCasters[SHADOW_MAP_LAYER_WORLD]);
                mShadowWorldKeyCached = mShadowWorldKeyAccum;
                mShadowWorldCapture = false;
                // The one place the cached lists change. Everything downstream reads the counter instead of
                // the megabytes behind it (see ShadowMapLayerContentKey).
                mShadowWorldCacheGeneration++;
                BuildShadowWorldChunks(); // the spans index into the list that was just swapped in
            } else if (mShadowWorldKeyAccum != mShadowWorldKeyCached) {
                // Different geometry ran this frame than the cache was built from. The frame is already past
                // the point where it could have been captured, so arm the rebuild for the next one -- the
                // cache is one frame stale across a room change, the same lag the actors carry permanently.
                mShadowWorldCapture = true;
            }
        }
        // A zero signature means no world casters were bracketed at all this frame (paused, a cutscene, a
        // menu). That is not a geometry change, so the cache is left exactly as it is rather than being
        // rebuilt as empty and then rebuilt again the moment the room comes back -- which would flicker the
        // scenery shadows off and on.
        mShadowMapCasters[SHADOW_MAP_LAYER_WORLD].clear(); // keeps capacity for the next rebuild
        mShadowAlphaCasters[SHADOW_MAP_LAYER_WORLD].clear();
        mShadowWorldKeyAccum = 0;

        mShadowMapCastersReady[SHADOW_MAP_LAYER_ACTORS].swap(mShadowMapCasters[SHADOW_MAP_LAYER_ACTORS]);
        mShadowMapCasters[SHADOW_MAP_LAYER_ACTORS].clear();
        // Scenery actors roll exactly like characters. They are drawn in the world layer, but they are not
        // in its cache and must not be: a gate that opens has to be recaptured where it now stands.
        mShadowSceneryReady.swap(mShadowSceneryCasters);
        mShadowSceneryCasters.clear();
        mShadowAlphaSceneryReady.swap(mShadowAlphaScenery);
        mShadowAlphaScenery.clear();
        mShadowAlphaReady[SHADOW_MAP_LAYER_ACTORS].swap(mShadowAlphaCasters[SHADOW_MAP_LAYER_ACTORS]);
        mShadowAlphaCasters[SHADOW_MAP_LAYER_ACTORS].clear();
        // The object mark indexes into the list that was just emptied, so it has to come back to the start
        // with it -- otherwise the first object of the new frame is measured against last frame's offset and
        // the size gate cannot roll it back.
        mShadowAlphaObjectMark = 0;
        mShadowObjectHasVerts = false;
    }

    if (!mShadowMapEnabled) {
        return;
    }

    // SOH [Enhancement] World box around everything the ACTOR layer will draw, handed to the backend for the
    // receiver shader.
    //
    // That layer holds the characters and nothing else -- a handful of small objects somewhere on the map --
    // while EVERY scenery pixel on screen was sampling it, sixteen fetches deep, to be told it was lit. The
    // shader can skip the whole kernel wherever no part of this box lies behind the receiver along the light,
    // which over a room is nearly all of it. Skipping is exact and not an approximation: a lookup that lands
    // where no actor was drawn reads the cleared depth and returns "lit", which is the value the skip
    // substitutes.
    //
    // Measured here rather than accumulated during capture because the capture path can roll an object back
    // out of the list after its vertices are in (the per-object size gate), and a box grown incrementally
    // cannot un-grow. This walks only the actor layer, which is the small one.
    //
    // Computed before the early exits below so that every SetShadowMapParams path -- including the ones that
    // report no cascades -- carries a box matching the frame it belongs to.
    //
    // The same two boxes then cull those lists per cascade further down. Neither list is chunked the way the
    // cached room mesh is -- both are small and rebuilt every frame, so one box each is the right granularity
    // -- but until now neither was tested at all, and both were re-submitted into all four cascades.
    //
    // Sentinel is the one the backend starts from (see GfxRenderingAPI::mShadowActorBoundsMin): large enough
    // that no world coordinate reaches it, small enough that the shader's own margin cannot flip the
    // comparison or overflow the arithmetic it feeds. An empty list keeps it and fails every test.
    float actorMin[3] = { 1e30f, 1e30f, 1e30f };
    float actorMax[3] = { -1e30f, -1e30f, -1e30f };
    float sceneryMin[3] = { 1e30f, 1e30f, 1e30f };
    float sceneryMax[3] = { -1e30f, -1e30f, -1e30f };
    {
        auto growBox = [](const std::vector<float>& verts, size_t stride, float* boxMin, float* boxMax) {
            for (size_t i = 0; i + 2 < verts.size(); i += stride) {
                for (int a = 0; a < 3; a++) {
                    boxMin[a] = std::min(boxMin[a], verts[i + a]);
                    boxMax[a] = std::max(boxMax[a], verts[i + a]);
                }
            }
        };
        growBox(mShadowMapCastersReady[SHADOW_MAP_LAYER_ACTORS], 3, actorMin, actorMax);
        growBox(mShadowSceneryReady, 3, sceneryMin, sceneryMax);

        // The box the SHADER is given also has to cover the layer's cutout casters, which the per-cascade
        // draw cull does not: the cutout ranges carry their own boxes and are tested one by one already. So
        // the union is built here and the opaque box above is left alone for the draws to use.
        //
        // Those ranges measured themselves as they were captured. Ones whose texture has since been evicted
        // are included even though the pass will skip them: over-covering only costs a kernel that could have
        // been skipped, while under-covering would drop a shadow.
        float shaderMin[3] = { actorMin[0], actorMin[1], actorMin[2] };
        float shaderMax[3] = { actorMax[0], actorMax[1], actorMax[2] };
        for (const ShadowAlphaRange& r : mShadowAlphaReady[SHADOW_MAP_LAYER_ACTORS].ranges) {
            if (r.vertexCount < 3) {
                continue;
            }
            for (int a = 0; a < 3; a++) {
                shaderMin[a] = std::min(shaderMin[a], r.min[a]);
                shaderMax[a] = std::max(shaderMax[a], r.max[a]);
            }
        }
        mRapi->SetShadowMapActorBounds(shaderMin, shaderMax);
    }

    // Refreshed here rather than read per triangle: the cutout pipeline is built lazily inside
    // ShadowMapConfigure below, so the answer only becomes true after the first successful configure, and
    // the captures it governs all happen after this point in the frame.
    mShadowAlphaSupported = mRapi->SupportsShadowMapAlphaCasters();
    if (!mRapi->ShadowMapConfigure(mShadowMapCascadeCount, mShadowMapResolution)) {
        // Backend could not give us the maps; report no cascades so the main pass does not sample a
        // texture that was never filled.
        mRapi->SetShadowMapParams(nullptr, nullptr, 0, mShadowMapBlendFraction, mShadowMapNormalOffset,
                                  mShadowMapStrength, mShadowMapFilterWidth, mShadowMapDebug,
                                  mShadowMapEdgeHardness, mShadowMapEdgeHardnessFar);
        return;
    }
    if (mShadowMapWorldCache.size() + mShadowMapCastersReady[SHADOW_MAP_LAYER_ACTORS].size() +
            mShadowSceneryReady.size() + mShadowAlphaWorldCache.verts.size() +
            mShadowAlphaSceneryReady.verts.size() + mShadowAlphaReady[SHADOW_MAP_LAYER_ACTORS].verts.size() <
        9) {
        mRapi->SetShadowMapParams(nullptr, nullptr, 0, mShadowMapBlendFraction, mShadowMapNormalOffset,
                                  mShadowMapStrength, mShadowMapFilterWidth, mShadowMapDebug,
                                  mShadowMapEdgeHardness, mShadowMapEdgeHardnessFar);
        return;
    }

    float invVp[4][4];
    if (!ShadowInvertMatrix(mRsp->P_matrix, invVp)) {
        mRapi->SetShadowMapParams(nullptr, nullptr, 0, mShadowMapBlendFraction, mShadowMapNormalOffset,
                                  mShadowMapStrength, mShadowMapFilterWidth, mShadowMapDebug,
                                  mShadowMapEdgeHardness, mShadowMapEdgeHardnessFar);
        return;
    }

    // Two points on the view axis give the camera position and the direction it looks.
    float nearC[3], farC[3];
    if (!ShadowUnproject(invVp, 0.0f, 0.0f, 0.0f, nearC) || !ShadowUnproject(invVp, 0.0f, 0.0f, 1.0f, farC)) {
        mRapi->SetShadowMapParams(nullptr, nullptr, 0, mShadowMapBlendFraction, mShadowMapNormalOffset,
                                  mShadowMapStrength, mShadowMapFilterWidth, mShadowMapDebug,
                                  mShadowMapEdgeHardness, mShadowMapEdgeHardnessFar);
        return;
    }
    // Lateral half-extent of the frustum at the near and far planes, from the actual corners. Without this
    // the cascades were sized by a guess that ignored the field of view entirely, so they covered a
    // narrow tube down the middle of the view: shadows vanished toward the screen edges and popped in and
    // out as the camera turned, because whole objects kept falling outside the cascade's footprint.
    float halfNear = 0.0f, halfFar = 0.0f;
    {
        // The screen is WIDER than the projection matrix says. Every transformed vertex goes through
        // AdjXForAspectRatio, which divides clip x by the aspect ratio so a 4:3 projection fills a
        // widescreen window -- meaning the edge of the screen is NOT at matrix NDC x = +/-1, it is further
        // out, by exactly the reciprocal of that division.
        //
        // Unprojecting the corners at +/-1 therefore measured the 4:3 frustum and missed everything the
        // widescreen view actually shows. The cascade came out up to 40% too narrow at 16:9, and a receiver
        // in the outer part of the screen fell outside the cascade's footprint, where the shader has nothing
        // to compare against and reports "lit" -- the shadow simply stops. Turning the camera sweeps a
        // shadow across that boundary, so it vanishes and comes back with no change to the scene.
        //
        // Derived from AdjXForAspectRatio itself rather than recomputing the ratio, so the two can never
        // disagree -- including the framebuffer case, where it is the identity and this is a no-op.
        const float adj = AdjXForAspectRatio(1.0f);
        const float kx = (std::fabs(adj) > 1e-6f) ? (1.0f / adj) : 1.0f;
        float c[3];
        const float corners[4][2] = { { -kx, -1.0f }, { kx, -1.0f }, { -kx, 1.0f }, { kx, 1.0f } };
        for (int i = 0; i < 4; i++) {
            if (ShadowUnproject(invVp, corners[i][0], corners[i][1], 0.0f, c)) {
                const float dx = c[0] - nearC[0], dy = c[1] - nearC[1], dz = c[2] - nearC[2];
                halfNear = std::max(halfNear, std::sqrt(dx * dx + dy * dy + dz * dz));
            }
            if (ShadowUnproject(invVp, corners[i][0], corners[i][1], 1.0f, c)) {
                const float dx = c[0] - farC[0], dy = c[1] - farC[1], dz = c[2] - farC[2];
                halfFar = std::max(halfFar, std::sqrt(dx * dx + dy * dy + dz * dz));
            }
        }
    }

    float viewDir[3] = { farC[0] - nearC[0], farC[1] - nearC[1], farC[2] - nearC[2] };
    float viewLen = std::sqrt(viewDir[0] * viewDir[0] + viewDir[1] * viewDir[1] + viewDir[2] * viewDir[2]);
    if (viewLen < 1e-6f) {
        mRapi->SetShadowMapParams(nullptr, nullptr, 0, mShadowMapBlendFraction, mShadowMapNormalOffset,
                                  mShadowMapStrength, mShadowMapFilterWidth, mShadowMapDebug,
                                  mShadowMapEdgeHardness, mShadowMapEdgeHardnessFar);
        return;
    }
    for (int i = 0; i < 3; i++) {
        viewDir[i] /= viewLen;
    }

    // Light basis. The up reference is world up unless the light is nearly vertical, where that would be
    // degenerate and any horizontal reference does.
    float lz[3] = { mShadowMapLightDir[0], mShadowMapLightDir[1], mShadowMapLightDir[2] };
    float lzLen = std::sqrt(lz[0] * lz[0] + lz[1] * lz[1] + lz[2] * lz[2]);
    if (lzLen < 1e-6f) {
        mRapi->SetShadowMapParams(nullptr, nullptr, 0, mShadowMapBlendFraction, mShadowMapNormalOffset,
                                  mShadowMapStrength, mShadowMapFilterWidth, mShadowMapDebug,
                                  mShadowMapEdgeHardness, mShadowMapEdgeHardnessFar);
        return;
    }
    for (int i = 0; i < 3; i++) {
        lz[i] /= lzLen;
    }
    // Hold the light direction still (see SHADOW_MAP_LIGHT_DIR_HYSTERESIS_COS). Everything below builds the
    // texel grid from these axes, and the grid only does its job if it is the same grid frame to frame --
    // the game's environment light turns continuously with the time of day, so left alone it rotates the
    // grid a fraction of a degree every frame and every shadow edge trembles.
    {
        float* held = mShadowMapLightDirHeld;
        const float heldLen = std::sqrt(held[0] * held[0] + held[1] * held[1] + held[2] * held[2]);
        const float dot = heldLen > 0.5f ? (held[0] * lz[0] + held[1] * lz[1] + held[2] * lz[2]) : -1.0f;
        if (dot < SHADOW_MAP_LIGHT_DIR_HYSTERESIS_COS) {
            for (int i = 0; i < 3; i++) {
                held[i] = lz[i];
            }
        }
        for (int i = 0; i < 3; i++) {
            lz[i] = held[i];
        }
    }
    float up[3] = { 0.0f, 1.0f, 0.0f };
    if (std::fabs(lz[1]) > 0.99f) {
        up[0] = 1.0f;
        up[1] = 0.0f;
    }
    float lx[3] = { up[1] * lz[2] - up[2] * lz[1], up[2] * lz[0] - up[0] * lz[2], up[0] * lz[1] - up[1] * lz[0] };
    float lxLen = std::sqrt(lx[0] * lx[0] + lx[1] * lx[1] + lx[2] * lx[2]);
    for (int i = 0; i < 3; i++) {
        lx[i] /= lxLen;
    }
    const float ly[3] = { lz[1] * lx[2] - lz[2] * lx[1], lz[2] * lx[0] - lz[0] * lx[2], lz[0] * lx[1] - lz[1] * lx[0] };

    float matrices[SHADOW_MAP_MAX_CASCADES * 16] = {};
    float splits[SHADOW_MAP_MAX_CASCADES] = {};
    float nearDist = 0.0f;

    for (int c = 0; c < mShadowMapCascadeCount; c++) {
        const float farDist = mShadowMapSplits[c] > nearDist ? mShadowMapSplits[c] : nearDist + 1.0f;
        splits[c] = farDist;

        // Sphere around this slice of the view axis: centred at its midpoint, with a radius that also
        // covers the frustum's lateral spread. Half the slice length is a deliberate over-estimate --
        // cheap, and erring large only wastes a little resolution while erring small clips shadows off.
        const float mid = (nearDist + farDist) * 0.5f;
        // Sphere that provably contains this slice of the view frustum. The lateral half-extent grows
        // linearly with distance, so take it at both ends of the slice and keep the larger; combined with
        // half the slice's length that gives a radius covering every corner.
        // Fitting a sphere rather than the corners themselves is deliberate: a sphere's radius does not
        // change as the camera turns, so the projection keeps its size frame to frame. Fitting the corners
        // would resize it constantly and every shadow edge would crawl.
        const float axisLen = viewLen > 1e-6f ? viewLen : 1.0f;
        const float halfAtNear = halfNear + (halfFar - halfNear) * std::clamp(nearDist / axisLen, 0.0f, 1.0f);
        const float halfAtFar = halfNear + (halfFar - halfNear) * std::clamp(farDist / axisLen, 0.0f, 1.0f);
        const float halfLen = (farDist - nearDist) * 0.5f;
        const float lateral = std::max(halfAtNear, halfAtFar);
        float radius = std::sqrt(halfLen * halfLen + lateral * lateral);

        // Hold the radius steady. Rotation-invariance alone is not enough to stop the edges crawling: the
        // fit above is derived from the near/far planes, which the game moves around, so the radius wobbles
        // a little every frame. The texel grid is sized 2*radius/resolution, so a wobbling radius means a
        // wobbling grid -- and the snapping below, which quantizes the centre in units of one texel, is then
        // measuring against a ruler that keeps changing length, which does nothing at all.
        //
        // Quantizing alone still flips between two neighbouring steps when the fit sits near a boundary, so
        // the held value only moves when it has to: up whenever the fit no longer fits, down only once the
        // fit is clearly smaller. It must never end up below the fitted radius or the cascade would clip
        // shadows at its edge.
        if (radius > 1e-4f) {
            const float step = std::exp2(std::floor(std::log2(radius)) - 3.0f);
            const float quantized = std::ceil(radius / step) * step;
            float& held = mShadowMapCascadeRadius[c];
            if (held <= 0.0f || radius > held || radius < held - 2.0f * step) {
                held = quantized;
            }
            radius = held;
        }
        float center[3] = { nearC[0] + viewDir[0] * mid, nearC[1] + viewDir[1] * mid, nearC[2] + viewDir[2] * mid };

        // Snap the centre to whole texels along the light's own axes (see the note above).
        const float texelWorldSize = (2.0f * radius) / (float)mShadowMapResolution;
        if (texelWorldSize > 0.0f) {
            float cx = center[0] * lx[0] + center[1] * lx[1] + center[2] * lx[2];
            float cy = center[0] * ly[0] + center[1] * ly[1] + center[2] * ly[2];
            float cz = center[0] * lz[0] + center[1] * lz[1] + center[2] * lz[2];
            cx = std::floor(cx / texelWorldSize) * texelWorldSize;
            cy = std::floor(cy / texelWorldSize) * texelWorldSize;
            // Depth too, not just the two lateral axes. The eye sits a fixed distance behind this point, so
            // an unsnapped depth slides every stored depth value by a fraction of a texel every frame --
            // and a comparison that was a hair on one side of the bias lands on the other, which flickers
            // exactly on the surfaces where the shadow is nearly tangent to the light.
            cz = std::floor(cz / texelWorldSize) * texelWorldSize;
            for (int i = 0; i < 3; i++) {
                center[i] = lx[i] * cx + ly[i] * cy + lz[i] * cz;
            }
        }

        // Pull the eye back far enough that casters above the slice still fall inside the depth range.
        // Margin proportional to the cascade rather than a fixed distance: a flat 1000 units of slack was
        // most of the depth range for a near cascade, and a D16 map spends its precision on whatever range
        // it is given. Scaling with the radius keeps every cascade's precision comparable.
        const float back = radius * 3.0f;
        const float eye[3] = { center[0] - lz[0] * back, center[1] - lz[1] * back, center[2] - lz[2] * back };
        const float zNear = 0.0f;
        const float zFar = back + radius * 2.0f;

        // view * ortho, folded into one row-vector matrix (world position * M -> clip).
        float* m = &matrices[c * 16];
        const float sx = 1.0f / radius;
        const float sy = 1.0f / radius;
        const float sz = 1.0f / (zFar - zNear);
        m[0] = lx[0] * sx;
        m[1] = ly[0] * sy;
        m[2] = lz[0] * sz;
        m[3] = 0.0f;
        m[4] = lx[1] * sx;
        m[5] = ly[1] * sy;
        m[6] = lz[1] * sz;
        m[7] = 0.0f;
        m[8] = lx[2] * sx;
        m[9] = ly[2] * sy;
        m[10] = lz[2] * sz;
        m[11] = 0.0f;
        m[12] = -(eye[0] * lx[0] + eye[1] * lx[1] + eye[2] * lx[2]) * sx;
        m[13] = -(eye[0] * ly[0] + eye[1] * ly[1] + eye[2] * ly[2]) * sy;
        m[14] = (-(eye[0] * lz[0] + eye[1] * lz[1] + eye[2] * lz[2]) - zNear) * sz;
        m[15] = 1.0f;

        nearDist = farDist;
    }

    // Render layer by layer, cascades within. Every cascade of a layer draws the same caster list, and the
    // backend skips re-uploading a list it already holds -- but only for consecutive calls. Walking
    // cascade-first alternated between the two layers on every call, so the check never matched and the
    // whole list was pushed to the GPU eight times a frame instead of twice. Each of those uploads
    // discards and reallocates the buffer, which is felt as a stutter once a room's mesh is large.
    // Both layers are still cleared and drawn even when empty, so a layer that had casters last frame and
    // none now comes back clear instead of holding stale depth.
    // Texture ids are resolved once per frame, not once per cascade: the lookup is the same for all eight
    // slices, and a range whose texture has been evicted must be skipped consistently across them.
    ResolveShadowAlphaTextures(mShadowAlphaWorldCache);
    ResolveShadowAlphaTextures(mShadowAlphaSceneryReady);
    ResolveShadowAlphaTextures(mShadowAlphaReady[SHADOW_MAP_LAYER_ACTORS]);

    // One summary of each layer's caster geometry, computed after the ids above are resolved because they
    // are part of what gets drawn. Handed to the backend so a slice whose casters AND matrix are both
    // unchanged is left holding the image it already has instead of being cleared and redrawn -- which for
    // the world layer, in a room where only the camera moves, is every cascade every frame.
    const uint64_t layerContentKeys[SHADOW_MAP_LAYERS] = { ShadowMapLayerContentKey(SHADOW_MAP_LAYER_WORLD),
                                                           ShadowMapLayerContentKey(SHADOW_MAP_LAYER_ACTORS) };

    // Can any part of this span land inside the cascade's footprint? Standard conservative box-against-slab
    // test: the box's centre projects to a point and its half-extents project to a radius, so the span is
    // rejected only when the whole box sits off one side. Erring towards keeping a span is harmless; erring
    // the other way would drop a caster, so nothing here may be tightened into an exact test.
    //
    // The two lateral axes are tested on BOTH sides: the rasterizer discards anything outside the viewport
    // whatever its depth, so a span entirely off to one side contributes nothing and skipping it changes no
    // pixel.
    //
    // The depth axis is tested on ONE side only, and the asymmetry is not an oversight.
    //   NEAR side (in front of the cascade, towards the light): never tested. The depth pass runs with depth
    //     clipping disabled on purpose -- a caster above the cascade's slice still has to occlude, and
    //     clipping it away is exactly the shadow the slice exists to record.
    //   FAR side (past the cascade's far plane, away from the light): safe to reject, and free, since the
    //     projection of the box onto that axis is already being computed. The viewport clamps such a span to
    //     depth 1.0, the depth test is LESS, and the slice was cleared to 1.0 -- so it fails the test and
    //     writes nothing. The texel keeps the clear value either way, and every receiver the cascade covers
    //     has ndc z <= 1.0, which reads as lit against it. Drawing the span and skipping it therefore leave
    //     the map bit-identical.
    //
    // The matrix is row-vector (world * M), so the x column is m[0], m[4], m[8] and the translation m[12].
    // w is exactly 1 -- the projection is orthographic by construction -- so clip xyz IS ndc xyz. Depth runs
    // 0 to 1 (the matrix is built for that convention directly, above), the lateral axes -1 to 1.
    auto boxVisible = [](const float* bmin, const float* bmax, const float* m) {
        const float cx = (bmin[0] + bmax[0]) * 0.5f;
        const float cy = (bmin[1] + bmax[1]) * 0.5f;
        const float cz = (bmin[2] + bmax[2]) * 0.5f;
        const float hx = (bmax[0] - bmin[0]) * 0.5f;
        const float hy = (bmax[1] - bmin[1]) * 0.5f;
        const float hz = (bmax[2] - bmin[2]) * 0.5f;
        for (int axis = 0; axis < 3; axis++) {
            const float a0 = m[0 + axis], a1 = m[4 + axis], a2 = m[8 + axis];
            const float centre = (cx * a0) + (cy * a1) + (cz * a2) + m[12 + axis];
            const float radius = (hx * std::fabs(a0)) + (hy * std::fabs(a1)) + (hz * std::fabs(a2));
            if (centre - radius > 1.0f) {
                return false;
            }
            if (axis < 2 && centre + radius < -1.0f) {
                return false;
            }
        }
        return true;
    };

    for (int l = 0; l < SHADOW_MAP_LAYERS; l++) {
        // The world layer draws from the cache, which usually holds the same vector contents as last frame --
        // so on top of skipping the capture, the backend's "same list as the previous call" check also skips
        // the upload across frames, not just across cascades.
        const std::vector<float>& casters =
            (l == SHADOW_MAP_LAYER_WORLD) ? mShadowMapWorldCache : mShadowMapCastersReady[l];
        const ShadowAlphaCasters& alpha =
            (l == SHADOW_MAP_LAYER_WORLD) ? mShadowAlphaWorldCache : mShadowAlphaReady[l];
        // Scenery actors ride in the world layer's slices from their own buffer slot. Separate slots is what
        // keeps the two lists from evicting each other: the cached room mesh stays uploaded across cascades
        // and frames while the scenery list beside it is replaced every frame.
        const bool sceneryHere = (l == SHADOW_MAP_LAYER_WORLD);
        for (int c = 0; c < mShadowMapCascadeCount; c++) {
            // False means this slice already holds exactly what the calls below would draw into it. Nothing
            // may be submitted then -- the backend has not cleared it, has not set the depth pipeline up,
            // and is not the render target.
            if (!mRapi->ShadowMapBeginCascade(l, c, &matrices[c * 16], layerContentKeys[l])) {
                continue;
            }
            if (casters.size() >= 9) {
                const size_t casterVerts = casters.size() / 3;
                if (l == SHADOW_MAP_LAYER_WORLD && !mShadowWorldChunks.empty()) {
                    // Adjacent surviving spans are merged into one draw, so a cascade that keeps everything
                    // still issues exactly one call -- the cull can cost draw calls only where it is also
                    // saving whole spans of geometry.
                    //
                    // Short rejected runs are BRIDGED rather than split around, which is what makes a fine
                    // span size safe. Splitting on every rejection ties the draw-call count to how finely the
                    // spans interleave, and a mesh whose submission order alternates across the cascade edge
                    // would pay one call per span -- trading a rasterisation saving for a more expensive CPU
                    // one. A draw call costs far more than pushing a couple of hundred extra triangles
                    // through a pass with no pixel shader, so a gap smaller than that is simply drawn.
                    // Drawing it is always correct: the spans are contiguous in the buffer, and a rejected
                    // span was only ever rejected as an optimisation.
                    const float* m = &matrices[c * 16];
                    size_t runFirst = 0, runCount = 0, gapCount = 0;
                    for (const ShadowCasterChunk& ch : mShadowWorldChunks) {
                        if (boxVisible(ch.min, ch.max, m)) {
                            if (runCount == 0) {
                                runFirst = ch.firstVertex;
                                gapCount = 0; // nothing to bridge back to
                            }
                            runCount += gapCount + ch.vertexCount;
                            gapCount = 0;
                        } else if (runCount != 0) {
                            gapCount += ch.vertexCount;
                            if (gapCount > kShadowChunkBridgeTriangles * 3) {
                                mRapi->ShadowMapDrawCasters(casters.data(), casterVerts, SHADOW_MAP_CASTER_SLOT_MAIN,
                                                            runFirst, runCount);
                                runCount = 0;
                                gapCount = 0;
                            }
                        }
                    }
                    if (runCount != 0) {
                        mRapi->ShadowMapDrawCasters(casters.data(), casterVerts, SHADOW_MAP_CASTER_SLOT_MAIN, runFirst,
                                                    runCount);
                    }
                } else if (l == SHADOW_MAP_LAYER_WORLD || boxVisible(actorMin, actorMax, &matrices[c * 16])) {
                    // The actor list gets one box for the whole layer rather than spans: it holds the
                    // characters, which stand together, so cutting it up would only add tests. A cascade the
                    // characters are nowhere near now issues no draw at all instead of re-transforming every
                    // one of them to have the viewport throw them away.
                    mRapi->ShadowMapDrawCasters(casters.data(), casterVerts, SHADOW_MAP_CASTER_SLOT_MAIN);
                }
            }
            // Scenery actors are per-frame and uncached, and were going into all four cascades untested. Same
            // one-box treatment as the characters above.
            if (sceneryHere && mShadowSceneryReady.size() >= 9 &&
                boxVisible(sceneryMin, sceneryMax, &matrices[c * 16])) {
                mRapi->ShadowMapDrawCasters(mShadowSceneryReady.data(), mShadowSceneryReady.size() / 3,
                                            SHADOW_MAP_CASTER_SLOT_SCENERY);
            }
            // Alpha-cutout casters second, so the one big opaque batch keeps the fast path to itself and the
            // pipeline switch happens once per cascade rather than being interleaved.
            if (alpha.VertexCount() >= 3) {
                mRapi->ShadowMapUploadAlphaCasters(alpha.verts.data(), alpha.VertexCount());
                for (const ShadowAlphaRange& r : alpha.ranges) {
                    if (r.textureId != UINT32_MAX && r.vertexCount >= 3 &&
                        boxVisible(r.min, r.max, &matrices[c * 16])) {
                        mRapi->ShadowMapDrawAlphaRange(r.textureId, r.firstVertex, r.vertexCount);
                    }
                }
            }
            if (sceneryHere && mShadowAlphaSceneryReady.VertexCount() >= 3) {
                mRapi->ShadowMapUploadAlphaCasters(mShadowAlphaSceneryReady.verts.data(),
                                                  mShadowAlphaSceneryReady.VertexCount());
                for (const ShadowAlphaRange& r : mShadowAlphaSceneryReady.ranges) {
                    if (r.textureId != UINT32_MAX && r.vertexCount >= 3 &&
                        boxVisible(r.min, r.max, &matrices[c * 16])) {
                        mRapi->ShadowMapDrawAlphaRange(r.textureId, r.firstVertex, r.vertexCount);
                    }
                }
            }
        }
    }

    mRapi->ShadowMapEndPass();

    // Caster census, once a second while a debug mode is on. Whether a vanished shadow is a caster that
    // stopped being captured or a receiver that stopped sampling is invisible from the picture alone, and
    // these four numbers separate them: a shadow that disappears while its layer's count holds steady was
    // captured and not sampled, and one whose count drops was never submitted.
    if (mShadowMapDebug > 0.5f) {
        static int sCensusFrames = 0;
        if (++sCensusFrames >= 60) {
            sCensusFrames = 0;
            SPDLOG_INFO("Shadow map casters: world {} tris (+{} cutout in {} batches), scenery {} tris (+{} "
                        "cutout in {} batches), actors {} tris (+{} cutout in {} batches)",
                        mShadowMapWorldCache.size() / 9, mShadowAlphaWorldCache.VertexCount() / 3,
                        mShadowAlphaWorldCache.ranges.size(), mShadowSceneryReady.size() / 9,
                        mShadowAlphaSceneryReady.VertexCount() / 3, mShadowAlphaSceneryReady.ranges.size(),
                        mShadowMapCastersReady[SHADOW_MAP_LAYER_ACTORS].size() / 9,
                        mShadowAlphaReady[SHADOW_MAP_LAYER_ACTORS].VertexCount() / 3,
                        mShadowAlphaReady[SHADOW_MAP_LAYER_ACTORS].ranges.size());
        }
    }
    mRapi->SetShadowMapParams(matrices, splits, mShadowMapCascadeCount, mShadowMapBlendFraction,
                              mShadowMapNormalOffset, mShadowMapStrength, mShadowMapFilterWidth,
                              mShadowMapDebug, mShadowMapEdgeHardness, mShadowMapEdgeHardnessFar);
}

void Interpreter::RenderShadowVolumes() {
    const float coreAlpha = std::clamp(mToonShadowAlpha, 0.0f, 1.0f);
    auto clearAccums = [this] {
        for (int b = 0; b < kShadowBands; b++) {
            mShadowVolumeAccum[b].clear();
            mShadowVolumeKind[b].clear();
        }
    };
    size_t accumTotal = 0;
    for (int b = 0; b < kShadowBands; b++) {
        accumTotal += mShadowVolumeAccum[b].size();
    }
    if (accumTotal < 9 || coreAlpha <= 0.0f) {
        clearAccums();
        return;
    }

    const uint64_t savedCombine = mRdp->combine_mode;
    const uint32_t savedOtherL = mRdp->other_mode_l;
    const uint32_t savedOtherH = mRdp->other_mode_h;
    const uint32_t savedGeo = mRsp->geometry_mode;
    const bool savedToon = mRdp->toon;
    const bool savedToonShadow = mRdp->toon_shadow;
    const bool savedGray = mRdp->grayscale;
    const struct RGBA savedPrim = mRdp->prim_color;

    mRdp->toon = false;
    mRdp->toon_shadow = false; // the volume geometry must not be re-captured
    mRdp->grayscale = false;
    mRdp->other_mode_h = (savedOtherH & ~(3U << G_MDSFT_CYCLETYPE)) | G_CYC_1CYCLE;
    GfxDpSetCombineMode(color_comb(0, 0, 0, G_CCMUX_PRIMITIVE), alpha_comb(0, 0, 0, G_ACMUX_PRIMITIVE), 0, 0);

    const uint32_t cullFront = get_attr(CULL_FRONT);
    const uint32_t cullBack = get_attr(CULL_BACK);
    const uint32_t cullBoth = get_attr(CULL_BOTH);
    const GfxClipParameters clip = mRapi->GetClipParameters();
    // How many opacity steps the soft edge uses (EdgeSoftness rings + the core). Leftover ring geometry from
    // a mid-frame softness change draws at the lightest step rather than being dropped.
    const int bandsTotal = std::clamp(mShadowEdgeSoftness, 0, kShadowBands - 1) + 1;

    // Copy triangle t's three cached verts into the scratch slots GfxSpTri1 reads.
    auto loadTri = [&](size_t t) {
        mRsp->loaded_vertices[MAX_VERTICES + 0] = mShadowXform[t * 3 + 0];
        mRsp->loaded_vertices[MAX_VERTICES + 1] = mShadowXform[t * 3 + 1];
        mRsp->loaded_vertices[MAX_VERTICES + 2] = mShadowXform[t * 3 + 2];
    };
    size_t triTotal = 0; // triangles of the band currently staged in mShadowXform

    // SOH [Enhancement] Actor shadow: batched volume submission. The volume is thousands of identical-state
    // triangles; routing each through GfxSpTri1 repays the full combiner/shader/mode resolution per triangle —
    // the measured CPU bottleneck. Instead, resolve+load all render state ONCE per band (draw a single triangle
    // through the normal path with culling off, then discard it — nothing is submitted because its state-change
    // Flushes fire on an empty buffer and we zero the buffer afterward), learn the exact per-vertex VBO layout
    // from what it wrote, and fill the vertex buffer directly for the rest. The z-fail pass is two-sided
    // (no culling — the GPU's facing picks increment vs decrement per triangle), so the software cull
    // below only fires for geometry modes that carry cull bits. The capture must rerun each band
    // because the band's composite re-resolves the pipeline with its own state. mBufVbo is backend-agnostic,
    // so this single interpreter-side change is correct on all three backends.
    size_t shadowStride = 0;                           // floats per vertex (0 = layout unexpected -> fallback)
    float shadowColorBlock[VBO_MAX_FLOATS_PER_VERTEX]; // constant non-position vertex data (the flat prim color)

    auto appendShadowVert = [&](const LoadedVertex& v) {
        float z = v.z;
        if (clip.z_is_from_0_to_1) {
            z = (z + v.w) / 2.0f;
        }
        mBufVbo[mBufVboLen++] = v.x;
        mBufVbo[mBufVboLen++] = clip.invertY ? -v.y : v.y;
        mBufVbo[mBufVboLen++] = z;
        mBufVbo[mBufVboLen++] = v.w;
        for (size_t f = 4; f < shadowStride; f++) {
            mBufVbo[mBufVboLen++] = shadowColorBlock[f];
        }
    };
    // Software backface cull, identical to GfxSpTri1's, so each pass keeps the same faces it did per-triangle.
    auto faceCulled = [&](const LoadedVertex& v1, const LoadedVertex& v2, const LoadedVertex& v3) -> bool {
        const uint32_t cullType = mRsp->geometry_mode & cullBoth;
        if (cullType == 0) {
            return false;
        }
        const float dx1 = (v1.x / v1.w) - (v2.x / v2.w), dy1 = (v1.y / v1.w) - (v2.y / v2.w);
        const float dx2 = (v3.x / v3.w) - (v2.x / v2.w), dy2 = (v3.y / v3.w) - (v2.y / v2.w);
        float cross = (dx1 * dy2) - (dy1 * dx2);
        if ((v1.w < 0) ^ (v2.w < 0) ^ (v3.w < 0)) {
            cross = -cross;
        }
        if (ucode_handler_index == UcodeHandlers::ucode_f3dex2 &&
            (mRsp->extra_geometry_mode & G_EX_INVERT_CULLING) == 1) {
            cross = -cross;
        }
        if (cullType == cullFront) {
            return cross <= 0;
        }
        if (cullType == cullBack) {
            return cross >= 0;
        }
        return true; // cull_both
    };
    // Fill + submit the whole volume for the current pass: fast direct-write path, or the per-triangle GfxSpTri1
    // fallback if the captured layout was unexpected (shadowStride == 0).
    auto drawVolume = [&]() {
        if (shadowStride == 0) {
            for (size_t t = 0; t < triTotal; t++) {
                loadTri(t);
                GfxSpTri1(MAX_VERTICES + 0, MAX_VERTICES + 1, MAX_VERTICES + 2, false);
            }
        } else {
            for (size_t t = 0; t < triTotal; t++) {
                const LoadedVertex& a = mShadowXform[t * 3 + 0];
                const LoadedVertex& b = mShadowXform[t * 3 + 1];
                const LoadedVertex& c = mShadowXform[t * 3 + 2];
                if (faceCulled(a, b, c)) {
                    continue;
                }
                appendShadowVert(a), appendShadowVert(b), appendShadowVert(c);
                if (++mBufVboNumTris == MAX_TRI_BUFFER) {
                    Flush();
                }
            }
        }
        Flush();
    };

    for (int band = 0; band < kShadowBands; band++) {
        const std::vector<float>& vol = mShadowVolumeAccum[band];
        if (vol.size() < 9) {
            continue;
        }
        // Band alpha: full-opacity core, one step lighter per ring.
        const int step = std::min(band, bandsTotal - 1);
        const uint8_t bandA =
            (uint8_t)(coreAlpha * ((float)(bandsTotal - step) / (float)bandsTotal) * 255.0f);
        if (bandA == 0) {
            continue;
        }

        // Transform the band to clip space ONCE; the two stencil passes and the debug overlay reuse it
        // instead of re-projecting every vertex per pass. Vertex color is left undefined: the combine outputs
        // PRIMITIVE (set per pass via prim_color), so the per-vertex shade color is never used.
        const size_t vertCount = vol.size() / 3;
        mShadowXform.resize(vertCount);
        for (size_t vi = 0; vi < vertCount; vi++) {
            const float wx = vol[vi * 3 + 0], wy = vol[vi * 3 + 1], wz = vol[vi * 3 + 2];
            LoadedVertex& d = mShadowXform[vi];
            d.x = AdjXForAspectRatio((wx * mRsp->P_matrix[0][0]) + (wy * mRsp->P_matrix[1][0]) +
                                     (wz * mRsp->P_matrix[2][0]) + mRsp->P_matrix[3][0]);
            d.y = (wx * mRsp->P_matrix[0][1]) + (wy * mRsp->P_matrix[1][1]) + (wz * mRsp->P_matrix[2][1]) +
                  mRsp->P_matrix[3][1];
            d.z = (wx * mRsp->P_matrix[0][2]) + (wy * mRsp->P_matrix[1][2]) + (wz * mRsp->P_matrix[2][2]) +
                  mRsp->P_matrix[3][2];
            d.w = (wx * mRsp->P_matrix[0][3]) + (wy * mRsp->P_matrix[1][3]) + (wz * mRsp->P_matrix[2][3]) +
                  mRsp->P_matrix[3][3];
            d.u = d.v = 0;
            d.clip_rej = 0;
        }
        triTotal = vertCount / 3;

        // Shared mask render state (both z-fail passes): flat transparent black, depth-tested XLU. Then
        // resolve it into the pipeline and learn the VBO layout via the discarded setup triangle (see the
        // batched-submission note above) — required per band, since the previous band's composite re-resolved
        // the pipeline with its own state.
        mRdp->prim_color = { 0, 0, 0, 0 };
        mRdp->other_mode_l = G_RM_AA_ZB_XLU_SURF | G_RM_AA_ZB_XLU_SURF2;
        shadowStride = 0;
        {
            const uint32_t geoSaved = mRsp->geometry_mode;
            mRsp->geometry_mode = G_ZBUFFER; // no cull -> GfxSpTri1 runs its full setup for this triangle
            Flush();
            loadTri(0);
            GfxSpTri1(MAX_VERTICES + 0, MAX_VERTICES + 1, MAX_VERTICES + 2, false);
            mRsp->geometry_mode = geoSaved;
            const size_t stride = (mBufVboNumTris > 0) ? (mBufVboLen / 3) : 0;
            if (stride > 4 && stride <= VBO_MAX_FLOATS_PER_VERTEX) {
                for (size_t f = 4; f < stride; f++) {
                    shadowColorBlock[f] = mBufVbo[f]; // vertex 0's data past the 4 position floats (x,y,z,w)
                }
                shadowStride = stride;
            }
            mBufVboLen = 0, mBufVboNumTris = 0; // discard the setup triangle (re-drawn, culled, in the passes)
        }

        // z-fail mask, ONE two-sided pass: the GPU's facing picks the op per triangle (wrap-increment
        // one facing, wrap-decrement the other), so the volume is submitted once with culling off
        // instead of as a cull-front/cull-back pass pair — half the iteration, VBO fill and draw
        // calls. Wrap ops + the composite's nonzero test make primitive order and facing polarity
        // irrelevant; only "opposite ops per facing" matters (see StencilMode::VolumeIncrDecr).
        mRsp->geometry_mode = G_ZBUFFER;
        Flush();
        mRapi->SetStencilMode((int)StencilMode::VolumeIncrDecr);
        drawVolume();

        // composite (self-clearing); full-screen clip-space quad, no depth test.
        mRdp->prim_color = { 0, 0, 0, bandA };
        mRdp->other_mode_l = G_RM_AA_XLU_SURF | G_RM_AA_XLU_SURF2;
        mRsp->geometry_mode = 0;
        mRapi->SetStencilMode((int)StencilMode::Composite);
        {
            const float qx[4] = { -1.0f, 1.0f, 1.0f, -1.0f }, qy[4] = { -1.0f, -1.0f, 1.0f, 1.0f };
            for (int k = 0; k < 4; k++) {
                LoadedVertex* d = &mRsp->loaded_vertices[MAX_VERTICES + k];
                d->x = qx[k], d->y = qy[k], d->z = 0.0f, d->w = 1.0f;
                d->u = d->v = 0;
                d->color = mRdp->prim_color;
                d->clip_rej = 0;
            }
            GfxSpTri1(MAX_VERTICES + 0, MAX_VERTICES + 1, MAX_VERTICES + 2, false);
            GfxSpTri1(MAX_VERTICES + 0, MAX_VERTICES + 2, MAX_VERTICES + 3, false);
        }
        Flush();

        // debug overlay: draw this band's volumes translucently (black caps, blue walls), no stencil/depth.
        if (mShadowShowVolume) {
            mRapi->SetStencilMode((int)StencilMode::Off);
            mRsp->geometry_mode = cullBack;
            mRdp->other_mode_l = G_RM_AA_XLU_SURF | G_RM_AA_XLU_SURF2;
            const std::vector<uint8_t>& kinds = mShadowVolumeKind[band];
            for (int batch = 0; batch < 2; batch++) {
                mRdp->prim_color = (batch == 0) ? RGBA{ 0, 0, 0, 128 } : RGBA{ 40, 90, 255, 128 };
                for (size_t t = 0; (t * 3) + 3 <= vertCount; t++) {
                    const bool isCap = t < kinds.size() && kinds[t] == 0;
                    if (isCap != (batch == 0)) {
                        continue;
                    }
                    loadTri(t);
                    GfxSpTri1(MAX_VERTICES + 0, MAX_VERTICES + 1, MAX_VERTICES + 2, false);
                }
                Flush();
            }
        }
    }

    mRapi->SetStencilMode((int)StencilMode::Off);
    mRdp->prim_color = savedPrim;
    mRdp->combine_mode = savedCombine;
    mRdp->other_mode_l = savedOtherL;
    mRdp->other_mode_h = savedOtherH;
    mRsp->geometry_mode = savedGeo;
    mRdp->toon = savedToon;
    mRdp->toon_shadow = savedToonShadow;
    mRdp->grayscale = savedGray;

    clearAccums();
}

void Interpreter::GfxDpSetGrayscaleColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->grayscale_color.r = r;
    mRdp->grayscale_color.g = g;
    mRdp->grayscale_color.b = b;
    mRdp->grayscale_color.a = a;
}

void Interpreter::GfxDpSetEnvColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->env_color.r = r;
    mRdp->env_color.g = g;
    mRdp->env_color.b = b;
    mRdp->env_color.a = a;
}

void Interpreter::GfxDpSetPrimColor(uint8_t m, uint8_t l, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->prim_lod_fraction = l;
    mRdp->prim_color.r = r;
    mRdp->prim_color.g = g;
    mRdp->prim_color.b = b;
    mRdp->prim_color.a = a;
}

void Interpreter::GfxDpSetFogColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->fog_color.r = r;
    mRdp->fog_color.g = g;
    mRdp->fog_color.b = b;
    mRdp->fog_color.a = a;
}

void Interpreter::GfxDpSetBlendColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    // TODO: Implement this command..
}

void Interpreter::GfxDpSetFillColor(uint32_t packed_color) {
    uint16_t col16 = (uint16_t)packed_color;
    uint32_t r = col16 >> 11;
    uint32_t g = (col16 >> 6) & 0x1f;
    uint32_t b = (col16 >> 1) & 0x1f;
    uint32_t a = col16 & 1;
    mRdp->fill_color.r = SCALE_5_8(r);
    mRdp->fill_color.g = SCALE_5_8(g);
    mRdp->fill_color.b = SCALE_5_8(b);
    mRdp->fill_color.a = a * 255;
}

void Interpreter::GfxDrawRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry) {
    uint32_t saved_other_mode_h = mRdp->other_mode_h;
    uint32_t cycle_type = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE));

    if (cycle_type == G_CYC_COPY) {
        mRdp->other_mode_h = (mRdp->other_mode_h & ~(3U << G_MDSFT_TEXTFILT)) | G_TF_POINT;
    }

    // U10.2 coordinates
    float ulxf = ulx;
    float ulyf = uly;
    float lrxf = lrx;
    float lryf = lry;

    ulxf = ulxf / (4.0f * HALF_SCREEN_WIDTH(mActiveFrameBuffer)) - 1.0f;
    ulyf = -(ulyf / (4.0f * HALF_SCREEN_HEIGHT(mActiveFrameBuffer))) + 1.0f;
    lrxf = lrxf / (4.0f * HALF_SCREEN_WIDTH(mActiveFrameBuffer)) - 1.0f;
    lryf = -(lryf / (4.0f * HALF_SCREEN_HEIGHT(mActiveFrameBuffer))) + 1.0f;

    ulxf = AdjXForAspectRatio(ulxf);
    lrxf = AdjXForAspectRatio(lrxf);

    struct LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    struct LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    struct LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    struct LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];

    ul->x = ulxf;
    ul->y = ulyf;
    ul->z = -1.0f;
    ul->w = 1.0f;

    ll->x = ulxf;
    ll->y = lryf;
    ll->z = -1.0f;
    ll->w = 1.0f;

    lr->x = lrxf;
    lr->y = lryf;
    lr->z = -1.0f;
    lr->w = 1.0f;

    ur->x = lrxf;
    ur->y = ulyf;
    ur->z = -1.0f;
    ur->w = 1.0f;

    // The coordinates for texture rectangle shall bypass the viewport setting
    struct XYWidthHeight default_viewport;
    if (!mFbActive) {
        default_viewport = { 0, (int16_t)mNativeDimensions.height, mNativeDimensions.width, mNativeDimensions.height };
    } else {
        default_viewport = { 0, (int16_t)mActiveFrameBuffer->second.orig_height, mActiveFrameBuffer->second.orig_width,
                             mActiveFrameBuffer->second.orig_height };
    }

    struct XYWidthHeight viewport_saved = mRdp->viewport;
    uint32_t geometry_mode_saved = mRsp->geometry_mode;

    AdjustVIewportOrScissor(&default_viewport);

    mRdp->viewport = default_viewport;
    mRdp->viewport_or_scissor_changed = true;
    mRsp->geometry_mode = 0;

    GfxSpTri1(MAX_VERTICES + 0, MAX_VERTICES + 1, MAX_VERTICES + 3, true);
    GfxSpTri1(MAX_VERTICES + 1, MAX_VERTICES + 2, MAX_VERTICES + 3, true);

    mRsp->geometry_mode = geometry_mode_saved;
    mRdp->viewport = viewport_saved;
    mRdp->viewport_or_scissor_changed = true;

    if (cycle_type == G_CYC_COPY) {
        mRdp->other_mode_h = saved_other_mode_h;
    }
}

void Interpreter::GfxDpTextureRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, uint8_t tile, int16_t uls,
                                        int16_t ult, int16_t dsdx, int16_t dtdy, bool flip) {
    // printf("render %d at %d\n", tile, lrx);
    uint64_t saved_combine_mode = mRdp->combine_mode;
    if ((mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_COPY) {
        // Per RDP Command Summary Set Tile's shift s and this dsdx should be set to 4 texels
        // Divide by 4 to get 1 instead
        dsdx >>= 2;

        // Color combiner is turned off in copy mode
        GfxDpSetCombineMode(color_comb(0, 0, 0, G_CCMUX_TEXEL0), alpha_comb(0, 0, 0, G_ACMUX_TEXEL0), 0, 0);

        // Per documentation one extra pixel is added in this modes to each edge
        lrx += 1 << 2;
        lry += 1 << 2;
    }

    // uls and ult are S10.5
    // dsdx and dtdy are S5.10
    // lrx, lry, ulx, uly are U10.2
    // lrs, lrt are S10.5
    if (flip) {
        dsdx = -dsdx;
        dtdy = -dtdy;
    }
    int16_t width = !flip ? lrx - ulx : lry - uly;
    int16_t height = !flip ? lry - uly : lrx - ulx;
    float lrs = ((uls << 7) + dsdx * width) >> 7;
    float lrt = ((ult << 7) + dtdy * height) >> 7;

    LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];
    ul->u = uls;
    ul->v = ult;
    lr->u = lrs;
    lr->v = lrt;
    if (!flip) {
        ll->u = uls;
        ll->v = lrt;
        ur->u = lrs;
        ur->v = ult;
    } else {
        ll->u = lrs;
        ll->v = ult;
        ur->u = uls;
        ur->v = lrt;
    }

    uint8_t saved_tile = mRdp->first_tile_index;
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = tile;

    GfxDrawRectangle(ulx, uly, lrx, lry);
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = saved_tile;
    mRdp->combine_mode = saved_combine_mode;
}

void Interpreter::GfxDpImageRectangle(int32_t tile, int32_t w, int32_t h, int32_t ulx, int32_t uly, int16_t uls,
                                      int16_t ult, int32_t lrx, int32_t lry, int16_t lrs, int16_t lrt) {

    LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];
    ul->u = uls * 32;
    ul->v = ult * 32;
    lr->u = lrs * 32;
    lr->v = lrt * 32;
    ll->u = uls * 32;
    ll->v = lrt * 32;
    ur->u = lrs * 32;
    ur->v = ult * 32;

    // ensure we have the correct texture size, format and starting position
    mRdp->texture_tile[tile].siz = G_IM_SIZ_8b;
    mRdp->texture_tile[tile].fmt = G_IM_FMT_RGBA;
    mRdp->texture_tile[tile].cms = 0;
    mRdp->texture_tile[tile].cmt = 0;
    mRdp->texture_tile[tile].shifts = 0;
    mRdp->texture_tile[tile].shiftt = 0;
    mRdp->texture_tile[tile].uls = 0 * 4;
    mRdp->texture_tile[tile].ult = 0 * 4;
    mRdp->texture_tile[tile].lrs = w * 4;
    mRdp->texture_tile[tile].lrt = h * 4;
    mRdp->texture_tile[tile].line_size_bytes = w << (mRdp->texture_tile[tile].siz >> 1);

    auto& loadtex = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index];
    loadtex.full_image_line_size_bytes = loadtex.line_size_bytes = mRdp->texture_tile[tile].line_size_bytes;
    loadtex.size_bytes = loadtex.orig_size_bytes = loadtex.line_size_bytes * h;

    uint8_t saved_tile = mRdp->first_tile_index;
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = tile;

    GfxDrawRectangle(ulx, uly, lrx, lry);
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = saved_tile;
}

void Interpreter::GfxDpFillRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry) {
    if (mRdp->color_image_address == mRdp->z_buf_address) {
        // Don't clear Z buffer here since we already did it with glClear
        return;
    }
    uint32_t mode = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE));

    // OTRTODO: This is a bit of a hack for widescreen screen fades, but it'll work for now...
    if (ulx == 0 && uly == 0 && lrx == (319 * 4) && lry == (239 * 4)) {
        ulx = -1024;
        uly = -1024;
        lrx = 2048;
        lry = 2048;
    }

    if (mode == G_CYC_COPY || mode == G_CYC_FILL) {
        // Per documentation one extra pixel is added in this modes to each edge
        lrx += 1 << 2;
        lry += 1 << 2;
    }

    for (int i = MAX_VERTICES; i < MAX_VERTICES + 4; i++) {
        LoadedVertex* v = &mRsp->loaded_vertices[i];
        v->color = mRdp->fill_color;
    }

    uint64_t saved_combine_mode = mRdp->combine_mode;

    if (mode == G_CYC_FILL) {
        GfxDpSetCombineMode(color_comb(0, 0, 0, G_CCMUX_SHADE), alpha_comb(0, 0, 0, G_ACMUX_SHADE), 0, 0);
    }

    GfxDrawRectangle(ulx, uly, lrx, lry);
    mRdp->combine_mode = saved_combine_mode;
}

void Interpreter::GfxDpSetZImage(void* zBufAddr) {
    mRdp->z_buf_address = zBufAddr;
}

void Interpreter::GfxDpSetColorImage(uint32_t format, uint32_t size, uint32_t width, void* address) {
    mRdp->color_image_address = address;
}

void Interpreter::GfxSpSetOtherMode(uint32_t shift, uint32_t num_bits, uint64_t mode) {
    uint64_t mask = (((uint64_t)1 << num_bits) - 1) << shift;
    uint64_t om = mRdp->other_mode_l | ((uint64_t)mRdp->other_mode_h << 32);
    om = (om & ~mask) | mode;
    mRdp->other_mode_l = (uint32_t)om;
    mRdp->other_mode_h = (uint32_t)(om >> 32);
}

void Interpreter::GfxDpSetOtherMode(uint32_t h, uint32_t l) {
    mRdp->other_mode_h = h;
    mRdp->other_mode_l = l;
}

void Interpreter::Gfxs2dexBgCopy(F3DuObjBg* bg) {
    /*
    bg->b.imageX = 0;
    bg->b.imageW = width * 4;
    bg->b.frameX = frameX * 4;
    bg->b.imageY = 0;
    bg->b.imageH = height * 4;
    bg->b.frameY = frameY * 4;
    bg->b.imagePtr = source;
    bg->b.imageLoad = G_BGLT_LOADTILE;
    bg->b.imageFmt = fmt;
    bg->b.imageSiz = siz;
    bg->b.imagePal = 0;
    bg->b.imageFlip = 0;
    */

    uintptr_t data = (uintptr_t)bg->b.imagePtr;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if ((bool)gfx_check_image_signature((char*)data)) {
        std::shared_ptr<Fast::Texture> tex = std::static_pointer_cast<Fast::Texture>(
            Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess((char*)data));
        texFlags = tex->Flags;
        rawTexMetadata.width = tex->Width;
        rawTexMetadata.height = tex->Height;
        rawTexMetadata.h_byte_scale = tex->HByteScale;
        rawTexMetadata.v_pixel_scale = tex->VPixelScale;
        rawTexMetadata.type = tex->Type;
        rawTexMetadata.resource = tex;
        data = (uintptr_t) reinterpret_cast<char*>(tex->ImageData);
    }

    s16 dsdx = 4 << 10;
    s16 uls = bg->b.imageX << 3;
    // Flip flag only flips horizontally
    if (bg->b.imageFlip == G_BG_FLAG_FLIPS) {
        dsdx = -dsdx;
        uls = (bg->b.imageW - bg->b.imageX) << 3;
    }

    SUPPORT_CHECK(bg->b.imageSiz == G_IM_SIZ_16b);
    GfxDpSetTextureImage(G_IM_FMT_RGBA, G_IM_SIZ_16b, 0, nullptr, texFlags, rawTexMetadata, (void*)data);
    GfxDpSetTile(G_IM_FMT_RGBA, G_IM_SIZ_16b, 0, 0, G_TX_LOADTILE, 0, 0, 0, 0, 0, 0, 0);
    GfxDpLoadBlock(G_TX_LOADTILE, 0, 0, (bg->b.imageW * bg->b.imageH >> 4) - 1, 0);
    GfxDpSetTile(bg->b.imageFmt, G_IM_SIZ_16b, bg->b.imageW >> 4, 0, G_TX_RENDERTILE, bg->b.imagePal, 0, 0, 0, 0, 0, 0);
    GfxDpSetTileSize(G_TX_RENDERTILE, 0, 0, bg->b.imageW, bg->b.imageH);
    GfxDpTextureRectangle(bg->b.frameX, bg->b.frameY, bg->b.frameX + bg->b.imageW - 4, bg->b.frameY + bg->b.imageH - 4,
                          G_TX_RENDERTILE, uls, bg->b.imageY << 3, dsdx, 1 << 10, false);
}

void Interpreter::Gfxs2dexBg1cyc(F3DuObjBg* bg) {
    uintptr_t data = (uintptr_t)bg->b.imagePtr;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if ((bool)gfx_check_image_signature((char*)data)) {
        std::shared_ptr<Fast::Texture> tex = std::static_pointer_cast<Fast::Texture>(
            Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess((char*)data));
        texFlags = tex->Flags;
        rawTexMetadata.width = tex->Width;
        rawTexMetadata.height = tex->Height;
        rawTexMetadata.h_byte_scale = tex->HByteScale;
        rawTexMetadata.v_pixel_scale = tex->VPixelScale;
        rawTexMetadata.type = tex->Type;
        rawTexMetadata.resource = tex;
        data = (uintptr_t) reinterpret_cast<char*>(tex->ImageData);
    }

    // TODO: Implement bg scaling correctly
    s16 uls = bg->b.imageX >> 2;
    s16 lrs = bg->b.imageW >> 2;

    s16 dsdxRect = 1 << 10;
    s16 ulsRect = bg->b.imageX << 3;
    // Flip flag only flips horizontally
    if (bg->b.imageFlip == G_BG_FLAG_FLIPS) {
        dsdxRect = -dsdxRect;
        ulsRect = (bg->b.imageW - bg->b.imageX) << 3;
    }

    GfxDpSetTextureImage(bg->b.imageFmt, bg->b.imageSiz, bg->b.imageW >> 2, nullptr, texFlags, rawTexMetadata,
                         (void*)data);
    GfxDpSetTile(bg->b.imageFmt, bg->b.imageSiz, 0, 0, G_TX_LOADTILE, 0, 0, 0, 0, 0, 0, 0);
    GfxDpLoadBlock(G_TX_LOADTILE, 0, 0, (bg->b.imageW * bg->b.imageH >> 4) - 1, 0);
    GfxDpSetTile(bg->b.imageFmt, bg->b.imageSiz, (((lrs - uls) * bg->b.imageSiz) + 7) >> 3, 0, G_TX_RENDERTILE,
                 bg->b.imagePal, 0, 0, 0, 0, 0, 0);
    GfxDpSetTileSize(G_TX_RENDERTILE, 0, 0, bg->b.imageW, bg->b.imageH);

    GfxDpTextureRectangle(bg->b.frameX, bg->b.frameY, bg->b.frameW, bg->b.frameH, G_TX_RENDERTILE, ulsRect,
                          bg->b.imageY << 3, dsdxRect, 1 << 10, false);
}

void Interpreter::Gfxs2dexRecyCopy(F3DuObjSprite* spr) {
    s16 dsdx = 4 << 10;
    [[maybe_unused]] s16 uls = spr->s.objX << 3;
    // Flip flag only flips horizontally
    if (spr->s.imageFlags == G_BG_FLAG_FLIPS) {
        dsdx = -dsdx;
        uls = (spr->s.imageW - spr->s.objX) << 3;
    }

    int realX = spr->s.objX >> 2;
    int realY = spr->s.objY >> 2;
    int realW = (((spr->s.imageW)) >> 5);
    int realH = (((spr->s.imageH)) >> 5);
    float realSW = spr->s.scaleW / 1024.0f;
    float realSH = spr->s.scaleH / 1024.0f;

    int testX = (realX + (realW / realSW));
    int testY = (realY + (realH / realSH));

    GfxDpTextureRectangle(realX << 2, realY << 2, testX << 2, testY << 2, G_TX_RENDERTILE,
                          (s32)mRdp->texture_tile[0].uls << 3, (s32)mRdp->texture_tile[0].ult << 3,
                          (float)(1 << 10) * realSW, (float)(1 << 10) * realSH, false);
}

void* Interpreter::SegAddr(uintptr_t w1) {
    // Segmented?
    if (w1 & 1) {
        uint32_t segNum = (uint32_t)(w1 >> 24);

        uint32_t offset = w1 & 0x00FFFFFE;

        if (mSegmentPointers[segNum] != 0) {
            return (void*)(mSegmentPointers[segNum] + offset);
        } else {
            return (void*)w1;
        }
    } else {
        return (void*)w1;
    }
}

#define C0(pos, width) ((cmd->words.w0 >> (pos)) & ((1U << width) - 1))
#define C1(pos, width) ((cmd->words.w1 >> (pos)) & ((1U << width) - 1))

void GfxExecStack::start(F3DGfx* dlist) {
    while (!cmd_stack.empty())
        cmd_stack.pop();
    gfx_path.clear();
    cmd_stack.push(dlist);
    disp_stack.clear();
}

void GfxExecStack::stop() {
    while (!cmd_stack.empty())
        cmd_stack.pop();
    gfx_path.clear();
}

F3DGfx*& GfxExecStack::currCmd() {
    return cmd_stack.top();
}

void GfxExecStack::openDisp(const char* file, int line) {
    disp_stack.push_back({ file, line });
}
void GfxExecStack::closeDisp() {
    disp_stack.pop_back();
}
const std::vector<GfxExecStack::CodeDisp>& GfxExecStack::getDisp() const {
    return disp_stack;
}

void GfxExecStack::branch(F3DGfx* caller) {
    F3DGfx* old = cmd_stack.top();
    cmd_stack.pop();
    cmd_stack.push(nullptr);
    cmd_stack.push(old);

    gfx_path.push_back(caller);
}

void GfxExecStack::call(F3DGfx* caller, F3DGfx* callee) {
    cmd_stack.push(callee);
    gfx_path.push_back(caller);
}

F3DGfx* GfxExecStack::ret() {
    F3DGfx* cmd = cmd_stack.top();

    cmd_stack.pop();
    if (!gfx_path.empty()) {
        gfx_path.pop_back();
    }

    while (cmd_stack.size() > 0 && cmd_stack.top() == nullptr) {
        cmd_stack.pop();
        if (!gfx_path.empty()) {
            gfx_path.pop_back();
        }
    }
    return cmd;
}

void gfx_set_framebuffer(int fb, float noise_scale);
void gfx_reset_framebuffer();
void gfx_copy_framebuffer(int fb_dst_id, int fb_src_id, bool copyOnce, bool* hasCopiedPtr);

// The main type of the handler function. These function will take a pointer to a pointer to a Gfx. It needs to be a
// double pointer because we sometimes need to increment and decrement the underlying pointer Returns false if the
// current opcode should be incremented after the handler ends.
typedef bool (*GfxOpcodeHandlerFunc)(F3DGfx** gfx);

bool gfx_load_ucode_handler_f3dex2(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->mRsp->fog_mul = 0;
    gfx->mRsp->fog_offset = 0;
    return false;
}

bool gfx_cull_dl_handler_f3dex2(F3DGfx** cmd) {
    // TODO:
    return false;
}

bool gfx_marker_handler_otr(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    (*cmd0)++;
    F3DGfx* cmd = (*cmd0);
    gfx->mMarkerOn = true;
    return false;
}

bool gfx_invalidate_tex_cache_handler_f3dex2(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    const uintptr_t texAddr = (*cmd)->words.w1;

    if (texAddr == 0) {
        gfx->TextureCacheClear();
    } else {
        gfx->TextureCacheDelete((const uint8_t*)texAddr);
    }
    return false;
}

bool gfx_noop_handler_f3dex2(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    const char* filename = (const char*)(cmd)->words.w1;
    uint32_t p = C0(16, 8);
    uint32_t l = C0(0, 16);
    if (p == 7) {
        g_exec_stack.openDisp(filename, l);
    } else if (p == 8) {
        if (g_exec_stack.disp_stack.size() == 0) {
            SPDLOG_WARN("CLOSE_DISPS without matching open {}:{}", p, l);
        } else {
            g_exec_stack.closeDisp();
        }
    }
    return false;
}

bool gfx_mtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t mtxAddr = cmd->words.w1;

    gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, (const int32_t*)gfx->SegAddr(mtxAddr));
    return false;
}
// Seems to be the same for all other non F3DEX2 microcodes...
bool gfx_mtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t mtxAddr = cmd->words.w1;

    gfx->GfxSpMatrix(C0(16, 8), (const int32_t*)gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_mtx_otr_filepath_handler_custom_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* fileName = (const char*)cmd->words.w1;
    const int32_t* mtx = (const int32_t*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(
        (const char*)fileName);

    if (mtx != NULL) {
        gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, mtx);
    }

    return false;
}

bool gfx_mtx_otr_filepath_handler_custom_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* fileName = (const char*)cmd->words.w1;
    const int32_t* mtx = (const int32_t*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(
        (const char*)fileName);

    if (mtx != NULL) {
        gfx->GfxSpMatrix(C0(16, 8), mtx);
    }

    return false;
}

bool gfx_mtx_otr_filepath_handler_custom(F3DGfx** cmd0) {
    if (ucode_handler_index == ucode_f3dex2) {
        return gfx_mtx_otr_filepath_handler_custom_f3dex2(cmd0);
    } else {
        return gfx_mtx_otr_filepath_handler_custom_f3d(cmd0);
    }
}

bool gfx_mtx_otr_handler_custom_f3dex2(F3DGfx** cmd0) {
    (*cmd0)++;
    F3DGfx* cmd = *cmd0;

    const uint64_t hash = ((uint64_t)cmd->words.w0 << 32) + cmd->words.w1;
    const int32_t* mtx =
        (const int32_t*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);

    if (mtx != NULL) {
        Interpreter* gfx = mInstance.lock().get();
        cmd--;
        gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, mtx);
        cmd++;
    }

    return false;
}

bool gfx_mtx_otr_handler_custom_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    (*cmd0)++;
    F3DGfx* cmd = *cmd0;

    const uint64_t hash = ((uint64_t)cmd->words.w0 << 32) + cmd->words.w1;
    const int32_t* mtx =
        (const int32_t*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);
    if (mtx != nullptr) {
        cmd--;
        gfx->GfxSpMatrix(C0(16, 8), mtx);
        cmd++;
    }
    return false;
}

bool gfx_mtx_otr_handler_custom(F3DGfx** cmd0) {
    if (ucode_handler_index == ucode_f3dex2) {
        return gfx_mtx_otr_handler_custom_f3dex2(cmd0);
    } else {
        return gfx_mtx_otr_handler_custom_f3d(cmd0);
    }
}

bool gfx_pop_mtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpPopMatrix((uint32_t)(cmd->words.w1 / 64));

    return false;
}

bool gfx_pop_mtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpPopMatrix(1);

    return false;
}

bool gfx_movemem_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovememF3dex2(C0(0, 8), C0(8, 8) * 8, gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_movemem_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovememF3d(C0(16, 8), 0, gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_movemem_handler_otr(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    const uint8_t index = C1(24, 8);
    const uint8_t offset = C1(16, 8);
    const uint8_t hasOffset = C1(8, 8);

    (*cmd0)++;

    const uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

    if (ucode_handler_index == ucode_f3dex2) {
        gfx->GfxSpMovememF3dex2(index, offset,
                                Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash));
    } else {
        auto light = (Fast::LightEntry*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);
        uintptr_t data = (uintptr_t)&light->Ambient;
        gfx->GfxSpMovememF3d(index, offset, (void*)(data + (hasOffset == 1 ? 0x8 : 0)));
    }
    return false;
}

int16_t Interpreter::CreateShader(const std::string& path) {
    std::shared_ptr<Ship::ResourceInitData> initData = std::make_shared<Ship::ResourceInitData>();
    initData->Path = path;
    initData->IsCustom = false;
    initData->ByteOrder = Ship::Endianness::Native;
    auto shader = Ship::Context::GetInstance()->GetResourceManager()->GetArchiveManager()->LoadFile(path);
    if (shader == nullptr || !shader->IsLoaded) {
        return -1;
    }
    shader_ids.push_back(std::string(shader->Buffer->data()));
    return shader_ids.size() - 1;
}

bool gfx_set_shader_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();

    F3DGfx* cmd = *cmd0;
    char* file = (char*)cmd->words.w1;

    if (file == nullptr) {
        gfx->mRsp->current_shader = { 0, 0, false };
        return false;
    }

    const auto path = std::string(file);
    const auto shaderId = gfx->CreateShader(path);
    gfx->mRdp->current_shader = { true, shaderId, (uint8_t)C0(16, 1) };
    return false;
}

bool gfx_moveword_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovewordF3dex2(C0(16, 8), C0(0, 16), cmd->words.w1);

    return false;
}

bool gfx_moveword_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovewordF3d(C0(0, 8), C0(8, 16), cmd->words.w1);

    return false;
}

bool gfx_texture_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTexture(C1(16, 16), C1(0, 16), C0(11, 3), C0(8, 3), C0(1, 7));

    return false;
}

// Seems to be the same for all other non F3DEX2 microcodes...
bool gfx_texture_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTexture(C1(16, 16), C1(0, 16), C0(11, 3), C0(8, 3), C0(0, 8));

    return false;
}

// Almost all versions of the microcode have their own version of this opcode
bool gfx_vtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    gfx->GfxSpVertex(C0(10, 6), C0(17, 7), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpVertex((C0(0, 16)) / sizeof(F3DVtx), C0(16, 4), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_hash_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    // Offset added to the start of the vertices
    const uintptr_t offset = (*cmd0)->words.w1;
    // This is a two-part display list command, so increment the instruction pointer so we can get the CRC64
    // hash from the second
    (*cmd0)++;
    const uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

    // We need to know if the offset is a cached pointer or not. An offset greater than one million is not a
    // real offset, so it must be a real pointer
    if (offset > 0xFFFFF) {
        (*cmd0)--;
        F3DGfx* cmd = *cmd0;
        gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), (F3DVtx*)offset);
        (*cmd0)++;
    } else {
        F3DVtx* vtx = (F3DVtx*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);

        if (vtx != NULL) {
            vtx = (F3DVtx*)((char*)vtx + offset);

            (*cmd0)--;
            F3DGfx* cmd = *cmd0;

            // TODO: WTF??
            cmd->words.w1 = (uintptr_t)vtx;

            gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), vtx);
            (*cmd0)++;
        }
    }
    return false;
}

bool gfx_vtx_otr_filepath_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    char* fileName = (char*)cmd->words.w1;
    (*cmd0)++;
    cmd = *cmd0;
    size_t vtxCnt = cmd->words.w0;
    size_t vtxIdxOff = cmd->words.w1 >> 16;
    size_t vtxDataOff = cmd->words.w1 & 0xFFFF;
    F3DVtx* vtx =
        (F3DVtx*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer((const char*)fileName);
    vtx += vtxDataOff;

    gfx->GfxSpVertex(vtxCnt, vtxIdxOff, vtx);
    return false;
}

bool gfx_dl_otr_filepath_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    char* fileName = (char*)cmd->words.w1;
    F3DGfx* nDL =
        (F3DGfx*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer((const char*)fileName);

    if (C0(16, 1) == 0 && nDL != nullptr) {
        g_exec_stack.call(*cmd0, nDL);
    } else {
        if (nDL != nullptr) {
            (*cmd0) = nDL;
            g_exec_stack.branch(cmd);
            return true; // shortcut cmd increment
        } else {
            assert(0 && "???");
            // gfx_path.pop_back();
            // cmd = cmd_stack.top();
            // cmd_stack.pop();
        }
    }
    return false;
}

// The original F3D microcode doesn't seem to have this opcode. Glide handles it as part of moveword
bool gfx_modify_vtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    gfx->GfxSpModifyVertex(C0(1, 15), C0(16, 8), (uint32_t)cmd->words.w1);
    return false;
}

// F3D, F3DEX, and F3DEX2 do the same thing but F3DEX2 has its own opcode number
bool gfx_dl_handler_common(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    F3DGfx* subGFX = (F3DGfx*)gfx->SegAddr(cmd->words.w1);
    if (C0(16, 1) == 0) {
        // Push return address
        if (subGFX != nullptr) {
            g_exec_stack.call(*cmd0, subGFX);
        }
    } else {
        (*cmd0) = subGFX;
        g_exec_stack.branch(cmd);
        return true; // shortcut cmd increment
    }
    return false;
}

bool gfx_dl_otr_hash_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    if (C0(16, 1) == 0) {
        // Push return address
        (*cmd0)++;

        uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

        F3DGfx* gfx = (F3DGfx*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);

        if (gfx != 0) {
            g_exec_stack.call(cmd, gfx);
        }
    } else {
        Interpreter* gfx = mInstance.lock().get();
        assert(0 && "????");
        (*cmd0) = (F3DGfx*)gfx->SegAddr((*cmd0)->words.w1);
        return true;
    }
    return false;
}
bool gfx_dl_index_handler(F3DGfx** cmd0) {
    // Compute seg addr by converting an index value to a offset value
    // handling 32 vs 64 bit size differences for Gfx
    // adding 1 to trigger the segaddr flow
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = (*cmd0);
    uint8_t segNum = (uint8_t)(cmd->words.w1 >> 24);
    uint32_t index = (uint32_t)(cmd->words.w1 & 0x00FFFFFF);
    uintptr_t segAddr = (segNum << 24) | (index * sizeof(F3DGfx)) + 1;

    F3DGfx* subGFX = (F3DGfx*)gfx->SegAddr(segAddr);
    if (C0(16, 1) == 0) {
        // Push return address
        if (subGFX != nullptr) {
            g_exec_stack.call((*cmd0), subGFX);
        }
    } else {
        (*cmd0) = subGFX;
        g_exec_stack.branch(cmd);
        return true; // shortcut cmd increment
    }
    return false;
}

// TODO handle special OTR opcodes later...
bool gfx_pushcd_handler_custom(F3DGfx** cmd0) {
    gfx_push_current_dir((char*)(*cmd0)->words.w1);
    return false;
}

// TODO handle special OTR opcodes later...
bool gfx_branch_z_otr_handler_f3dex2(F3DGfx** cmd0) {
    // Push return address
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = (*cmd0);

    uint8_t vbidx = (uint8_t)((*cmd0)->words.w0 & 0x00000FFF);
    uint32_t zval = (uint32_t)((*cmd0)->words.w1);

    (*cmd0)++;

    if (gfx->mRsp->loaded_vertices[vbidx].z <= zval ||
        (gfx->mRsp->extra_geometry_mode & G_EX_ALWAYS_EXECUTE_BRANCH) != 0) {
        uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

        F3DGfx* gfx = (F3DGfx*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);

        if (gfx != 0) {
            (*cmd0) = gfx;
            g_exec_stack.branch(cmd);
            return true; // shortcut cmd increment
        }
    }
    return false;
}

// F3D, F3DEX, and F3DEX2 do the same thing but F3DEX2 has its own opcode number
bool gfx_end_dl_handler_common(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->mMarkerOn = false;
    *cmd0 = g_exec_stack.ret();
    return true;
}

bool gfx_set_prim_depth_handler_rdp(F3DGfx** cmd) {
    // TODO Implement this command...
    return false;
}

// Only on F3DEX2
bool gfx_geometry_mode_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(~C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_set_geometry_mode_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(0, (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_clear_geometry_mode_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode((uint32_t)cmd->words.w1, 0);
    return false;
}

bool gfx_tri1_otr_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();

    F3DGfx* cmd = *cmd0;
    uint8_t v00 = (uint8_t)(cmd->words.w0 & 0x0000FFFF);
    uint8_t v01 = (uint8_t)(cmd->words.w1 >> 16);
    uint8_t v02 = (uint8_t)(cmd->words.w1 & 0x0000FFFF);
    gfx->GfxSpTri1(v00, v01, v02, false);

    return false;
}

bool gfx_tri1_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C0(16, 8) / 2, C0(8, 8) / 2, C0(0, 8) / 2, false);

    return false;
}

bool gfx_tri1_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C1(17, 7), C1(9, 7), C1(1, 7), false);

    return false;
}

bool gfx_tri1_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C1(16, 8) / 10, C1(8, 8) / 10, C1(0, 8) / 10, false);

    return false;
}

// F3DEX, and F3DEX2 share a tri2 function, however F3DEX has a different quad function.
bool gfx_tri2_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C0(17, 7), C0(9, 7), C0(1, 7), false);
    gfx->GfxSpTri1(C1(17, 7), C1(9, 7), C1(1, 7), false);
    return false;
}

bool gfx_quad_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C0(16, 8) / 2, C0(8, 8) / 2, C0(0, 8) / 2, false);
    gfx->GfxSpTri1(C1(16, 8) / 2, C1(8, 8) / 2, C1(0, 8) / 2, false);
    return false;
}

bool gfx_quad_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C1(16, 8) / 2, C1(8, 8) / 2, C1(0, 8) / 2, false);
    gfx->GfxSpTri1(C1(16, 8) / 2, C1(0, 8) / 2, C1(24, 8) / 2, false);
    return false;
}

bool gfx_othermode_l_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(31 - C0(8, 8) - C0(0, 8), C0(0, 8) + 1, cmd->words.w1);

    return false;
}

bool gfx_othermode_l_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(C0(8, 8), C0(0, 8), cmd->words.w1);

    return false;
}

bool gfx_othermode_h_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(63 - C0(8, 8) - C0(0, 8), C0(0, 8) + 1, (uint64_t)cmd->words.w1 << 32);

    return false;
}

// Only on F3DEX and older
bool gfx_set_geometry_mode_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(0, (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_clear_geometry_mode_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode((uint32_t)cmd->words.w1, 0);
    return false;
}

bool gfx_othermode_h_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(C0(8, 8) + 32, C0(0, 8), (uint64_t)cmd->words.w1 << 32);

    return false;
}

bool gfx_set_timg_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t i = (uintptr_t)gfx->SegAddr(cmd->words.w1);

    char* imgData = (char*)i;
    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetdata = {};

    if ((i & 1) != 1) {
        if (gfx_check_image_signature(imgData) == 1) {
            std::shared_ptr<Fast::Texture> tex = std::static_pointer_cast<Fast::Texture>(
                Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess(imgData));

            if (tex == nullptr) {
                (*cmd0)++;
                return false;
            }

            i = (uintptr_t) reinterpret_cast<char*>(tex->ImageData);
            texFlags = tex->Flags;
            rawTexMetdata.width = tex->Width;
            rawTexMetdata.height = tex->Height;
            rawTexMetdata.h_byte_scale = tex->HByteScale;
            rawTexMetdata.v_pixel_scale = tex->VPixelScale;
            rawTexMetdata.type = tex->Type;
            rawTexMetdata.resource = tex;
        }
    }

    gfx->GfxDpSetTextureImage(C0(21, 3), C0(19, 2), C0(0, 12) + 1, imgData, texFlags, rawTexMetdata, (void*)i);

    return false;
}

bool gfx_set_timg_otr_hash_handler_custom(F3DGfx** cmd0) {
    uintptr_t addr = (*cmd0)->words.w1;
    (*cmd0)++;
    uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (uint64_t)(*cmd0)->words.w1;

    const char* fileName = Ship::Context::GetInstance()->GetResourceManager()->GetArchiveManager()->HashToCString(hash);
    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if (fileName == nullptr) {
        (*cmd0)++;
        return false;
    }

    std::shared_ptr<Fast::Texture> texture =
        std::static_pointer_cast<Fast::Texture>(Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess(
            Ship::Context::GetInstance()->GetResourceManager()->GetArchiveManager()->HashToCString(hash)));
    if (texture != nullptr) {
        texFlags = texture->Flags;
        rawTexMetadata.width = texture->Width;
        rawTexMetadata.height = texture->Height;
        rawTexMetadata.h_byte_scale = texture->HByteScale;
        rawTexMetadata.v_pixel_scale = texture->VPixelScale;
        rawTexMetadata.type = texture->Type;
        rawTexMetadata.resource = texture;

        // OTRTODO: We have disabled caching for now to fix a texture corruption issue with HD texture
        // support. In doing so, there is a potential performance hit since we are not caching lookups. We
        // need to do proper profiling to see whether or not it is worth it to keep the caching system.

        char* tex = reinterpret_cast<char*>(texture->ImageData);

        if (tex != nullptr) {
            (*cmd0)--;
            uintptr_t oldData = (*cmd0)->words.w1;
            // TODO: wtf??
            (*cmd0)->words.w1 = (uintptr_t)tex;

            // if (ourHash != (uint64_t)-1) {
            //     auto res = ResourceLoad(ourHash);
            // }

            (*cmd0)++;
        }

        (*cmd0)--;
        F3DGfx* cmd = (*cmd0);
        uint32_t fmt = C0(21, 3);
        uint32_t size = C0(19, 2);
        uint32_t width = C0(0, 12) + 1;

        if (tex != NULL) {
            Interpreter* gfx = mInstance.lock().get();
            gfx->GfxDpSetTextureImage(fmt, size, width, fileName, texFlags, rawTexMetadata, tex);
        }
    } else {
        SPDLOG_ERROR("G_SETTIMG_OTR_HASH: Texture is null");
    }

    (*cmd0)++;
    return false;
}

bool gfx_set_timg_otr_filepath_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    const char* fileName = (char*)cmd->words.w1;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    std::shared_ptr<Fast::Texture> texture = std::static_pointer_cast<Fast::Texture>(
        Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess(fileName));
    if (texture != nullptr) {
        Interpreter* gfx = mInstance.lock().get();
        texFlags = texture->Flags;
        rawTexMetadata.width = texture->Width;
        rawTexMetadata.height = texture->Height;
        rawTexMetadata.h_byte_scale = texture->HByteScale;
        rawTexMetadata.v_pixel_scale = texture->VPixelScale;
        rawTexMetadata.type = texture->Type;
        rawTexMetadata.resource = texture;

        uint32_t fmt = C0(21, 3);
        uint32_t size = C0(19, 2);
        uint32_t width = C0(0, 12) + 1;

        gfx->GfxDpSetTextureImage(fmt, size, width, fileName, texFlags, rawTexMetadata,
                                  reinterpret_cast<char*>(texture->ImageData));
    } else {
        SPDLOG_ERROR("G_SETTIMG_OTR_FILEPATH: Texture is null");
    }
    return false;
}

bool gfx_set_fb_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();
    gfx->Flush();

    if (cmd->words.w1) {
        gfx->SetFrameBuffer((int32_t)cmd->words.w1, 1.0f);
        gfx->mActiveFrameBuffer = gfx->mFrameBuffers.find((int32_t)cmd->words.w1);
        gfx->mFbActive = true;
    } else {
        gfx->ResetFrameBuffer();
        gfx->mFbActive = false;
        gfx->mActiveFrameBuffer = gfx->mFrameBuffers.end();
    }
    return false;
}

bool gfx_reset_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->Flush();
    gfx->mFbActive = false;
    gfx->mActiveFrameBuffer = gfx->mFrameBuffers.end();
    gfx->mRapi->StartDrawToFramebuffer(gfx->mRendersToFb ? gfx->mGameFb : 0,
                                       (float)gfx->mCurDimensions.height / gfx->mNativeDimensions.height);
    // Force viewport and scissor to reapply against the main framebuffer, in case a previous smaller
    // framebuffer truncated the values
    gfx->mRdp->viewport_or_scissor_changed = true;
    gfx->mRenderingState.viewport = {};
    gfx->mRenderingState.scissor = {};
    return false;
}

bool gfx_copy_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    bool* hasCopiedPtr = (bool*)cmd->words.w1;

    gfx->Flush();
    gfx->CopyFrameBuffer(C0(11, 11), C0(0, 11), (bool)C0(22, 1), hasCopiedPtr);
    return false;
}

bool gfx_read_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    int32_t width, height;
    [[maybe_unused]] int32_t ulx, uly;
    uint16_t* rgba16Buffer = (uint16_t*)cmd->words.w1;
    int fbId = C0(0, 8);
    bool bswap = C0(8, 1);
    ++(*cmd0);
    cmd = *cmd0;
    // Specifying the upper left origin value is unused and unsupported at the renderer level
    ulx = C0(0, 16);
    uly = C0(16, 16);
    width = C1(0, 16);
    height = C1(16, 16);

    gfx->Flush();
    gfx->mRapi->ReadFramebufferToCPU(fbId, width, height, rgba16Buffer);

#ifndef IS_BIGENDIAN
    // byteswap the output to BE
    if (bswap) {
        for (size_t i = 0; i < (size_t)width * height; i++) {
            rgba16Buffer[i] = BE16SWAP(rgba16Buffer[i]);
        }
    }
#endif

    return false;
}

bool gfx_register_blended_texture_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    // Flush incase we are replacing a previous blended texture that hasn't been finialized to the GPU
    gfx->Flush();

    char* timg = (char*)cmd->words.w1;

    ++(*cmd0);
    cmd = *cmd0;

    uint8_t* mask = (uint8_t*)cmd->words.w0;
    uint8_t* replacementTex = (uint8_t*)cmd->words.w1;

    if (!gfx_check_image_signature(timg)) {
        SPDLOG_ERROR(
            "OTR_G_REGBLENDEDTEX: Texture is not a valid OTR resource name, unable to register blended texture");
        return false;
    }

    // With no mask, we should clear the blended texture
    if (mask == nullptr) {
        gfx->UnregisterBlendedTexture(timg);
    } else {
        gfx->RegisterBlendedTexture(timg, mask, replacementTex);
    }

    return false;
}

bool gfx_set_timg_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->Flush();
    gfx->mRapi->SelectTextureFb((uint32_t)cmd->words.w1);
    gfx->mRdp->textures_changed[0] = false;
    gfx->mRdp->textures_changed[1] = false;
    return false;
}

bool gfx_set_grayscale_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->grayscale = cmd->words.w1;
    return false;
}

// SOH [Enhancement] Toon lighting per-draw marker (mirrors grayscale).
bool gfx_set_toon_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    // SOH [Enhancement] Actor shadow: draw the last object's drop shadow before the bracket toggles, then
    // disarm the shadow pass (each object re-arms via gSPToonShadow). Run on both edges so a stale enable
    // can never leak across the actor-loop boundary.
    gfx->FlushToonShadow();
    gfx->mRdp->toon_shadow = false;

    gfx->mRdp->toon = cmd->words.w1;
    // A fresh key must be supplied (per object) after each toon-on; clear any stale one.
    gfx->mRsp->toon_key_valid = false;
    return false;
}

// SOH [Enhancement] Per-object toon key light: world-space direction (3x s8 / 127) + color (3x u8 / 255).
bool gfx_set_toon_key_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    // SOH [Enhancement] The toon key light is pushed to the GPU once per batch (see Flush) as a single
    // world-space direction + color. Several objects that share texture/state are otherwise batched into
    // one draw, so they would all be lit by whichever object's key was set last — each object picks its
    // own key (the nearest light, or the sun), so a shared batch would mislight all but the last. Flush
    // the pending geometry now, with the PREVIOUS object's key still in effect, so every toon object
    // becomes its own correctly-lit batch.
    gfx->Flush();

    int8_t dx = (cmd->words.w0 >> 16) & 0xFF;
    int8_t dy = (cmd->words.w0 >> 8) & 0xFF;
    int8_t dz = (cmd->words.w0 >> 0) & 0xFF;
    gfx->mRsp->toon_key_dir[0] = dx / 127.0f;
    gfx->mRsp->toon_key_dir[1] = dy / 127.0f;
    gfx->mRsp->toon_key_dir[2] = dz / 127.0f;
    gfx->mRsp->toon_key_color[0] = ((cmd->words.w1 >> 16) & 0xFF) / 255.0f;
    gfx->mRsp->toon_key_color[1] = ((cmd->words.w1 >> 8) & 0xFF) / 255.0f;
    gfx->mRsp->toon_key_color[2] = ((cmd->words.w1 >> 0) & 0xFF) / 255.0f;
    gfx->mRsp->toon_key_valid = true;
    // The key changes the effective light, so force a recompute on the next vertex.
    gfx->mRsp->lights_changed = true;
    return false;
}

// SOH [Enhancement] Per-object actor shadow marker. The normal bytes are only an arm flag (nonzero = this
// object casts a shadow, zero = it doesn't) — the renderer builds the volume from the captured feet and the
// per-object toon key direction, not from a floor plane. w1 carries the eased 0..1 shadow size. Emitting this
// per object also bounds each object's captured geometry: the previous object's volume is built here before
// the next one arms.
bool gfx_set_toon_shadow_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    // Arm flag: any nonzero normal byte arms the shadow for this object; all-zero disarms it. nx:ny carry an
    // s16 "feet-clamp" world Y (or TOON_SHADOW_NO_CLAMP); nz is the arm marker that keeps the shadow armed even
    // when the clamp bytes happen to be zero.
    uint8_t nx = (cmd->words.w0 >> 16) & 0xFF;
    uint8_t ny = (cmd->words.w0 >> 8) & 0xFF;
    uint8_t nz = (cmd->words.w0 >> 0) & 0xFF;

    float sizeOrSentinel;
    uint32_t w1Bits = (uint32_t)cmd->words.w1;
    memcpy(&sizeOrSentinel, &w1Bits, sizeof(sizeOrSentinel));

    // Sentinel (gSPToonShadowFlush): a zero normal with this magic w1 means "render the frame's accumulated
    // shadow volumes now" (emitted at the pre-actor hook so shadows land only on the environment).
    if ((nx | ny | nz) == 0 && sizeOrSentinel <= -1.0e29f) {
        // SOH [Enhancement] Cascaded shadow maps: the world-caster bracket rides further down the same
        // sentinel range, so it has to be tested BEFORE the flush -- the flush's own test (<= -1e29) would
        // otherwise swallow both of these.
        // Tested before everything else: it is a veto, and the values sit past the rest of the range so the
        // chain below never has to make room for them.
        if (sizeOrSentinel <= -7.5e30f) {
            gfx->mRdp->shadow_no_cast = true; // gSPShadowMapCasterOff
            return false;
        }
        if (sizeOrSentinel <= -6.5e30f) {
            gfx->mRdp->shadow_no_cast = false; // gSPShadowMapCasterOn
            return false;
        }
        if (sizeOrSentinel <= -5.5e30f) {
            gfx->RenderShadowMap(); // gSPShadowMapFlush
            return false;
        }
        if (sizeOrSentinel <= -4.5e30f) {
            gfx->mRdp->shadow_no_receive = false; // gSPShadowMapReceiveOn
            return false;
        }
        if (sizeOrSentinel <= -3.5e30f) {
            gfx->mRdp->shadow_no_receive = true; // gSPShadowMapReceiveOff
            return false;
        }
        if (sizeOrSentinel <= -2.5e30f) {
            gfx->mRdp->shadow_world_caster = false; // gSPShadowMapWorldCasterEnd
            return false;
        }
        if (sizeOrSentinel <= -1.5e30f) {
            gfx->mRdp->shadow_world_caster = true; // gSPShadowMapWorldCasterBegin
            return false;
        }
        // A scenery actor casts into the WORLD layer, so the bracket sets the world flag too -- that is what
        // puts a tree's shadow on the player instead of only on the ground. The scenery flag beside it
        // carries the cutout permission the world flag deliberately does not (see the RDP field).
        if (sizeOrSentinel <= -1.3e30f) {
            gfx->mRdp->shadow_scenery_caster = true; // gSPShadowMapSceneryCasterBegin
            return false;
        }
        if (sizeOrSentinel <= -1.1e30f) {
            gfx->mRdp->shadow_scenery_caster = false; // gSPShadowMapSceneryCasterEnd
            return false;
        }
        // Stencil volumes only. The shadow map used to share this hook, but it must not: this fires
        // BEFORE the actors draw, so it could only ever render casters captured a frame earlier, and the
        // room -- drawn earlier still -- then sampled the result a frame after that. Two frames of lag on
        // a character's own shadow, which reads as the shadow sliding along behind him. It has its own
        // sentinel now (gSPShadowMapFlush), emitted once the actors are done.
        gfx->RenderShadowVolumes();
        return false;
    }

    gfx->FlushToonShadow(); // build + accumulate the previous object's volume

    gfx->mRsp->toon_shadow_size = sizeOrSentinel;
    gfx->mRdp->toon_shadow = (nx | ny | nz) != 0;
    // Decode the s16 feet clamp packed in nx:ny (TOON_SHADOW_NO_CLAMP = leave the feet at the captured
    // geometry). Set alongside toon_shadow_size so the deferred FlushToonShadow reads this object's value.
    int16_t feetClamp = (int16_t)(((uint16_t)nx << 8) | (uint16_t)ny);
    gfx->mRsp->toon_shadow_clamp_feet = (feetClamp != -32768);
    gfx->mRsp->toon_shadow_feet_clamp_y = (float)feetClamp;
    // Snapshot THIS object's key direction (set by the gSPToonKey just before this command) so its
    // deferred shadow flush uses it, not whatever later object last touched toon_key_dir.
    gfx->mRsp->toon_shadow_dir[0] = gfx->mRsp->toon_key_dir[0];
    gfx->mRsp->toon_shadow_dir[1] = gfx->mRsp->toon_key_dir[1];
    gfx->mRsp->toon_shadow_dir[2] = gfx->mRsp->toon_key_dir[2];
    return false;
}

// SOH [Enhancement] World light casting: set the stencil mode for the stencil light-volume technique.
// Mirrors the toon-key handler's flush-then-set: the mode change must not retroactively apply to already
// batched geometry, so flush the pending tris (under the previous mode) before switching.
bool gfx_set_stencil_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->Flush();
    gfx->mRapi->SetStencilMode((int)cmd->words.w1);
    return false;
}

bool gfx_load_block_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadBlock(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_load_tile_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadTile(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_tile_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetTile(C0(21, 3), C0(19, 2), C0(9, 9), C0(0, 9), C1(24, 3), C1(20, 4), C1(18, 2), C1(14, 4), C1(10, 4),
                      C1(8, 2), C1(4, 4), C1(0, 4));
    return false;
}

bool gfx_set_tile_size_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetTileSize(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_tile_size_interp_handler_rdp(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();

    if (gfx->mInterpolationIndex == gfx->mInterpolationIndexTarget) {
        int tile = C1(24, 3);
        gfx->GfxDpSetTileSize(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
        ++(*cmd0);
        memcpy(&gfx->mRdp->texture_tile[tile].uls, &(*cmd0)->words.w0, sizeof(float));
        memcpy(&gfx->mRdp->texture_tile[tile].ult, &(*cmd0)->words.w1, sizeof(float));
        ++(*cmd0);
        memcpy(&gfx->mRdp->texture_tile[tile].lrs, &(*cmd0)->words.w0, sizeof(float));
        memcpy(&gfx->mRdp->texture_tile[tile].lrt, &(*cmd0)->words.w1, sizeof(float));
    } else {
        ++(*cmd0);
        ++(*cmd0);
    }

    return false;
}

bool gfx_set_interpolation_index_target(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();

    gfx->mInterpolationIndexTarget = cmd->words.w1;
    return false;
}

bool gfx_load_tlut_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadTlut(C1(24, 3), C1(14, 10));
    return false;
}

bool gfx_set_env_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetEnvColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_prim_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetPrimColor(C0(8, 8), C0(0, 8), C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_fog_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetFogColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_blend_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetBlendColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_fill_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetFillColor((uint32_t)cmd->words.w1);
    return false;
}

bool gfx_set_intensity_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetGrayscaleColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_combine_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetCombineMode(
        color_comb(C0(20, 4), C1(28, 4), C0(15, 5), C1(15, 3)), alpha_comb(C0(12, 3), C1(12, 3), C0(9, 3), C1(9, 3)),
        color_comb(C0(5, 4), C1(24, 4), C0(0, 5), C1(6, 3)), alpha_comb(C1(21, 3), C1(3, 3), C1(18, 3), C1(0, 3)));
    return false;
}

bool gfx_tex_rect_and_flip_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);
    int32_t lrx, lry, tile, ulx, uly;
    uint32_t uls, ult, dsdx, dtdy;

    lrx = C0(12, 12);
    lry = C0(0, 12);
    tile = C1(24, 3);
    ulx = C1(12, 12);
    uly = C1(0, 12);
    // TODO make sure I don't need to increment cmd0
    ++(*cmd0);
    cmd = *cmd0;
    uls = C1(16, 16);
    ult = C1(0, 16);
    ++(*cmd0);
    cmd = *cmd0;
    dsdx = C1(16, 16);
    dtdy = C1(0, 16);

    gfx->GfxDpTextureRectangle(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, opcode == RDP_G_TEXRECTFLIP);
    return false;
}

bool gfx_tex_rect_wide_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);
    int32_t lrx, lry, tile, ulx, uly;
    uint32_t uls, ult, dsdx, dtdy;

    lrx = static_cast<int32_t>((C0(0, 24) << 8)) >> 8;
    lry = static_cast<int32_t>((C1(0, 24) << 8)) >> 8;
    tile = C1(24, 3);
    ++(*cmd0);
    cmd = *cmd0;
    ulx = static_cast<int32_t>((C0(0, 24) << 8)) >> 8;
    uly = static_cast<int32_t>((C1(0, 24) << 8)) >> 8;
    ++(*cmd0);
    cmd = *cmd0;
    uls = C0(16, 16);
    ult = C0(0, 16);
    dsdx = C1(16, 16);
    dtdy = C1(0, 16);
    gfx->GfxDpTextureRectangle(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, opcode == RDP_G_TEXRECTFLIP);
    return false;
}

bool gfx_image_rect_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int16_t tile, iw, ih;
    int16_t x0, y0, s0, t0;
    int16_t x1, y1, s1, t1;
    tile = C0(0, 3);
    iw = C1(16, 16);
    ih = C1(0, 16);
    cmd = ++(*cmd0);
    x0 = C0(16, 16);
    y0 = C0(0, 16);
    s0 = C1(16, 16);
    t0 = C1(0, 16);
    cmd = ++(*cmd0);
    x1 = C0(16, 16);
    y1 = C0(0, 16);
    s1 = C1(16, 16);
    t1 = C1(0, 16);
    gfx->GfxDpImageRectangle(tile, iw, ih, x0, y0, s0, t0, x1, y1, s1, t1);

    return false;
}

bool gfx_fill_rect_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpFillRectangle(C1(12, 12), C1(0, 12), C0(12, 12), C0(0, 12));
    return false;
}

bool gfx_fill_wide_rect_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);
    int32_t lrx, lry, ulx, uly;

    lrx = (int32_t)(C0(0, 24) << 8) >> 8;
    lry = (int32_t)(C1(0, 24) << 8) >> 8;
    cmd = ++(*cmd0);
    ulx = (int32_t)(C0(0, 24) << 8) >> 8;
    uly = (int32_t)(C1(0, 24) << 8) >> 8;
    gfx->GfxDpFillRectangle(ulx, uly, lrx, lry);

    return false;
}

bool gfx_SetScissor_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetScissor(C1(24, 2), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_z_img_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetZImage(gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_set_c_img_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetColorImage(C0(21, 3), C0(19, 2), C0(0, 11), gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_rdp_set_other_mode_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetOtherMode(C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

bool gfx_bg_copy_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    if (!gfx->mMarkerOn) {
        gfx->Gfxs2dexBgCopy((F3DuObjBg*)cmd->words.w1); // not gfx->SegAddr here it seems
    }
    return false;
}

bool gfx_bg_1cyc_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->Gfxs2dexBg1cyc((F3DuObjBg*)cmd->words.w1);
    return false;
}

bool gfx_obj_rectangle_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    if (!gfx->mMarkerOn) {
        gfx->Gfxs2dexRecyCopy((F3DuObjSprite*)cmd->words.w1); // not gfx->SegAddr here it seems
    }
    return false;
}

bool gfx_extra_geometry_mode_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxSpExtraGeometryMode(~C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

bool gfx_stubbed_command_handler(F3DGfx** cmd0) {
    return false;
}

bool gfx_spnoop_command_handler_f3dex2(F3DGfx** cmd0) {
    return false;
}

class UcodeHandler {
  public:
    inline constexpr UcodeHandler(
        std::initializer_list<std::pair<int8_t, std::pair<const char*, GfxOpcodeHandlerFunc>>> initializer) {
        std::fill(std::begin(mHandlers), std::end(mHandlers),
                  std::pair<const char*, GfxOpcodeHandlerFunc>(nullptr, nullptr));

        for (const auto& [opcode, handler] : initializer) {
            mHandlers[static_cast<uint8_t>(opcode)] = handler;
        }
    }

    inline bool contains(int8_t opcode) const {
        return mHandlers[static_cast<uint8_t>(opcode)].first != nullptr;
    }

    inline std::pair<const char*, GfxOpcodeHandlerFunc> at(int8_t opcode) const {
        return mHandlers[static_cast<uint8_t>(opcode)];
    }

  private:
    std::pair<const char*, GfxOpcodeHandlerFunc> mHandlers[std::numeric_limits<uint8_t>::max() + 1];
};

static constexpr UcodeHandler rdpHandlers = {
    { RDP_G_SETTARGETINTERPINDEX,
      { "G_SETTARGETINTERPINDEX", gfx_set_interpolation_index_target } }, // G_SETTARGETINTERPINDEX
    { RDP_G_SETTILESIZE_INTERP,
      { "G_SETTILESIZE_INTERP", gfx_set_tile_size_interp_handler_rdp } },            // G_SETTILESIZE_INTERP
    { RDP_G_TEXRECT, { "G_TEXRECT", gfx_tex_rect_and_flip_handler_rdp } },           // G_TEXRECT (-28)
    { RDP_G_TEXRECTFLIP, { "G_TEXRECTFLIP", gfx_tex_rect_and_flip_handler_rdp } },   // G_TEXRECTFLIP (-27)
    { RDP_G_RDPLOADSYNC, { "mRdpLOADSYNC", gfx_stubbed_command_handler } },          // mRdpLOADSYNC (-26)
    { RDP_G_RDPPIPESYNC, { "mRdpPIPESYNC", gfx_stubbed_command_handler } },          // mRdpPIPESYNC (-25)
    { RDP_G_RDPTILESYNC, { "mRdpTILESYNC", gfx_stubbed_command_handler } },          // mRdpPIPESYNC (-24)
    { RDP_G_RDPFULLSYNC, { "mRdpFULLSYNC", gfx_stubbed_command_handler } },          // mRdpFULLSYNC (-23)
    { RDP_G_SETSCISSOR, { "G_SETSCISSOR", gfx_SetScissor_handler_rdp } },            // G_SETSCISSOR (-19)
    { RDP_G_SETPRIMDEPTH, { "G_SETPRIMDEPTH", gfx_set_prim_depth_handler_rdp } },    // G_SETPRIMDEPTH (-18)
    { RDP_G_RDPSETOTHERMODE, { "mRdpSETOTHERMODE", gfx_rdp_set_other_mode_rdp } },   // mRdpSETOTHERMODE (-17)
    { RDP_G_LOADTLUT, { "G_LOADTLUT", gfx_load_tlut_handler_rdp } },                 // G_LOADTLUT (-16)
    { RDP_G_SETTILESIZE, { "G_SETTILESIZE", gfx_set_tile_size_handler_rdp } },       // G_SETTILESIZE (-14)
    { RDP_G_LOADBLOCK, { "G_LOADBLOCK", gfx_load_block_handler_rdp } },              // G_LOADBLOCK (-13)
    { RDP_G_LOADTILE, { "G_LOADTILE", gfx_load_tile_handler_rdp } },                 // G_LOADTILE (-12)
    { RDP_G_SETTILE, { "G_SETTILE", gfx_set_tile_handler_rdp } },                    // G_SETTILE (-11)
    { RDP_G_FILLRECT, { "G_FILLRECT", gfx_fill_rect_handler_rdp } },                 // G_FILLRECT (-10)
    { RDP_G_SETFILLCOLOR, { "G_SETFILLCOLOR", gfx_set_fill_color_handler_rdp } },    // G_SETFILLCOLOR (-9)
    { RDP_G_SETFOGCOLOR, { "G_SETFOGCOLOR", gfx_set_fog_color_handler_rdp } },       // G_SETFOGCOLOR (-8)
    { RDP_G_SETBLENDCOLOR, { "G_SETBLENDCOLOR", gfx_set_blend_color_handler_rdp } }, // G_SETBLENDCOLOR (-7)
    { RDP_G_SETPRIMCOLOR, { "G_SETPRIMCOLOR", gfx_set_prim_color_handler_rdp } },    // G_SETPRIMCOLOR (-6)
    { RDP_G_SETENVCOLOR, { "G_SETENVCOLOR", gfx_set_env_color_handler_rdp } },       // G_SETENVCOLOR (-5)
    { RDP_G_SETCOMBINE, { "G_SETCOMBINE", gfx_set_combine_handler_rdp } },           // G_SETCOMBINE (-4)
    { RDP_G_SETTIMG, { "G_SETTIMG", gfx_set_timg_handler_rdp } },                    // G_SETTIMG (-3)
    { RDP_G_SETZIMG, { "G_SETZIMG", gfx_set_z_img_handler_rdp } },                   // G_SETZIMG (-2)
    { RDP_G_SETCIMG, { "G_SETCIMG", gfx_set_c_img_handler_rdp } },                   // G_SETCIMG (-1)
};

static constexpr UcodeHandler otrHandlers = {
    { OTR_G_SETTIMG_OTR_HASH,
      { "G_SETTIMG_OTR_HASH", gfx_set_timg_otr_hash_handler_custom } },       // G_SETTIMG_OTR_HASH (0x20)
    { OTR_G_SETFB, { "G_SETFB", gfx_set_fb_handler_custom } },                // G_SETFB (0x21)
    { OTR_G_RESETFB, { "G_RESETFB", gfx_reset_fb_handler_custom } },          // G_RESETFB (0x22)
    { OTR_G_SETTIMG_FB, { "G_SETTIMG_FB", gfx_set_timg_fb_handler_custom } }, // G_SETTIMG_FB (0x23)
    { OTR_G_VTX_OTR_FILEPATH,
      { "G_VTX_OTR_FILEPATH", gfx_vtx_otr_filepath_handler_custom } }, // G_VTX_OTR_FILEPATH (0x24)
    { OTR_G_SETTIMG_OTR_FILEPATH,
      { "G_SETTIMG_OTR_FILEPATH", gfx_set_timg_otr_filepath_handler_custom } }, // G_SETTIMG_OTR_FILEPATH (0x25)
    { OTR_G_TRI1_OTR, { "G_TRI1_OTR", gfx_tri1_otr_handler_f3dex2 } },          // G_TRI1_OTR (0x26)
    { OTR_G_DL_OTR_FILEPATH, { "G_DL_OTR_FILEPATH", gfx_dl_otr_filepath_handler_custom } }, // G_DL_OTR_FILEPATH (0x27)
    { OTR_G_PUSHCD, { "G_PUSHCD", gfx_pushcd_handler_custom } },                            // G_PUSHCD (0x28)
    { OTR_G_MTX_OTR_FILEPATH,
      { "G_MTX_OTR_FILEPATH", gfx_mtx_otr_filepath_handler_custom } },          // G_MTX_OTR_FILEPATH (0x29)
    { OTR_G_DL_OTR_HASH, { "G_DL_OTR_HASH", gfx_dl_otr_hash_handler_custom } }, // G_DL_OTR_HASH (0x31)
    { OTR_G_VTX_OTR_HASH, { "G_VTX_OTR_HASH", gfx_vtx_hash_handler_custom } },  // G_VTX_OTR_HASH (0x32)
    { OTR_G_MARKER, { "G_MARKER", gfx_marker_handler_otr } },                   // G_MARKER (0X33)
    { OTR_G_INVALTEXCACHE, { "G_INVALTEXCACHE", gfx_invalidate_tex_cache_handler_f3dex2 } }, // G_INVALTEXCACHE (0X34)
    { OTR_G_BRANCH_Z_OTR, { "G_BRANCH_Z_OTR", gfx_branch_z_otr_handler_f3dex2 } },           // G_BRANCH_Z_OTR (0x35)
    { OTR_G_MTX_OTR, { "G_MTX_OTR", gfx_mtx_otr_handler_custom } },                          // G_MTX_OTR (0x36)
    { OTR_G_TEXRECT_WIDE, { "G_TEXRECT_WIDE", gfx_tex_rect_wide_handler_custom } },          // G_TEXRECT_WIDE (0x37)
    { OTR_G_FILLWIDERECT, { "G_FILLWIDERECT", gfx_fill_wide_rect_handler_custom } },         // G_FILLWIDERECT (0x38)
    { OTR_G_SETGRAYSCALE, { "G_SETGRAYSCALE", gfx_set_grayscale_handler_custom } },          // G_SETGRAYSCALE (0x39)
    { OTR_G_EXTRAGEOMETRYMODE,
      { "G_EXTRAGEOMETRYMODE", gfx_extra_geometry_mode_handler_custom } }, // G_EXTRAGEOMETRYMODE (0x3a)
    { OTR_G_COPYFB, { "G_COPYFB", gfx_copy_fb_handler_custom } },          // G_COPYFB (0x3b)
    { OTR_G_IMAGERECT, { "G_IMAGERECT", gfx_image_rect_handler_custom } }, // G_IMAGERECT (0x3c)
    { OTR_G_DL_INDEX, { "G_DL_INDEX", gfx_dl_index_handler } },            // G_DL_INDEX (0x3d)
    { OTR_G_READFB, { "G_READFB", gfx_read_fb_handler_custom } },          // G_READFB (0x3e)
    { OTR_G_REGBLENDEDTEX,
      { "G_REGBLENDEDTEX", gfx_register_blended_texture_handler_custom } },         // G_REGBLENDEDTEX (0x3f)
    { OTR_G_SETINTENSITY, { "G_SETINTENSITY", gfx_set_intensity_handler_custom } }, // G_SETINTENSITY (0x40)
    { OTR_G_SETTOON, { "G_SETTOON", gfx_set_toon_handler_custom } },                // G_SETTOON (0x41)
    { OTR_G_SETTOONKEY, { "G_SETTOONKEY", gfx_set_toon_key_handler_custom } },      // G_SETTOONKEY (0x4a)
    { OTR_G_SETTOONSHADOW,
      { "G_SETTOONSHADOW", gfx_set_toon_shadow_handler_custom } }, // G_SETTOONSHADOW (0x4b) actor shadow
    { OTR_G_SETSTENCIL, { "G_SETSTENCIL", gfx_set_stencil_handler_custom } }, // G_SETSTENCIL (0x46)
    { OTR_G_MOVEMEM_HASH, { "OTR_G_MOVEMEM_HASH", gfx_movemem_handler_otr } },      // OTR_G_MOVEMEM_HASH
    { OTR_G_LOAD_SHADER, { "G_LOAD_SHADER", gfx_set_shader_custom } },
};

static constexpr UcodeHandler f3dex2Handlers = {
    { F3DEX2_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX2_G_SPNOOP, { "G_SPNOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX2_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX2_G_MTX, { "G_MTX", gfx_mtx_handler_f3dex2 } },
    { F3DEX2_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3dex2 } },
    { F3DEX2_G_MOVEMEM, { "G_MOVEMEM", gfx_movemem_handler_f3dex2 } },
    { F3DEX2_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3dex2 } },
    { F3DEX2_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3dex2 } },
    { F3DEX2_G_VTX, { "G_VTX", gfx_vtx_handler_f3dex2 } },
    { F3DEX2_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX2_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX2_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX2_G_GEOMETRYMODE, { "G_GEOMETRYMODE", gfx_geometry_mode_handler_f3dex2 } },
    { F3DEX2_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3dex2 } },
    { F3DEX2_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX2_G_QUAD, { "G_QUAD", gfx_quad_handler_f3dex2 } },
    { F3DEX2_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3dex2 } },
    { F3DEX2_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3dex2 } },
};

static constexpr UcodeHandler f3dexHandlers = {
    { F3DEX_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX_G_MTX, { "G_MTX", gfx_mtx_handler_f3d } },
    { F3DEX_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3d } },
    { F3DEX_G_MOVEMEM, { "G_POPMEM", gfx_movemem_handler_f3d } },
    { F3DEX_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3d } },
    { F3DEX_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3d } },
    { F3DEX_G_SETGEOMETRYMODE, { "G_SETGEOMETRYMODE", gfx_set_geometry_mode_handler_f3dex } },
    { F3DEX_G_CLEARGEOMETRYMODE, { "G_CLEARGEOMETRYMODE", gfx_clear_geometry_mode_handler_f3dex } },
    { F3DEX_G_VTX, { "G_VTX", gfx_vtx_handler_f3dex } },
    { F3DEX_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3dex } },
    { F3DEX_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX_G_SPNOOP, { "G_SPNOOP", gfx_spnoop_command_handler_f3dex2 } },
    { F3DEX_G_RDPHALF_1, { "mRdpHALF_1", gfx_stubbed_command_handler } },
    { F3DEX_G_QUAD, { "G_QUAD", gfx_quad_handler_f3dex } },
};

static constexpr UcodeHandler f3dHandlers = {
    { F3DEX_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX_G_MTX, { "G_MTX", gfx_mtx_handler_f3d } },
    { F3DEX_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3d } },
    { F3DEX_G_MOVEMEM, { "G_POPMEM", gfx_movemem_handler_f3d } },
    { F3DEX_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3d } },
    { F3DEX_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3d } },
    { F3DEX_G_SETGEOMETRYMODE, { "G_SETGEOMETRYMODE", gfx_set_geometry_mode_handler_f3dex } },
    { F3DEX_G_CLEARGEOMETRYMODE, { "G_CLEARGEOMETRYMODE", gfx_clear_geometry_mode_handler_f3dex } },
    { F3DEX_G_VTX, { "G_VTX", gfx_vtx_handler_f3d } },
    { F3DEX_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3d } },
    { F3DEX_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX_G_SPNOOP, { "G_SPNOOP", gfx_spnoop_command_handler_f3dex2 } },
    { F3DEX_G_RDPHALF_1, { "mRdpHALF_1", gfx_stubbed_command_handler } },
};

// LUSTODO: These S2DEX commands have different opcode numbers on F3DEX2 vs other ucodes. More research needs to be done
// to see if the implementations are different.
static constexpr UcodeHandler s2dexHandlers = {
    { F3DEX2_G_BG_COPY, { "G_BG_COPY", gfx_bg_copy_handler_s2dex } },
    { F3DEX2_G_BG_1CYC, { "G_BG_1CYC", gfx_bg_1cyc_handler_s2dex } },
    { F3DEX2_G_OBJ_RENDERMODE, { "G_OBJ_RENDERMODE", gfx_stubbed_command_handler } },
    { F3DEX2_G_OBJ_RECTANGLE_R, { "G_OBJ_RECTANGLE_R", gfx_stubbed_command_handler } },
    { F3DEX2_G_OBJ_RECTANGLE, { "G_OBJ_RECTANGLE", gfx_obj_rectangle_handler_s2dex } },
    { F3DEX2_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX2_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
};

static constexpr std::array ucode_handlers = {
    &f3dHandlers,    // ucode_f3db
    &f3dHandlers,    // ucode_f3d
    &f3dexHandlers,  // ucode_f3dex
    &f3dexHandlers,  // ucode_f3dexb
    &f3dex2Handlers, // ucode_f3dex2
    &s2dexHandlers,  // ucode_s2dex
};

const char* GfxGetOpcodeName(int8_t opcode) {
    if (otrHandlers.contains(opcode)) {
        return otrHandlers.at(opcode).first;
    }

    if (rdpHandlers.contains(opcode)) {
        return rdpHandlers.at(opcode).first;
    }

    if (ucode_handler_index < ucode_handlers.size()) {
        if (ucode_handlers[ucode_handler_index]->contains(opcode)) {
            return ucode_handlers[ucode_handler_index]->at(opcode).first;
        } else {
            SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, for loaded ucode: {}", (uint8_t)opcode,
                            (uint32_t)ucode_handler_index);
        }
    } else {
        SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, invalid ucode: {}", (uint8_t)opcode, (uint32_t)ucode_handler_index);
    }

    return nullptr;
}

// TODO, implement a system where we can get the current opcode handler by writing to the GWords. If the powers that be
// are OK with that...
static void gfx_set_ucode_handler(UcodeHandlers ucode) {
    // Loaded ucode must be in range of the supported ucode_handlers
    assert(ucode < ucode_max);
    Interpreter* gfx = mInstance.lock().get();
    ucode_handler_index = ucode;

    // Reset some RSP state values upon ucode load to deal with hardware quirks discovered by emulators
    switch (ucode) {
        case ucode_f3d:
        case ucode_f3db:
        case ucode_f3dex:
        case ucode_f3dexb:
        case ucode_f3dex2:
            gfx->mRsp->fog_mul = 0;
            gfx->mRsp->fog_offset = 0;
            break;
        default:
            break;
    }
}

static void gfx_step() {
    auto& cmd = g_exec_stack.currCmd();
    auto cmd0 = cmd;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);

#ifdef USE_GBI_TRACE
    if (cmd->words.trace.valid &&
        Ship::Context::GetInstance()->GetConsoleVariables()->GetInteger("gEnableGFXTrace", 0)) {
#define TRACE                                  \
    "\n====================================\n" \
    " - CMD: {:02X}\n"                         \
    " - Path: {}:{}\n"                         \
    " - W0: {:08X}\n"                          \
    " - W1: {:08X}\n"                          \
    "===================================="
        SPDLOG_INFO(TRACE, (uint8_t)opcode, cmd->words.trace.file, cmd->words.trace.idx, cmd->words.w0, cmd->words.w1);
    }
#endif

    if (opcode == F3DEX2_G_LOAD_UCODE) {
        gfx_set_ucode_handler((UcodeHandlers)(cmd->words.w0 & 0xFFFFFF));
        ++cmd;
        return;
        // Instead of having a handler for each ucode for switching ucode, just check for it early and return.
    }

    if (otrHandlers.contains(opcode)) {
        if (otrHandlers.at(opcode).second(&cmd)) {
            return;
        }
    } else if (rdpHandlers.contains(opcode)) {
        if (rdpHandlers.at(opcode).second(&cmd)) {
            return;
        }
    } else if (ucode_handler_index < ucode_handlers.size()) {
        if (ucode_handlers[ucode_handler_index]->contains(opcode)) {
            if (ucode_handlers[ucode_handler_index]->at(opcode).second(&cmd)) {
                return;
            }
        } else {
            SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, for loaded ucode: {}", (uint8_t)opcode,
                            (uint32_t)ucode_handler_index);
        }
    } else {
        SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, invalid ucode: {}", (uint8_t)opcode, (uint32_t)ucode_handler_index);
    }

    ++cmd;
}

void Interpreter::SpReset() {
    mRsp->modelview_matrix_stack_size = 1;
    mRsp->current_num_lights = 2;
    mRsp->lights_changed = true;
    mRsp->lookat[0].dir[0] = 0;
    mRsp->lookat[0].dir[1] = 127;
    mRsp->lookat[0].dir[2] = 0;
    mRsp->lookat[1].dir[0] = 127;
    mRsp->lookat[1].dir[1] = 0;
    mRsp->lookat[1].dir[2] = 0;
    CalculateNormalDir(&mRsp->lookat[0], mRsp->current_lookat_coeffs[0]);
    CalculateNormalDir(&mRsp->lookat[1], mRsp->current_lookat_coeffs[1]);
}

void Interpreter::GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY) {
    mWapi->GetDimensions(width, height, posX, posY);
}

void Interpreter::Init(class GfxWindowBackend* wapi, class GfxRenderingAPI* rapi, const char* game_name,
                       bool start_in_fullscreen, uint32_t width, uint32_t height, uint32_t posX, uint32_t posY) {
    mWapi = wapi;
    mRapi = rapi;
    mWapi->Init(game_name, rapi->GetName(), start_in_fullscreen, width, height, posX, posY);
    mRapi->Init();
    mRapi->UpdateFramebufferParameters(0, width, height, 1, false, true, true, true);
    mCurDimensions.internal_mul =
        Ship::Context::GetInstance()->GetConsoleVariables()->GetFloat(CVAR_INTERNAL_RESOLUTION, 1);
    mMsaaLevel = Ship::Context::GetInstance()->GetConsoleVariables()->GetInteger(CVAR_MSAA_VALUE, 1);

    mCurDimensions.width = width;
    mCurDimensions.height = height;

    mGameFb = mRapi->CreateFramebuffer();
    mGameFbMsaaResolved = mRapi->CreateFramebuffer();

    mNativeDimensions.width = SCREEN_WIDTH;
    mNativeDimensions.height = SCREEN_HEIGHT;

    for (int i = 0; i < MAX_SEGMENT_POINTERS; i++) {
        mSegmentPointers[i] = 0;
    }

    if (mTexUploadBuffer == nullptr) {
        // We cap texture max to 8k, because why would you need more?
        int max_tex_size = std::min(8192, mRapi->GetMaxTextureSize());
        mTexUploadBuffer = (uint8_t*)malloc(max_tex_size * max_tex_size * 4);
    }

    ucode_handler_index = UcodeHandlers::ucode_f3dex2;
}

void Interpreter::Destroy() {
    // TODO: should also destroy rapi, and any other resources acquired in fast3d
    free(mTexUploadBuffer);
    mWapi->Destroy();

    // Texture cache and loaded textures store references to Resources which need to be unreferenced.
    TextureCacheClear();
    mRdp->texture_to_load.raw_tex_metadata.resource = nullptr;
    mRdp->loaded_texture[0].raw_tex_metadata.resource = nullptr;
    mRdp->loaded_texture[1].raw_tex_metadata.resource = nullptr;
}

GfxRenderingAPI* Interpreter::GetCurrentRenderingAPI() {
    return mRapi;
}

void Interpreter::HandleWindowEvents() {
    mWapi->HandleEvents();
}

bool Interpreter::IsFrameReady() {
    return mWapi->IsFrameReady();
}

bool Interpreter::ViewportMatchesRendererResolution() {
#ifdef __APPLE__
    // Always treat the viewport as not matching the render resolution on mac
    // to avoid issues with retina scaling.
    return false;
#else
    if (mCurDimensions.width == mGameWindowViewport.width && mCurDimensions.height == mGameWindowViewport.height) {
        return true;
    }
    return false;
#endif
}

void Interpreter::StartFrame() {
    mWapi->GetDimensions(&mGfxCurrentWindowDimensions.width, &mGfxCurrentWindowDimensions.height, &mCurWindowPosX,
                         &mCurWindowPosY);
    if (mCurDimensions.height == 0) {
        // Avoid division by zero
        mCurDimensions.height = 1;
    }
    mCurDimensions.aspect_ratio = (float)mCurDimensions.width / (float)mCurDimensions.height;

    // Update the framebuffer sizes when the viewport or native dimension changes
    if (mCurDimensions.width != mPrvDimensions.width || mCurDimensions.height != mPrvDimensions.height ||
        mNativeDimensions.width != mPrevNativeDimensions.width ||
        mNativeDimensions.height != mPrevNativeDimensions.height) {

        for (auto& fb : mFrameBuffers) {
            uint32_t width = fb.second.orig_width, height = fb.second.orig_height;
            if (fb.second.resize) {
                AdjustWidthHeightForScale(width, height, fb.second.native_width, fb.second.native_height);
            }
            if (width != fb.second.applied_width || height != fb.second.applied_height) {
                mRapi->UpdateFramebufferParameters(fb.first, width, height, 1, true, true, true, true);
                fb.second.applied_width = width;
                fb.second.applied_height = height;
            }
        }
    }

    mPrvDimensions = mCurDimensions;
    mPrevNativeDimensions = mNativeDimensions;
    if (!ViewportMatchesRendererResolution() || mMsaaLevel > 1) {
        mRendersToFb = true;
        if (!ViewportMatchesRendererResolution()) {
            mRapi->UpdateFramebufferParameters(mGameFb, mCurDimensions.width, mCurDimensions.height, mMsaaLevel, true,
                                               true, true, true);
        } else {
            // MSAA framebuffer needs to be resolved to an equally sized target when complete, which must therefore
            // match the window size
            mRapi->UpdateFramebufferParameters(mGameFb, mGfxCurrentWindowDimensions.width,
                                               mGfxCurrentWindowDimensions.height, mMsaaLevel, false, true, true, true);
        }
        if (mMsaaLevel > 1 && !ViewportMatchesRendererResolution()) {
            mRapi->UpdateFramebufferParameters(mGameFbMsaaResolved, mCurDimensions.width, mCurDimensions.height, 1,
                                               false, false, false, false);
        }
    } else {
        mRendersToFb = false;
    }

    mFbActive = false;
}

GfxExecStack g_exec_stack = {};

void Interpreter::RunGuiOnly() {
    SpReset();

    mGetPixelDepthPending.clear();
    mGetPixelDepthCached.clear();

    mRapi->UpdateFramebufferParameters(0, mGfxCurrentWindowDimensions.width, mGfxCurrentWindowDimensions.height, 1,
                                       false, true, true, !mRendersToFb);
    mRapi->StartFrame();
    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, (float)mCurDimensions.height / mNativeDimensions.height);
    mRapi->ClearFramebuffer(false, true);
    mRdp->viewport_or_scissor_changed = true;
    mRenderingState.viewport = {};
    mRenderingState.scissor = {};

    Flush();
    mGfxFrameBuffer = 0;

    if (mRendersToFb) {
        mRapi->StartDrawToFramebuffer(0, 1);
        mRapi->ClearFramebuffer(true, true);
        if (mMsaaLevel > 1) {
            if (!ViewportMatchesRendererResolution()) {
                mRapi->ResolveMSAAColorBuffer(mGameFbMsaaResolved, mGameFb);
                mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFbMsaaResolved);
            } else {
                mRapi->ResolveMSAAColorBuffer(0, mGameFb);
            }
        } else {
            mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFb);
        }
    } else if (mFbActive) {
        // Failsafe reset to main framebuffer to prevent softlocking the renderer
        mFbActive = 0;
        mRapi->StartDrawToFramebuffer(0, 1);

        assert(0 && "active framebuffer was never reset back to original");
    }
}

void Interpreter::Run(Gfx* commands, const std::unordered_map<Mtx*, MtxF>& mtx_replacements) {
    SpReset();

    mGetPixelDepthPending.clear();
    mGetPixelDepthCached.clear();

    mCurMtxReplacements = &mtx_replacements;

    mRapi->UpdateFramebufferParameters(0, mGfxCurrentWindowDimensions.width, mGfxCurrentWindowDimensions.height, 1,
                                       false, true, true, !mRendersToFb);
    mRapi->StartFrame();
    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, (float)mCurDimensions.height / mNativeDimensions.height);
    mRapi->ClearFramebuffer(false, true);
    mRdp->viewport_or_scissor_changed = true;
    mRenderingState.viewport = {};
    mRenderingState.scissor = {};

    auto dbg = Ship::Context::GetInstance()->GetGfxDebugger();
    g_exec_stack.start((F3DGfx*)commands);
    while (!g_exec_stack.cmd_stack.empty()) {
        auto cmd = g_exec_stack.cmd_stack.top();

        if (dbg->IsDebugging()) {
            g_exec_stack.gfx_path.push_back(cmd);
            if (dbg->HasBreakPoint(g_exec_stack.gfx_path)) {
                // On a breakpoint with the active framebuffer still set, we need to reset back to prevent
                // soft locking the renderer
                if (mFbActive) {
                    mFbActive = 0;
                    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, 1);
                }

                break;
            }
            g_exec_stack.gfx_path.pop_back();
        }
        gfx_step();
    }

    Flush();
    mGfxFrameBuffer = 0;
    currentDir = std::stack<std::string>();

    if (mRendersToFb) {
        mRapi->StartDrawToFramebuffer(0, 1);
        mRapi->ClearFramebuffer(true, true);
        if (mMsaaLevel > 1) {
            if (!ViewportMatchesRendererResolution()) {
                mRapi->ResolveMSAAColorBuffer(mGameFbMsaaResolved, mGameFb);
                mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFbMsaaResolved);
            } else {
                mRapi->ResolveMSAAColorBuffer(0, mGameFb);
            }
        } else {
            mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFb);
        }
    } else if (mFbActive) {
        // Failsafe reset to main framebuffer to prevent softlocking the renderer
        mFbActive = 0;
        mRapi->StartDrawToFramebuffer(0, 1);

        assert(0 && "active framebuffer was never reset back to original");
    }
}

void Interpreter::EndFrame() {
    mRapi->EndFrame();
    mWapi->SwapBuffersBegin();
    mRapi->FinishRender();
    mWapi->SwapBuffersEnd();
}

void gfx_set_target_ucode(UcodeHandlers ucode) {
    ucode_handler_index = ucode;
}

int Interpreter::GetTargetFps() {
    return mWapi->GetTargetFps();
}

void Interpreter::SetTargetFps(int fps) {
    mWapi->SetTargetFps(fps);
}

void Interpreter::SetMaxFrameLatency(int latency) {
    mWapi->SetMaxFrameLatency(latency);
}

int Interpreter::CreateFrameBuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                                   uint8_t resize) {
    uint32_t orig_width = width, orig_height = height;
    if (resize) {
        AdjustWidthHeightForScale(width, height, native_width, native_height);
    }

    int fb = mRapi->CreateFramebuffer();
    mRapi->UpdateFramebufferParameters(fb, width, height, 1, true, true, true, true);

    mFrameBuffers[fb] = {
        orig_width, orig_height, width, height, native_width, native_height, static_cast<bool>(resize)
    };
    return fb;
}

void Interpreter::SetFrameBuffer(int fb, float noiseScale) {
    mRapi->StartDrawToFramebuffer(fb, noiseScale);
    mRapi->ClearFramebuffer(false, true);
}

void Interpreter::CopyFrameBuffer(int fb_dst_id, int fb_src_id, bool copyOnce, bool* hasCopiedPtr) {
    // Do not copy again if we have already copied before
    if (copyOnce && hasCopiedPtr != nullptr && *hasCopiedPtr) {
        return;
    }

    if (fb_src_id == 0 && mRendersToFb) {
        // read from the framebuffer we've been rendering to
        fb_src_id = mGameFb;
    }

    int srcX0, srcY0, srcX1, srcY1;
    int dstX0, dstY0, dstX1, dstY1;

    // When rendering to the main window buffer or MSAA is enabled with a buffer size equal to the view port,
    // then the source coordinates must account for any docked ImGui elements
    if (fb_src_id == 0 || (mMsaaLevel > 1 && mCurDimensions.width == mGameWindowViewport.width &&
                           mCurDimensions.height == mGameWindowViewport.height)) {
        srcX0 = mGameWindowViewport.x;
        srcY0 = mGameWindowViewport.y;
        srcX1 = mGameWindowViewport.x + mGameWindowViewport.width;
        srcY1 = mGameWindowViewport.y + mGameWindowViewport.height;
    } else {
        srcX0 = 0;
        srcY0 = 0;
        srcX1 = mCurDimensions.width;
        srcY1 = mCurDimensions.height;
    }

    dstX0 = 0;
    dstY0 = 0;
    dstX1 = mCurDimensions.width;
    dstY1 = mCurDimensions.height;

    mRapi->CopyFramebuffer(fb_dst_id, fb_src_id, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1);

    // Set the copied pointer if we have one
    if (hasCopiedPtr != nullptr) {
        *hasCopiedPtr = true;
    }
}

void Interpreter::ResetFrameBuffer() {
    mRapi->StartDrawToFramebuffer(0, (float)mCurDimensions.height / mNativeDimensions.height);
}

void Interpreter::AdjustPixelDepthCoordinates(float& x, float& y) {
    x = x * RATIO_X(mActiveFrameBuffer, mCurDimensions) -
        (mNativeDimensions.width * RATIO_X(mActiveFrameBuffer, mCurDimensions) - mCurDimensions.width) / 2;
    y *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
    if (!mRendersToFb || (mMsaaLevel > 1 && mCurDimensions.width == mGameWindowViewport.width &&
                          mCurDimensions.height == mGameWindowViewport.height)) {
        x += mGameWindowViewport.x;
        y += mGfxCurrentWindowDimensions.height - (mGameWindowViewport.y + mGameWindowViewport.height);
    }
}

void Interpreter::GetPixelDepthPrepare(float x, float y) {
    AdjustPixelDepthCoordinates(x, y);
    mGetPixelDepthPending.emplace(x, y);
}

uint16_t Interpreter::GetPixelDepth(float x, float y) {
    AdjustPixelDepthCoordinates(x, y);

    if (auto it = mGetPixelDepthCached.find(std::make_pair(x, y)); it != mGetPixelDepthCached.end()) {
        return it->second;
    }

    mGetPixelDepthPending.emplace(x, y);

    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res =
        mRapi->GetPixelDepth(mRendersToFb ? mGameFb : 0, mGetPixelDepthPending);
    mGetPixelDepthCached.merge(res);
    mGetPixelDepthPending.clear();

    return mGetPixelDepthCached.find(std::make_pair(x, y))->second;
}

void gfx_push_current_dir(char* path) {
    if (gfx_check_image_signature(path) == 1)
        path = &path[7];

    currentDir.push(GetPathWithoutFileName(path));
}

int32_t gfx_check_image_signature(const char* imgData) {
    uintptr_t i = (uintptr_t)(imgData);

    if ((i & 1) == 1) {
        return 0;
    }

    if (i != 0) {
        return Ship::Context::GetInstance()->GetResourceManager()->OtrSignatureCheck(imgData);
    }

    return 0;
}

void Interpreter::RegisterBlendedTexture(const char* name, uint8_t* mask, uint8_t* replacement) {
    if (gfx_check_image_signature(name)) {
        name += 7;
    }

    if (gfx_check_image_signature(reinterpret_cast<char*>(replacement))) {
        Fast::Texture* tex = std::static_pointer_cast<Fast::Texture>(
                                 Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess(
                                     reinterpret_cast<char*>(replacement)))
                                 .get();

        replacement = tex->ImageData;
    }

    mMaskedTextures[name] = MaskedTextureEntry{ mask, replacement };
}

void Interpreter::UnregisterBlendedTexture(const char* name) {
    if (gfx_check_image_signature(name)) {
        name += 7;
    }

    mMaskedTextures.erase(name);
}

// New getters and setters
void Interpreter::SetNativeDimensions(float width, float height) {
    mNativeDimensions.width = width;
    mNativeDimensions.height = height;
}

void Interpreter::SetResolutionMultiplier(float multiplier) {
    mCurDimensions.internal_mul = multiplier;
}

void Interpreter::SetMsaaLevel(uint32_t level) {
    mMsaaLevel = level;
}

void Interpreter::GetCurDimensions(uint32_t* width, uint32_t* height) {
    *width = mCurDimensions.width;
    *height = mCurDimensions.height;
}

} // namespace Fast

void gfx_cc_get_features(uint64_t shader_id0, uint32_t shader_id1, struct CCFeatures* cc_features) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {
                cc_features->c[i][j][k] = shader_id0 >> i * 32 + j * 16 + k * 4 & 0xf;
            }
        }
    }

    cc_features->opt_alpha = (shader_id1 & SHADER_OPT(ALPHA)) != 0;
    cc_features->opt_fog = (shader_id1 & SHADER_OPT(FOG)) != 0;
    cc_features->opt_texture_edge = (shader_id1 & SHADER_OPT(TEXTURE_EDGE)) != 0;
    cc_features->opt_noise = (shader_id1 & SHADER_OPT(NOISE)) != 0;
    cc_features->opt_2cyc = (shader_id1 & SHADER_OPT(_2CYC)) != 0;
    cc_features->opt_alpha_threshold = (shader_id1 & SHADER_OPT(ALPHA_THRESHOLD)) != 0;
    cc_features->opt_invisible = (shader_id1 & SHADER_OPT(INVISIBLE)) != 0;
    cc_features->opt_grayscale = (shader_id1 & SHADER_OPT(GRAYSCALE)) != 0;
    cc_features->opt_toon = (shader_id1 & SHADER_OPT(TOON)) != 0; // SOH [Enhancement] toon lighting
    // SOH [Enhancement] cascaded shadow maps: this draw samples the cascade array
    cc_features->opt_shadow_map = (shader_id1 & SHADER_OPT(SHADOW_MAP)) != 0;

    cc_features->clamp[0][0] = shader_id1 & SHADER_OPT(TEXEL0_CLAMP_S);
    cc_features->clamp[0][1] = shader_id1 & SHADER_OPT(TEXEL0_CLAMP_T);
    cc_features->clamp[1][0] = shader_id1 & SHADER_OPT(TEXEL1_CLAMP_S);
    cc_features->clamp[1][1] = shader_id1 & SHADER_OPT(TEXEL1_CLAMP_T);

    if (shader_id1 & SHADER_OPT(USE_SHADER)) {
        // SOH [Enhancement] 17->18 for the TOON opt bit, 18->19 for SHADOW_MAP. Must match the encode in
        // the ColorCombinerKey build; a mismatch silently selects the wrong shader for every draw.
        cc_features->shader_id = (shader_id1 >> 19) & 0x1FFF;
    }

    cc_features->usedTextures[0] = false;
    cc_features->usedTextures[1] = false;
    cc_features->used_masks[0] = false;
    cc_features->used_masks[1] = false;
    cc_features->used_blend[0] = false;
    cc_features->used_blend[1] = false;
    cc_features->numInputs = 0;

    for (int c = 0; c < 2; c++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                if (cc_features->c[c][i][j] >= SHADER_INPUT_1 && cc_features->c[c][i][j] <= SHADER_INPUT_7) {
                    if (cc_features->c[c][i][j] > cc_features->numInputs) {
                        cc_features->numInputs = cc_features->c[c][i][j];
                    }
                }
                if (cc_features->c[c][i][j] == SHADER_TEXEL0 || cc_features->c[c][i][j] == SHADER_TEXEL0A) {
                    cc_features->usedTextures[0] = true;
                    if (cc_features->opt_2cyc) {
                        cc_features->usedTextures[1] = true;
                    }
                }
                if (cc_features->c[c][i][j] == SHADER_TEXEL1 || cc_features->c[c][i][j] == SHADER_TEXEL1A) {
                    cc_features->usedTextures[1] = true;
                    if (cc_features->opt_2cyc) {
                        cc_features->usedTextures[0] = true;
                    }
                }
            }
        }
    }

    for (int c = 0; c < 2; c++) {
        cc_features->do_single[c][0] = cc_features->c[c][0][2] == SHADER_0;
        cc_features->do_single[c][1] = cc_features->c[c][1][2] == SHADER_0;
        cc_features->do_multiply[c][0] = cc_features->c[c][0][1] == SHADER_0 && cc_features->c[c][0][3] == SHADER_0;
        cc_features->do_multiply[c][1] = cc_features->c[c][1][1] == SHADER_0 && cc_features->c[c][1][3] == SHADER_0;
        cc_features->do_mix[c][0] = cc_features->c[c][0][1] == cc_features->c[c][0][3];
        cc_features->do_mix[c][1] = cc_features->c[c][1][1] == cc_features->c[c][1][3];
        cc_features->color_alpha_same[c] = (shader_id0 >> c * 32 & 0xffff) == (shader_id0 >> c * 32 + 16 & 0xffff);
    }

    if (cc_features->usedTextures[0] && shader_id1 & SHADER_OPT(TEXEL0_MASK)) {
        cc_features->used_masks[0] = true;
    }
    if (cc_features->usedTextures[1] && shader_id1 & SHADER_OPT(TEXEL1_MASK)) {
        cc_features->used_masks[1] = true;
    }

    if (cc_features->usedTextures[0] && shader_id1 & SHADER_OPT(TEXEL0_BLEND)) {
        cc_features->used_blend[0] = true;
    }
    if (cc_features->usedTextures[1] && shader_id1 & SHADER_OPT(TEXEL1_BLEND)) {
        cc_features->used_blend[1] = true;
    }
}

extern "C" int gfx_create_framebuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                                      uint8_t resize) {
    return Fast::mInstance.lock().get()->CreateFrameBuffer(width, height, native_width, native_height, resize);
}

extern "C" void gfx_texture_cache_clear() {
    Fast::mInstance.lock().get()->TextureCacheClear();
}
