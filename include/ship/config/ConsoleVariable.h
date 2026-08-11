#pragma once

#include "ship/utils/color.h"
#include <nlohmann/json.hpp>
#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <string>
#include <string_view>
#include <functional>

namespace Ship {
typedef enum class ConsoleVariableType { Integer, Float, String, Color, Color24 } ConsoleVariableType;

typedef struct CVar {
    ConsoleVariableType Type;
    union {
        int32_t Integer;
        float Float;
        char* String = nullptr;
        Color_RGBA8 Color;
        Color_RGB8 Color24;
    };
    ~CVar() {
        if (Type == ConsoleVariableType::String && String != nullptr) {
            free(String);
        }
    }
} CVar;

class ConsoleVariable {
  public:
    ConsoleVariable();
    ~ConsoleVariable();

    // Deliberately still hands out a shared_ptr. Returning the raw CVar* instead measured as a 0.1 ns
    // saving out of the ~18 ns this change is worth -- nothing -- while the audio thread reads CVars
    // (audio_playback.c, audio_synthesis.c) and the owning reference is what keeps a variable alive
    // there if the main thread clears it mid-read. Not a trade worth making.
    std::shared_ptr<CVar> Get(const char* name);

    int32_t GetInteger(const char* name, int32_t defaultValue);
    float GetFloat(const char* name, float defaultValue);
    const char* GetString(const char* name, const char* defaultValue);
    Color_RGBA8 GetColor(const char* name, Color_RGBA8 defaultValue);
    Color_RGB8 GetColor24(const char* name, Color_RGB8 defaultValue);

    void SetInteger(const char* name, int32_t value);
    void SetFloat(const char* name, float value);
    void SetString(const char* name, const char* value);
    void SetColor(const char* name, Color_RGBA8 value);
    void SetColor24(const char* name, Color_RGB8 value);

    void RegisterInteger(const char* name, int32_t defaultValue);
    void RegisterFloat(const char* name, float defaultValue);
    void RegisterString(const char* name, const char* defaultValue);
    void RegisterColor(const char* name, Color_RGBA8 defaultValue);
    void RegisterColor24(const char* name, Color_RGB8 defaultValue);

    void ClearVariable(const char* name);
    void ClearBlock(const char* name);
    void CopyVariable(const char* from, const char* to);

    void Save();
    void Load();

  protected:
    void LoadFromPath(std::string path,
                      nlohmann::detail::iteration_proxy<nlohmann::detail::iter_impl<nlohmann::json>> items);
    void LoadLegacy();

  private:
    // Transparent hashing so a lookup can take the const char* the caller already has. With a plain
    // unordered_map<std::string, ...> every find() has to build a std::string first, and CVar names
    // ("gCosmetics.Navi.IdlePrimary.Changed") run well past the small-string buffer -- so the hot read
    // path was doing a heap allocation and a free per lookup, purely to throw the string away again.
    struct TransparentStringHash {
        using is_transparent = void;
        size_t operator()(std::string_view name) const noexcept {
            return std::hash<std::string_view>{}(name);
        }
    };

    std::unordered_map<std::string, std::shared_ptr<CVar>, TransparentStringHash, std::equal_to<>> mVariables;
};
} // namespace Ship
