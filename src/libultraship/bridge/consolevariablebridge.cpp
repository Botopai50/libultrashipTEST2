#include "libultraship/bridge/consolevariablebridge.h"
#include "ship/Context.h"

std::shared_ptr<Ship::CVar> CVarGet(const char* name) {
    return Ship::Context::GetInstance()->GetConsoleVariables()->Get(name);
}

extern "C" {
// The read side goes through GetConsoleVariablesRaw rather than GetInstance()->GetConsoleVariables().
// These are called well over a thousand times a frame from game code -- z_parameter.c alone has 481 call
// sites, all of them running every frame for the HUD -- and the owning form costs four atomic
// read-modify-writes per call (a weak_ptr::lock compare-exchange loop, then a shared_ptr copy) to reach
// a pointer that is fixed for the lifetime of the game.
//
// The writes are left as they were: they happen when a setting changes, not per frame, so there is
// nothing to win and the owning form is the safer default.
int32_t CVarGetInteger(const char* name, int32_t defaultValue) {
    return Ship::Context::GetConsoleVariablesRaw()->GetInteger(name, defaultValue);
}

float CVarGetFloat(const char* name, float defaultValue) {
    return Ship::Context::GetConsoleVariablesRaw()->GetFloat(name, defaultValue);
}

const char* CVarGetString(const char* name, const char* defaultValue) {
    return Ship::Context::GetConsoleVariablesRaw()->GetString(name, defaultValue);
}

Color_RGBA8 CVarGetColor(const char* name, Color_RGBA8 defaultValue) {
    return Ship::Context::GetConsoleVariablesRaw()->GetColor(name, defaultValue);
}

Color_RGB8 CVarGetColor24(const char* name, Color_RGB8 defaultValue) {
    return Ship::Context::GetConsoleVariablesRaw()->GetColor24(name, defaultValue);
}

void CVarSetInteger(const char* name, int32_t value) {
    Ship::Context::GetInstance()->GetConsoleVariables()->SetInteger(name, value);
}

void CVarSetFloat(const char* name, float value) {
    Ship::Context::GetInstance()->GetConsoleVariables()->SetFloat(name, value);
}

void CVarSetString(const char* name, const char* value) {
    Ship::Context::GetInstance()->GetConsoleVariables()->SetString(name, value);
}

void CVarSetColor(const char* name, Color_RGBA8 value) {
    Ship::Context::GetInstance()->GetConsoleVariables()->SetColor(name, value);
}

void CVarSetColor24(const char* name, Color_RGB8 value) {
    Ship::Context::GetInstance()->GetConsoleVariables()->SetColor24(name, value);
}

void CVarRegisterInteger(const char* name, int32_t defaultValue) {
    Ship::Context::GetInstance()->GetConsoleVariables()->RegisterInteger(name, defaultValue);
}

void CVarRegisterFloat(const char* name, float defaultValue) {
    Ship::Context::GetInstance()->GetConsoleVariables()->RegisterFloat(name, defaultValue);
}

void CVarRegisterString(const char* name, const char* defaultValue) {
    Ship::Context::GetInstance()->GetConsoleVariables()->RegisterString(name, defaultValue);
}

void CVarRegisterColor(const char* name, Color_RGBA8 defaultValue) {
    Ship::Context::GetInstance()->GetConsoleVariables()->RegisterColor(name, defaultValue);
}

void CVarRegisterColor24(const char* name, Color_RGB8 defaultValue) {
    Ship::Context::GetInstance()->GetConsoleVariables()->RegisterColor24(name, defaultValue);
}

void CVarClear(const char* name) {
    Ship::Context::GetInstance()->GetConsoleVariables()->ClearVariable(name);
}

void CVarClearBlock(const char* name) {
    Ship::Context::GetInstance()->GetConsoleVariables()->ClearBlock(name);
}

void CVarCopy(const char* from, const char* to) {
    Ship::Context::GetInstance()->GetConsoleVariables()->CopyVariable(from, to);
}

void CVarLoad() {
    Ship::Context::GetInstance()->GetConsoleVariables()->Load();
}

void CVarSave() {
    Ship::Context::GetInstance()->GetConsoleVariables()->Save();
}
}
