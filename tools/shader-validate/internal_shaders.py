#!/usr/bin/env python3
"""Extract the renderer's INTERNAL HLSL shaders so they can be compiled.

default.shader.hlsl is not the only shader compiled at run time. The shadow system carries four more, as
C++ raw strings in gfx_direct3d11.cpp: the depth pass, the alpha-cutout caster pass, the moment
resolve/blur, and the screen-space mask. Nothing ever compiled them -- not the build, not CI, not
validate.sh, which only ever looked at the template. A mistake in one of them reaches a player as the same
CreateAndLoadNewShader crash, and that is exactly how one did.

Splices of the form  )" SHADOW_MAP_STR(NAME) R"(  are how those strings share a constant with the C++ side.
The value is read out of fast/shadow_map.h rather than written here, so this cannot validate a shader built
from different numbers than the game builds.

Writes one .hlsl per string into the output directory and prints, per line:
    <file> <entry> <profile>
"""
import os
import re
import sys

# Entry points and profiles each string is actually compiled with, mirroring the calls in
# gfx_direct3d11.cpp. Kept here rather than discovered, because a shader compiled at the wrong profile is
# a check that passes and a game that crashes.
SHADERS = {
    "kShadowDepthShaderSource": [("VSMain", "vs_4_0")],
    "kShadowAlphaDepthShaderSource": [("VSMain", "vs_4_0"), ("PSMain", "ps_4_0")],
    "kShadowMomentShaderSource": [("VSMain", "vs_4_0"), ("PSResolve", "ps_4_0"), ("PSBlur", "ps_4_0")],
    "kShadowMaskShaderSource": [
        ("VSPrepass", "vs_4_0"),
        ("VSFullscreen", "vs_4_0"),
        ("PSResolve", "ps_4_0"),
        ("PSBlur", "ps_4_0"),
    ],
}


def constants(header_path):
    """Every #define in shadow_map.h whose body is a plain number."""
    out = {}
    pattern = re.compile(r"^#define\s+(SHADOW_MAP_\w+)\s+([0-9][0-9a-zA-Z.+-]*)\s*$")
    with open(header_path, encoding="utf-8") as handle:
        for line in handle:
            found = pattern.match(line.strip())
            if found:
                out[found.group(1)] = found.group(2)
    return out


def main():
    if len(sys.argv) != 4:
        sys.stderr.write("usage: internal_shaders.py <gfx_direct3d11.cpp> <shadow_map.h> <outdir>\n")
        return 2
    source_path, header_path, outdir = sys.argv[1:4]
    source = open(source_path, encoding="utf-8").read()
    values = constants(header_path)
    os.makedirs(outdir, exist_ok=True)

    missing = []
    for name, entries in SHADERS.items():
        found = re.search(r'static const char\* ' + name + r' = R"\((.*?)\)";', source, re.S)
        if not found:
            missing.append(name)
            continue
        body = found.group(1)

        def splice(match):
            key = match.group(1)
            if key not in values:
                missing.append(name + ": " + key)
                return "0"
            return values[key]

        # Both spellings the file uses: the generic macro, and the one alias defined for the cutout.
        body = re.sub(r'\)"\s*SHADOW_MAP_STR\((\w+)\)\s*R"\(', splice, body)
        body = re.sub(r'\)"\s*SHADOW_MAP_ALPHA_CUTOUT_STR\s*R"\(',
                      lambda m: values.get("SHADOW_MAP_ALPHA_CUTOUT", "0.5f"), body)
        if ')"' in body or 'R"(' in body:
            missing.append(name + ": an unrecognised splice is left in the text")
            continue

        path = os.path.join(outdir, name + ".hlsl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        for entry, profile in entries:
            print(path, entry, profile)

    if missing:
        sys.stderr.write("could not extract: " + ", ".join(missing) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
