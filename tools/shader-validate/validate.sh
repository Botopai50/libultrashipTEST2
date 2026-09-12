#!/usr/bin/env bash
# Compiles the Direct3D shader template ahead of shipping it.
#
# Nothing in the ordinary build looks at that file: the renderer expands and compiles it at run time, so a
# mistake in it reaches a player as a crash out of CreateAndLoadNewShader the first time the affected
# variant is drawn. This expands it the way the renderer does and compiles every variant.
#
# Usage:  tools/shader-validate/validate.sh [path/to/libultraship]
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SRC="$ROOT/src/fast/backends/gfx_direct3d11.cpp"
SHADER="$ROOT/src/fast/shaders/directx/default.shader.hlsl"
SHADER_DIR="$ROOT/src/fast/shaders"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${TMPDIR:-/tmp}/shader-validate.$$"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

# prism comes from the CMake build tree; point PRISM_DIR at it if the build lives elsewhere.
PRISM_DIR="${PRISM_DIR:-}"
if [ -z "$PRISM_DIR" ]; then
  PRISM_DIR="$(find "$ROOT" -maxdepth 4 -type d -name prism-src -print -quit 2>/dev/null || true)"
fi
PRISM_LIB="$(find "${PRISM_DIR%/*}" -name libprism.a -print -quit 2>/dev/null || true)"
if [ -z "$PRISM_DIR" ] || [ -z "$PRISM_LIB" ]; then
  echo "prism not found. Configure the CMake build once first, or set PRISM_DIR." >&2
  exit 2
fi

# The combiner helpers are SLICED out of the renderer rather than copied, so this can never validate a
# shader expanded differently from the one the game builds.
# Anchored to the whole line, so the marker names can be mentioned in prose nearby without ending the slice.
awk '/^\/\/ PRISM-HELPERS-BEGIN$/{f=1;next} /^\/\/ PRISM-HELPERS-END$/{f=0} f' "$SRC" > "$WORK/prism_helpers.inc"
if [ ! -s "$WORK/prism_helpers.inc" ]; then
  echo "PRISM-HELPERS markers not found in $SRC -- did they get renamed?" >&2
  exit 2
fi

echo "building the preprocessor..."
c++ -std=gnu++20 -O1 -I"$PRISM_DIR/src" -I"$PRISM_DIR" -I"$WORK" -I"$ROOT/include" \
    -o "$WORK/prism_driver" "$HERE/prism_driver.cpp" "$PRISM_LIB"

# Pick a compiler: fxc is the real one and wins whenever it is on PATH.
if command -v fxc.exe >/dev/null 2>&1 || command -v fxc >/dev/null 2>&1; then
  FXC="$(command -v fxc.exe || command -v fxc)"
  compile() { "$FXC" /nologo /T "$3" /E "$2" /O2 /Fo NUL "$1" 2>&1; }
  echo "compiler: fxc ($FXC)"
elif command -v wine >/dev/null 2>&1 && command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
  x86_64-w64-mingw32-gcc -O1 -o "$WORK/hlslc.exe" "$HERE/hlslc.c"
  export WINEDEBUG="${WINEDEBUG:--all}"
  compile() { wine "$WORK/hlslc.exe" "$1" "$2" "$3" 2>/dev/null; }
  echo "compiler: Wine d3dcompiler_47 (not authoritative -- see README)"
else
  echo "no HLSL compiler. Install the Windows SDK (fxc), or wine + mingw-w64." >&2
  exit 2
fi

# Every option set is built twice where it makes a difference: the shadow kernel exists in a Shader Model
# 4.0 form that fetches its four texels one at a time, and a 4.1 form that gathers the 2x2 footprint in one
# instruction. The renderer picks between them from the adapter's feature level, so BOTH have to compile --
# and each only compiles against its own profile, which is the point of pairing them here.
# Whether one compile passed. The compiler's EXIT STATUS decides, and the message only confirms.
#
# This used to be "the output does not mention an error", which reads a compiler that printed nothing as a
# compiler that was happy -- so a crashed, aborted or missing compiler turned the whole suite green. That
# is not a hypothetical: hlslc under Wine can abort on the failure path, and when it did, every variant in
# the run reported ok with no output at all. A check that cannot fail is worse than no check.
#
# Returns 0 for pass, 1 for fail, and prints the compiler's complaint on failure.
check_one() {
    local label="$1" file="$2" entry="$3" profile="$4"
    local out status=0
    out="$(compile "$file" "$entry" "$profile" 2>&1)" || status=$?
    if [ "$status" -ne 0 ] || printf '%s' "$out" | grep -qiE "error|FAILED"; then
        echo "FAIL  $label  $entry  $profile"
        if [ -n "$out" ]; then
            printf '%s\n' "$out" | grep -iE "error|FAILED" | head -5
        else
            echo "      the compiler produced no output and exited $status"
        fi
        return 1
    fi
    echo "ok    $label  $entry  $profile"
    return 0
}

# Prove the harness can tell a good shader from a bad one before trusting a word it says. BOTH directions,
# because either alone is satisfied by a compiler that is not working: a compiler that always fails passes
# the negative half, and a harness that reads silence as success passes the positive half. Checking only
# one is how a suite goes green while compiling nothing -- which is exactly what happened here.
printf 'float4 PSMain() : SV_TARGET { return float4(1.0, 1.0, 1.0, 1.0); }\n' > "$WORK/selftest_ok.hlsl"
printf 'float4 PSMain() : SV_TARGET { return thisIsNotDeclared; }\n' > "$WORK/selftest_bad.hlsl"
selftest_out="$(compile "$WORK/selftest_ok.hlsl" PSMain ps_4_0 2>&1)" || selftest_status=$?
selftest_status="${selftest_status:-0}"
if ! check_one "self-test (valid shader)" "$WORK/selftest_ok.hlsl" PSMain ps_4_0 >/dev/null 2>&1; then
    echo "the compiler rejected a trivially valid shader, so it is not working here." >&2
    echo "it exited ${selftest_status} and said:" >&2
    printf '%s\n' "${selftest_out:-(nothing at all)}" >&2
    exit 2
fi
if check_one "self-test (broken shader)" "$WORK/selftest_bad.hlsl" PSMain ps_4_0 >/dev/null 2>&1; then
    echo "the compiler accepted a shader with an undeclared identifier -- the harness is not working." >&2
    exit 2
fi
echo "self-test: the harness passes valid shaders and fails broken ones"

fail=0
for combo in ":4_0" "t:4_0" "s:4_0" "ts:4_0" "sf:4_0" "ts2:4_0" "sa:4_0" "san:4_0" \
             "sg:4_1" "tsg:4_1" "sfg:4_1" "ts2g:4_1" "sag:4_1" "sang:4_1"; do
  opts="${combo%%:*}"
  model="${combo##*:}"
  "$WORK/prism_driver" "$SHADER" "$SHADER_DIR" "$opts" > "$WORK/v.hlsl"
  for entry in VSMain PSMain; do
    case "$entry" in VSMain) profile="vs_$model" ;; *) profile="ps_$model" ;; esac
    check_one "options='${opts:-none}'" "$WORK/v.hlsl" "$entry" "$profile" || fail=1
  done
done

# The template is not the only shader compiled at run time. The shadow system carries four more as C++ raw
# strings -- the depth pass, the cutout caster pass, the moment resolve/blur, and the screen-space mask --
# and until this section existed nothing compiled any of them. A mistake there reaches a player as the same
# CreateAndLoadNewShader crash the template's would, and that is exactly how one did.
#
# Extracted rather than copied, with the shared constants read out of fast/shadow_map.h, so this cannot
# drift into validating shaders built from different numbers than the game builds.
if command -v python3 >/dev/null 2>&1; then
  echo "internal shaders:"
  if python3 "$HERE/internal_shaders.py" "$SRC" "$ROOT/include/fast/shadow_map.h" "$WORK/internal" \
       > "$WORK/internal.list" 2> "$WORK/internal.err"; then
    while read -r file entry profile; do
      [ -n "$file" ] || continue
      check_one "$(basename "$file" .hlsl)" "$file" "$entry" "$profile" || fail=1
    done < "$WORK/internal.list"
  else
    echo "FAIL  could not extract the internal shaders:" >&2
    cat "$WORK/internal.err" >&2
    fail=1
  fi
else
  echo "python3 absent -- the internal shaders were NOT checked." >&2
fi

[ "$fail" -eq 0 ] && echo "all variants compile" || echo "SOME VARIANTS FAILED"
exit "$fail"
