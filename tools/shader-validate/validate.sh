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

# One profile, Shader Model 4.0, which is what the renderer now compiles everything against. The shadow
# kernel used to have a second 4.1 form so it could use Texture2DArray.Gather, and every option set was built
# twice to keep both honest; SampleCmpLevelZero does the same fetch at 4.0, so there is one form again.
fail=0
for combo in ":4_0" "t:4_0" "s:4_0" "ts:4_0" "sf:4_0" "ts2:4_0" "sa:4_0" "san:4_0"; do
  opts="${combo%%:*}"
  model="${combo##*:}"
  "$WORK/prism_driver" "$SHADER" "$SHADER_DIR" "$opts" > "$WORK/v.hlsl"
  for entry in VSMain PSMain; do
    case "$entry" in VSMain) profile="vs_$model" ;; *) profile="ps_$model" ;; esac
    out="$(compile "$WORK/v.hlsl" "$entry" "$profile" || true)"
    if printf '%s' "$out" | grep -qiE "error|FAILED"; then
      echo "FAIL  options='${opts:-none}'  $entry  $profile"
      printf '%s\n' "$out" | grep -iE "error|FAILED" | head -5
      fail=1
    else
      echo "ok    options='${opts:-none}'  $entry  $profile"
    fi
  done
done

[ "$fail" -eq 0 ] && echo "all variants compile" || echo "SOME VARIANTS FAILED"
exit "$fail"
