#!/usr/bin/env bash
# Compiles the Direct3D shader template ahead of shipping it.
#
# Nothing in the ordinary build looks at that file: the renderer expands and compiles it at run time, so a
# mistake in it reaches a player as a crash out of CreateAndLoadNewShader the first time the affected
# variant is drawn. This expands it the way the renderer does and compiles every variant.
#
# Runs in one piece locally, and splits in two for CI, because the two halves want different machines:
# expanding needs a C++ compiler and prism, which are easy on Linux; compiling wants fxc.exe, which is only
# real on Windows. See --expand-only / --compile-only.
#
# Usage:
#   validate.sh [path/to/libultraship]                     expand and compile here
#   validate.sh [path] --expand-only DIR                   write the expansions to DIR and stop
#   validate.sh --compile-only DIR                         compile what is already in DIR (no prism needed)
set -euo pipefail

MODE="both"
STAGE_DIR=""
ROOT_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --expand-only) MODE="expand"; STAGE_DIR="${2:?--expand-only needs a directory}"; shift 2 ;;
    --compile-only) MODE="compile"; STAGE_DIR="${2:?--compile-only needs a directory}"; shift 2 ;;
    *) ROOT_ARG="$1"; shift ;;
  esac
done

ROOT="${ROOT_ARG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SRC="$ROOT/src/fast/backends/gfx_direct3d11.cpp"
SHADER="$ROOT/src/fast/shaders/directx/default.shader.hlsl"
SHADER_DIR="$ROOT/src/fast/shaders"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${TMPDIR:-/tmp}/shader-validate.$$"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

# Every option set that has a distinct expansion, and the profile it must compile against.
#
# Both halves of the alpha split are built. 'O' is the opaque combiner, whose vertex inputs are float3 where
# the alpha one's are float4; the game compiles plenty of those and for a long time none of them were
# checked here.
#
# One profile, Shader Model 4.0, which is the floor the renderer accepts (feature level 10_0) and therefore
# what it compiles everything against. Nothing in the template may need 4.1 -- two things have wanted it,
# Texture2DArray.Gather and comparison sampling of the cascade array, and neither survived contact with that
# floor.
COMBOS=(":4_0" "t:4_0" "s:4_0" "ts:4_0" "sf:4_0" "ts2:4_0" "sa:4_0" "san:4_0" \
        "O:4_0" "sO:4_0" "tsO:4_0" "sfO:4_0" "ts2O:4_0" "sf2O:4_0")

expand() {
  # prism comes from the CMake build tree; point PRISM_DIR at it if the build lives elsewhere.
  PRISM_DIR="${PRISM_DIR:-}"
  if [ -z "$PRISM_DIR" ]; then
    PRISM_DIR="$(find "$ROOT" -maxdepth 4 -type d -name prism-src -print -quit 2>/dev/null || true)"
  fi
  PRISM_LIB="${PRISM_LIB:-$(find "${PRISM_DIR%/*}" -name 'libprism.a' -print -quit 2>/dev/null || true)}"
  if [ -z "$PRISM_DIR" ] || [ -z "$PRISM_LIB" ]; then
    echo "prism not found. Configure the CMake build once first, or set PRISM_DIR (and PRISM_LIB)." >&2
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

  mkdir -p "$STAGE_DIR"
  : > "$STAGE_DIR/manifest.txt"
  for combo in "${COMBOS[@]}"; do
    local opts="${combo%%:*}"
    local model="${combo##*:}"
    local name="${opts:-none}"
    "$WORK/prism_driver" "$SHADER" "$SHADER_DIR" "$opts" > "$STAGE_DIR/$name.hlsl"
    printf '%s %s\n' "$name" "$model" >> "$STAGE_DIR/manifest.txt"
  done
  echo "expanded ${#COMBOS[@]} variants into $STAGE_DIR"
}

pick_compiler() {
  # fxc is the real one and wins whenever it is on PATH. It is also the only one whose answer is worth
  # trusting: Wine's d3dcompiler_47 has accepted shader code that fxc rejects outright, which is how a
  # variant that cannot compile at ps_4_0 once shipped past this script.
  if command -v fxc.exe >/dev/null 2>&1 || command -v fxc >/dev/null 2>&1; then
    FXC="$(command -v fxc.exe || command -v fxc)"
    compile() { "$FXC" /nologo /T "$3" /E "$2" /O2 /Fo NUL "$1" 2>&1; }
    echo "compiler: fxc ($FXC)"
  elif command -v wine >/dev/null 2>&1 && command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
    x86_64-w64-mingw32-gcc -O1 -o "$WORK/hlslc.exe" "$HERE/hlslc.c"
    export WINEDEBUG="${WINEDEBUG:--all}"
    compile() { wine "$WORK/hlslc.exe" "$1" "$2" "$3" 2>/dev/null; }
    echo "compiler: Wine d3dcompiler_47 (NOT authoritative -- see README)"
  else
    echo "no HLSL compiler. Install the Windows SDK (fxc), or wine + mingw-w64." >&2
    exit 2
  fi
}

compile_stage() {
  pick_compiler
  local fail=0
  while read -r name model; do
    [ -n "$name" ] || continue
    for entry in VSMain PSMain; do
      case "$entry" in VSMain) profile="vs_$model" ;; *) profile="ps_$model" ;; esac
      out="$(compile "$STAGE_DIR/$name.hlsl" "$entry" "$profile" || true)"
      if printf '%s' "$out" | grep -qiE "error|FAILED"; then
        echo "FAIL  options='$name'  $entry  $profile"
        printf '%s\n' "$out" | grep -iE "error|FAILED" | head -5
        fail=1
      else
        echo "ok    options='$name'  $entry  $profile"
      fi
    done
  done < "$STAGE_DIR/manifest.txt"
  [ "$fail" -eq 0 ] && echo "all variants compile" || echo "SOME VARIANTS FAILED"
  return "$fail"
}

case "$MODE" in
  expand)  expand ;;
  compile)
    [ -f "$STAGE_DIR/manifest.txt" ] || { echo "no manifest in $STAGE_DIR -- run --expand-only first" >&2; exit 2; }
    compile_stage
    ;;
  both)
    STAGE_DIR="$WORK/stage"
    expand
    compile_stage
    ;;
esac
