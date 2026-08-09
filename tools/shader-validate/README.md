# shader-validate

Compiles the Direct3D shader template before it ships.

## Why this exists

`src/fast/shaders/directx/default.shader.hlsl` is not a source file the build compiles. It is a template
full of preprocessor directives that the renderer expands at run time, once per shader variant, and hands
to `D3DCompile` the first time a draw needs that variant. Nothing in the ordinary build, and nothing in CI,
ever looks at it.

So a mistake in it ships. It surfaces as the game throwing out of `CreateAndLoadNewShader` — an unhandled
exception, which is a crash — at the moment the affected variant is first drawn. That can be a long way
from the change that caused it: a fault anywhere in the file fails the `vs_4_0` pass first, because FXC
compiles the whole file once per entry point, so a mistake in pixel-shader-only code reports against the
vertex shader.

This is the check that was missing. It expands the template the way the renderer does and compiles every
combination that matters.

## Running it

    tools/shader-validate/validate.sh

Configure the CMake build once first, so prism is available; set `PRISM_DIR` if it lives somewhere unusual.

Exit status is 0 when every variant compiles and 1 otherwise, so it can gate a build.

It needs an HLSL compiler and picks the best one present:

| | |
|---|---|
| `fxc` on PATH | the compiler the game runs. Authoritative. Ships with the Windows SDK. |
| `wine` + `mingw-w64` | Wine's `d3dcompiler_47`, a different implementation. See the limits below. |

## What it does and does not prove

It compiles the real template, expanded by the renderer's real combiner helpers — those are sliced out of
`gfx_direct3d11.cpp` between its `PRISM-HELPERS` markers rather than copied, so this cannot drift into
validating a shader the game does not build. Both entry points are compiled, across the toon, shadow-map,
fog and two-cycle combinations.

Two limits are worth stating plainly, because a check that is trusted past its reach is worse than no check:

- **The combiner is representative, not exhaustive.** One textured input with alpha. The game builds a
  different combiner per material, and this compiles one of them. It exercises every hand-written part of
  the file, which is where mistakes are made; it does not prove every combiner the game can generate
  expands correctly.
- **Wine's compiler is not FXC.** It reliably catches undeclared identifiers, type errors and bad
  signatures — the class of mistake that otherwise reaches a player as a crash. It is not authoritative
  about Shader Model 4's own limits, which is where this file has had trouble before: no dynamic indexing
  of vectors, no gradient instructions under non-uniform control flow, and flow the compiler cannot resolve
  producing "cannot map expression to ps_4_0". For a change that goes near those, run it under `fxc`.

  This has been measured, not assumed. Wine's `d3dcompiler_47` **does not enforce profile restrictions at
  all**: it accepts a dynamically indexed constant-buffer array at `ps_4_0`, which real FXC rejects, and it
  accepts `Texture2DArray.Gather` at `ps_4_0`, which is a Shader Model 4.1 intrinsic. So a green run here
  says "this is valid HLSL", not "this is valid at this profile". Anything whose correctness rests on the
  profile — the gathered shadow kernel does — needs a guarantee elsewhere. That one has two: the emission
  and the target profile are driven by a single flag so they cannot disagree, and the renderer probes the
  intrinsic against the real compiler and the real driver at startup before committing to it.

## Files

- `validate.sh` — slices the helpers, builds the preprocessor, compiles each variant.
- `prism_driver.cpp` — expands the template through the project's own prism context.
- `hlslc.c` — calls `D3DCompile`; built with mingw and run under Wine when `fxc` is absent.
