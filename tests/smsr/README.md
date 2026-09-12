# SMSR regressions

Standalone Windows tests; no game assets or SoH build required. Run from the application checkout:

```sh
cmake -S libultraship/tests/smsr -B build-smsr
cmake --build build-smsr --config Release --target smsr_tests smsr_prism_driver smsr_clipmap_tests smsr_upload_tests smsr_light_frame_tests smsr_depth_tests
ctest --test-dir build-smsr -C Release -R "^smsr_" --output-on-failure
```

`smsr_warp` extracts the production `ShadowProjection`, SMSR functions and dispatch function, compiles
them with the native HLSL compiler, and renders synthetic cases using Direct3D 11 WARP. A test stub
identifies calls to the original filter. Tests cover all twelve visibility cases, signed normalization,
eight diagonal orientations, binary output, search truncation, layer/texel selection and an R16 receiver
plane. `smsr-comparison.ppm` shows the unfiltered baseline on the left and SMSR on the right.

`smsr_clipmap_config` compiles production allocation and configuration routines with a real WARP device.
Only pipeline creation and the optional static cache are stubbed. It checks texture/view descriptors
and the cascade ladder's bounds.

`smsr_full_shaders` uses the project's Prism driver and native D3DCompile for six complete VS/PS shader
variants. Reflection compares the SMSR uniform offset with the production C++ buffer declaration.
The separate `prism` dependency test is not part of this suite; use the `^smsr_` filter above.

`smsr_upload_freshness` compiles the production opaque/cutout upload routines and reads back real WARP
buffers. It reproduces stale data when a room allocation returns with the same pointer and vertex count
after skipped uploads, then verifies generation invalidation repairs both buffers while preserving reuse
for an unchanged generation. It does not reproduce a particular in-game scene.

`smsr_light_frame_stability` checks rotation-minimizing light axes, the solar orbit near noon,
crossing the old up-axis threshold, antipodal directions, orthonormality and bit-identical reuse
when the light is stationary.

`smsr_depth_stability` runs the production rasterizer-state function on WARP. It rasterizes a planar
receiver across 64 D16 quantization phases and reads back the depths: a zero-bias control reproduces
temporal self-shadowing, while the quantization margin avoids it. It also checks bounded displacement,
continued occlusion by a separate surface, and a nonzero bias clamp at large cascade/clipmap extents.

For offline configuration, set `FETCHCONTENT_SOURCE_DIR_PRISM` and `FETCHCONTENT_SOURCE_DIR_SPDLOG` to
existing dependency checkouts. Prism is pinned to the revision used by libultraship. These tests verify
shader behavior on synthetic fixtures; they do not measure in-game performance or validate every material.
