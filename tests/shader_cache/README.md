# DirectX shader cache regression tests

The CMake target extracts the production header, hash and IO routines from `gfx_direct3d11.cpp`,
then compiles them with MSVC. It fails configuration if the extraction boundaries cannot be found.
No game assets, renderer dependencies or physical GPU are required.

```powershell
cmake -S tests/shader_cache -B build-shader-cache-tests -A x64
cmake --build build-shader-cache-tests --config Release
ctest --test-dir build-shader-cache-tests -C Release --output-on-failure
```

Covers cache round trips, key and length mismatches, old versions, payload corruption, truncated and
oversized files, repairing existing entries, concurrent reads/writes, and temporary-file cleanup.
This validates cache IO, not HLSL compilation, shader appearance or FPS.
