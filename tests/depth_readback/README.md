# Direct3D 11 adaptive depth readback test

Compiles the production readback helper and exercises it on a real Direct3D 11 WARP device.
No ROM, game assets or physical GPU is required. This validates readback values and lifetime,
not the visual appearance or FPS of the game.

```powershell
cmake -S tests/depth_readback -B build-depth-tests -A x64
cmake --build build-depth-tests --config Release
ctest --test-dir build-depth-tests -C Release --output-on-failure
```

Checks synchronous cold reads, reuse of a previous stationary query, completion of ring copies,
changed coordinates with identical counts, expiration, explicit reset, disabled mode and batch sizes.
The test waits for a GPU event to make completion deterministic; production does not spin or flush.
