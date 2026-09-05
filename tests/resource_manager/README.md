# ResourceManager regression test

This standalone target compiles the production ResourceManager, Resource and Utils sources with
the actual thread pool. An in-memory archive/decoder test double replaces file IO and parsing.
It does not test graphics, archive formats or game assets.

```sh
cmake -S tests/resource_manager -B build-resource-tests
cmake --build build-resource-tests --config Release
ctest --test-dir build-resource-tests -C Release --output-on-failure
```

The first configure downloads pinned dependency releases. The warm-hit check uses a preconstructed
resource identifier and counts allocations on the calling thread. It checks zero allocations for
the synchronous hit and uses the async hit as a positive control; this is not a game FPS benchmark.

Behavior checks cover alternate/exact selection, OTR prefixes, dirty and unloaded resources,
missing files, explicit initData, owner/parent cache isolation, mixed synchronous/asynchronous
callers and retry after a decoder exception.
