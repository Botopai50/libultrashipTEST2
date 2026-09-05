#include <windows.h>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "shader_cache_io.inc"

static void Check(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char** argv) {
    try {
        Check(argc == 2, "test directory argument required");
        const std::filesystem::path dir(argv[1]);
        std::filesystem::create_directories(dir);
        const auto path = (dir / "entry.bin").string();
        std::vector<uint8_t> vs, ps;
        const std::vector<uint8_t> a(256, 0x35), b(512, 0x79);
        auto store = [&](const auto& vertex, const auto& pixel) {
            ShaderCacheStore(path, 11, 22, 100, vertex.data(), vertex.size(), pixel.data(), pixel.size());
        };
        auto load = [&] { return ShaderCacheLoad(path, 11, 22, 100, vs, ps); };
        store(a, b);
        Check(load() && vs == a && ps == b, "round trip");
        Check(!ShaderCacheLoad(path, 12, 22, 100, vs, ps), "source/flags hash mismatch");
        Check(!ShaderCacheLoad(path, 11, 23, 100, vs, ps), "second hash mismatch");
        Check(!ShaderCacheLoad(path, 11, 22, 101, vs, ps), "source length mismatch");
        Check(vs.empty() && ps.empty(), "miss must clear previously loaded bytecode");
        store(b, a);
        Check(load() && vs == b && ps == a, "replace existing entry on Windows");

        // An intact header with a damaged payload must be a miss, not invalid bytecode for the GPU.
        {
            std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
            f.seekp(sizeof(ShaderCacheHeader) + 7);
            f.put(0);
        }
        Check(!load() && vs.empty() && ps.empty(), "payload corruption");
        store(a, b);
        Check(load() && vs == a && ps == b, "repair corrupted entry");

        std::filesystem::resize_file(path, sizeof(ShaderCacheHeader) + 1);
        Check(!load(), "truncated payload");
        store(a, b);
        {
            std::ofstream f(path, std::ios::binary | std::ios::app);
            f.put(0);
        }
        Check(!load(), "trailing bytes");
        store(a, b);
        {
            std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
            f.seekp(sizeof(uint32_t));
            const uint32_t oldVersion = 1;
            f.write(reinterpret_cast<const char*>(&oldVersion), sizeof(oldVersion));
        }
        Check(!load(), "old format rejected");
        store(a, b);
        Check(load(), "old format replaced");

        std::atomic<bool> finished{ false };
        std::thread writer([&] {
            for (int i = 0; i < 200; ++i) {
                if (i % 2) {
                    store(a, b);
                } else {
                    store(b, a);
                }
            }
            finished = true;
        });
        bool mixedPair = false;
        do {
            if (load() && !((vs == a && ps == b) || (vs == b && ps == a))) {
                mixedPair = true;
            }
        } while (!finished);
        writer.join();
        Check(!mixedPair, "concurrent reader saw a mixed shader pair");
        store(a, b);
        Check(load() && vs == a && ps == b, "cache remains writable after concurrent access");
        for (const auto& file : std::filesystem::directory_iterator(dir)) {
            Check(file.path().extension() != ".tmp", "temporary file leaked");
        }
        std::filesystem::remove(path);
        std::cout << "PASS: round trip, invalidation, corruption, replacement and concurrent access\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
