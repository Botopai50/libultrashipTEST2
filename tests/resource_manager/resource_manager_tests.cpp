// Exercise the production ResourceManager, resource objects and thread pool with an in-memory
// archive/decoder boundary. No game assets, graphics device or private-member access are needed.
#include "ship/resource/ResourceManager.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>

namespace {
thread_local bool countAllocations = false;
thread_local size_t allocationCount = 0;
} // namespace

void* operator new(std::size_t size) {
    if (countAllocations) {
        ++allocationCount;
    }
    if (void* memory = std::malloc(size ? size : 1)) {
        return memory;
    }
    throw std::bad_alloc();
}
void operator delete(void* memory) noexcept {
    std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

namespace {
using namespace Ship;
void Check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct TestResource final : Resource<char> {
    using Resource::Resource;
    char value = 0;
    char* GetPointer() override {
        return &value;
    }
    size_t GetPointerSize() override {
        return 1;
    }
};

struct TestArchive final : Archive {
    using Archive::Archive;
    std::shared_ptr<File> LoadFile(const std::string&) override {
        return nullptr;
    }
    std::shared_ptr<File> LoadFile(uint64_t) override {
        return nullptr;
    }
    bool Open() override {
        return true;
    }
    bool Close() override {
        return true;
    }
    bool WriteFile(const std::string&, const std::vector<uint8_t>&) override {
        return false;
    }
};

std::mutex backendMutex;
std::condition_variable backendCv;
std::unordered_map<std::string, size_t> reads;
bool loaderEntered = false;
bool releaseLoader = false;
std::atomic<bool> failNextDecode = false;
} // namespace

// Test doubles only for archive IO and decoding. Cache selection, dirty handling, normalization,
// queueing, futures and in-flight cleanup come from the production .cpp compiled by this target.
namespace Ship {
Archive::Archive(const std::string& path) : mPath(path) {
}
Archive::~Archive() = default;
const std::string& Archive::GetPath() {
    return mPath;
}
ArchiveManager::ArchiveManager() = default;
ArchiveManager::~ArchiveManager() = default;
void ArchiveManager::Init(const std::vector<std::string>&, const std::unordered_set<uint32_t>&) {
}
bool ArchiveManager::IsLoaded() {
    return true;
}
std::shared_ptr<File> ArchiveManager::LoadFile(const std::string& path) {
    std::lock_guard lock(backendMutex);
    ++reads[path];
    if (path.starts_with("missing") || (path.starts_with("alt/") && path != "alt/item")) {
        return nullptr;
    }
    return std::make_shared<File>();
}
std::shared_ptr<std::vector<std::string>> ArchiveManager::ListFiles(const std::list<std::string>&,
                                                                    const std::list<std::string>&) {
    return std::make_shared<std::vector<std::string>>();
}
const std::string* ArchiveManager::HashToString(uint64_t) const {
    return nullptr;
}
std::shared_ptr<Archive> ArchiveManager::GetArchiveFromFile(const std::string&) {
    return nullptr;
}
bool ArchiveManager::WriteFile(std::shared_ptr<Archive>, const std::string&, const std::vector<uint8_t>&) {
    return false;
}
ResourceLoader::ResourceLoader() = default;
ResourceLoader::~ResourceLoader() = default;
std::shared_ptr<IResource> ResourceLoader::LoadResource(std::string path, std::shared_ptr<File>,
                                                        std::shared_ptr<ResourceInitData> initData) {
    if (path == "concurrent") {
        std::unique_lock lock(backendMutex);
        loaderEntered = true;
        backendCv.notify_all();
        if (!backendCv.wait_for(lock, std::chrono::seconds(5), [] { return releaseLoader; })) {
            throw std::runtime_error("decoder gate timed out");
        }
    }
    if (failNextDecode.exchange(false)) {
        throw std::runtime_error("test decode failure");
    }
    if (!initData) {
        initData = std::make_shared<ResourceInitData>();
        initData->Path = path;
    }
    return std::make_shared<TestResource>(initData);
}
} // namespace Ship

int main() {
    try {
        ResourceManager manager;
        const int reserved = std::max(0, static_cast<int>(std::thread::hardware_concurrency()) - 2);
        manager.Init({}, {}, reserved);
        const ResourceIdentifier item{ "item", 0, nullptr };
        auto original = manager.LoadResource(item);
        Check(original != nullptr, "cold synchronous load");
        Check(manager.LoadResource(item) == original, "warm hit identity");
        Check(manager.LoadResourceAsync(item).get() == original, "async warm hit identity");

        // A preconstructed identifier isolates the eliminated future allocation from path allocation.
        allocationCount = 0;
        countAllocations = true;
        for (int i = 0; i < 1000; ++i) {
            auto cached = manager.LoadResource(item);
            if (cached != original) {
                std::abort();
            }
        }
        countAllocations = false;
        const size_t syncAllocations = allocationCount;
        allocationCount = 0;
        countAllocations = true;
        for (int i = 0; i < 1000; ++i) {
            auto cached = manager.LoadResourceAsync(item).get();
            if (cached != original) {
                std::abort();
            }
        }
        countAllocations = false;
        std::cout << "1000 warm hits: sync allocations=" << syncAllocations << ", async allocations=" << allocationCount
                  << '\n';
        Check(syncAllocations == 0, "synchronous cache hit allocated");
        Check(allocationCount > syncAllocations, "allocation counter positive control");

        manager.SetAltAssetsEnabled(true);
        auto alternate = manager.LoadResource("alt/item", true);
        Check(manager.LoadResource(item) == alternate, "alternate asset selection");
        Check(manager.LoadResource(item, true) == original, "exact bypasses alternate");
        Check(manager.LoadResource("__OTR__item") == alternate, "OTR prefix with alternate");
        Check(manager.LoadResource("__OTR__item", true) == original, "OTR prefix with exact");
        alternate->Dirty();
        auto newAlternate = manager.LoadResource(item);
        Check(newAlternate != alternate && !newAlternate->IsDirty(), "dirty alternate reload");
        Check(manager.LoadResource("fallback")->GetInitData()->Path == "fallback", "missing alternate fallback");
        manager.SetAltAssetsEnabled(false);
        Check(manager.LoadResource(item) == original, "alternate toggle off");

        original->Dirty();
        auto reloaded = manager.LoadResource(item);
        Check(reloaded != original && !reloaded->IsDirty(), "dirty resource reload");
        manager.UnloadResource(item);
        Check(manager.LoadResource(item) != reloaded, "unloaded resource reload");
        Check(manager.LoadResource("missing/file") == nullptr, "missing file result");

        auto metadata = std::make_shared<ResourceInitData>();
        metadata->Path = "custom metadata";
        Check(manager.LoadResource("__OTR__metadata-sync", true, metadata)->GetInitData() == metadata,
              "prefixed synchronous initData preserved");
        Check(manager.LoadResourceAsync("__OTR__metadata-async", true, BS::pr::normal, metadata).get()->GetInitData() ==
                  metadata,
              "prefixed asynchronous initData preserved");

        auto archiveA = std::make_shared<TestArchive>("archive-a");
        auto archiveB = std::make_shared<TestArchive>("archive-b");
        const ResourceIdentifier ownedA{ "owned", 1, archiveA };
        const ResourceIdentifier ownedB{ "owned", 2, archiveA };
        const ResourceIdentifier parentB{ "owned", 1, archiveB };
        auto resourceA = manager.LoadResource(ownedA);
        Check(manager.LoadResource(ownedB) != resourceA, "owner cache isolation");
        Check(manager.LoadResource(parentB) != resourceA, "parent cache isolation");
        Check(manager.LoadResource({ "__OTR__owned", 1, archiveA }) == resourceA, "prefix preserves owner/parent");

        auto first = manager.LoadResourceAsync("concurrent");
        {
            std::unique_lock lock(backendMutex);
            Check(backendCv.wait_for(lock, std::chrono::seconds(5), [] { return loaderEntered; }),
                  "worker did not enter decoder");
        }
        auto second = manager.LoadResourceAsync("__OTR__concurrent");
        auto synchronous = std::async(std::launch::async, [&] { return manager.LoadResource("concurrent"); });
        {
            std::lock_guard lock(backendMutex);
            releaseLoader = true;
        }
        backendCv.notify_all();
        auto loaded = first.get();
        Check(second.get() == loaded && synchronous.get() == loaded, "mixed callers share result");
        {
            std::lock_guard lock(backendMutex);
            Check(reads["concurrent"] == 1, "duplicate IO for in-flight load");
        }

        failNextDecode = true;
        bool threw = false;
        try {
            manager.LoadResource("retry");
        } catch (const std::runtime_error&) { threw = true; }
        Check(threw, "decode exception was swallowed");
        Check(manager.LoadResource("retry") != nullptr, "failed in-flight entry was retained");
        std::cout << "PASS: cache, allocation, alt/exact, dirty/unload, metadata, ownership, concurrency and retry\n";
        return 0;
    } catch (const std::exception& error) {
        countAllocations = false;
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
