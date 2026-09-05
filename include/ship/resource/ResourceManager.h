#pragma once

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <list>
#include <vector>
#include <mutex>
#include <queue>
#include <variant>
#include "ship/resource/Resource.h"
#include "ship/resource/ResourceLoader.h"
#include "ship/resource/archive/Archive.h"
#include "ship/resource/archive/ArchiveManager.h"

#define BS_THREAD_POOL_ENABLE_PRIORITY
#define BS_THREAD_POOL_ENABLE_PAUSE
#include <BS_thread_pool.hpp>

namespace Ship {
struct File;

struct ResourceFilter {
    ResourceFilter(const std::list<std::string>& includeMasks, const std::list<std::string>& excludeMasks,
                   const uintptr_t owner, const std::shared_ptr<Archive> parent);

    const std::list<std::string> IncludeMasks;
    const std::list<std::string> ExcludeMasks;
    const uintptr_t Owner = 0;
    const std::shared_ptr<Archive> Parent = nullptr;
};

struct ResourceIdentifier {
    friend struct ResourceIdentifierHash;

    ResourceIdentifier(const std::string& path, const uintptr_t owner, const std::shared_ptr<Archive> parent);
    bool operator==(const ResourceIdentifier& rhs) const;

    // Must be an exact path. Passing a path with a wildcard will return a fail state
    const std::string Path = "";
    const uintptr_t Owner = 0;
    const std::shared_ptr<Archive> Parent = nullptr;

  private:
    size_t GetHash() const;
    size_t CalculateHash();
    size_t mHash;
};

struct ResourceIdentifierHash {
    size_t operator()(const ResourceIdentifier& rcd) const;
};

class ResourceManager {
    friend class ResourceLoader;
    typedef enum class ResourceLoadError { None, NotCached, NotFound } ResourceLoadError;

  public:
    ResourceManager();
    void Init(const std::vector<std::string>& archivePaths, const std::unordered_set<uint32_t>& validHashes,
              int32_t reservedThreadCount = 1);
    ~ResourceManager();

    bool IsLoaded();

    std::shared_ptr<ArchiveManager> GetArchiveManager();
    std::shared_ptr<ResourceLoader> GetResourceLoader();

    std::shared_ptr<IResource> GetCachedResource(const std::string& filePath, bool loadExact = false);
    std::shared_ptr<IResource> GetCachedResource(const ResourceIdentifier& identifier, bool loadExact = false);
    std::shared_ptr<IResource> LoadResource(const std::string& filePath, bool loadExact = false,
                                            std::shared_ptr<ResourceInitData> initData = nullptr);
    std::shared_ptr<IResource> LoadResource(const ResourceIdentifier& identifier, bool loadExact = false,
                                            std::shared_ptr<ResourceInitData> initData = nullptr);
    std::shared_ptr<IResource> LoadResource(uint64_t crc, bool loadExact = false,
                                            std::shared_ptr<ResourceInitData> initData = nullptr);
    std::shared_ptr<IResource> LoadResourceProcess(const std::string& filePath, bool loadExact = false,
                                                   std::shared_ptr<ResourceInitData> initData = nullptr);
    std::shared_ptr<IResource> LoadResourceProcess(const ResourceIdentifier& identifier, bool loadExact = false,
                                                   std::shared_ptr<ResourceInitData> initData = nullptr);
    std::shared_future<std::shared_ptr<IResource>>
    LoadResourceAsync(const std::string& filePath, bool loadExact = false, BS::priority_t priority = BS::pr::normal,
                      std::shared_ptr<ResourceInitData> initData = nullptr);
    std::shared_future<std::shared_ptr<IResource>>
    LoadResourceAsync(const ResourceIdentifier& identifier, bool loadExact = false,
                      BS::priority_t priority = BS::pr::normal, std::shared_ptr<ResourceInitData> initData = nullptr);
    size_t UnloadResource(const ResourceIdentifier& identifier);
    size_t UnloadResource(const std::string& filePath);
    bool WriteResource(const ResourceIdentifier& identifier, const std::vector<uint8_t>& data, bool unloadFile);

    std::shared_ptr<std::vector<std::shared_ptr<IResource>>> LoadResources(const std::string& searchMask);
    std::shared_ptr<std::vector<std::shared_ptr<IResource>>> LoadResources(const ResourceFilter& filter);
    std::shared_future<std::shared_ptr<std::vector<std::shared_ptr<IResource>>>>
    LoadResourcesAsync(const std::string& searchMask, BS::priority_t priority = BS::pr::normal);
    std::shared_future<std::shared_ptr<std::vector<std::shared_ptr<IResource>>>>
    LoadResourcesAsync(const ResourceFilter& filter, BS::priority_t priority = BS::pr::normal);

    void DirtyResources(const std::string& searchMask);
    void DirtyResources(const ResourceFilter& filter);
    void UnloadResources(const std::string& searchMask);
    void UnloadResources(const ResourceFilter& filter);
    void UnloadResourcesAsync(const std::string& searchMask, BS::priority_t priority = BS::pr::normal);
    void UnloadResourcesAsync(const ResourceFilter& filter, BS::priority_t priority = BS::pr::normal);

    bool OtrSignatureCheck(const char* fileName);
    bool IsAltAssetsEnabled();
    void SetAltAssetsEnabled(bool isEnabled);
    std::shared_ptr<File> LoadFileProcess(const ResourceIdentifier& identifier);
    std::shared_ptr<File> LoadFileProcess(const std::string& filePath);

    size_t GetResourceSize(std::shared_ptr<IResource> resource);
    size_t GetResourceSize(const char* name);
    size_t GetResourceSize(uint64_t crc);

    bool GetResourceIsCustom(std::shared_ptr<IResource> resource);
    bool GetResourceIsCustom(const char* name);
    bool GetResourceIsCustom(uint64_t crc);

    void* GetResourceRawPointer(std::shared_ptr<IResource> resource);
    void* GetResourceRawPointer(const char* name);
    void* GetResourceRawPointer(uint64_t crc);

  protected:
    std::shared_ptr<std::vector<std::shared_ptr<IResource>>> LoadResourcesProcess(const ResourceFilter& filter);
    void UnloadResourcesProcess(const ResourceFilter& filter);
    std::variant<ResourceLoadError, std::shared_ptr<IResource>> CheckCache(const ResourceIdentifier& identifier,
                                                                           bool loadExact = false);
    std::variant<ResourceLoadError, std::shared_ptr<IResource>> CheckCache(const std::string& filePath,
                                                                           bool loadExact = false);

    std::shared_ptr<IResource> GetCachedResource(std::variant<ResourceLoadError, std::shared_ptr<IResource>> cacheLine);

  private:
    // Called after signature normalization and a cache miss by both public loading APIs.
    std::shared_future<std::shared_ptr<IResource>>
    QueueResourceLoad(const ResourceIdentifier& identifier, bool loadExact, BS::priority_t priority,
                      std::shared_ptr<ResourceInitData> initData);

    // A load that has been queued but has not finished yet, so a second request for the same thing can wait
    // on the first instead of repeating it.
    //
    // Without this the duplicate work still happened in full -- read, decompress, parse, allocate -- and was
    // thrown away at the end: LoadResourceProcess rechecks the cache after building the resource and drops
    // whatever it just made if another thread got there first. The result was always correct; it was the
    // work that was wasted, and on a room load with several workers pulling the same shared assets that is
    // the same file decompressed several times over.
    //
    // Keyed with loadExact alongside the identifier because the two are a different question: the exact
    // load skips alt-asset redirection, so sharing one future between them would hand a caller the resource
    // it specifically asked not to get.
    struct InFlightKey {
        ResourceIdentifier Identifier;
        bool LoadExact;
        bool operator==(const InFlightKey& rhs) const {
            return LoadExact == rhs.LoadExact && Identifier == rhs.Identifier;
        }
    };
    struct InFlightKeyHash {
        size_t operator()(const InFlightKey& key) const {
            return ResourceIdentifierHash{}(key.Identifier) ^ (key.LoadExact ? 0x9e3779b97f4a7c15ULL : 0ULL);
        }
    };
    // Removes the entry when the load finishes, however it finishes -- a normal return or an exception.
    // A nested type of ResourceManager rather than a local struct inside the worker lambda: a local class
    // declared inside a lambda reaching for the enclosing class's private members is exactly the kind of
    // access question compilers disagree about, and there is nothing to gain by asking it.
    struct InFlightGuard {
        ResourceManager* Manager;
        InFlightKey Key;
        ~InFlightGuard();
    };

    std::mutex mInFlightMutex;
    std::unordered_map<InFlightKey, std::shared_future<std::shared_ptr<IResource>>, InFlightKeyHash> mInFlight;

    std::unordered_map<ResourceIdentifier, std::variant<ResourceLoadError, std::shared_ptr<IResource>>,
                       ResourceIdentifierHash>
        mResourceCache;
    std::shared_ptr<ResourceLoader> mResourceLoader;
    std::shared_ptr<ArchiveManager> mArchiveManager;
    std::shared_ptr<BS::thread_pool> mThreadPool;
    std::mutex mMutex;
    bool mAltAssetsEnabled = false;
    // Private information for which owner and archive are default.
    uintptr_t mDefaultCacheOwner = 0;
    std::shared_ptr<Archive> mDefaultCacheArchive = nullptr;
};
} // namespace Ship
