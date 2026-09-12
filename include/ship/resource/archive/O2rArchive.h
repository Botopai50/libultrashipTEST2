#pragma once

#undef _DLL

#include <string>
#include <stdint.h>
#include <string>
#include <unordered_map>

#include "zip.h"

#include "ship/resource/File.h"
#include "ship/resource/Resource.h"
#include "ship/resource/archive/Archive.h"

namespace Ship {
struct File;

class O2rArchive final : virtual public Archive {
  public:
    O2rArchive(const std::string& archivePath);
    ~O2rArchive();

    bool Open();
    bool Close();
    bool WriteFile(const std::string& filename, const std::vector<uint8_t>& data);

    std::shared_ptr<File> LoadFile(const std::string& filePath);
    std::shared_ptr<File> LoadFile(uint64_t hash);

  private:
    // Rebuilds mEntryIndex from the archive as it stands right now. Must be called after anything that
    // reopens the zip, because every entry index is invalidated by that.
    void RebuildEntryIndex();

    zip_t* mZipArchive;
    // Entry name -> zip entry index, built once at Open() from the same enumeration that populates the VFS.
    // LoadFile() used to call zip_name_locate() on every single load to recover a number this class already
    // walked past at startup. Names come straight from zip_get_name(), so a name present in the archive is
    // present here; the lookup falls back to zip_name_locate() anyway, which keeps this strictly a shortcut
    // rather than a second source of truth about what the archive contains.
    std::unordered_map<std::string, zip_int64_t> mEntryIndex;
};
} // namespace Ship
