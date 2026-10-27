#pragma once

#include <filesystem>
#include <string>

// Local-filesystem durability boundary: synchronize file contents before
// publishing a replacement, then synchronize the containing directory.
// Appends synchronize before returning; a torn record must be rejected by
// the caller's record decoder, never interpreted as a committed record.
bool common_agent_write_durable_file(const std::filesystem::path & path,
        const std::string & text, bool append, std::string & error);

bool common_agent_sync_directory(const std::filesystem::path & path, std::string & error);
bool common_agent_sync_directory_ancestors(const std::filesystem::path & path, std::string & error);
