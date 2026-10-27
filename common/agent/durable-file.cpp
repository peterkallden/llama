#include "agent/durable-file.h"

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

bool common_agent_sync_directory(const std::filesystem::path & path, std::string & error) {
#ifdef _WIN32
    // MoveFileExW with WRITE_THROUGH is the publication barrier below.
    (void) path;
    (void) error;
    return true;
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) { error = "cannot open durable directory: " + std::string(std::strerror(errno)); return false; }
    const bool ok = ::fsync(fd) == 0;
    if (!ok) error = "cannot synchronize durable directory: " + std::string(std::strerror(errno));
    ::close(fd);
    return ok;
#endif
}

bool common_agent_sync_directory_ancestors(const std::filesystem::path & path, std::string & error) {
    std::error_code ec;
    auto directory = std::filesystem::absolute(path, ec);
    if (ec) { error = "cannot resolve durable directory: " + ec.message(); return false; }
    for (;;) {
        if (!common_agent_sync_directory(directory, error)) return false;
        const auto parent = directory.parent_path();
        if (parent.empty() || parent == directory) return true;
        directory = parent;
    }
}

bool common_agent_write_durable_file(const std::filesystem::path & path,
        const std::string & text, bool append, std::string & error) {
    error.clear();
    static std::atomic<unsigned long long> sequence{0};
    auto target = path;
    if (!append) {
#ifdef _WIN32
        const auto pid = GetCurrentProcessId();
#else
        const auto pid = ::getpid();
#endif
        target += ".tmp-" + std::to_string(pid) + "-" + std::to_string(++sequence);
    }
#ifdef _WIN32
    HANDLE fd = CreateFileW(target.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
        append ? OPEN_ALWAYS : CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fd == INVALID_HANDLE_VALUE) { error = "cannot open durable file"; return false; }
    bool ok = !append || SetFilePointerEx(fd, {}, nullptr, FILE_END);
    size_t offset = 0;
    while (ok && offset < text.size()) {
        DWORD written = 0;
        const DWORD count = static_cast<DWORD>(std::min<size_t>(text.size() - offset, 1024U * 1024U));
        ok = WriteFile(fd, text.data() + offset, count, &written, nullptr) && written != 0;
        offset += written;
    }
    if (ok) ok = FlushFileBuffers(fd);
    CloseHandle(fd);
    if (ok && !append) ok = MoveFileExW(target.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
#else
    const int fd = ::open(target.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC |
        (append ? O_APPEND : O_EXCL), 0600);
    if (fd < 0) { error = "cannot open durable file: " + std::string(std::strerror(errno)); return false; }
    bool ok = true;
    size_t offset = 0;
    while (offset < text.size()) {
        const auto written = ::write(fd, text.data() + offset, text.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) { ok = false; break; }
        offset += static_cast<size_t>(written);
    }
    if (ok) ok = ::fsync(fd) == 0;
    if (!ok) error = "cannot synchronize durable file: " + std::string(std::strerror(errno));
    ::close(fd);
    if (ok && !append) {
        std::error_code ec;
        std::filesystem::rename(target, path, ec);
        ok = !ec;
        if (ec) error = "cannot publish durable file: " + ec.message();
    }
#endif
    if (!ok) {
        if (error.empty()) error = "cannot write or publish durable file";
        if (!append) { std::error_code ignored; std::filesystem::remove(target, ignored); }
        return false;
    }
    return common_agent_sync_directory(path.parent_path().empty() ? "." : path.parent_path(), error);
}
