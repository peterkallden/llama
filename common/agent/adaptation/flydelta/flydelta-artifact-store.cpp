#include "agent/adaptation/flydelta/flydelta-artifact-store.h"

#include <chrono>
#include <fstream>
#include <iterator>

namespace {

bool valid_relative_artifact_path(
        const std::filesystem::path & path,
        std::string & error) {
    if (path.empty() || path.is_absolute() || path.extension() != ".flyd" ||
            path.lexically_normal() != path || path.filename() == "." ||
            path.filename() == "..") {
        error = "FlyDelta artifact path must be a normalized relative .flyd path";
        return false;
    }
    for (const auto & component : path) {
        if (component == "..") {
            error = "FlyDelta artifact path traversal is not allowed";
            return false;
        }
    }
    return true;
}

bool read_bounded_file(
        const std::filesystem::path & path,
        size_t max_bytes,
        std::string & text,
        std::string & error) {
    std::error_code ec;
    if (std::filesystem::is_symlink(path, ec) || ec) {
        error = "FlyDelta artifact path must not be a symlink";
        return false;
    }
    if (!std::filesystem::is_regular_file(path, ec) || ec) {
        error = "FlyDelta artifact is not a regular file";
        return false;
    }
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size > max_bytes) {
        error = "FlyDelta artifact file exceeds its byte bound";
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "could not open FlyDelta artifact";
        return false;
    }
    text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    if (input.bad() || text.size() != size || text.size() > max_bytes) {
        error = "could not read FlyDelta artifact within its byte bound";
        return false;
    }
    return true;
}

} // namespace

common_flydelta_artifact_store::common_flydelta_artifact_store(
        std::filesystem::path root_value,
        common_flydelta_artifact_store_config config_value)
    : root(std::move(root_value)), config(config_value) {}

bool common_flydelta_artifact_store::resolve(
        const std::filesystem::path & relative_path,
        std::filesystem::path & resolved,
        std::string & error) const {
    error.clear();
    if (!valid_relative_artifact_path(relative_path, error) || root.empty() || !root.is_absolute()) {
        if (error.empty()) error = "FlyDelta artifact store requires an absolute root";
        return false;
    }
    resolved = root / relative_path;
    std::error_code ec;
    if (std::filesystem::exists(resolved, ec) && std::filesystem::is_symlink(resolved, ec)) {
        error = "FlyDelta artifact path must not be a symlink";
        return false;
    }
    return !ec;
}

bool common_flydelta_artifact_store::write(
        const std::filesystem::path & relative_path,
        const common_flydelta_artifact & artifact,
        std::string & error) const {
    error.clear();
    std::filesystem::path path;
    if (!resolve(relative_path, path, error) ||
            !common_flydelta_artifact_validate(
                artifact, config.max_weights, config.max_serialized_bytes, error)) return false;

    common_flydelta_artifact normalized = artifact;
    normalized.content_hash = common_flydelta_artifact_hash(normalized);
    const std::string text = common_flydelta_artifact_to_json(normalized);
    if (text.size() > config.max_serialized_bytes) {
        error = "FlyDelta artifact exceeds serialized byte bound";
        return false;
    }

    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        if (ec) { error = "could not inspect existing FlyDelta artifact"; return false; }
        std::string existing;
        if (!read_bounded_file(path, config.max_serialized_bytes, existing, error)) return false;
        common_flydelta_artifact parsed;
        if (!common_flydelta_artifact_from_json(
                existing, config.max_weights, config.max_serialized_bytes, parsed, error)) return false;
        if (parsed.content_hash != normalized.content_hash) {
            error = "FlyDelta artifact path is immutable";
            return false;
        }
        return true;
    }

    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) { error = "could not create FlyDelta artifact directory"; return false; }
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temporary = path.string() + ".tmp-" + std::to_string(stamp);
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) { error = "could not create temporary FlyDelta artifact"; return false; }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.close();
    if (!output) {
        std::error_code cleanup;
        std::filesystem::remove(temporary, cleanup);
        error = "could not write temporary FlyDelta artifact";
        return false;
    }
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        std::error_code cleanup;
        std::filesystem::remove(temporary, cleanup);
        error = "could not atomically install FlyDelta artifact";
        return false;
    }
    return true;
}

bool common_flydelta_artifact_store::read(
        const std::filesystem::path & relative_path,
        common_flydelta_artifact & artifact,
        std::string & error) const {
    std::filesystem::path path;
    if (!resolve(relative_path, path, error)) return false;
    std::string text;
    if (!read_bounded_file(path, config.max_serialized_bytes, text, error)) return false;
    return common_flydelta_artifact_from_json(
        text, config.max_weights, config.max_serialized_bytes, artifact, error);
}
