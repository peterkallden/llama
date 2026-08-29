#include "tools/agent/host/agent-host-config.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string environment_value(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr ? value : std::string();
}

void set_environment(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

void unset_environment(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

struct environment_guard {
    std::string llama_agent_config = environment_value("LLAMA_AGENT_CONFIG");
    std::string xdg_config_home = environment_value("XDG_CONFIG_HOME");
    std::string home = environment_value("HOME");

    ~environment_guard() {
        if (llama_agent_config.empty()) unset_environment("LLAMA_AGENT_CONFIG");
        else set_environment("LLAMA_AGENT_CONFIG", llama_agent_config);
        if (xdg_config_home.empty()) unset_environment("XDG_CONFIG_HOME");
        else set_environment("XDG_CONFIG_HOME", xdg_config_home);
        if (home.empty()) unset_environment("HOME");
        else set_environment("HOME", home);
    }
};

bool write_config(const std::filesystem::path & path) {
    std::ofstream output(path);
    output << R"({"model":{"path":"fake.gguf"}})";
    return output.good();
}

bool expect_path(
        const std::string & explicit_path,
        const std::filesystem::path & expected,
        const char * label) {
    std::string path;
    std::string error;
    if (!resolve_agent_host_config_path(explicit_path, path, error) ||
            std::filesystem::path(path) != expected) {
        std::fprintf(stderr, "%s resolution failed: %s\n", label, error.c_str());
        return false;
    }
    return true;
}

} // namespace

int main() {
    environment_guard guard;
    const auto root = std::filesystem::temp_directory_path() / "llama-agent-config-discovery-smoke";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    std::filesystem::create_directories(root / "xdg" / "llama-agent");
    std::filesystem::create_directories(root / "home" / ".config" / "llama-agent");

    const auto explicit_path = root / "explicit.json";
    const auto environment_path = root / "environment.json";
    const auto xdg_path = root / "xdg" / "llama-agent" / "config.json";
    const auto home_path = root / "home" / ".config" / "llama-agent" / "config.json";
    if (!write_config(explicit_path) || !write_config(environment_path) ||
            !write_config(xdg_path) || !write_config(home_path)) {
        std::fprintf(stderr, "failed to create config discovery fixtures\n");
        return 1;
    }

    set_environment("LLAMA_AGENT_CONFIG", environment_path.string());
    if (!expect_path(explicit_path.string(), explicit_path, "explicit")) return 1;
    if (!expect_path("", environment_path, "environment")) return 1;

    unset_environment("LLAMA_AGENT_CONFIG");
    set_environment("XDG_CONFIG_HOME", (root / "xdg").string());
    set_environment("HOME", (root / "missing-home").string());
    if (!expect_path("", xdg_path, "XDG")) return 1;

    unset_environment("XDG_CONFIG_HOME");
    set_environment("HOME", (root / "home").string());
    if (!expect_path("", home_path, "HOME")) return 1;

    std::string path;
    std::string error;
    set_environment("LLAMA_AGENT_CONFIG", (root / "missing.json").string());
    if (resolve_agent_host_config_path("", path, error) ||
            error.find("LLAMA_AGENT_CONFIG") == std::string::npos) {
        std::fprintf(stderr, "missing explicit environment config was accepted\n");
        return 1;
    }

    agent_host_config processor_config;
    const nlohmann::ordered_json processor_policy_config = {
        {"resources", {
            {"processor_policies", {
                {"pdf.page_image", {
                    {"execution", "sandbox_required"},
                    {"backend", "kubernetes"},
                    {"image", "registry.example/pdf-worker@sha256:test"},
                    {"expected_version", "mupdf-1.26"},
                }},
                {"pdf.text", {
                    {"execution", "local_preferred"},
                    {"backend", "auto"},
                    {"executable", "mutool"},
                }},
                {"odt.text", {
                    {"execution", "local_preferred"},
                    {"backend", "auto"},
                    {"executable", "pandoc"},
                    {"expected_version", "pandoc 3.10.1"},
                }},
                {"html.text", {
                    {"execution", "sandbox_required"},
                    {"backend", "docker"},
                    {"image", "registry.example/document-worker@sha256:test"},
                }},
                {"xlsx.workbook", {
                    {"execution", "local_preferred"},
                    {"backend", "auto"},
                    {"executable", "python"},
                    {"script", "scripts/agent-xlsx-to-json.py"},
                }},
            }},
        }},
    };
    if (!parse_agent_host_config_json(processor_policy_config, processor_config, error) ||
            !validate_agent_host_config(processor_config, error) ||
            processor_config.resource_processor_policies.size() != 5 ||
            processor_config.resource_processor_policies.at("pdf.page_image").execution != "sandbox_required" ||
            processor_config.resource_processor_policies.at("pdf.page_image").backend != "kubernetes" ||
            processor_config.resource_processor_policies.at("odt.text").executable != "pandoc" ||
            processor_config.resource_processor_policies.at("html.text").backend != "docker" ||
            processor_config.resource_processor_policies.at("xlsx.workbook").script != "scripts/agent-xlsx-to-json.py") {
        std::fprintf(stderr, "resource processor execution policy was not parsed: %s\n", error.c_str());
        return 1;
    }
    const auto serialized = agent_host_config_to_json(processor_config);
    if (!serialized.contains("resources") ||
            !serialized["resources"].contains("processor_policies") ||
            serialized["resources"]["processor_policies"]["pdf.text"]["executable"] != "mutool") {
        std::fprintf(stderr, "resource processor execution policy was not serialized\n");
        return 1;
    }

    agent_host_config multimodal_config;
    const nlohmann::ordered_json multimodal_json = {
        {"model", {
            {"path", "models/qwen2-vl.gguf"},
            {"mmproj", "models/qwen2-vl-mmproj.gguf"},
        }},
    };
    if (!parse_agent_host_config_json(multimodal_json, multimodal_config, error) ||
            multimodal_config.model_path != "models/qwen2-vl.gguf" ||
            multimodal_config.mmproj_path != "models/qwen2-vl-mmproj.gguf") {
        std::fprintf(stderr, "multimodal model configuration was not parsed: %s\n", error.c_str());
        return 1;
    }
    const auto multimodal_serialized = agent_host_config_to_json(multimodal_config);
    if (multimodal_serialized["model"]["mmproj"] != "models/qwen2-vl-mmproj.gguf") {
        std::fprintf(stderr, "multimodal model configuration was not serialized\n");
        return 1;
    }

    agent_host_config catalog_config;
    const nlohmann::ordered_json catalog_json = {
        {"model", {{"path", "legacy.gguf"}}},
        {"models", {
            {"schema_version", 1},
            {"directory", "/models"},
            {"bases", {
                {"small", {{"kind", "generation"}, {"path", "small.gguf"}, {"load", "resident"}}},
                {"nomic", {{"kind", "embedding"}, {"path", "nomic.gguf"}}},
            }},
            {"profiles", {
                {"agent-default", {{"base", "small"}, {"context_size", 4096}}},
            }},
            {"routing", {{"default_profile", "agent-default"}, {"embedding_model", "nomic"}}},
            {"limits", {{"max_loaded_generation_models", 2}, {"model_eviction", "lru"}}},
        }},
    };
    if (!parse_agent_host_config_json(catalog_json, catalog_config, error) ||
            !validate_agent_host_config(catalog_config, error) ||
            catalog_config.model_path != "legacy.gguf" ||
            catalog_config.model_catalog.bases.size() != 2 ||
            catalog_config.model_catalog.profiles.size() != 1 ||
            catalog_config.model_catalog.embedding_model_id != "nomic") {
        std::fprintf(stderr, "model catalog configuration was not parsed: %s\n", error.c_str());
        return 1;
    }
    const auto catalog_serialized = agent_host_config_to_json(catalog_config);
    if (!catalog_serialized.contains("models") ||
            catalog_serialized["models"]["profiles"]["agent-default"]["base"] != "small") {
        std::fprintf(stderr, "model catalog configuration was not serialized\n");
        return 1;
    }

    agent_host_config invalid_processor_config;
    const nlohmann::ordered_json invalid_policy_config = {
        {"resources", {
            {"processor_policies", {
                {"pdf.page_image", {
                    {"execution", "sandbox_required"},
                    {"backend", "local"},
                }},
            }},
        }},
    };
    if (parse_agent_host_config_json(invalid_policy_config, invalid_processor_config, error)) {
        std::fprintf(stderr, "invalid resource processor execution policy was accepted\n");
        return 1;
    }

    std::printf("config_discovery=passed\n");
    return 0;
}
