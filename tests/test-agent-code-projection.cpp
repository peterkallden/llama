#include "agent-clangd-projection.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace {
using json = agent_clangd_json;
void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
json range(int line, int column) {
    return {{"start", {{"line", line}, {"character", column}}},
        {"end", {{"line", line}, {"character", column + 3}}}};
}
}

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("llama-code-projection-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(root);
        const auto selected = json{{"uri", agent_clangd_file_uri(root / "src" / u8"file å#.cpp")},
            {"range", range(0, 0)}, {"selectionRange", range(3, 5)}, {"name", "target"}};
        const auto other = json{{"uri", agent_clangd_file_uri(root / "src" / "caller.cpp")},
            {"range", range(1, 0)}, {"selectionRange", range(2, 4)}, {"name", "caller"}};
        json projected;
        std::string error;
        require(agent_clangd_project_location(selected, root.string(), projected, error), "encoded source location must be accepted");
        require(projected["path"] == u8"src/file å#.cpp" && projected["line"] == 4 && projected["column"] == 6,
            "projection must decode the path and use the symbol selection range");
        require(agent_clangd_project_call({{"from", other}, {"fromRanges", json::array({range(8, 11)})}},
            selected, true, root.string(), projected, error), "raw incoming call must obtain its selected target");
        require(projected["from"]["name"] == "caller" && projected["to"]["name"] == "target" &&
            projected["call_sites"][0]["path"] == "src/caller.cpp" && projected["call_sites"][0]["line"] == 9,
            "incoming call direction and source anchors must be preserved");
        require(agent_clangd_project_call({{"to", other}, {"fromRanges", json::array({range(9, 2)})}},
            selected, false, root.string(), projected, error), "raw outgoing call must obtain its selected caller");
        require(projected["from"]["name"] == "target" && projected["to"]["name"] == "caller" &&
            projected["call_sites"][0]["path"] == u8"src/file å#.cpp", "outgoing call sites belong to the selected caller");
        auto invalid = selected;
        invalid["selectionRange"]["start"]["line"] = "3";
        require(!agent_clangd_project_location(invalid, root.string(), projected, error), "malformed coordinate must be rejected");
        invalid = selected;
        invalid["uri"] = agent_clangd_file_uri(root.parent_path() / "outside.cpp");
        require(!agent_clangd_project_location(invalid, root.string(), projected, error), "external source location must be rejected");
        for (const auto * uri : {"file://remote/src.cpp", "file:///tmp/%00.cpp", "file:///tmp/%zz.cpp"}) {
            invalid["uri"] = uri;
            require(!agent_clangd_project_location(invalid, root.string(), projected, error), "invalid local file URI must be rejected");
        }
        const auto linked = json{{"targetUri", selected["uri"]}, {"targetSelectionRange", range(5, 7)}};
        require(agent_clangd_project_location(linked, root.string(), projected, error) && projected["line"] == 6,
            "LocationLink must use its target selection range");
        require(!agent_clangd_project_call({{"from", other}, {"fromRanges", json::array()}}, selected, true,
            root.string(), projected, error), "missing call sites must not produce an evidenced relation");
        std::filesystem::remove_all(root);
        return 0;
    } catch (const std::exception & exception) {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
        std::fprintf(stderr, "%s\n", exception.what());
        return 1;
    }
}
