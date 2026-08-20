#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LLAMA_AGENT_BOOTSTRAP_SCRIPT_PATH
#error "LLAMA_AGENT_BOOTSTRAP_SCRIPT_PATH must be defined"
#endif

namespace {

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::set<std::string> expected_tools() {
    return {
        "calculator", "time_now", "memory_search", "memory_get",
        "memory_inspect", "memory_conflict_check", "memory_remember",
        "memory_propose_update", "memory_propose_forget", "memory_link",
        "memory_compact_propose", "plan_get", "plan_propose",
        "repository.list", "repository.search", "repository.read",
        "repository.diff", "repository.log", "repository.status",
        "repository.changed_files", "workspace.list", "workspace.read",
        "workspace.search", "workspace.patch", "diagnostics.compile",
        "diagnostics.symbol", "diagnostics.references",
        "diagnostics.call_hierarchy", "diagnostics.test_failures",
        "diagnostics.format", "diagnostics.include_graph", "dataset.list",
        "dataset.inspect", "dataset.schema", "dataset.sample",
        "dataset.validate", "data.query", "data.filter", "data.aggregate",
        "data.join", "data.transform", "statistics.describe",
        "statistics.outliers", "statistics.value_counts",
        "artifact.export", "resource_read", "web_search", "web_fetch",
        "development.build", "development.test",
    };
}

std::set<std::string> read_script_tools(const std::string & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("could not open bootstrap script: " + path);
    }

    std::set<std::string> tools;
    bool in_external_section = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.find("External catalog tools") != std::string::npos) {
            in_external_section = true;
            continue;
        }
        if (in_external_section && line.find("Use --enable-tools") != std::string::npos) {
            break;
        }
        if (!in_external_section) continue;

        line = trim(line);
        if (line.empty()) continue;

        size_t start = 0;
        while (start < line.size()) {
            const size_t comma = line.find(',', start);
            const std::string tool = trim(line.substr(
                start, comma == std::string::npos ? std::string::npos : comma - start));
            if (!tool.empty()) tools.insert(tool);
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    }
    return tools;
}

void print_difference(
        const char * label,
        const std::set<std::string> & values) {
    if (values.empty()) return;
    std::cerr << label;
    for (const auto & value : values) std::cerr << " " << value;
    std::cerr << "\n";
}

} // namespace

int main() {
    try {
        const auto expected = expected_tools();
        const auto actual = read_script_tools(LLAMA_AGENT_BOOTSTRAP_SCRIPT_PATH);

        std::set<std::string> missing;
        std::set<std::string> extra;
        std::set_difference(
            expected.begin(), expected.end(), actual.begin(), actual.end(),
            std::inserter(missing, missing.end()));
        std::set_difference(
            actual.begin(), actual.end(), expected.begin(), expected.end(),
            std::inserter(extra, extra.end()));

        if (!missing.empty() || !extra.empty()) {
            print_difference("missing tools:", missing);
            print_difference("unexpected tools:", extra);
            return 1;
        }

        std::cout << "agent bootstrap tool list is consistent ("
                  << actual.size() << " external tools)\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "agent bootstrap tool list check failed: "
                  << error.what() << "\n";
        return 1;
    }
}
