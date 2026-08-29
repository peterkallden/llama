#include "agent-learning-lifecycle-store.h"

#include "agent/adaptation/lifecycle-store.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void usage(const char * executable) {
    std::cerr << "usage: " << executable << " --backend BACKEND [--path PATH] --list\n"
              << "       " << executable << " --backend BACKEND --path PATH --append FILE\n";
}

bool read_file(const std::string & path, std::string & text, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "cannot read lifecycle record: " + path; return false; }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) { error = "failed reading lifecycle record: " + path; return false; }
    text = contents.str();
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string backend = "auto";
    std::string path;
    std::string append_path;
    bool list = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--backend" && index + 1 < argc) backend = argv[++index];
        else if (argument == "--path" && index + 1 < argc) path = argv[++index];
        else if (argument == "--list") list = true;
        else if (argument == "--append" && index + 1 < argc) append_path = argv[++index];
        else { usage(argv[0]); return 2; }
    }
    if (list != append_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    std::string error;
    auto store = make_agent_learning_lifecycle_store(backend, path, error);
    if (!store) {
        std::cerr << "lifecycle store error: " << error << '\n';
        return 1;
    }
    if (list) {
        const auto records = store->list(error);
        if (!error.empty()) {
            std::cerr << "lifecycle list error: " << error << '\n';
            return 1;
        }
        for (const auto & record : records) std::cout << common_learning_lifecycle_to_json(record) << '\n';
        return 0;
    }

    std::string text;
    common_learning_lifecycle_record record;
    if (!read_file(append_path, text, error) ||
            !common_learning_lifecycle_from_json(text, record, error) ||
            !common_learning_lifecycle_validate(record, 4 * 1024 * 1024, error) ||
            !store->append(record, error)) {
        std::cerr << "lifecycle append error: " << error << '\n';
        return 1;
    }
    std::cout << "appended event_id=" << record.event_id << " subject_id=" << record.subject_id << '\n';
    return 0;
}
