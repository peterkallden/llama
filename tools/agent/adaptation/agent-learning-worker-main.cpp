#include "agent-learning-worker.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

void usage(const char * executable) {
    std::cerr << "usage: " << executable << " --queue PATH [--max-artifact-bytes N]\n";
}

bool parse_size(const char * text, size_t & value) {
    if (text == nullptr || *text == '\0') return false;
    char * end = nullptr;
    const auto parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    value = static_cast<size_t>(parsed);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::filesystem::path queue_root;
    agent_learning_worker_limits limits;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--queue" && index + 1 < argc) {
            queue_root = argv[++index];
        } else if (argument == "--max-artifact-bytes" && index + 1 < argc) {
            if (!parse_size(argv[++index], limits.max_artifact_bytes)) {
                usage(argv[0]);
                return 2;
            }
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (queue_root.empty()) {
        usage(argv[0]);
        return 2;
    }

    agent_learning_worker_report report;
    std::string error;
    if (!agent_learning_worker_run_once(queue_root, limits, report, error)) {
        std::cerr << "worker error: " << error << '\n';
        return 1;
    }
    std::cout << "state=" << agent_learning_worker_job_state_name(report.state);
    if (!report.job_id.empty()) std::cout << " job_id=" << report.job_id;
    if (!report.safe_summary.empty()) std::cout << " detail=" << report.safe_summary;
    std::cout << '\n';
    return report.state == agent_learning_worker_job_state::failed ? 1 : 0;
}
