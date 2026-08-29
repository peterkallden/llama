#include "agent-learning-worker.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace {

void usage(const char * executable) {
    std::cerr << "usage: " << executable << " --queue PATH [--max-job-bytes N] [--max-artifact-bytes N] [--max-corpus-bytes N] [--max-manifest-bytes N] [--max-result-bytes N] [--max-runtime-seconds N] [--worker-id ID]\n"
              << "       " << executable << " --queue PATH --watch [--poll-seconds N] [--max-jobs N] [worker options]\n"
              << "       " << executable << " --queue PATH --trainer-command KIND EXECUTABLE [--trainer-argument KIND VALUE ...]\n"
              << "       " << executable << " --capabilities\n";
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
    bool show_capabilities = false;
    bool watch = false;
    size_t poll_seconds = 5;
    size_t max_jobs = 0;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--capabilities") {
            show_capabilities = true;
        } else if (argument == "--watch") {
            watch = true;
        } else if (argument == "--queue" && index + 1 < argc) {
            queue_root = argv[++index];
        } else if (argument == "--max-artifact-bytes" && index + 1 < argc) {
            if (!parse_size(argv[++index], limits.max_artifact_bytes)) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--max-job-bytes" && index + 1 < argc) {
            if (!parse_size(argv[++index], limits.max_job_bytes) || limits.max_job_bytes == 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--max-corpus-bytes" && index + 1 < argc) {
            if (!parse_size(argv[++index], limits.max_corpus_bytes)) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--max-manifest-bytes" && index + 1 < argc) {
            if (!parse_size(argv[++index], limits.max_manifest_bytes) || limits.max_manifest_bytes == 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--max-result-bytes" && index + 1 < argc) {
            if (!parse_size(argv[++index], limits.max_result_bytes) || limits.max_result_bytes == 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--max-runtime-seconds" && index + 1 < argc) {
            if (!parse_size(argv[++index], limits.max_job_runtime_seconds)) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--worker-id" && index + 1 < argc) {
            limits.worker_id = argv[++index];
        } else if (argument == "--poll-seconds" && index + 1 < argc) {
            if (!parse_size(argv[++index], poll_seconds) || poll_seconds == 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--max-jobs" && index + 1 < argc) {
            if (!parse_size(argv[++index], max_jobs) || max_jobs == 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--trainer-command" && index + 2 < argc) {
            const std::string kind = argv[++index];
            const std::string executable = argv[++index];
            if (kind.empty() || executable.empty() || limits.external_trainer_commands.count(kind) != 0) {
                usage(argv[0]);
                return 2;
            }
            limits.external_trainer_commands[kind] = {executable};
        } else if (argument == "--trainer-argument" && index + 2 < argc) {
            const std::string kind = argv[++index];
            const std::string value = argv[++index];
            const auto trainer = limits.external_trainer_commands.find(kind);
            if (trainer == limits.external_trainer_commands.end() || value.empty()) {
                usage(argv[0]);
                return 2;
            }
            trainer->second.push_back(value);
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (show_capabilities) {
        if (!queue_root.empty() || watch || max_jobs != 0) {
            usage(argv[0]);
            return 2;
        }
        std::cout << agent_learning_worker_capabilities_json(limits) << '\n';
        return 0;
    }
    if (queue_root.empty()) {
        usage(argv[0]);
        return 2;
    }

    size_t completed_jobs = 0;
    for (;;) {
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
        if (report.state == agent_learning_worker_job_state::failed) return 1;
        if (report.state != agent_learning_worker_job_state::idle) ++completed_jobs;
        if (!watch || (max_jobs != 0 && completed_jobs >= max_jobs)) return 0;
        if (report.state == agent_learning_worker_job_state::idle) {
            std::this_thread::sleep_for(std::chrono::seconds(poll_seconds));
        }
    }
}
