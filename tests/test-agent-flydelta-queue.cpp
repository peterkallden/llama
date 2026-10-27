#include "agent/adaptation/flydelta/flydelta-queue.h"
#include "agent/durable-file.h"

#include <filesystem>
#include <nlohmann/json.hpp>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
struct child_cleanup {
    pid_t pid;
    ~child_cleanup() { if (pid > 0) { ::kill(pid, SIGKILL); ::waitpid(pid, nullptr, 0); } }
};
#endif

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_job make_job() {
    common_flydelta_experiment_job job;
    job.id = "flydelta://job/queue-1";
    job.kind = common_flydelta_experiment_job_kind::basis;
    job.seed.id = "evidence://repair/queue-1/flydelta";
    job.seed.behavior_key = "tool_use/diagnostics/missing-argument";
    job.seed.source = common_adaptation_evidence_source::tool_repair;
    job.seed.scope.namespace_id = "local";
    job.seed.scope.project_id = "project";
    job.seed.scope.session_id = "session";
    job.seed.task_fingerprint = "sha256:task";
    job.seed.model_profile_fingerprint = "sha256:model";
    job.seed.tokenizer_fingerprint = "sha256:tokenizer";
    job.seed.template_fingerprint = "sha256:template";
    job.seed.execution_context_fingerprint = "sha256:execution-context";
    job.seed.baseline_ref = "execution:failed";
    job.seed.candidate_ref = "execution:repaired";
    job.seed.verifier_ref = "verifier:v1";
    job.seed.evidence_ref = "evidence://repair/queue-1";
    job.seed.transaction_ids = {"learning://failed", "learning://repaired"};
    job.behavior_delta_ids = {"flydelta://delta/queue-1"};
    job.code_revision = "test:v1";
    return job;
}

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / "llama-agent-flydelta-queue-test";
    std::error_code ignored;
    fs::remove_all(root, ignored);
    std::string error;
    const auto limits = common_flydelta_experiment_queue_limits{};
    const auto job = make_job();

#ifndef _WIN32
    // Advisory ownership must reject another process, not merely another
    // thread. A stale lock file alone must never prevent later ownership.
    auto owner = common_flydelta_acquire_queue_owner(root, error);
    CHECK(owner != nullptr);
    const auto contender = ::fork();
    CHECK(contender >= 0);
    if (contender == 0) {
        std::string child_error;
        const auto second = common_flydelta_acquire_queue_owner(root, child_error);
        ::_exit(!second && child_error.find("LOCK_BUSY") != std::string::npos ? 0 : 1);
    }
    int child_status = 0;
    CHECK(::waitpid(contender, &child_status, 0) == contender);
    CHECK(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0);
    owner.reset();
    CHECK(fs::exists(root / ".owner.lock"));
    owner = common_flydelta_acquire_queue_owner(root, error);
    CHECK(owner != nullptr);
    owner.reset();

    // A fork-only child must not retain the parent's queue lock after the
    // parent releases it (CLOEXEC alone would not cover this case).
    owner = common_flydelta_acquire_queue_owner(root, error);
    CHECK(owner != nullptr);
    int released[2];
    CHECK(::pipe(released) == 0);
    const auto inheritor = ::fork();
    CHECK(inheritor >= 0);
    if (inheritor == 0) {
        ::close(released[1]);
        char marker;
        if (::read(released[0], &marker, 1) != 1) ::_exit(2);
        std::string child_error;
        const auto acquired = common_flydelta_acquire_queue_owner(root, child_error);
        ::_exit(acquired ? 0 : 1);
    }
    ::close(released[0]);
    owner.reset();
    CHECK(::write(released[1], "1", 1) == 1);
    ::close(released[1]);
    CHECK(::waitpid(inheritor, &child_status, 0) == inheritor);
    CHECK(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0);

    const auto crash_root = root / "crash-owner";
    int ready[2];
    CHECK(::pipe(ready) == 0);
    const auto crashing = ::fork();
    CHECK(crashing >= 0);
    if (crashing == 0) {
        ::close(ready[0]);
        std::string child_error;
        const auto held = common_flydelta_acquire_queue_owner(crash_root, child_error);
        const char result = held ? '1' : '0';
        if (::write(ready[1], &result, 1) != 1) ::_exit(2);
        ::close(ready[1]);
        for (;;) ::pause();
    }
    child_cleanup cleanup{crashing};
    ::close(ready[1]);
    char result = '0';
    CHECK(::read(ready[0], &result, 1) == 1 && result == '1');
    ::close(ready[0]);
    CHECK(::kill(crashing, SIGKILL) == 0);
    CHECK(::waitpid(crashing, &child_status, 0) == crashing);
    cleanup.pid = 0;
    CHECK(common_flydelta_acquire_queue_owner(crash_root, error) != nullptr);
#endif

    CHECK(common_flydelta_experiment_queue_enqueue(root, job, limits, error));
    CHECK(fs::directory_iterator(root / "pending") != fs::directory_iterator{});
    CHECK(!common_flydelta_experiment_queue_enqueue(root, job, limits, error));

    common_flydelta_claimed_experiment_job claimed;
    CHECK(common_flydelta_experiment_queue_claim_next(root, limits, claimed, error));
    CHECK(claimed.job.id == job.id);
    CHECK(claimed.job.seed.scope.session_id == job.seed.scope.session_id);
    CHECK(fs::exists(root / "running" / claimed.queue_key / "job.json"));
    CHECK(common_flydelta_experiment_queue_complete(root, claimed,
        common_flydelta_experiment_queue_state::succeeded, "basis candidate ready", limits, error));
    CHECK(fs::exists(root / "succeeded" / claimed.queue_key / "state.json"));

    common_flydelta_claimed_experiment_job empty;
    CHECK(common_flydelta_experiment_queue_claim_next(root, limits, empty, error));
    CHECK(empty.queue_key.empty());

    auto abandoned_job = make_job();
    abandoned_job.id = "flydelta://job/abandoned";
    CHECK(common_flydelta_experiment_queue_enqueue(root, abandoned_job, limits, error));
    CHECK(common_flydelta_experiment_queue_claim_next(root, limits, claimed, error));
    CHECK(common_flydelta_experiment_queue_recover(root, limits, error));
    CHECK(common_flydelta_experiment_queue_claim_next(root, limits, claimed, error));
    CHECK(claimed.job.id == abandoned_job.id);
    // Crash after writing the terminal acknowledgement, before directory move.
    const auto terminal = nlohmann::ordered_json{{"state", "succeeded"},
        {"job_id", abandoned_job.id}, {"safe_summary", "durable"}}.dump();
    CHECK(common_agent_write_durable_file(root / "running" / claimed.queue_key / "state.json", terminal, false, error));
    CHECK(common_flydelta_experiment_queue_recover(root, limits, error));
    CHECK(fs::exists(root / "succeeded" / claimed.queue_key));
    CHECK(common_flydelta_experiment_queue_claim_next(root, limits, empty, error) && empty.queue_key.empty());

    auto replay_job = make_job();
    replay_job.id = "flydelta://job/physical-replay-bound";
    auto replay_limits = limits;
    replay_limits.max_execution_attempts = 2;
    CHECK(common_flydelta_experiment_queue_enqueue(root, replay_job, replay_limits, error));
    CHECK(common_flydelta_experiment_queue_claim_next(root, replay_limits, claimed, error));
    const auto replay_key = claimed.queue_key;
    CHECK(common_flydelta_experiment_queue_recover(root, replay_limits, error));
    CHECK(common_flydelta_experiment_queue_claim_next(root, replay_limits, claimed, error));
    CHECK(common_flydelta_experiment_queue_recover(root, replay_limits, error));
    CHECK(!common_flydelta_experiment_queue_claim_next(root, replay_limits, claimed, error));
    CHECK(error.find("retry limit") != std::string::npos);
    CHECK(fs::exists(root / "failed" / replay_key));

    auto small_limits = limits;
    small_limits.max_job_bytes = 8;
    auto oversized = make_job();
    oversized.id = "flydelta://job/queue-oversized";
    CHECK(!common_flydelta_experiment_queue_enqueue(root, oversized, small_limits, error));

    fs::remove_all(root, ignored);
    return 0;
}
