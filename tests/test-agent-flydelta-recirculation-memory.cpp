#include "agent/adaptation/flydelta/flydelta-recirculation-memory.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <filesystem>

int main() {
    common_learning_in_memory_lifecycle_store journal;
    common_agent_scope scope;
    scope.namespace_id = "local";
    scope.project_id = "project-a";
    scope.session_id = "session-a";
    common_agent_recirculation_request request;
    request.source_layer = 7;
    request.destination_layer = 2;
    request.rounds = 2;
    common_agent_recirculation_observation observation;
    observation.attempted = true;
    observation.committed = true;
    observation.target_position = 18;
    observation.completed_rounds = 2;
    observation.baseline_entropy = 0.2;
    observation.refined_entropy = 0.19;
    std::string error;
    assert(common_flydelta_recirculation_remember(journal, scope, "model-a", "context-a",
        request, observation, error));
    assert(common_flydelta_recirculation_remember(journal, scope, "model-a", "context-a",
        request, observation, error));
    assert(journal.list(error).size() == 1);
    common_agent_recirculation_request recalled;
    bool found = false;
    assert(common_flydelta_recirculation_recall(journal, scope, "model-a", "context-a",
        recalled, found, error) && found);
    assert(recalled.source_layer == 7 && recalled.destination_layer == 2 && recalled.rounds == 2);
    assert(common_flydelta_recirculation_recall(journal, scope, "model-b", "context-a",
        recalled, found, error) && !found);
    assert(common_flydelta_recirculation_recall(journal, scope, "model-a", "context-b",
        recalled, found, error) && !found);
    auto other_scope = scope;
    other_scope.session_id = "session-b";
    assert(common_flydelta_recirculation_recall(journal, other_scope, "model-a", "context-a",
        recalled, found, error) && !found);
    auto project_scope = scope;
    project_scope.memory_scope = common_memory_scope::project;
    assert(common_flydelta_recirculation_remember(journal, project_scope, "model-a", "project-context",
        request, observation, error));
    auto another_session = project_scope;
    another_session.session_id = "session-c";
    assert(common_flydelta_recirculation_recall(journal, another_session, "model-a", "project-context",
        recalled, found, error) && found);
    assert(common_flydelta_recirculation_recall(journal, scope, "model-a", "project-context",
        recalled, found, error) && !found);
    auto uncommitted = observation;
    uncommitted.committed = false;
    assert(!common_flydelta_recirculation_remember(journal, scope, "model-a", "context-a",
        request, uncommitted, error));
    auto weak = observation;
    weak.refined_entropy = 0.1995;
    assert(!common_flydelta_recirculation_remember(journal, scope, "model-a", "context-a",
        request, weak, error));
    auto updated = request;
    updated.alpha = 0.2f;
    assert(common_flydelta_recirculation_remember(journal, scope, "model-a", "context-a",
        updated, observation, error));
    assert(common_flydelta_recirculation_recall(journal, scope, "model-a", "context-a",
        recalled, found, error) && found && recalled.alpha == 0.2f);
    assert(common_flydelta_recirculation_suppress(journal, scope, "model-a", "context-a",
        updated, error));
    assert(common_flydelta_recirculation_recall(journal, scope, "model-a", "context-a",
        recalled, found, error) && !found);
    auto replacement = request;
    replacement.alpha = 0.3f;
    assert(common_flydelta_recirculation_remember(journal, scope, "model-a", "context-a",
        replacement, observation, error));
    assert(common_flydelta_recirculation_recall(journal, scope, "model-a", "context-a",
        recalled, found, error) && found && recalled.alpha == 0.3f);
    const auto records = journal.list(error);
    assert(records.size() == 5);
    for (const auto & record : records) {
        assert(record.kind == common_learning_lifecycle_kind::flydelta_experiment);
        assert(record.status == common_learning_lifecycle_status::observed);
    }
    const auto path = std::filesystem::temp_directory_path() /
        ("recirculation-memory-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".jsonl");
    {
        common_learning_jsonl_lifecycle_store durable;
        assert(durable.open(path, error));
        assert(common_flydelta_recirculation_remember(durable, project_scope, "model-a",
            "durable-context", request, observation, error));
    }
    {
        common_learning_jsonl_lifecycle_store reopened;
        assert(reopened.open(path, error));
        assert(common_flydelta_recirculation_recall(reopened, another_session, "model-a",
            "durable-context", recalled, found, error) && found);
    }
    std::filesystem::remove(path);
}
