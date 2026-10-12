#include "flydelta-teaching-material-store.h"

#include <filesystem>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_teaching_material_identity identity() {
    common_flydelta_teaching_material_identity value;
    value.model_profile_fingerprint = "model:v1";
    value.tokenizer_fingerprint = "tokenizer:v1";
    value.template_fingerprint = "template:v1";
    value.capture_layout_revision = "capture:v1";
    value.execution_context_fingerprint = "context:v1";
    value.scope_fingerprint = "scope:v1";
    value.verifier_revision = "verifier:v1";
    return value;
}

static common_flydelta_teaching_relation relation(const char * id) {
    common_flydelta_teaching_relation value;
    value.id = id;
    value.teaching_key = "dataset.grouped_sum.v1";
    value.source = common_adaptation_evidence_source::procedure_blueprint;
    value.behavior_key = "tool_choice/dataset/grouped_sum";
    value.scope.namespace_id = "local";
    value.scope.session_id = "session";
    value.task_fingerprint = std::string("task:") + id;
    value.baseline_ref = std::string("baseline:") + id;
    value.conditioned_ref = std::string("conditioned:") + id;
    value.control_ref = std::string("control:") + id;
    value.verifier_ref = "verifier:grouped-sum:v1";
    value.evidence_ref = std::string("evidence:") + id;
    value.status = common_flydelta_teaching_relation_status::resolved;
    value.baseline_origin = common_flydelta_teaching_origin::host_derived;
    value.conditioned_origin = common_flydelta_teaching_origin::host_derived;
    value.control_origin = common_flydelta_teaching_origin::host_counterfactual;
    value.confidence = 1.0f;
    value.host_approved = true;
    return value;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() / "llama-agent-flydelta-teaching-material.cozo";
    std::error_code ignored;
    fs::remove(path, ignored);
    std::string error;
    auto runtime = make_agent_flydelta_teaching_material_runtime(
            "cozo", path.string(), identity(), error);
    CHECK(runtime && error.empty());
    CHECK(runtime->observe_relation(relation("one"), error));
    CHECK(runtime->observe_relation(relation("two"), error));
    const auto group_ref = runtime->store().groups().front().group_ref;
    CHECK(runtime->observe_trajectory(group_ref, "trajectory://one", error));
    CHECK(runtime->observe_trajectory(group_ref, "trajectory://two", error));

    runtime.reset();
    auto reopened = make_agent_flydelta_teaching_material_runtime(
            "cozo", path.string(), identity(), error);
    CHECK(reopened && error.empty());
    bool relation_ready = false;
    bool trajectory_ready = false;
    CHECK(reopened->inspect_group(group_ref, relation_ready, trajectory_ready, error));
    CHECK(relation_ready && trajectory_ready);
    CHECK(reopened->store().groups().front().trajectory_refs.size() == 2);
    fs::remove(path, ignored);
    return 0;
}
