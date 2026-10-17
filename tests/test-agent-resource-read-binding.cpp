#include "agent/runtime/agent-runtime.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <nlohmann/json.hpp>

#include <cassert>

namespace {

common_plan_observation read_observation(const std::string & id, common_runtime_resource_ref ref) {
    common_plan_observation observation;
    observation.id = id;
    observation.source = "tool_result";
    observation.resource_refs.push_back(std::move(ref));
    return observation;
}

void bind_read_provenance(common_plan_observation & observation,
        const common_agent_resource_read_binding & binding,
        const std::string & uri = "resource://text") {
    observation.resource_read_provenance = common_plan_resource_read_provenance{
        binding.task_fingerprint,
        common_agent_resource_read_intent_name(binding.intent),
        uri,
        "sha256:source-content",
        "text/markdown",
        "resource://source"};
}

common_plan_step read_step(const std::string & tool, const nlohmann::json & arguments) {
    common_plan_step step;
    step.id = "read-step";
    step.status = common_plan_step_status::completed;
    step.tool_call = common_plan_tool_call{tool, arguments.dump()};
    return step;
}

} // namespace

int main() {
    common_agent_request request;
    common_agent_resource_read_binding binding;
    binding.intent = common_agent_resource_read_intent::orientation;
    binding.resource_uris = {"resource://source", "resource://text"};
    binding.task_fingerprint = "task-orientation-1";
    binding.require_full_coverage = true;
    binding.targets.push_back({"resource://text", "sha256:source-content", "text/markdown",
        "resource://source", 100, true});
    request.resource_read_binding = binding;
    common_agent_input_resource input;
    input.resource.uri = "resource://text";
    input.resource.size_bytes = 100;
    input.resource.mime_type = "text/markdown";
    input.resource.lineage.parent_uri = "resource://source";
    request.input_resources.push_back(input);
    common_agent_input_resource unrelated_input;
    unrelated_input.resource.uri = "resource://other";
    request.input_resources.push_back(unrelated_input);

    common_plan_state plan;
    plan.steps.push_back(read_step("resource.inspect", {{"uri", "resource://source"}}));
    plan.observations.push_back(read_observation("tool:read-step:1", input.resource));
    assert(!common_agent_plan_satisfies_resource_read_binding(plan, request));

    plan.steps.clear();
    plan.steps.push_back(read_step("resource.read", {{"uri", "resource://unrelated"}, {"offset", 0}, {"max_bytes", 100}}));
    assert(!common_agent_plan_satisfies_resource_read_binding(plan, request));
    plan.steps.back().tool_call->arguments_json = R"({"uri":"resource://other","offset":0,"max_bytes":100})";
    plan.observations.clear();
    plan.observations.push_back(read_observation("tool:read-step:1", unrelated_input.resource));
    assert(!common_agent_plan_satisfies_resource_read_binding(plan, request));

    plan.steps.clear();
    plan.observations.clear();
    plan.steps.push_back(read_step("resource.read", {{"uri", "resource://text"}, {"offset", 0}, {"max_bytes", 99}}));
    auto full_read = read_observation("tool:read-step:1", input.resource);
    bind_read_provenance(full_read, *request.resource_read_binding);
    plan.observations.push_back(full_read);
    assert(!common_agent_plan_satisfies_resource_read_binding(plan, request));
    plan.steps.back().tool_call->arguments_json = R"({"uri":"resource://text","offset":0,"max_bytes":100})";
    plan.observations.back().resource_read_provenance->task_fingerprint = "stale-task";
    assert(!common_agent_plan_satisfies_resource_read_binding(plan, request));
    plan.observations.back().resource_read_provenance->task_fingerprint = "task-orientation-1";
    assert(common_agent_plan_satisfies_resource_read_binding(plan, request));
    plan.steps.back().tool_call->arguments_json = R"({"uri":"resource://source","offset":0,"max_bytes":100})";
    assert(!common_agent_plan_satisfies_resource_read_binding(plan, request));
    plan.steps.back().tool_call->arguments_json = R"({"uri":"resource://text","offset":0,"max_bytes":100})";
    assert(common_agent_plan_satisfies_resource_read_binding(plan, request));

    request.resource_read_binding->require_full_coverage = false;
    request.resource_read_binding->intent = common_agent_resource_read_intent::question;
    request.resource_read_binding->task_fingerprint = "task-question-1";
    plan.steps.back().tool_call->arguments_json = R"({"uri":"resource://text","offset":100,"max_bytes":10})";
    plan.observations.back().resource_read_provenance->task_fingerprint = "task-question-1";
    plan.observations.back().resource_read_provenance->intent = "question";
    assert(common_agent_plan_satisfies_resource_read_binding(plan, request));

    plan.steps.clear();
    plan.observations.clear();
    plan.steps.push_back(read_step("dataset.inspect", {{"resource", "resource://text"}}));
    auto dataset_inspection = read_observation("tool:read-step:1", input.resource);
    bind_read_provenance(dataset_inspection, *request.resource_read_binding);
    common_agent_dataset_ref dataset_ref;
    dataset_ref.uri = "dataset://unrelated";
    dataset_ref.name = "unrelated";
    dataset_ref.row_count = 1;
    dataset_ref.column_count = 1;
    dataset_ref.source_resource_uri = "resource://other";
    dataset_inspection.dataset_refs.push_back(dataset_ref);
    plan.observations.push_back(dataset_inspection);
    request.resource_read_binding->require_full_coverage = false;
    assert(!common_agent_plan_satisfies_resource_read_binding(plan, request));
    plan.observations.back().dataset_refs.front().source_resource_uri = "resource://text";
    assert(common_agent_plan_satisfies_resource_read_binding(plan, request));

    request.resource_read_binding->require_full_coverage = true;
    request.resource_read_binding->intent = common_agent_resource_read_intent::orientation;
    request.resource_read_binding->task_fingerprint = "task-orientation-1";
    plan.steps.clear();
    plan.observations.clear();
    for (size_t i = 0; i < 2; ++i) {
        common_runtime_resource_ref chunk;
        chunk.uri = "resource://chunk/" + std::to_string(i);
        chunk.lineage.parent_uri = "resource://text";
        chunk.lineage.chunk_index = i;
        chunk.lineage.chunk_count = 2;
        chunk.lineage.byte_offset = i * 50;
        chunk.lineage.byte_length = 50;
        chunk.lineage.derivation = "host.chunk.text.v1";
        auto observation = read_observation("resource_chunk:resource://text:" + std::to_string(i), chunk);
        observation.source = "resource_chunk";
        bind_read_provenance(observation, *request.resource_read_binding);
        plan.observations.push_back(std::move(observation));
    }
    assert(common_agent_plan_satisfies_resource_read_binding(plan, request));
    auto invalid_chunk_set = plan;
    invalid_chunk_set.observations.front().resource_refs.front().lineage.chunk_index = 2;
    assert(!common_agent_plan_satisfies_resource_read_binding(invalid_chunk_set, request));
    invalid_chunk_set = plan;
    invalid_chunk_set.observations.back().resource_refs.front().lineage.chunk_count = 3;
    assert(!common_agent_plan_satisfies_resource_read_binding(invalid_chunk_set, request));
    invalid_chunk_set = plan;
    invalid_chunk_set.observations.back().resource_refs.front().lineage.chunk_index = 0;
    assert(!common_agent_plan_satisfies_resource_read_binding(invalid_chunk_set, request));
    invalid_chunk_set = plan;
    invalid_chunk_set.observations.back().resource_refs.front().lineage.parent_uri = "resource://unrelated";
    assert(!common_agent_plan_satisfies_resource_read_binding(invalid_chunk_set, request));
    plan.observations.pop_back();
    assert(!common_agent_plan_satisfies_resource_read_binding(plan, request));
    plan.observations.clear();
    auto unresolved = read_observation("resource_chunk:resource://text:0", input.resource);
    unresolved.source = "resource_chunk_unresolved";
    plan.observations.push_back(std::move(unresolved));
    assert(!common_agent_plan_satisfies_resource_read_binding(plan, request));

    auto gap = plan;
    gap.observations.clear();
    for (size_t i = 0; i < 2; ++i) {
        common_runtime_resource_ref chunk;
        chunk.uri = "resource://gap/" + std::to_string(i);
        chunk.lineage.parent_uri = "resource://text";
        chunk.lineage.chunk_index = i;
        chunk.lineage.chunk_count = 2;
        chunk.lineage.byte_offset = i == 0 ? 0 : 60;
        chunk.lineage.byte_length = 40;
        chunk.lineage.derivation = "host.chunk.text.v1";
        auto observation = read_observation("gap:" + std::to_string(i), chunk);
        observation.source = "resource_chunk";
        bind_read_provenance(observation, *request.resource_read_binding);
        gap.observations.push_back(std::move(observation));
    }
    assert(!common_agent_plan_satisfies_resource_read_binding(gap, request));

    request.resource_read_binding->intent = common_agent_resource_read_intent::question;
    request.resource_read_binding->require_full_coverage = false;
    request.resource_read_binding->reusable_orientation_targets = {request.resource_read_binding->targets.front()};
    request.resource_read_binding->reusable_orientation_targets.front().content_fingerprint = "sha256:source-content";
    gap.observations.clear();
    assert(common_agent_plan_satisfies_resource_read_binding(gap, request));
    request.resource_read_binding->reusable_orientation_targets.front().content_fingerprint = "sha256:changed";
    assert(!common_agent_plan_satisfies_resource_read_binding(gap, request));
    request.resource_read_binding->reusable_orientation_targets.clear();

    // A processor may publish a new derived URI for each turn. A prior full
    // orientation remains reusable only through the same original parent and
    // content fingerprint, not merely because both resources are text.
    auto reprocessed = request;
    reprocessed.resource_read_binding->targets = {
        {"resource://text-v2", "sha256:source-content", "text/markdown", "resource://source", 100, true}};
    reprocessed.resource_read_binding->task_fingerprint = "task-question-new";
    common_plan_state prior_orientation;
    for (size_t i = 0; i < 2; ++i) {
        common_runtime_resource_ref chunk;
        chunk.uri = "resource://old-chunk/" + std::to_string(i);
        chunk.lineage.parent_uri = "resource://text-v1";
        chunk.lineage.chunk_index = i;
        chunk.lineage.chunk_count = 2;
        chunk.lineage.byte_offset = i * 50;
        chunk.lineage.byte_length = 50;
        chunk.lineage.derivation = "host.chunk.text.v1";
        auto observation = read_observation("old-orientation:" + std::to_string(i), chunk);
        observation.source = "resource_chunk";
        observation.summary = "finding: useful fact\nevidence: exact source sentence";
        common_agent_resource_read_binding old_binding = *request.resource_read_binding;
        old_binding.intent = common_agent_resource_read_intent::orientation;
        old_binding.task_fingerprint = "task-orientation-old";
        bind_read_provenance(observation, old_binding, "resource://text-v1");
        observation.resource_read_provenance->source_parent_uri = "resource://source";
        prior_orientation.observations.push_back(std::move(observation));
    }
    assert(common_agent_plan_satisfies_resource_read_binding(prior_orientation, reprocessed));
    prior_orientation.observations.front().resource_read_provenance->source_fingerprint = "sha256:changed";
    assert(!common_agent_plan_satisfies_resource_read_binding(prior_orientation, reprocessed));

    request.resource_read_binding->targets = {{"resource://text", "sha256:source-content", "text/markdown",
        "resource://source", 100, true}};
    request.resource_read_binding->targets.push_back(
        {"resource://second", "sha256:second", "text/plain", {}, 10, true});
    assert(!common_agent_plan_satisfies_resource_read_binding(gap, request));
    return 0;
}
