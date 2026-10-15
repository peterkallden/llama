#ifdef NDEBUG
#undef NDEBUG
#endif

#include "agent/context/conversation-context.h"
#include "agent/context/observation-projection.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/resource/dispatch/agent-resource-processor-dispatch.h"
#include "tools/agent/runtime/agent-session-context-store.h"

#include <cassert>
#include <memory>

int main() {
    using json = nlohmann::ordered_json;
    std::string error;
    auto blobs = std::make_shared<agent_in_memory_blob_store>();
    agent_catalogued_resource_store resources(blobs, std::make_unique<agent_in_memory_resource_catalog>());

    agent_session_context_store session(&resources, "ns-a", "session-a", true);
    common_agent_conversation_segment user;
    user.request_id = "request-1";
    user.turn_id = "turn-old";
    user.role = common_agent_conversation_role::user;
    user.content = "Always preserve this exact instruction.";
    assert(session.append(user, error));
    user.turn_id = "turn-retry";
    assert(session.append(user, error)); // request identity, not regenerated turn id, is idempotent

    common_agent_conversation_segment assistant;
    assistant.request_id = "request-1";
    assistant.turn_id = "turn-old";
    assistant.role = common_agent_conversation_role::assistant;
    assistant.content = "Recorded answer.";
    assistant.evidence_refs = {"resource://source/1", "observation:obs-1"};
    assert(session.append(assistant, error));

    auto conflict = user;
    conflict.content = "Conflicting retry.";
    assert(!session.append(conflict, error));
    assert(error.find("conflicting") != std::string::npos);

    std::vector<common_agent_conversation_segment> segments;
    std::optional<common_agent_context_checkpoint> checkpoint;
    uint64_t epoch = 0;
    assert(session.load(segments, checkpoint, epoch, error));
    assert(segments.size() == 2 && epoch == 1);

    common_agent_context_checkpoint saved;
    saved.summary = "The request has been answered.";
    saved.current_goal = "Continue the same task.";
    saved.decisions = {"Use the verified source."};
    saved.covered_turn_refs = {segments.front().resource_uri};
    saved.protected_instruction_refs = {segments.front().resource_uri};
    saved.plan_refs = {"plan:research-1"};
    saved.evidence_refs = {"observation:obs-1", "resource://source/1"};
    saved.memory_refs = {"memory:fact-1"};
    assert(session.save_checkpoint(saved, error));
    assert(saved.revision == 1 && saved.next_sequence == 3);

    common_agent_conversation_segment later_user;
    later_user.request_id = "request-2";
    later_user.turn_id = "turn-2";
    later_user.role = common_agent_conversation_role::user;
    later_user.content = "Continue with the source details.";
    assert(session.append(later_user, error));

    // Reconstruct the store facade over the same resource backend to model a
    // daemon/session-host restart while preserving immutable resources.
    agent_session_context_store restarted(&resources, "ns-a", "session-a", true);
    assert(restarted.load(segments, checkpoint, epoch, error));
    assert(segments.size() == 3 && checkpoint && checkpoint->revision == 1);
    auto composition = compose_common_agent_conversation_context(segments, &*checkpoint, 1, 4096);
    assert(composition.used_checkpoint);
    assert(composition.missing_protected_refs.empty());
    assert(common_agent_projected_conversation_context_chars(composition, 40) > composition.rendered_history.size() + 40);
    assert(composition.rendered_history.find("Always preserve this exact instruction.") != std::string::npos);
    assert(composition.rendered_history.find("resource://source/1") != std::string::npos);
    assert(composition.rendered_history.find("Use the verified source.") != std::string::npos);
    assert(composition.rendered_history.find("Plan ref: plan:research-1") != std::string::npos);
    assert(composition.rendered_history.find("Evidence ref: resource://source/1") != std::string::npos);
    assert(composition.rendered_history.find("Memory ref: memory:fact-1") != std::string::npos);
    assert(std::find(composition.included_sections.begin(), composition.included_sections.end(), "checkpoint") != composition.included_sections.end());
    assert(std::find(composition.included_sections.begin(), composition.included_sections.end(), "evidence_references") != composition.included_sections.end());

    auto too_small = compose_common_agent_conversation_context(segments, &*checkpoint, 1, 8);
    assert(!too_small.missing_protected_refs.empty());

    agent_session_context_store other_scope(&resources, "ns-b", "session-a", true);
    assert(other_scope.load(segments, checkpoint, epoch, error));
    assert(segments.empty() && !checkpoint);

    assert(restarted.begin_new_epoch(error));
    assert(restarted.load(segments, checkpoint, epoch, error));
    assert(epoch == 2 && segments.empty() && !checkpoint);

    agent_session_context_store transient(nullptr, "ns", "transient-session", false);
    common_agent_conversation_segment transient_user;
    transient_user.request_id = "t1";
    transient_user.turn_id = "t1";
    transient_user.role = common_agent_conversation_role::user;
    transient_user.content = "transient";
    assert(transient.append(transient_user, error));
    assert(transient.load(segments, checkpoint, epoch, error) && segments.size() == 1);
    assert(transient.begin_new_epoch(error));
    assert(transient.load(segments, checkpoint, epoch, error) && segments.empty() && epoch == 2);

    const json observation = {{"payload", "large"}, {"status", "ok"},
        {"resource_ref", "resource://important"}, {"details", json::array({1, 2, 3})}};
    const auto projected = common_agent_project_observation_json(observation, 0, 4, 4, 2, 64);
    assert(projected.is_object() && projected.at("status") == "ok");
    assert(projected.at("resource_ref") == "resource://important");
    assert(projected.at("details").is_array());

    agent_host_config config;
    assert(parse_agent_host_config_json(json::object(), config, error));
    assert(config.sandbox.defaults.limits.max_output_bytes == 65536);
    assert(agent_resource_processor_default_sandbox_output_bytes(
        "resource.processor.pdf.page_image") == 4 * 1024 * 1024);
    assert(agent_resource_processor_default_sandbox_output_bytes(
        "resource.processor.ocr.tesseract") == 4 * 1024 * 1024);
    assert(agent_resource_processor_default_sandbox_output_bytes("developer-build") == 0);

    agent_host_config processor_config;
    const json processor_limits = {
        {"sandbox", {
            {"workspace", {{"root", "workspaces"}}},
            {"classes", {
                {"resource.processor.pdf.page_image", {{"filesystem", "artifact_write"}}},
                {"resource.processor.ocr.tesseract", {{"max_output_bytes", 2097152}}},
            }},
        }},
    };
    assert(parse_agent_host_config_json(processor_limits, processor_config, error));
    assert(processor_config.sandbox.classes.at("resource.processor.pdf.page_image")
        .limits.max_output_bytes == 4 * 1024 * 1024);
    assert(processor_config.sandbox.classes.at("resource.processor.ocr.tesseract")
        .limits.max_output_bytes == 2097152);
    const auto serialized = agent_host_config_to_json(config);
    const auto & management = serialized.at("runtime").at("context_management");
    assert(management.at("enabled") == true);
    assert(management.at("persist_session") == true);
    assert(management.at("semantic_compaction") == true);
    assert(management.at("recent_turns") == 2);
    assert(management.at("summary_max_tokens") == 512);
    assert(management.at("max_summary_calls_per_turn") == 1);

    agent_host_config disabled_config;
    assert(parse_agent_host_config_json(json{{"runtime", {{"context_management", {
        {"enabled", false}, {"persist_session", false}, {"semantic_compaction", false},
        {"recent_turns", 4}, {"summary_max_tokens", 256},
        {"max_summary_calls_per_turn", 0}, {"max_history_chars", 4096}}}}}},
        disabled_config, error));
    const auto disabled_json = agent_host_config_to_json(disabled_config);
    const auto & disabled = disabled_json.at("runtime").at("context_management");
    assert(disabled.at("enabled") == false);
    assert(disabled.at("persist_session") == false);
    assert(disabled.at("semantic_compaction") == false);
    assert(disabled.at("recent_turns") == 4);
    assert(disabled.at("summary_max_tokens") == 256);
    assert(disabled.at("max_summary_calls_per_turn") == 0);
    assert(disabled.at("max_history_chars") == 4096);
}
