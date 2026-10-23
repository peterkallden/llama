#include "research-workspace-adapters.h"

#include "agent/thinking/research/research-identity.h"
#include "agent/thinking/research/research-workspace-serialization.h"
#include "agent/tooling/adapters/support/adapter-support.h"
#include "resource/resource-retention.h"
#include "hash/hash.h"

#include <algorithm>
#include <ctime>

namespace {
using json = common_adapter_json;

bool promote_workspace_sources(
        const common_native_tool_bindings & bindings,
        const common_agent_scope & scope,
        common_agent_research_workspace & workspace,
        std::string & error) {
    if (scope.project_id.empty()) return true;
    const auto authority = make_agent_resource_read_authority(bindings.resource_runtime,
        static_cast<int64_t>(std::time(nullptr)));
    size_t copied_bytes = 0;
    constexpr size_t max_total_bytes = 16 * 1024 * 1024;
    for (auto & source : workspace.sources) {
        if (!source.resource_ref) continue;
        agent_resource_retention_target target;
        target.scope = common_runtime_resource_scope::project;
        target.namespace_id = scope.namespace_id;
        target.session_id = scope.session_id;
        target.project_id = scope.project_id;
        target.turn_id = scope.turn_id;
        target.source_tool = "research.workspace.publish";
        target.derivation = "research-source-project-promotion:v1";
        target.max_bytes = std::min<size_t>(4 * 1024 * 1024, max_total_bytes - copied_bytes);
        agent_resource_descriptor retained;
        bool copied = false;
        if (!retain_agent_resource(*bindings.resource_runtime.store, *source.resource_ref, authority,
                target, retained, copied, error)) return false;
        static_cast<common_runtime_resource_ref &>(*source.resource_ref) = retained;
        if (!retained.sha256.empty()) source.content_hash = "sha256:" + retained.sha256;
        if (copied) copied_bytes += retained.size_bytes;
    }
    error.clear();
    return true;
}

std::string research_content_fingerprint(const common_agent_research_workspace & workspace) {
    auto value = common_agent_research_workspace_to_json(workspace);
    value.erase("workspace_id");
    value.erase("draft_id");
    value.erase("revision");
    value.erase("base_revision");
    value.erase("request_id");
    value.erase("turn_id");
    value.erase("session_id");
    value.erase("plan_id");
    value.erase("plan_step_id");
    value["scope"].erase("session_id");
    value["scope"].erase("turn_id");
    for (auto & source : value["sources"]) {
        if (source.contains("resource_ref") && source["resource_ref"].is_object()) {
            // Resource IDs are allocated per copy. The source hash and retained
            // metadata identify content; a newly allocated URI does not.
            source["resource_ref"].erase("uri");
        }
    }
    const std::string bytes = value.dump();
    return hash_sha256_hex(bytes.data(), bytes.size());
}

common_tool_execution_result publish_research_workspace(const common_native_tool_bindings & bindings) {
    if (bindings.data_store == nullptr || bindings.resource_runtime.store == nullptr)
        return common_adapter_execution_failure("tool.research.workspace.publish.unavailable",
            "research persistence is unavailable", "The research workspace could not be published.");
    const auto & scope = bindings.tool_scope;
    if (scope.namespace_id.empty() || scope.session_id.empty())
        return common_adapter_validation_failure("tool.research.workspace.publish.scope_invalid",
            "publishing research requires a namespace and authenticated chat session");
    const std::string research_id = common_agent_research_identity(scope);
    const std::string draft_id = common_agent_research_draft_identity(research_id, scope.session_id);
    common_agent_research_draft_ref draft;
    std::string error;
    if (!bindings.data_store->get_research_draft(draft_id, scope, draft, error))
        return common_adapter_not_found_failure("tool.research.workspace.publish.draft_missing", std::move(error),
            "There is no saved research draft to publish from this chat.");
    if (draft.research_id != research_id || draft.owner_session_id != scope.session_id)
        return common_adapter_validation_failure("tool.research.workspace.publish.draft_scope",
            "the saved draft is not owned by the active chat and research scope");
    const auto authority = make_agent_resource_read_authority(bindings.resource_runtime,
        static_cast<int64_t>(std::time(nullptr)));
    std::string workspace_json;
    if (!bindings.resource_runtime.store->read_text(draft.payload_uri, authority, 4 * 1024 * 1024,
            workspace_json, error))
        return common_adapter_not_found_failure("tool.research.workspace.publish.draft_unavailable", std::move(error),
            "The saved research draft is no longer accessible in this scope.");
    if (hash_sha256_hex(workspace_json.data(), workspace_json.size()) != draft.payload_fingerprint)
        return common_adapter_execution_failure("tool.research.workspace.publish.fingerprint_mismatch",
            "saved research draft fingerprint does not match its revision index",
            "The saved research draft failed its integrity check.");
    const auto parsed_workspace = json::parse(workspace_json, nullptr, false);
    common_agent_research_workspace workspace;
    if (parsed_workspace.is_discarded() ||
            !common_agent_research_workspace_from_json(parsed_workspace, workspace, error))
        return common_adapter_execution_failure("tool.research.workspace.publish.invalid_draft", std::move(error),
            "The saved research draft is invalid.");
    if (workspace.research_id != draft.research_id || workspace.draft_id != draft.draft_id ||
            workspace.base_revision != draft.base_revision || workspace.revision != draft.draft_revision ||
            workspace.scope.namespace_id != scope.namespace_id || workspace.scope.project_id != scope.project_id ||
            workspace.session_id != scope.session_id)
        return common_adapter_validation_failure("tool.research.workspace.publish.draft_identity",
            "saved research draft payload does not match its host-owned identity");

    if (!promote_workspace_sources(bindings, scope, workspace, error))
        return common_adapter_execution_failure("tool.research.workspace.publish.source_unavailable", std::move(error),
            "The research revision cannot be published because one or more source resources cannot be safely retained in the project.");

    const std::string publication_material = json::array({scope.namespace_id, scope.project_id,
        scope.project_id.empty() ? scope.session_id : std::string{}, research_id, draft_id,
        draft.base_revision, research_content_fingerprint(workspace)}).dump();
    const std::string publication_id = "publication:" +
        hash_sha256_hex(publication_material.data(), publication_material.size()).substr(0, 40);
    common_agent_research_revision_ref current;
    bool has_current = bindings.data_store->get_research_revision(research_id, 0, scope, current, error);
    if (!has_current && error != "research workspace was not found in this scope")
        return common_adapter_execution_failure("tool.research.workspace.publish.revision_read_failed", std::move(error),
            "The current published research revision could not be read.");
    error.clear();
    if (has_current) {
        const auto current_authority = make_agent_resource_read_authority(bindings.resource_runtime,
            static_cast<int64_t>(std::time(nullptr)));
        std::string current_json;
        common_agent_research_workspace current_workspace;
        const auto current_value = [&]() {
            if (!bindings.resource_runtime.store->read_text(current.payload_uri, current_authority,
                    4 * 1024 * 1024, current_json, error)) return json();
            if (hash_sha256_hex(current_json.data(), current_json.size()) != current.payload_fingerprint) {
                error = "published research revision fingerprint does not match its revision index";
                return json();
            }
            auto parsed = json::parse(current_json, nullptr, false);
            if (parsed.is_discarded() || !common_agent_research_workspace_from_json(parsed, current_workspace, error)) return json();
            return parsed;
        }();
        if (current_value.is_discarded() || !error.empty())
            return common_adapter_execution_failure("tool.research.workspace.publish.current_revision_invalid", std::move(error),
                "The current published research revision could not be verified.");
        if (current_workspace.research_id != research_id || current_workspace.revision != current.revision ||
                current_workspace.scope.namespace_id != scope.namespace_id || current_workspace.scope.project_id != scope.project_id)
            return common_adapter_validation_failure("tool.research.workspace.publish.current_revision_scope",
                "the current published research payload does not match its host revision and scope");
        if (research_content_fingerprint(current_workspace) == research_content_fingerprint(workspace)) {
            json result{{"research_id", research_id}, {"revision", current.revision},
                {"resource_uri", current.payload_uri}, {"publication_id", current.publication_id},
                {"already_published", true}, {"draft_reset", draft.base_revision == current.revision}};
            if (draft.base_revision != current.revision)
                result["draft_reset_warning"] = "The current project revision already contains this research state, but this chat draft remains based on an older revision.";
            return common_adapter_success_json(result);
        }
    }
    const uint64_t actual_revision = has_current ? current.revision : 0;
    if (actual_revision != draft.base_revision)
        return common_adapter_failure("tool.research.workspace.publish.conflict", common_tool_failure_class::conflict,
            false, "This research changed after the draft was created. Keep the draft and open a fresh chat from the latest revision.",
            "research revision conflict: expected " + std::to_string(draft.base_revision) +
            ", found " + std::to_string(actual_revision));

    auto published_workspace = workspace;
    published_workspace.workspace_id = research_id + ":revision:" + std::to_string(draft.base_revision + 1);
    published_workspace.draft_id.clear();
    published_workspace.base_revision = 0;
    published_workspace.revision = draft.base_revision + 1;
    published_workspace.request_id.clear();
    published_workspace.turn_id.clear();
    published_workspace.session_id.clear();
    published_workspace.plan_id.clear();
    published_workspace.plan_step_id.clear();
    if (!scope.project_id.empty()) published_workspace.scope.session_id = "project:" + scope.project_id;
    published_workspace.scope.turn_id.clear();
    std::string published_json;
    if (!common_agent_research_workspace_validate(published_workspace, error))
        return common_adapter_execution_failure("tool.research.workspace.publish.invalid_revision", std::move(error),
            "The research revision could not be validated for publication.");
    published_json = common_agent_research_workspace_to_json(published_workspace).dump();
    const std::string fingerprint = hash_sha256_hex(published_json.data(), published_json.size());
    agent_resource_put_request put;
    put.name = "research-" + research_id + "-r" + std::to_string(published_workspace.revision) + ".json";
    put.description = "Immutable published research revision.";
    put.mime_type = "application/vnd.llama-agent.research-workspace+json";
    put.text = published_json;
    put.scope = scope.project_id.empty() ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
    put.namespace_id = scope.namespace_id;
    put.project_id = scope.project_id;
    put.session_id = scope.session_id;
    put.turn_id = scope.turn_id;
    put.source_provider = "native";
    put.source_tool = "research.workspace.publish";
    put.created_at = static_cast<int64_t>(std::time(nullptr));
    put.metadata.purpose = "published_research_revision";
    put.metadata.content_summary = "Immutable published research workspace revision.";
    put.metadata.content_hash = fingerprint;
    put.metadata.processing_cache_key = publication_id;
    put.lineage.parent_uri = draft.payload_uri;
    put.lineage.derivation = "research-workspace-publish:v1";
    apply_agent_resource_runtime(bindings.resource_runtime, put);
    agent_resource_descriptor resource;
    if (!bindings.resource_runtime.store->put_text(put, resource, error) || resource.uri.empty())
        return common_adapter_execution_failure("tool.research.workspace.publish.resource_write_failed", std::move(error),
            "The immutable research revision could not be saved.");
    common_agent_research_revision_ref next;
    next.research_id = research_id;
    next.revision = published_workspace.revision;
    next.parent_revision = draft.base_revision;
    next.payload_uri = resource.uri;
    next.payload_fingerprint = fingerprint;
    next.publication_id = publication_id;
    next.published_at_ms = static_cast<int64_t>(std::time(nullptr)) * 1000;
    common_agent_research_revision_ref published;
    bool already_published = false;
    if (!bindings.data_store->publish_research_revision(scope, next, draft.base_revision,
            published, already_published, error)) {
        if (error == "research workspace revision conflict")
            return common_adapter_failure("tool.research.workspace.publish.conflict", common_tool_failure_class::conflict,
                false, "The research revision changed during publication. The saved chat draft remains available.",
                std::move(error));
        return common_adapter_execution_failure("tool.research.workspace.publish.publication_failed", std::move(error),
            "The research publication could not be confirmed. The saved chat draft remains available; retry the same publication to recover its outcome.");
    }

    // Advance only this chat's draft after its revision was published. This
    // keeps the chat writable while preserving all other chats' independent drafts.
    auto continued_workspace = workspace;
    continued_workspace.base_revision = published.revision;
    continued_workspace.revision = draft.draft_revision + 1;
    std::string continued_json;
    common_agent_research_draft_ref next_draft = draft;
    bool draft_reset = false;
    std::string reset_error;
    if (common_agent_research_workspace_validate(continued_workspace, reset_error)) {
        continued_json = common_agent_research_workspace_to_json(continued_workspace).dump();
        next_draft.draft_revision = continued_workspace.revision;
        next_draft.payload_fingerprint = hash_sha256_hex(continued_json.data(), continued_json.size());
        next_draft.updated_at_ms = static_cast<int64_t>(std::time(nullptr)) * 1000;
        agent_resource_put_request draft_put = put;
        draft_put.name = "research-draft-" + draft_id + "-" + std::to_string(next_draft.draft_revision) + ".json";
        draft_put.description = "Immutable chat-owned research draft after publication.";
        draft_put.text = continued_json;
        draft_put.scope = common_runtime_resource_scope::session;
        draft_put.turn_id = scope.turn_id;
        draft_put.source_tool = "research.workspace.publish";
        draft_put.metadata.purpose = "research_workspace_draft";
        draft_put.metadata.content_hash = next_draft.payload_fingerprint;
        draft_put.metadata.processing_cache_key = research_id + ":" + draft_id + ":" + next_draft.payload_fingerprint;
        draft_put.lineage.parent_uri = published.payload_uri;
        draft_put.lineage.derivation = "research-workspace-draft-after-publish:v1";
        agent_resource_descriptor draft_resource;
        if (bindings.resource_runtime.store->put_text(draft_put, draft_resource, reset_error) && !draft_resource.uri.empty()) {
            next_draft.payload_uri = draft_resource.uri;
            common_agent_research_draft_ref saved_draft;
            bool already_saved = false;
            draft_reset = bindings.data_store->save_research_draft(scope, next_draft,
                draft.draft_revision, saved_draft, already_saved, reset_error);
        }
    }
    json result{{"research_id", research_id}, {"revision", published.revision},
        {"resource_uri", published.payload_uri}, {"publication_id", publication_id},
        {"already_published", already_published}, {"draft_reset", draft_reset}};
    if (!draft_reset) result["draft_reset_warning"] = reset_error.empty()
        ? "The revision was published, but this chat draft could not be advanced automatically."
        : reset_error;
    return common_adapter_success_json(result);
}
}

bool common_try_register_research_workspace_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    if (definition.executor_id != "builtin.research.workspace.publish") return false;
    installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
        json arguments;
        std::string parse_error;
        if (!common_adapter_parse_object(input, arguments, parse_error) || !arguments.empty())
            return common_adapter_validation_failure("tool.research.workspace.publish.invalid_arguments",
                parse_error.empty() ? "research.workspace.publish takes no arguments" : std::move(parse_error));
        return publish_research_workspace(bindings);
    }, error, false, true);
    return true;
}
