#include "agent-research-workspace-host.h"
#include "agent/thinking/research/research-identity.h"
#include "agent/thinking/research/research-workspace-serialization.h"
#include "agent/thinking/research/research-workspace.h"
#include "resource/resource-retention.h"
#include "hash/hash.h"
#include <algorithm>
#include <ctime>
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

namespace {
common_agent_scope research_scope(const common_agent_runtime_session_host_turn_request & request) {
    common_agent_scope scope;
    scope.namespace_id = request.namespace_id;
    scope.session_id = request.session_id;
    scope.project_id = request.project_id;
    scope.turn_id = request.turn_id;
    return scope;
}

bool read_research_workspace(agent_resource_store & resources, const std::string & uri,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_research_workspace & workspace, std::string & error,
        const std::string & expected_fingerprint = {}) {
    agent_resource_read_authority authority;
    authority.namespace_id = request.namespace_id;
    authority.session_id = request.session_id;
    authority.project_id = request.project_id;
    authority.turn_id = request.turn_id;
    authority.now = static_cast<int64_t>(std::time(nullptr));
    std::string text;
    if (!resources.read_text(uri, authority, 4 * 1024 * 1024, text, error)) return false;
    if (!expected_fingerprint.empty() && hash_sha256_hex(text.data(), text.size()) != expected_fingerprint) {
        error = "saved research payload fingerprint does not match its revision index";
        return false;
    }
    const auto parsed = json::parse(text, nullptr, false);
    if (parsed.is_discarded() || !common_agent_research_workspace_from_json(parsed, workspace, error)) return false;
    if (workspace.scope.namespace_id != request.namespace_id || workspace.scope.project_id != request.project_id ||
            (request.project_id.empty() && workspace.scope.session_id != request.session_id)) {
        error = "saved research payload scope does not match the active chat";
        return false;
    }
    error.clear();
    return true;
}

} // namespace

bool agent_prepare_research_workspace(agent_research_host_resources & runtime,
        common_agent_runtime_session_host_turn_request & request, std::string & error) {
    if (runtime.data_store == nullptr || runtime.resource_store == nullptr) return true;
    if (request.session_id.empty() || request.namespace_id.empty()) {
        error = "persistent research requires a session and namespace scope";
        return false;
    }
    const auto scope = research_scope(request);
    const std::string research_id = common_agent_research_identity(scope);
    const std::string draft_id = common_agent_research_draft_identity(research_id, request.session_id);
    common_agent_research_draft_ref draft;
    const bool has_draft = runtime.data_store->get_research_draft(draft_id, scope, draft, error);
    // Older/in-memory data-store implementations may not support persistent
    // research yet. Keep ordinary turns usable there; configured Cozo errors
    // remain explicit and are not treated as an empty draft.
    if (!has_draft && error.empty()) return true;
    if (!has_draft && error != "research draft was not found in this scope") return false;
    error.clear();
    common_agent_research_workspace seed;
    bool has_seed = false;
    if (has_draft) {
        if (draft.research_id != research_id || draft.owner_session_id != request.session_id ||
                !read_research_workspace(*runtime.resource_store, draft.payload_uri, request, seed, error,
                    draft.payload_fingerprint)) return false;
        if (seed.research_id != research_id || seed.draft_id != draft_id || seed.base_revision != draft.base_revision ||
                seed.revision != draft.draft_revision) {
            error = "saved research draft metadata does not match its payload";
            return false;
        }
        has_seed = true;
    } else {
        common_agent_research_revision_ref current;
        if (runtime.data_store->get_research_revision(research_id, 0, scope, current, error)) {
            if (!read_research_workspace(*runtime.resource_store, current.payload_uri, request, seed, error,
                    current.payload_fingerprint)) return false;
            if (seed.research_id != research_id || seed.revision != current.revision) {
                error = "published research metadata does not match its payload";
                return false;
            }
            has_seed = true;
            draft.research_id = research_id;
            draft.draft_id = draft_id;
            draft.owner_session_id = request.session_id;
            draft.base_revision = current.revision;
            draft.draft_revision = 0;
        } else if (error == "research workspace was not found in this scope" || error.empty()) {
            error.clear();
            draft.research_id = research_id;
            draft.draft_id = draft_id;
            draft.owner_session_id = request.session_id;
            draft.base_revision = 0;
            draft.draft_revision = 0;
        } else return false;
    }
    request.research_draft = draft;
    if (has_seed) request.research_workspace_seed = std::move(seed);
    error.clear();
    return true;
}

namespace {
bool retain_research_sources_for_chat(agent_research_host_resources & runtime,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_research_workspace & workspace, std::string & error,
        std::map<std::string, agent_resource_descriptor> * retained_cache = nullptr) {
    agent_resource_read_authority authority;
    authority.namespace_id = request.namespace_id;
    authority.session_id = request.session_id;
    authority.project_id = request.project_id;
    authority.turn_id = request.turn_id;
    authority.now = static_cast<int64_t>(std::time(nullptr));
    size_t copied_bytes = 0;
    constexpr size_t max_total_bytes = 32 * 1024 * 1024;
    for (auto & source : workspace.sources) {
        if (!source.resource_ref) continue;
        agent_resource_retention_target target;
        target.scope = common_runtime_resource_scope::session;
        target.namespace_id = request.namespace_id;
        target.session_id = request.session_id;
        target.project_id = request.project_id;
        target.turn_id = request.turn_id;
        target.source_tool = "research.workspace.persist";
        target.derivation = "research-source-chat-retention:v1";
        target.max_bytes = std::min<size_t>(10 * 1024 * 1024, max_total_bytes - copied_bytes);
        agent_resource_descriptor retained;
        bool copied = false;
        const auto original_uri = source.resource_ref->uri;
        const auto cached = retained_cache ? retained_cache->find(original_uri) : std::map<std::string, agent_resource_descriptor>::iterator{};
        if (retained_cache && cached != retained_cache->end()) {
            // Re-authorize the retained object on every checkpoint; a cache
            // entry is never itself authority to read a resource.
            if (!runtime.resource_store->stat(cached->second.uri, authority, retained, error) ||
                    retained.sha256 != cached->second.sha256) {
                if (error.empty()) error = "retained research source changed or became unavailable";
                return false;
            }
            if (retained.size_bytes > target.max_bytes) {
                error = "research sources exceed the configured retention byte limit";
                return false;
            }
            copied = true;
        } else {
            if (!retain_agent_resource(*runtime.resource_store, *source.resource_ref, authority,
                    target, retained, copied, error)) return false;
            if (retained_cache && copied) (*retained_cache)[original_uri] = retained;
        }
        common_agent_research_rebind_resource(workspace, source, retained);
        if (!retained.sha256.empty()) source.content_hash = "sha256:" + retained.sha256;
        if (copied) copied_bytes += retained.size_bytes;
    }
    error.clear();
    return true;
}

} // namespace

bool agent_persist_research_workspace(agent_research_host_resources & runtime,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_session_host_turn_result & result, std::string & error,
        common_agent_research_draft_ref * advanced,
        std::map<std::string, agent_resource_descriptor> * retained_cache) {
    if (!result.research_workspace_checkpoint) return true;
    if (!request.research_draft) {
        const auto & workspace = result.research_workspace_checkpoint->workspace;
        // prepare_research_workspace deliberately leaves unsupported stores
        // unbound. Such a turn may still return an ephemeral checkpoint.
        if (workspace.draft_id.empty() && workspace.base_revision == 0 &&
                workspace.research_id == workspace.workspace_id && workspace.session_id == request.session_id)
            return true;
        error = "research checkpoint was produced without a persistent host draft binding";
        return false;
    }
    if (runtime.data_store == nullptr || runtime.resource_store == nullptr) {
        error = "research completed without its persistent host binding";
        return false;
    }
    auto workspace = result.research_workspace_checkpoint->workspace;
    const auto & current = *request.research_draft;
    std::string mismatched_fields;
    const auto mismatch = [&](bool differs, const char * field) {
        if (!differs) return;
        if (!mismatched_fields.empty()) mismatched_fields += ", ";
        mismatched_fields += field;
    };
    mismatch(workspace.research_id != current.research_id, "research_id");
    mismatch(workspace.draft_id != current.draft_id, "draft_id");
    mismatch(workspace.base_revision != current.base_revision, "base_revision");
    mismatch(workspace.session_id != request.session_id, "session_id");
    if (!mismatched_fields.empty()) {
        error = "research checkpoint does not belong to the active persistent draft; mismatched fields: " + mismatched_fields;
        return false;
    }
    if (!retain_research_sources_for_chat(runtime, request, workspace, error, retained_cache)) return false;
    workspace.revision = current.draft_revision + 1;
    const std::string payload = common_agent_research_workspace_to_json(workspace).dump();
    const std::string fingerprint = hash_sha256_hex(payload.data(), payload.size());
    agent_resource_put_request put;
    put.name = "research-draft-" + current.draft_id + "-" + std::to_string(workspace.revision) + ".json";
    put.description = "Immutable chat-owned research workspace draft revision.";
    put.mime_type = "application/vnd.llama-agent.research-workspace+json";
    put.text = payload;
    put.scope = common_runtime_resource_scope::session;
    put.namespace_id = request.namespace_id;
    put.project_id = request.project_id;
    put.session_id = request.session_id;
    put.turn_id = request.turn_id;
    put.source_provider = "native";
    put.source_tool = "research.workspace.persist";
    put.created_at = static_cast<int64_t>(std::time(nullptr));
    put.metadata.purpose = "research_workspace_draft";
    put.metadata.content_summary = "Persistent active research draft.";
    put.metadata.content_hash = fingerprint;
    put.metadata.processing_cache_key = current.research_id + ":" + current.draft_id + ":" + fingerprint;
    put.lineage.parent_uri = current.payload_uri;
    put.lineage.derivation = "research-workspace-draft:v1";
    agent_resource_descriptor stored;
    if (!runtime.resource_store->put_text(put, stored, error) || stored.uri.empty()) {
        if (error.empty()) error = "research draft resource store returned an empty reference";
        return false;
    }
    common_agent_research_draft_ref next = current;
    next.draft_revision = workspace.revision;
    next.payload_uri = stored.uri;
    next.payload_fingerprint = fingerprint;
    next.updated_at_ms = static_cast<int64_t>(std::time(nullptr)) * 1000;
    common_agent_research_draft_ref saved;
    bool already_saved = false;
    if (!runtime.data_store->save_research_draft(research_scope(request), next,
            current.draft_revision, saved, already_saved, error)) return false;
    if (advanced) *advanced = saved;
    result.research_workspace_checkpoint->workspace = std::move(workspace);
    error.clear();
    return true;
}
