#include "agent-session-context-store.h"

#include <algorithm>
#include <ctime>

using json = nlohmann::ordered_json;

namespace {
constexpr const char * k_turn_purpose = "agent.context.turn.v1";
constexpr const char * k_checkpoint_purpose = "agent.context.checkpoint.v1";
constexpr const char * k_epoch_purpose = "agent.context.epoch.v1";

std::string hash_text(const std::string & value) {
    uint64_t hash = 14695981039346656037ull;
    for (const unsigned char byte : value) { hash ^= byte; hash *= 1099511628211ull; }
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(16, '0');
    for (size_t i = 0; i < result.size(); ++i) result[result.size() - i - 1] = digits[(hash >> (i * 4)) & 0xf];
    return result;
}

json checkpoint_json(const common_agent_context_checkpoint & checkpoint) {
    return {{"schema_version", checkpoint.schema_version}, {"namespace_id", checkpoint.namespace_id},
        {"session_id", checkpoint.session_id}, {"revision", checkpoint.revision},
        {"previous_revision", checkpoint.previous_revision}, {"next_sequence", checkpoint.next_sequence},
        {"epoch", checkpoint.epoch},
        {"fingerprint", checkpoint.fingerprint}, {"summary", checkpoint.summary},
        {"current_goal", checkpoint.current_goal}, {"decisions", checkpoint.decisions},
        {"open_questions", checkpoint.open_questions}, {"tried_alternatives", checkpoint.tried_alternatives},
        {"covered_turn_refs", checkpoint.covered_turn_refs},
        {"protected_instruction_refs", checkpoint.protected_instruction_refs},
        {"plan_refs", checkpoint.plan_refs}, {"evidence_refs", checkpoint.evidence_refs},
        {"memory_refs", checkpoint.memory_refs}};
}

bool read_checkpoint(const json & value, common_agent_context_checkpoint & out) {
    if (!value.is_object() || value.value("schema_version", 0u) != 1u) return false;
    try {
        out.schema_version = 1;
        out.namespace_id = value.at("namespace_id").get<std::string>();
        out.session_id = value.at("session_id").get<std::string>();
        out.revision = value.at("revision").get<uint64_t>();
        out.previous_revision = value.value("previous_revision", uint64_t(0));
        out.next_sequence = value.value("next_sequence", uint64_t(1));
        out.epoch = value.value("epoch", uint64_t(1));
        out.fingerprint = value.value("fingerprint", std::string());
        out.summary = value.value("summary", std::string());
        out.current_goal = value.value("current_goal", std::string());
        out.decisions = value.value("decisions", std::vector<std::string>());
        out.open_questions = value.value("open_questions", std::vector<std::string>());
        out.tried_alternatives = value.value("tried_alternatives", std::vector<std::string>());
        out.covered_turn_refs = value.value("covered_turn_refs", std::vector<std::string>());
        out.protected_instruction_refs = value.value("protected_instruction_refs", std::vector<std::string>());
        out.plan_refs = value.value("plan_refs", std::vector<std::string>());
        out.evidence_refs = value.value("evidence_refs", std::vector<std::string>());
        out.memory_refs = value.value("memory_refs", std::vector<std::string>());
        return !out.namespace_id.empty() && !out.session_id.empty() && out.revision > 0 && out.next_sequence > 0;
    } catch (...) { return false; }
}
}

agent_session_context_store::agent_session_context_store(agent_resource_store * resource_store,
        std::string ns, std::string session, bool should_persist)
    : store(resource_store), namespace_id(std::move(ns)), session_id(std::move(session)), persist(should_persist) {}

bool agent_session_context_store::list_internal(std::vector<agent_resource_descriptor> & resources,
        std::string & error) const {
    resources.clear();
    if (!persist || store == nullptr) { error.clear(); return true; }
    agent_resource_read_authority authority;
    authority.namespace_id = namespace_id;
    authority.session_id = session_id;
    authority.now = static_cast<int64_t>(std::time(nullptr));
    std::vector<agent_resource_descriptor> all;
    if (!store->list(authority, all, error)) return false;
    for (auto & descriptor : all) {
        if (descriptor.metadata.purpose.rfind("agent.context.", 0) == 0 &&
                descriptor.namespace_id == namespace_id && descriptor.session_id == session_id) {
            resources.push_back(std::move(descriptor));
        }
    }
    error.clear();
    return true;
}

bool agent_session_context_store::put_internal(const std::string & name, const std::string & key,
        const json & payload, agent_resource_descriptor & out, std::string & error) const {
    if (!persist || store == nullptr) { error.clear(); return true; }
    agent_resource_put_request request;
    request.name = name;
    request.description = "Host-owned private conversation context record.";
    request.mime_type = "application/json";
    request.text = payload.dump();
    request.scope = common_runtime_resource_scope::session;
    request.namespace_id = namespace_id;
    request.session_id = session_id;
    request.source_provider = "agent.context";
    request.source_tool = "agent.context.store";
    request.created_at = static_cast<int64_t>(std::time(nullptr));
    request.metadata.purpose = name;
    request.metadata.processing_cache_key = key;
    request.metadata.content_hash = hash_text(request.text);
    return store->put_text(request, out, error);
}

bool agent_session_context_store::append(common_agent_conversation_segment segment, std::string & error) {
    if (!persist || store == nullptr) {
        for (const auto & old : transient_segments) {
            if (old.request_id != segment.request_id || old.role != segment.role) continue;
            if (old.content != segment.content || old.evidence_refs != segment.evidence_refs) {
                error = "request_id was reused with conflicting conversation content";
                return false;
            }
            error.clear();
            return true;
        }
        segment.sequence = transient_segments.empty() ? 1 : transient_segments.back().sequence + 1;
        segment.recorded_at = static_cast<int64_t>(std::time(nullptr));
        segment.content_hash = hash_text(segment.content);
        segment.resource_uri = "transient://" + namespace_id + "/" + session_id + "/" + std::to_string(segment.sequence);
        transient_segments.push_back(std::move(segment));
        error.clear();
        return true;
    }
    std::vector<common_agent_conversation_segment> existing_segments;
    std::optional<common_agent_context_checkpoint> existing_checkpoint;
    uint64_t current_epoch = 1;
    if (!load(existing_segments, existing_checkpoint, current_epoch, error)) return false;
    const std::string cache_key = "turn:" + std::to_string(current_epoch) + ":" + segment.request_id + ":" + common_agent_conversation_role_name(segment.role);
    std::vector<agent_resource_descriptor> resources;
    if (!list_internal(resources, error)) return false;
    for (const auto & descriptor : resources) {
        if (descriptor.metadata.purpose != k_turn_purpose || descriptor.metadata.processing_cache_key != cache_key) continue;
        agent_resource_read_authority authority;
        authority.namespace_id = namespace_id; authority.session_id = session_id;
        authority.now = static_cast<int64_t>(std::time(nullptr));
        std::string stored;
        if (!store->read_text(descriptor.uri, authority, 1024 * 1024, stored, error)) return false;
        const auto old = json::parse(stored, nullptr, false);
        if (!old.is_object() || old.value("content", std::string()) != segment.content ||
                old.value("evidence_refs", std::vector<std::string>()) != segment.evidence_refs) {
            error = "request_id was reused with conflicting conversation content";
            return false;
        }
        error.clear();
        return true;
    }

    // The lane serializes turns; sequence is allocated from persisted current
    // epoch data so an interrupted daemon resumes without an in-memory cursor.
    uint64_t next = 1;
    for (const auto & old : existing_segments) next = std::max(next, old.sequence + 1);
    segment.sequence = next;
    segment.recorded_at = static_cast<int64_t>(std::time(nullptr));
    segment.content_hash = hash_text(segment.content);
    const json payload = {{"schema_version", 1}, {"request_id", segment.request_id},
        {"turn_id", segment.turn_id}, {"sequence", segment.sequence}, {"recorded_at", segment.recorded_at},
        {"epoch", current_epoch},
        {"role", common_agent_conversation_role_name(segment.role)}, {"content", segment.content},
        {"content_hash", segment.content_hash}, {"evidence_refs", segment.evidence_refs}};
    agent_resource_descriptor saved;
    return put_internal(k_turn_purpose, cache_key, payload, saved, error);
}

bool agent_session_context_store::load(std::vector<common_agent_conversation_segment> & segments,
        std::optional<common_agent_context_checkpoint> & checkpoint, uint64_t & epoch,
        std::string & error) const {
    segments.clear(); checkpoint.reset(); epoch = 1;
    if (!persist || store == nullptr) {
        segments = transient_segments;
        checkpoint = transient_checkpoint;
        epoch = transient_epoch;
        error.clear();
        return true;
    }
    std::vector<agent_resource_descriptor> resources;
    if (!list_internal(resources, error)) return false;
    uint64_t latest_epoch_revision = 0;
    std::vector<std::pair<uint64_t, common_agent_conversation_segment>> epoch_segments;
    std::vector<common_agent_context_checkpoint> checkpoints;
    for (const auto & descriptor : resources) {
        if (descriptor.metadata.purpose != k_turn_purpose && descriptor.metadata.purpose != k_checkpoint_purpose && descriptor.metadata.purpose != k_epoch_purpose) continue;
        agent_resource_read_authority authority;
        authority.namespace_id = namespace_id; authority.session_id = session_id;
        authority.now = static_cast<int64_t>(std::time(nullptr));
        std::string stored;
        if (!store->read_text(descriptor.uri, authority, 2 * 1024 * 1024, stored, error)) return false;
        const auto value = json::parse(stored, nullptr, false);
        if (!value.is_object() || value.value("schema_version", 0) != 1) continue;
        if (descriptor.metadata.purpose == k_epoch_purpose) {
            const auto revision = value.value("revision", uint64_t(0));
            if (revision > latest_epoch_revision) { latest_epoch_revision = revision; epoch = value.value("epoch", uint64_t(1)); }
        } else if (descriptor.metadata.purpose == k_turn_purpose) {
            common_agent_conversation_segment segment;
            const auto role = value.value("role", std::string());
            if (role != "user" && role != "assistant") continue;
            segment.resource_uri = descriptor.uri;
            segment.request_id = value.value("request_id", std::string());
            segment.turn_id = value.value("turn_id", std::string());
            segment.sequence = value.value("sequence", uint64_t(0));
            segment.recorded_at = value.value("recorded_at", int64_t(0));
            segment.role = role == "user" ? common_agent_conversation_role::user : common_agent_conversation_role::assistant;
            segment.content = value.value("content", std::string());
            segment.content_hash = value.value("content_hash", hash_text(segment.content));
            segment.evidence_refs = value.value("evidence_refs", std::vector<std::string>());
            const auto event_epoch = value.value("epoch", uint64_t(1));
            if (!segment.content.empty() && segment.sequence > 0) epoch_segments.emplace_back(event_epoch, std::move(segment));
        } else {
            common_agent_context_checkpoint candidate;
            if (!read_checkpoint(value, candidate) || candidate.namespace_id != namespace_id || candidate.session_id != session_id) continue;
            checkpoints.push_back(std::move(candidate));
        }
    }
    for (auto & entry : epoch_segments) if (entry.first == epoch) segments.push_back(std::move(entry.second));
    std::sort(segments.begin(), segments.end(), [](const auto & a, const auto & b) { return a.sequence < b.sequence; });
    std::sort(checkpoints.begin(), checkpoints.end(), [](const auto & a, const auto & b) { return a.revision > b.revision; });
    for (auto & candidate : checkpoints) {
        const bool has_parent = candidate.previous_revision == 0 || std::any_of(checkpoints.begin(), checkpoints.end(), [&](const auto & item) {
            return item.revision == candidate.previous_revision && item.epoch == candidate.epoch;
        });
        if (candidate.epoch != epoch || candidate.previous_revision >= candidate.revision || !has_parent) continue;
        std::vector<common_agent_conversation_segment> covered_segments;
        bool valid = true;
        for (const auto & ref : candidate.covered_turn_refs) {
            auto found = std::find_if(segments.begin(), segments.end(), [&](const auto & segment) { return segment.resource_uri == ref; });
            if (found == segments.end()) {
                valid = false;
                break;
            }
            covered_segments.push_back(*found);
        }
        if (!valid || (!candidate.fingerprint.empty() && candidate.fingerprint != common_agent_context_fingerprint(covered_segments))) continue;
        bool protected_valid = true;
        for (const auto & ref : candidate.protected_instruction_refs) {
            const auto found = std::find_if(segments.begin(), segments.end(), [&](const auto & segment) {
                return segment.resource_uri == ref && segment.role == common_agent_conversation_role::user;
            });
            if (found == segments.end()) { protected_valid = false; break; }
        }
        if (protected_valid) { checkpoint = std::move(candidate); break; }
    }
    error.clear();
    return true;
}

bool agent_session_context_store::save_checkpoint(common_agent_context_checkpoint & checkpoint, std::string & error) {
    checkpoint.namespace_id = namespace_id;
    checkpoint.session_id = session_id;
    std::vector<common_agent_conversation_segment> current_segments;
    std::optional<common_agent_context_checkpoint> previous;
    uint64_t current_epoch = 1;
    if (!load(current_segments, previous, current_epoch, error)) return false;
    checkpoint.epoch = current_epoch;
    checkpoint.previous_revision = previous ? previous->revision : 0;
    checkpoint.next_sequence = current_segments.empty() ? 1 : current_segments.back().sequence + 1;
    std::vector<common_agent_conversation_segment> covered_segments;
    for (const auto & ref : checkpoint.covered_turn_refs) {
        auto found = std::find_if(current_segments.begin(), current_segments.end(), [&](const auto & segment) { return segment.resource_uri == ref; });
        if (found == current_segments.end()) { error = "checkpoint refers to a missing conversation segment"; return false; }
        covered_segments.push_back(*found);
    }
    for (const auto & ref : checkpoint.protected_instruction_refs) {
        const auto found = std::find_if(current_segments.begin(), current_segments.end(), [&](const auto & segment) {
            return segment.resource_uri == ref && segment.role == common_agent_conversation_role::user;
        });
        if (found == current_segments.end()) { error = "checkpoint protects a missing non-user turn"; return false; }
    }
    checkpoint.fingerprint = common_agent_context_fingerprint(covered_segments);
    if (!persist || store == nullptr) {
        checkpoint.revision = ++transient_checkpoint_revision;
        checkpoint.previous_revision = checkpoint.revision - 1;
        transient_checkpoint_revision = checkpoint.revision;
        transient_checkpoint = checkpoint;
        error.clear();
        return true;
    }
    std::vector<agent_resource_descriptor> resources;
    if (!list_internal(resources, error)) return false;
    uint64_t max_revision = 0;
    for (const auto & descriptor : resources) {
        if (descriptor.metadata.purpose != k_checkpoint_purpose) continue;
        agent_resource_read_authority authority; authority.namespace_id = namespace_id; authority.session_id = session_id;
        authority.now = static_cast<int64_t>(std::time(nullptr));
        std::string stored;
        if (!store->read_text(descriptor.uri, authority, 2 * 1024 * 1024, stored, error)) return false;
        const auto value = json::parse(stored, nullptr, false);
        if (value.is_object()) max_revision = std::max(max_revision, value.value("revision", uint64_t(0)));
    }
    checkpoint.revision = max_revision + 1;
    agent_resource_descriptor saved;
    return put_internal(k_checkpoint_purpose, "checkpoint:" + std::to_string(checkpoint.revision), checkpoint_json(checkpoint), saved, error);
}

bool agent_session_context_store::begin_new_epoch(std::string & error) {
    if (!persist || store == nullptr) {
        ++transient_epoch;
        transient_segments.clear();
        transient_checkpoint.reset();
        transient_checkpoint_revision = 0;
        error.clear();
        return true;
    }
    std::vector<agent_resource_descriptor> resources;
    if (!list_internal(resources, error)) return false;
    // A session starts implicitly at epoch 1 before its first reset marker.
    // Reset must therefore create epoch 2 even when no prior marker exists.
    uint64_t epoch = 2, revision = 0;
    for (const auto & descriptor : resources) {
        if (descriptor.metadata.purpose != k_epoch_purpose) continue;
        agent_resource_read_authority authority; authority.namespace_id = namespace_id; authority.session_id = session_id;
        authority.now = static_cast<int64_t>(std::time(nullptr));
        std::string stored;
        if (!store->read_text(descriptor.uri, authority, 1024 * 1024, stored, error)) return false;
        const auto value = json::parse(stored, nullptr, false);
        if (value.is_object() && value.value("revision", uint64_t(0)) > revision) {
            revision = value.value("revision", uint64_t(0)); epoch = value.value("epoch", uint64_t(1)) + 1;
        }
    }
    agent_resource_descriptor saved;
    return put_internal(k_epoch_purpose, "epoch:" + std::to_string(revision + 1),
        {{"schema_version", 1}, {"revision", revision + 1}, {"epoch", epoch}}, saved, error);
}
