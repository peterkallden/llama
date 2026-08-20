#include "memory/memory-policy.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>

static constexpr size_t k_memory_policy_max_content_chars = 512;
static constexpr size_t k_memory_policy_max_rationale_chars = 240;
static constexpr float k_memory_policy_duplicate_threshold = 0.97f;
static constexpr float k_memory_policy_conflict_threshold = 0.92f;

static std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return value;
}

static std::string squeeze_space(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    bool in_space = false;
    for (unsigned char c : value) {
        if (std::isspace(c)) {
            if (!in_space && !out.empty()) {
                out.push_back(' ');
            }
            in_space = true;
        } else {
            out.push_back((char) std::tolower(c));
            in_space = false;
        }
    }
    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

static bool contains_any(const std::string & haystack, const std::vector<std::string> & needles) {
    for (const auto & needle : needles) {
        if (haystack.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool finite_unit_interval(float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

static bool auto_store_kind_allowed(common_memory_kind kind) {
    switch (kind) {
        case common_memory_kind::fact:
        case common_memory_kind::preference:
        case common_memory_kind::procedure:
        case common_memory_kind::constraint:
        case common_memory_kind::decision:
            return true;
        case common_memory_kind::episode:
        case common_memory_kind::observation:
        case common_memory_kind::reflection:
        case common_memory_kind::goal:
            return false;
    }
    return false;
}

static bool scope_is_valid(const common_memory_remember_request & request, std::string & reason) {
    if (request.namespace_id.empty()) {
        reason = "memory namespace must not be empty";
        return false;
    }
    if (request.scope == common_memory_scope::turn && request.turn_id.empty()) {
        reason = "turn-scoped memory requires a turn identity";
        return false;
    }
    if (request.scope == common_memory_scope::session && request.session_id.empty()) {
        reason = "session-scoped memory requires a session identity";
        return false;
    }
    if (request.scope == common_memory_scope::project && request.project_id.empty()) {
        reason = "project-scoped memory requires a project identity";
        return false;
    }
    if (request.scope == common_memory_scope::global && !request.global_opt_in) {
        reason = "global memory requires explicit local single-user opt-in";
        return false;
    }
    return true;
}

static bool looks_sensitive(const std::string & normalized) {
    return contains_any(normalized, {
        "password",
        "passphrase",
        "secret",
        "api key",
        "apikey",
        "token",
        "ssh key",
        "private key",
        "bearer ",
        "credential",
        "login",
        "bank account",
        "credit card",
        "social security",
        "ssn",
        "BEGIN PRIVATE KEY",
    });
}

static bool looks_policy_manipulative(const std::string & normalized) {
    return contains_any(normalized, {
        "ignore previous",
        "ignore earlier",
        "system prompt",
        "developer message",
        "tool behavior",
        "policy override",
        "change the policy",
        "always remember",
        "must store",
    });
}

static std::string make_record_id(const common_memory_remember_request & request, int64_t now) {
    std::hash<std::string> hasher;
    const size_t digest = hasher(
        request.content + "|" +
        common_memory_kind_name(request.kind) + "|" +
        request.rationale + "|" +
        std::to_string(now));

    std::ostringstream oss;
    oss << "remember-" << now << "-";
    oss << std::hex << std::setw(10) << std::setfill('0') << (uint64_t) digest;
    return oss.str();
}

const char * common_memory_remember_decision_name(common_memory_remember_decision decision) {
    switch (decision) {
        case common_memory_remember_decision::accept:    return "accept";
        case common_memory_remember_decision::reject:    return "reject";
        case common_memory_remember_decision::duplicate: return "duplicate";
        case common_memory_remember_decision::conflict:  return "conflict";
    }
    return "reject";
}

common_memory_remember_result common_memory_evaluate_remember_request(
        common_memory_store & store,
        const common_memory_remember_request & request,
        const std::vector<float> & embedding,
        int64_t now,
        std::string & error) {
    error.clear();

    common_memory_remember_result result;
    result.decision = common_memory_remember_decision::reject;

    if (request.content.empty() || request.content.size() > k_memory_policy_max_content_chars) {
        result.reason = "content must contain between 1 and 512 characters";
        return result;
    }
    if (request.rationale.size() > k_memory_policy_max_rationale_chars) {
        result.reason = "rationale must contain at most 240 characters";
        return result;
    }
    if (!finite_unit_interval(request.importance) || !finite_unit_interval(request.confidence)) {
        result.reason = "importance and confidence must be finite values between 0 and 1";
        return result;
    }
    if (!auto_store_kind_allowed(request.kind)) {
        result.reason = std::string("kind is not eligible for automatic storage: ") + common_memory_kind_name(request.kind);
        return result;
    }
    if (!scope_is_valid(request, result.reason)) {
        return result;
    }

    const std::string normalized_content = squeeze_space(request.content);
    if (normalized_content.size() < 8) {
        result.reason = "content is too short to store reliably";
        return result;
    }
    if (looks_sensitive(normalized_content)) {
        result.reason = "content appears to contain secrets or credentials";
        return result;
    }
    if (looks_policy_manipulative(normalized_content)) {
        result.reason = "content appears to manipulate policy or tool behavior";
        return result;
    }

    common_memory_retrieval retrieval(store);
    common_memory_query query;
    query.text = request.content;
    query.embedding = embedding;
    query.limit = 3;
    query.minimum_score = 0.50f;
    query.scope = request.scope;
    query.namespace_id = request.namespace_id;
    query.session_id = request.session_id;
    query.project_id = request.project_id;
    query.turn_id = request.turn_id;
    query.global_opt_in = request.global_opt_in;
    result.related_hits = retrieval.retrieve(query, error);
    if (!error.empty()) {
        result.reason = "duplicate scan failed";
        return result;
    }

    for (const auto & hit : result.related_hits) {
        const std::string normalized_hit = squeeze_space(hit.memory.content);
        if (normalized_hit == normalized_content || hit.semantic_score >= k_memory_policy_duplicate_threshold) {
            result.decision = common_memory_remember_decision::duplicate;
            result.reason = "similar memory already exists";
            return result;
        }
        if (hit.memory.kind == request.kind && hit.semantic_score >= k_memory_policy_conflict_threshold) {
            result.decision = common_memory_remember_decision::conflict;
            result.reason = "similar stored memory may conflict and should not be overwritten automatically";
            return result;
        }
    }

    common_memory_record record;
    record.id = make_record_id(request, now);
    record.kind = request.kind;
    record.content = request.content;
    record.embedding = embedding;
    record.created_at = now;
    record.accessed_at = now;
    record.importance = request.importance;
    record.confidence = request.confidence;
    record.scope = request.scope;
    record.namespace_id = request.namespace_id;
    record.session_id = request.session_id;
    record.project_id = request.project_id;
    record.turn_id = request.turn_id;
    record.metadata["policy_version"] = "memory_remember_v1";
    record.metadata["policy_decision"] = "accept";
    record.metadata["policy_reason"] = "accepted_low_risk_memory";
    record.metadata["source_role"] = request.source_role.empty() ? "assistant" : lowercase(request.source_role);
    if (!request.source_turn_id.empty()) {
        record.metadata["source_turn_id"] = request.source_turn_id;
    }
    if (!request.rationale.empty()) {
        record.metadata["remember_rationale"] = request.rationale;
    }
    result.decision = common_memory_remember_decision::accept;
    result.reason = "accepted low-risk memory";
    result.record = std::move(record);
    return result;
}
