#pragma once

#include "resource/resource-contract.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

enum class common_agent_conversation_role { user, assistant };

inline const char * common_agent_conversation_role_name(common_agent_conversation_role role) {
    return role == common_agent_conversation_role::user ? "user" : "assistant";
}

struct common_agent_conversation_segment {
    std::string resource_uri;
    std::string request_id;
    std::string turn_id;
    std::string content_hash;
    uint64_t sequence = 0;
    int64_t recorded_at = 0;
    common_agent_conversation_role role = common_agent_conversation_role::user;
    std::string content;
    std::vector<std::string> evidence_refs;
};

struct common_agent_context_checkpoint {
    uint32_t schema_version = 1;
    std::string namespace_id;
    std::string session_id;
    uint64_t revision = 0;
    uint64_t previous_revision = 0;
    uint64_t next_sequence = 1;
    uint64_t epoch = 1;
    std::string fingerprint;
    std::string summary;
    std::string current_goal;
    std::vector<std::string> decisions;
    std::vector<std::string> open_questions;
    std::vector<std::string> tried_alternatives;
    std::vector<std::string> covered_turn_refs;
    // Exact user text and source references are retained as evidence. This
    // section is host-protected; a generated summary cannot add authority.
    std::vector<std::string> protected_instruction_refs;
    std::vector<std::string> plan_refs;
    std::vector<std::string> evidence_refs;
    std::vector<std::string> memory_refs;
};

struct common_agent_context_composition {
    std::string rendered_history;
    std::vector<std::string> included_refs;
    std::vector<std::string> omitted_refs;
    std::vector<std::string> included_sections;
    size_t estimated_tokens = 0;
    bool estimate_is_heuristic = true;
    bool used_checkpoint = false;
    std::vector<std::string> missing_protected_refs;
    bool checkpoint_material_omitted = false;
};

inline size_t common_agent_projected_conversation_context_chars(
        const common_agent_context_composition & composition,
        size_t current_request_chars) {
    static const std::string context_prefix = "Conversation context (source records remain available to the host):\n";
    static const std::string current_request_prefix = "\nCurrent request:\n";
    const size_t framing_chars = composition.rendered_history.empty()
        ? 0 : context_prefix.size() + current_request_prefix.size();
    return composition.rendered_history.size() + current_request_chars + framing_chars;
}

inline std::string common_agent_context_fingerprint(const std::vector<common_agent_conversation_segment> & segments) {
    // Stable FNV-1a over ordered source identity and content hashes. This is a
    // corruption/change detector, not a cryptographic authenticity claim.
    uint64_t hash = 14695981039346656037ull;
    auto add = [&](const std::string & value) {
        for (const unsigned char byte : value) { hash ^= byte; hash *= 1099511628211ull; }
        hash ^= 0xff; hash *= 1099511628211ull;
    };
    for (const auto & segment : segments) {
        add(segment.resource_uri);
        add(segment.request_id);
        add(segment.content_hash);
        add(std::to_string(segment.sequence));
    }
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(16, '0');
    for (size_t i = 0; i < result.size(); ++i) result[result.size() - i - 1] = digits[(hash >> (i * 4)) & 0xf];
    return result;
}

inline common_agent_context_composition compose_common_agent_conversation_context(
        const std::vector<common_agent_conversation_segment> & segments,
        const common_agent_context_checkpoint * checkpoint,
        size_t recent_turn_count,
        size_t max_chars) {
    common_agent_context_composition out;
    out.used_checkpoint = checkpoint != nullptr;
    std::vector<const common_agent_conversation_segment *> ordered;
    ordered.reserve(segments.size());
    for (const auto & segment : segments) ordered.push_back(&segment);
    std::sort(ordered.begin(), ordered.end(), [](const auto * a, const auto * b) {
        return a->sequence < b->sequence;
    });
    std::vector<const common_agent_conversation_segment *> visible;
    for (const auto * segment : ordered) {
        if (checkpoint != nullptr && std::find(checkpoint->covered_turn_refs.begin(), checkpoint->covered_turn_refs.end(), segment->resource_uri) != checkpoint->covered_turn_refs.end()) continue;
        visible.push_back(segment);
    }
    const size_t recent_segments = recent_turn_count * 2;
    const size_t first_recent = visible.size() > recent_segments ? visible.size() - recent_segments : 0;
    std::set<std::string> selected_refs;
    for (size_t i = first_recent; i < visible.size(); ++i) selected_refs.insert(visible[i]->resource_uri);
    // Fill unused room with the nearest older source segments. This means
    // recent_turns is a priority, not an unconditional truncation threshold.
    size_t used_chars = 0;
    for (const auto * segment : ordered) {
        if (checkpoint != nullptr && std::find(checkpoint->protected_instruction_refs.begin(), checkpoint->protected_instruction_refs.end(), segment->resource_uri) != checkpoint->protected_instruction_refs.end()) {
            used_chars += segment->content.size() + segment->resource_uri.size() + 40;
        }
    }
    if (out.used_checkpoint) used_chars += checkpoint->summary.size() + checkpoint->current_goal.size() + 100;
    if (checkpoint != nullptr) {
        for (const auto & value : checkpoint->decisions) used_chars += value.size() + 16;
        for (const auto & value : checkpoint->open_questions) used_chars += value.size() + 20;
        for (const auto & value : checkpoint->tried_alternatives) used_chars += value.size() + 12;
        for (const auto & ref : checkpoint->plan_refs) used_chars += ref.size() + 12;
        for (const auto & ref : checkpoint->evidence_refs) used_chars += ref.size() + 14;
        for (const auto & ref : checkpoint->memory_refs) used_chars += ref.size() + 13;
    }
    for (size_t i = first_recent; i > 0; --i) {
        const auto * candidate = visible[i - 1];
        if (selected_refs.count(candidate->resource_uri) != 0) continue;
        const size_t estimate = candidate->content.size() + candidate->resource_uri.size() + 32;
        if (used_chars + estimate > max_chars) continue;
        selected_refs.insert(candidate->resource_uri);
        used_chars += estimate;
    }
    std::set<std::string> rendered_refs;
    std::set<std::string> included_sections;
    // Exact user-authored instructions are retained ahead of generated
    // summaries and recent dialogue. They remain evidence, not host policy.
    if (checkpoint != nullptr) {
        for (const auto & ref : checkpoint->protected_instruction_refs) {
            const auto found = std::find_if(ordered.begin(), ordered.end(), [&](const auto * segment) {
                return segment->resource_uri == ref && segment->role == common_agent_conversation_role::user;
            });
            if (found == ordered.end()) {
                out.missing_protected_refs.push_back(ref);
                continue;
            }
            const auto * segment = *found;
            std::string row = "Protected user instruction [" + ref + "] (verbatim):\n" + segment->content;
            if (!segment->evidence_refs.empty()) {
                row += "\nevidence refs: ";
                for (size_t i = 0; i < segment->evidence_refs.size(); ++i) {
                    if (i != 0) row += ", ";
                    row += segment->evidence_refs[i];
                }
            }
            row += "\n";
            if (out.rendered_history.size() + row.size() > max_chars) {
                out.missing_protected_refs.push_back(ref);
                continue;
            }
            out.rendered_history += row;
            out.included_refs.push_back(ref);
            rendered_refs.insert(ref);
            included_sections.insert("protected_user_instructions");
        }
    }
    if (out.used_checkpoint) {
        auto append_checkpoint = [&](const std::string & row) {
            if (out.rendered_history.size() + row.size() > max_chars) { out.checkpoint_material_omitted = true; return; }
            out.rendered_history += row;
        };
        if (!checkpoint->summary.empty()) append_checkpoint("Earlier conversation summary (orientation; source turns remain authoritative):\n" + checkpoint->summary + "\n");
        if (!checkpoint->current_goal.empty()) append_checkpoint("Current goal: " + checkpoint->current_goal + "\n");
        for (const auto & decision : checkpoint->decisions) append_checkpoint("Decision: " + decision + "\n");
        for (const auto & question : checkpoint->open_questions) append_checkpoint("Open question: " + question + "\n");
        for (const auto & tried : checkpoint->tried_alternatives) append_checkpoint("Tried: " + tried + "\n");
        for (const auto & ref : checkpoint->plan_refs) append_checkpoint("Plan ref: " + ref + "\n");
        for (const auto & ref : checkpoint->evidence_refs) append_checkpoint("Evidence ref: " + ref + "\n");
        for (const auto & ref : checkpoint->memory_refs) append_checkpoint("Memory ref: " + ref + "\n");
        if (!checkpoint->summary.empty() || !checkpoint->current_goal.empty() || !checkpoint->decisions.empty() ||
                !checkpoint->open_questions.empty() || !checkpoint->tried_alternatives.empty() ||
                !checkpoint->plan_refs.empty() || !checkpoint->evidence_refs.empty() || !checkpoint->memory_refs.empty())
            included_sections.insert("checkpoint");
    }
    for (size_t i = 0; i < visible.size(); ++i) {
        const auto * segment = visible[i];
        if (selected_refs.count(segment->resource_uri) == 0) { out.omitted_refs.push_back(segment->resource_uri); continue; }
        if (rendered_refs.count(segment->resource_uri) != 0) continue;
        std::string row = std::string(common_agent_conversation_role_name(segment->role)) + " [" + segment->resource_uri + "]: " + segment->content;
        if (!segment->evidence_refs.empty()) {
            row += "\nevidence refs: ";
            for (size_t ref = 0; ref < segment->evidence_refs.size(); ++ref) {
                if (ref != 0) row += ", ";
                row += segment->evidence_refs[ref];
            }
        }
        row += "\n";
        if (out.rendered_history.size() + row.size() > max_chars) {
            out.omitted_refs.push_back(segment->resource_uri);
            continue;
        }
        out.rendered_history += row;
        out.included_refs.push_back(segment->resource_uri);
        included_sections.insert("conversation_turns");
        if (!segment->evidence_refs.empty()) included_sections.insert("evidence_references");
    }
    out.included_sections.assign(included_sections.begin(), included_sections.end());
    out.estimated_tokens = (out.rendered_history.size() + 3) / 4;
    return out;
}
