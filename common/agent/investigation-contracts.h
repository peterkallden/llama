#pragma once

#include "agent/agent-scope.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// Host-owned, versioned investigation facts. These records contain references
// to source evidence; they never replace the resource store as byte authority.
enum class common_investigation_entity_kind {
    user, host, process, ip, file, hash, service, trace, message, event, unknown
};

enum class common_investigation_relation_kind {
    authenticated_to, communicated_with, spawned, accessed, modified,
    participates_in_trace, follows, correlated_with, unknown
};
const char * common_investigation_entity_kind_name(common_investigation_entity_kind);
const char * common_investigation_relation_kind_name(common_investigation_relation_kind);

enum class common_investigation_claim_state {
    observation, correlation, hypothesis, reviewed_conclusion
};

enum class common_investigation_source_kind {
    search_connector, document, log_file
};

enum class common_investigation_statement_kind {
    statement, reported_event, definition, interpretation
};

enum class common_investigation_comparison_kind {
    supports, contradicts, qualifies, same_topic, source_dependency, inconclusive
};

struct common_investigation_source {
    std::string id;
    common_investigation_source_kind kind = common_investigation_source_kind::document;
    std::string resource_uri;
    std::string representation_uri;
    std::string provider_id;
    std::string display_name;
    std::string mime_type;
    std::string content_hash;
    std::string representation_hash;
    std::string original_uri;
    std::string original_parent_uri;
    std::string processor_version;
    std::string processing_status = "ready";
    std::vector<std::string> processing_limitations;
    uint64_t source_revision = 1;
    bool complete = true;
};

struct common_investigation_source_anchor {
    std::string source_id;
    std::string resource_uri;
    uint64_t byte_offset = 0;
    uint64_t byte_length = 0;
    int64_t page_number = 0;
    std::string quote;
    bool quote_verified = false;
};

struct common_investigation_statement {
    std::string id;
    common_investigation_statement_kind kind = common_investigation_statement_kind::statement;
    std::string text;
    common_investigation_source_anchor anchor;
    std::vector<std::string> actor_ids;
    std::string reported_time_text;
    std::string extraction_origin = "model_proposal";
};

struct common_investigation_comparison {
    std::string id;
    std::string left_statement_id;
    std::string right_statement_id;
    common_investigation_comparison_kind kind = common_investigation_comparison_kind::inconclusive;
    std::string rationale;
    std::vector<std::string> source_ids;
    std::string assessment_origin = "model_proposal";
};

struct common_investigation_analysis_record {
    std::string id;
    std::string kind;
    std::string algorithm_version;
    uint64_t published_revision = 0;
    std::vector<std::string> source_ids;
    std::string parameters_fingerprint;
    std::string completeness = "complete";
    std::vector<std::string> limitations;
};

struct common_investigation_time {
    int64_t event_time_ms = 0;
    int64_t collected_time_ms = 0;
    uint64_t uncertainty_ms = 0;
    bool has_event_time = false;
};

struct common_investigation_entity {
    std::string id;
    std::string source_id;
    common_investigation_entity_kind kind = common_investigation_entity_kind::unknown;
    std::string value;
    std::string display_name;
    std::string host_identity;
    int64_t valid_from_ms = 0;
    int64_t valid_to_ms = 0;
};

struct common_investigation_event {
    std::string id;
    std::string source_id;
    std::string source_document_id;
    std::string event_type;
    common_investigation_time time;
    std::vector<std::string> entity_ids;
    std::string raw_resource_uri;
    std::string normalized_resource_uri;
    std::map<std::string, std::string> entity_roles;
    std::string source_index;
};

struct common_investigation_relation {
    std::string id;
    std::string source_id;
    std::string from_entity_id;
    std::string to_entity_id;
    common_investigation_relation_kind kind = common_investigation_relation_kind::unknown;
    int64_t first_seen_ms = 0;
    int64_t last_seen_ms = 0;
    bool has_time = false;
    uint64_t event_count = 0;
    std::string rule_id;
    std::string rule_revision;
    std::string confidence_label = "observed";
    std::vector<std::string> evidence_event_ids;
};

struct common_investigation_relation_rule {
    std::string id;
    std::string event_type;
    common_investigation_entity_kind from_kind = common_investigation_entity_kind::unknown;
    common_investigation_entity_kind to_kind = common_investigation_entity_kind::unknown;
    common_investigation_relation_kind relation_kind = common_investigation_relation_kind::unknown;
    std::string from_role;
    std::string to_role;
};

struct common_investigation_claim {
    std::string id;
    std::string text;
    common_investigation_claim_state state = common_investigation_claim_state::hypothesis;
    std::vector<std::string> supporting_event_ids;
    std::vector<std::string> contradicting_event_ids;
    std::vector<std::string> limitations;
};

struct common_investigation_case_revision {
    uint32_t schema_version = 2;
    std::string case_id;
    uint64_t revision = 1;
    std::string previous_revision_uri;
    std::string title;
    std::string question;
    common_agent_scope scope;
    std::string source_id;
    int64_t window_start_ms = 0;
    int64_t window_end_ms = 0;
    std::vector<common_investigation_entity> entities;
    std::vector<common_investigation_event> events;
    std::vector<common_investigation_relation> relations;
    std::vector<common_investigation_claim> claims;
    std::vector<common_investigation_source> sources;
    std::vector<common_investigation_statement> statements;
    std::vector<common_investigation_comparison> comparisons;
    std::vector<common_investigation_analysis_record> analyses;
    std::vector<std::string> gaps;
    std::string fingerprint;
};

struct common_investigation_case_summary {
    std::string case_id;
    uint64_t revision = 0;
    std::string title;
    std::string question;
    std::string source_id;
    int64_t window_start_ms = 0;
    int64_t window_end_ms = 0;
};

// A frozen report is indexed separately from the case evidence revision. The
// report resources remain immutable; this record only makes them discoverable
// again when the case is reopened.
struct common_investigation_report_index_entry {
    std::string case_id;
    uint64_t case_revision = 0;
    std::string source_id;
    std::string case_fingerprint;
    std::string content_fingerprint;
    std::string report_uri;
    std::string markdown_uri;
    std::string source_manifest_uri;
    std::vector<std::string> visualization_uris;
    std::vector<std::string> limitations;
    bool complete = false;
    int64_t created_at_ms = 0;
};

// A compact UI-to-runtime binding. It carries stable references only; the
// daemon reauthorizes the case and resolves every reference before prompting.
struct common_investigation_context_binding {
    std::string case_id;
    uint64_t revision = 0;
    std::vector<std::string> entity_ids;
    std::vector<std::string> relation_ids;
    std::vector<std::string> event_ids;
    std::vector<std::string> source_ids;
    std::string analysis_profile;
};

struct common_investigation_graph_query {
    std::string case_id;
    uint64_t case_revision = 0;
    std::string center_entity_id;
    size_t max_depth = 1;
    size_t max_nodes = 100;
    size_t max_edges = 200;
    int64_t window_start_ms = 0;
    int64_t window_end_ms = 0;
    std::vector<common_investigation_relation_kind> relation_kinds;
};

struct common_investigation_graph_result {
    uint64_t case_revision = 0;
    std::vector<common_investigation_entity> nodes;
    std::vector<common_investigation_relation> edges;
    bool truncated = false;
    std::vector<std::string> limitations;
};

struct common_investigation_analysis_budget {
    size_t max_expansions = 256;
    size_t max_path_length = 12;
    size_t max_candidates = 32;
};

struct common_investigation_candidate {
    std::string id;
    std::string operation;
    std::string entity_id;
    std::string reason;
    double deterministic_priority = 0.0;
    double posterior_priority = 0.0;
    size_t history_count = 0;
    double useful_history_weight = 0.0;
    double unhelpful_history_weight = 0.0;
};

struct common_investigation_node_metric {
    std::string entity_id;
    size_t degree = 0;
    size_t component = 0;
    double betweenness = 0.0;
};

struct common_investigation_graph_metrics {
    uint64_t case_revision = 0;
    std::vector<common_investigation_node_metric> nodes;
    size_t explored_states = 0;
    bool complete = false;
    bool betweenness_normalized = false;
};

bool common_investigation_case_validate(
    const common_investigation_case_revision &, std::string & error);
bool common_investigation_anchor_validate(
    const common_investigation_source_anchor &, const std::string & source_text,
    std::string & error);
nlohmann::ordered_json common_investigation_case_to_json(
    const common_investigation_case_revision &);
bool common_investigation_case_from_json(const nlohmann::ordered_json &,
    common_investigation_case_revision &, std::string & error);
bool common_investigation_project_context(
    const common_investigation_case_revision &,
    const common_investigation_context_binding &,
    nlohmann::ordered_json &, std::string & error);
bool common_investigation_graph_expand(
    const common_investigation_case_revision &,
    const common_investigation_graph_query &,
    common_investigation_graph_result &, std::string & error);
bool common_investigation_graph_shortest_path(
    const common_investigation_case_revision &,
    const std::string & start_id, const std::string & goal_id,
    const common_investigation_analysis_budget &,
    std::vector<std::string> & relation_ids, bool & complete,
    std::string & error);
bool common_investigation_graph_analyze(
    const common_investigation_case_revision &,
    const common_investigation_analysis_budget &,
    common_investigation_graph_metrics &, std::string & error);
bool common_investigation_temporal_path(
    const common_investigation_case_revision &,
    const std::string & start_id, const std::string & goal_id,
    int64_t not_before_ms, const common_investigation_analysis_budget &,
    std::vector<std::string> & relation_ids, bool & complete,
    std::string & error);
bool common_investigation_build_relations(
    const std::string & source_id,
    const std::vector<common_investigation_entity> & entities,
    const std::vector<common_investigation_event> & events,
    const std::vector<common_investigation_relation_rule> & rules,
    std::vector<common_investigation_relation> & relations,
    std::vector<std::string> & limitations,
    std::string & error);
std::vector<common_investigation_candidate> common_investigation_rank_candidates(
    std::vector<common_investigation_candidate>, size_t max_candidates);
