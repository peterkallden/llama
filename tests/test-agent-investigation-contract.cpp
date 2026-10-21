#include "agent/investigation-contracts.h"

#include <cmath>
#include <iostream>

namespace {
int failures = 0;
void check(bool condition, const char * message) {
    if (!condition) { std::cerr << message << '\n'; ++failures; }
}

common_investigation_case_revision sample() {
    common_investigation_case_revision value;
    value.case_id = "case-1"; value.title = "Identity investigation";
    value.question = "Where did the account authenticate?"; value.source_id = "elastic-a";
    value.window_start_ms = 100; value.window_end_ms = 1000;
    value.scope.namespace_id = "local"; value.scope.project_id = "project-a";
    value.entities = {{"user:1", "elastic-a", common_investigation_entity_kind::user, "u-1", "user", "", 0, 0},
                      {"host:1", "elastic-a", common_investigation_entity_kind::host, "h-1", "host", "", 0, 0},
                      {"host:2", "elastic-a", common_investigation_entity_kind::host, "h-2", "host", "", 0, 0}};
    value.events = {{"event:1", "elastic-a", "doc:1", "auth", {200, 300, 5, true}, {"user:1", "host:1"}, "resource://raw/1", "resource://normalized/1", {}},
                    {"event:2", "elastic-a", "doc:2", "auth", {400, 500, 5, true}, {"user:1", "host:2"}, "resource://raw/2", "resource://normalized/2", {}}};
    value.relations = {{"rel:1", "elastic-a", "user:1", "host:1", common_investigation_relation_kind::authenticated_to, 200, 200, true, 1, "auth-v1", "1", "observed", {"event:1"}},
                       {"rel:2", "elastic-a", "user:1", "host:2", common_investigation_relation_kind::authenticated_to, 400, 400, true, 1, "auth-v1", "1", "observed", {"event:2"}}};
    return value;
}
}

int main() {
    std::string error;
    auto value = sample();
    check(common_investigation_case_validate(value, error), "valid case must pass validation");
    const auto encoded = common_investigation_case_to_json(value);
    common_investigation_case_revision decoded;
    check(common_investigation_case_from_json(encoded, decoded, error) && decoded.case_id == value.case_id &&
          decoded.events.size() == value.events.size() && decoded.scope.project_id == value.scope.project_id,
          "case JSON roundtrip must preserve evidence, revision, and scope");

    common_investigation_context_binding binding;
    binding.case_id = value.case_id; binding.revision = value.revision;
    binding.entity_ids = {"user:1"}; binding.relation_ids = {"rel:1"}; binding.event_ids = {"event:1"};
    nlohmann::ordered_json context_projection;
    check(common_investigation_project_context(value, binding, context_projection, error) &&
          !context_projection.contains("entities") && context_projection["selected_entities"].size() == 1 &&
          context_projection["selected_entities"][0]["id"] == "user:1" &&
          context_projection["selected_relations"][0]["id"] == "rel:1" &&
          context_projection["selected_events"][0]["id"] == "event:1",
          "case context projection must resolve only explicitly selected stable references");
    auto stale_binding = binding; ++stale_binding.revision;
    check(!common_investigation_project_context(value, stale_binding, context_projection, error),
          "case context projection must reject a stale revision");
    auto invalid_binding = binding; invalid_binding.event_ids = {"missing-event"};
    check(!common_investigation_project_context(value, invalid_binding, context_projection, error),
          "case context projection must reject unknown evidence references");
    invalid_binding = binding; invalid_binding.entity_ids = {"user:1", "user:1"};
    check(!common_investigation_project_context(value, invalid_binding, context_projection, error),
          "case context projection must reject duplicate references");

    common_investigation_case_revision network_case;
    network_case.case_id = "network-case"; network_case.source_id = "elastic-a";
    network_case.title = "Network direction"; network_case.scope.namespace_id = "local";
    network_case.entities = {
        {"ip:a", "elastic-a", common_investigation_entity_kind::ip, "10.0.0.1", "10.0.0.1", "", 0, 0},
        {"ip:b", "elastic-a", common_investigation_entity_kind::ip, "10.0.0.2", "10.0.0.2", "", 0, 0}};
    common_investigation_event network_event{"event:net", "elastic-a", "doc:net", "network_connection",
        {500, 510, 0, true}, {"ip:a", "ip:b"}, {}, {}, {}};
    network_event.entity_roles = {{"source_ip", "ip:a"}, {"destination_ip", "ip:b"}};
    network_case.events = {network_event};
    std::vector<common_investigation_relation> network_relations;
    std::vector<std::string> network_gaps;
    const std::vector<common_investigation_relation_rule> network_rules = {{"network-v1", "network_connection",
        common_investigation_entity_kind::ip, common_investigation_entity_kind::ip,
        common_investigation_relation_kind::communicated_with, "source_ip", "destination_ip"}};
    check(common_investigation_build_relations("elastic-a", network_case.entities, network_case.events,
        network_rules, network_relations, network_gaps, error) && network_relations.size() == 1 &&
        network_relations[0].from_entity_id == "ip:a" && network_relations[0].to_entity_id == "ip:b",
        "directed network endpoints must produce one correctly oriented relation");

    common_investigation_graph_query query;
    query.case_id = value.case_id; query.case_revision = value.revision;
    query.center_entity_id = "user:1"; query.max_depth = 1;
    query.window_start_ms = 300; query.window_end_ms = 900;
    common_investigation_graph_result graph;
    check(common_investigation_graph_expand(value, query, graph, error), "graph expansion must succeed");
    check(graph.nodes.size() == 2 && graph.edges.size() == 1 && graph.edges[0].id == "rel:2",
          "graph expansion must respect time filter and be bounded");
    query.case_revision++;
    check(!common_investigation_graph_expand(value, query, graph, error), "stale graph revision must fail closed");

    std::vector<std::string> path; bool complete = false;
    check(common_investigation_graph_shortest_path(value, "host:1", "host:2", {}, path, complete, error),
          "bounded shortest path should execute");
    check(path.size() == 2 && complete, "shortest path should return its evidence relations");

    common_investigation_graph_metrics metrics;
    check(common_investigation_graph_analyze(value, {}, metrics, error),
          "bounded graph metrics should execute");
    check(metrics.complete && metrics.betweenness_normalized && metrics.nodes.size() == 3 &&
            metrics.nodes[0].degree == 2 && std::abs(metrics.nodes[0].betweenness - 1.0) < 1e-9,
          "component, degree and normalized betweenness metrics should be deterministic");
    common_investigation_analysis_budget tiny_budget;
    tiny_budget.max_expansions = 1;
    check(common_investigation_graph_analyze(value, tiny_budget, metrics, error) && !metrics.complete,
          "budget exhaustion must mark graph metrics partial");

    check(common_investigation_temporal_path(value, "user:1", "host:2", 100, {}, path, complete, error),
          "time-ordered path analysis should execute");
    check(complete && path == std::vector<std::string>({"rel:2"}),
          "temporal path should follow the observed direction of a relation");
    check(common_investigation_temporal_path(value, "host:2", "user:1", 100, {}, path, complete, error) &&
            complete && path.empty(),
          "temporal path must not traverse a directed relation backwards");
    auto out_of_order = value;
    out_of_order.relations[1].first_seen_ms = 150;
    out_of_order.relations[1].last_seen_ms = 150;
    check(common_investigation_temporal_path(out_of_order, "user:1", "host:2", 500, {}, path, complete, error) &&
            complete && path.empty(),
          "temporal path must not invent a chain across out-of-order events");

    std::vector<common_investigation_relation> rebuilt;
    std::vector<std::string> limitations;
    const std::vector<common_investigation_relation_rule> relation_rules = {
        {"auth-v1", "auth", common_investigation_entity_kind::user,
            common_investigation_entity_kind::host,
            common_investigation_relation_kind::authenticated_to, {}, {}},
    };
    check(common_investigation_build_relations("elastic-a", value.entities, value.events,
            relation_rules, rebuilt, limitations, error),
          "deterministic relation builder should process normalized events");
    check(rebuilt.size() == 2 && rebuilt[0].event_count == 1 &&
            rebuilt[0].confidence_label == "observed" &&
            rebuilt[0].evidence_event_ids.size() == 1,
          "derived edges must retain their rule and concrete event evidence");
    auto rebuilt_again = std::vector<common_investigation_relation>{};
    check(common_investigation_build_relations("elastic-a", value.entities, value.events,
            relation_rules, rebuilt_again, limitations, error) &&
            rebuilt_again.size() == rebuilt.size() && rebuilt_again[0].id == rebuilt[0].id,
          "relation IDs and ordering must be reproducible");
    auto malformed_roles = value.events;
    malformed_roles[0].entity_roles["source_ip"] = "missing-entity";
    check(!common_investigation_build_relations("elastic-a", value.entities, malformed_roles,
            relation_rules, rebuilt_again, limitations, error),
          "relation builder must reject malformed event role references without throwing");
    auto incomplete_role_rule = relation_rules;
    incomplete_role_rule[0].from_role = "principal";
    check(!common_investigation_build_relations("elastic-a", value.entities, value.events,
            incomplete_role_rule, rebuilt_again, limitations, error),
          "relation builder must reject rules with only one endpoint role");
    auto unknown_time_events = value.events;
    unknown_time_events[0].time.has_event_time = false;
    check(common_investigation_build_relations("elastic-a", value.entities, unknown_time_events,
            relation_rules, rebuilt_again, limitations, error) &&
            !limitations.empty(),
          "unknown event time must remain an explicit relation limitation");

    value.relations[0].evidence_event_ids = {"missing"};
    check(!common_investigation_case_validate(value, error), "unknown evidence must invalidate a relation");
    value = sample();
    value.relations[0].evidence_event_ids.clear();
    check(!common_investigation_case_validate(value, error), "relations without concrete evidence must be rejected");
    value = sample();
    value.events[0].entity_ids.push_back("missing");
    check(!common_investigation_case_validate(value, error), "unknown event entities must fail closed");

    std::vector<common_investigation_candidate> candidates = {
        {"a", "investigation.expand", "host:1", "deterministic", 0.8, 0.0, 0, 0, 0},
        {"b", "investigation.expand", "host:2", "history", 0.8, 0.0, 10, 9, 1},
    };
    const auto ranked = common_investigation_rank_candidates(std::move(candidates), 4);
    check(ranked.size() == 2 && ranked[0].id == "a" && ranked[1].posterior_priority < 0.8,
          "scoped outcome posterior should lower historically unhelpful work without overstating evidence");
    check(failures == 0, "all investigation contract checks must pass");
    return failures == 0 ? 0 : 1;
}
