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
    value.events = {{"event:1", "elastic-a", "doc:1", "auth", {200, 300, 5, true}, {"user:1", "host:1"}, "resource://raw/1", "resource://normalized/1"},
                    {"event:2", "elastic-a", "doc:2", "auth", {400, 500, 5, true}, {"user:1", "host:2"}, "resource://raw/2", "resource://normalized/2"}};
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

    std::vector<common_investigation_relation> rebuilt;
    std::vector<std::string> limitations;
    const std::vector<common_investigation_relation_rule> relation_rules = {
        {"auth-v1", "auth", common_investigation_entity_kind::user,
            common_investigation_entity_kind::host,
            common_investigation_relation_kind::authenticated_to},
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
    auto unknown_time_events = value.events;
    unknown_time_events[0].time.has_event_time = false;
    check(common_investigation_build_relations("elastic-a", value.entities, unknown_time_events,
            relation_rules, rebuilt_again, limitations, error) &&
            !limitations.empty(),
          "unknown event time must remain an explicit relation limitation");

    value.relations[0].evidence_event_ids = {"missing"};
    check(!common_investigation_case_validate(value, error), "unknown evidence must invalidate a relation");
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
