#include "agent/investigation-contracts.h"
#include "agent/investigation-sources.h"
#include "agent/investigation-log-parser.h"

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
    value.events = {{"event:1", "elastic-a", "doc:1", "auth", {200, 300, 5, true}, {"user:1", "host:1"}, "resource://raw/1", "resource://normalized/1", {}, {}},
                    {"event:2", "elastic-a", "doc:2", "auth", {400, 500, 5, true}, {"user:1", "host:2"}, "resource://raw/2", "resource://normalized/2", {}, {}}};
    value.relations = {{"rel:1", "elastic-a", "user:1", "host:1", common_investigation_relation_kind::authenticated_to, 200, 200, true, 1, "auth-v1", "1", "observed", {"event:1"}},
                       {"rel:2", "elastic-a", "user:1", "host:2", common_investigation_relation_kind::authenticated_to, 400, 400, true, 1, "auth-v1", "1", "observed", {"event:2"}}};
    return value;
}
}

int main() {
    std::string error;
    auto value = sample();
    value.events[0].source_index = "logs-2026.10";
    check(common_investigation_case_validate(value, error), "valid case must pass validation");
    auto multi_source = value;
    common_investigation_source secondary;
    secondary.id = "provider-elastic-b";
    secondary.kind = common_investigation_source_kind::search_connector;
    secondary.provider_id = "elastic-b";
    secondary.processing_status = "query_complete";
    multi_source.sources.push_back(secondary);
    auto secondary_entity = multi_source.entities.front();
    secondary_entity.id = "user:secondary";
    secondary_entity.source_id = "elastic-b";
    multi_source.entities.push_back(secondary_entity);
    check(common_investigation_case_validate(multi_source, error) &&
        common_investigation_search_providers(multi_source) == (std::vector<std::string>{"elastic-a", "elastic-b"}),
        "secondary provider domains must remain valid independently of source record identities");
    check(common_investigation_source_complete(secondary),
        "a complete collected search query must not be mistaken for an unprepared document");
    secondary.processing_status = "not_collected";
    check(!common_investigation_source_complete(secondary), "uncollected sources cannot imply complete coverage");
    multi_source.sources.push_back(multi_source.sources.front());
    multi_source.sources.back().id = "another-source-record";
    check(!common_investigation_case_validate(multi_source, error), "duplicate provider domains must be rejected");
    const auto encoded = common_investigation_case_to_json(value);
    common_investigation_case_revision decoded;
    check(common_investigation_case_from_json(encoded, decoded, error) && decoded.case_id == value.case_id &&
          decoded.events.size() == value.events.size() && decoded.events[0].source_index == "logs-2026.10" &&
          decoded.scope.project_id == value.scope.project_id,
          "case JSON roundtrip must preserve evidence, revision, and scope");
    auto legacy = encoded;
    legacy["schema_version"] = 1;
    legacy.erase("sources"); legacy.erase("statements"); legacy.erase("comparisons"); legacy.erase("analyses");
    legacy["events"][1].erase("source_index");
    common_investigation_case_revision legacy_decoded;
    check(common_investigation_case_from_json(legacy, legacy_decoded, error) &&
          legacy_decoded.events[1].source_index.empty(),
          "older case revisions without source indexes must remain readable");

    common_investigation_case_revision document_case;
    document_case.case_id = "document-case"; document_case.title = "Document investigation";
    document_case.question = "What do the selected reports say?"; document_case.scope.namespace_id = "local";
    common_investigation_source document_source;
    document_source.id = "source-report-a"; document_source.kind = common_investigation_source_kind::document;
    document_source.resource_uri = "resource://project/report-a";
    document_source.representation_uri = document_source.resource_uri;
    document_source.content_hash = "sha256:abc";
    document_source.processing_limitations = {"OCR text is derived and may be incomplete."};
    document_source.processing_status = "ready_with_limitations";
    document_source.complete = false;
    document_case.sources.push_back(document_source);
    check(common_investigation_case_validate(document_case, error),
        "document-only cases must not require a search provider or time window");
    auto document_roundtrip = common_investigation_case_to_json(document_case);
    common_investigation_case_revision document_decoded;
    check(common_investigation_case_from_json(document_roundtrip, document_decoded, error) &&
            document_decoded.source_id.empty() && document_decoded.sources.size() == 1 &&
            document_decoded.sources[0].processing_limitations == document_source.processing_limitations &&
            !document_decoded.sources[0].complete,
        "versioned case JSON must preserve typed document-only source processing limitations");
    const std::string unicode_source = "Prefix. Återgiven uppgift. Repeated: same quote. Repeated: same quote.";
    common_investigation_source_anchor anchor;
    anchor.source_id = "source-report-a"; anchor.resource_uri = "resource://project/report-a";
    anchor.byte_offset = unicode_source.find("same quote"); anchor.byte_length = std::string("same quote").size();
    anchor.quote = "same quote";
    check(common_investigation_anchor_validate(anchor, unicode_source, error),
        "anchor verification must use the exact declared byte position when quotes repeat");
    anchor.byte_offset = unicode_source.find("Återgiven");
    anchor.quote = "Återgiven"; anchor.byte_length = anchor.quote.size();
    check(common_investigation_anchor_validate(anchor, unicode_source, error),
        "anchor verification must preserve UTF-8 byte offsets");
    ++anchor.byte_offset;
    check(!common_investigation_anchor_validate(anchor, unicode_source, error),
        "anchor verification must reject offsets that split or miss UTF-8 text");
    auto invalid_document_case = document_case;
    common_investigation_statement statement;
    statement.id = "statement-1"; statement.text = "A quoted claim";
    statement.anchor.source_id = document_source.id; statement.anchor.resource_uri = document_source.representation_uri;
    statement.anchor.byte_offset = 0; statement.anchor.byte_length = 5; statement.anchor.quote = "wrong";
    statement.anchor.quote_verified = false;
    invalid_document_case.statements.push_back(statement);
    check(!common_investigation_case_validate(invalid_document_case, error),
        "unverified model-proposed source positions must not be persisted as evidence");
    auto invalid_actor_case = document_case;
    statement.anchor.quote_verified = true;
    statement.actor_ids = {"actor-mention-1", "actor-mention-1"};
    invalid_actor_case.statements.push_back(statement);
    check(!common_investigation_case_validate(invalid_actor_case, error),
        "duplicate actor mention references must not be persisted in one statement");

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
        {500, 510, 0, true}, {"ip:a", "ip:b"}, {}, {}, {}, {}};
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
    query.case_revision = value.revision; query.window_start_ms = 0; query.window_end_ms = 0;
    query.max_nodes = 1;
    check(common_investigation_graph_expand(value, query, graph, error) && graph.truncated &&
        graph.nodes.size() == 1 && graph.edges.empty(),
        "graph expansion must not return edges with an endpoint omitted by the node limit");
    query.max_nodes = 100;
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
    std::vector<common_investigation_relation> rebuilt_again;
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
    auto role_bound_rules = relation_rules;
    role_bound_rules[0].from_role = "user";
    role_bound_rules[0].to_role = "host";
    check(common_investigation_build_relations("elastic-a", value.entities, value.events,
            role_bound_rules, rebuilt_again, limitations, error) && rebuilt_again.size() == 2,
          "role-bound profiles should migrate legacy events when each endpoint kind is unambiguous");
    auto ambiguous_event = value.events;
    ambiguous_event[0].entity_ids.push_back("host:2");
    check(common_investigation_build_relations("elastic-a", value.entities, ambiguous_event,
            role_bound_rules, rebuilt_again, limitations, error) && rebuilt_again.size() == 1 &&
            rebuilt_again[0].evidence_event_ids == std::vector<std::string>({"event:2"}),
          "role-bound profiles must leave ambiguous legacy events unlinked rather than create cross-product edges");
    auto missing_roles = value.events;
    missing_roles[0].entity_roles = {{"unrelated_role", "host:1"}};
    check(common_investigation_build_relations("elastic-a", value.entities, missing_roles,
            role_bound_rules, rebuilt_again, limitations, error) && rebuilt_again.size() == 1 &&
            rebuilt_again[0].evidence_event_ids == std::vector<std::string>({"event:2"}) && !limitations.empty(),
          "modern events with missing declared roles must not infer new role bindings");
    auto arbitrary_roles = role_bound_rules;
    arbitrary_roles[0].from_role = "actor";
    arbitrary_roles[0].to_role = "target";
    check(common_investigation_build_relations("elastic-a", value.entities, value.events,
            arbitrary_roles, rebuilt_again, limitations, error) && rebuilt_again.empty(),
          "legacy migration must not invent bindings for arbitrary configured roles");
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
    // Declarative log parsing is independent of daemon, model and assertions.
    common_investigation_log_page log_page;
    const std::map<std::string,std::string> log_fields = {{"event.type","/event/type"},{"@timestamp","/@timestamp"},{"user.id","/user/id"}};
    const std::string log_text = "{\"event\":{\"type\":\"auth\"},\"user\":{\"id\":\"åsa\"}}\ninvalid\n{\"event\":{\"type\":\"network\"}}\n";
    check(common_investigation_parse_log(log_text,"jsonl",log_fields,0,2,log_page,error) &&
        log_page.rows.size() == 1 && log_page.rejected_count == 1 && !log_page.exhausted && log_page.next_record == 2,
        "log pages must count rejected records toward their scan budget and preserve continuation");
    const auto first_record_length = log_text.find('\n');
    check(log_page.rows[0]["@source.byte_length"] == first_record_length && log_page.rows[0]["user.id"] == "åsa",
        "log anchors must use original UTF-8 bytes rather than character indexes");
    check(common_investigation_parse_log(log_text,"jsonl",log_fields,2,2,log_page,error) &&
        log_page.exhausted && log_page.rows.size() == 1 && log_page.rows[0]["@source.line"] == 3,
        "continuation must reach the remaining original record without replaying earlier rows");
    check(!common_investigation_parse_log(log_text,"auto",log_fields,0,2,log_page,error),
        "unknown log formats must not trigger heuristic parser selection");
    const std::string csv = "type,user\r\nauth,\"Åsa, A\"\r\nnetwork,\"two\nlines\"\r\n";
    check(common_investigation_parse_log(csv,"csv",{{"event.type","type"},{"user.id","user"}},0,5,log_page,error) &&
        log_page.exhausted && log_page.rows.size() == 2 && log_page.rows[1]["user.id"] == "two\nlines" &&
        log_page.rows[1]["@source.line"] == 3,
        "CSV parsing must preserve quoted delimiters, embedded newlines and record positions");
    check(!common_investigation_parse_log("type,type\nauth,a\n","csv",{{"event.type","type"}},0,5,log_page,error),
        "ambiguous duplicate CSV headers must fail rather than guess a column");
    check(common_investigation_parse_log("{\"event\":{\"type\":[\"auth\"]}}\n","jsonl",log_fields,0,5,log_page,error) &&
        log_page.rows.empty() && log_page.rejected_count == 1,
        "structured values in scalar mappings must remain rejected evidence");

    auto actor_case = sample();
    common_investigation_source document;
    document.id = "document-a"; document.resource_uri = "resource://original/a";
    document.representation_uri = "resource://text/a";
    actor_case.sources.push_back(document);
    common_investigation_statement actor_statement;
    actor_statement.id = "statement-a"; actor_statement.kind = common_investigation_statement_kind::reported_event;
    actor_statement.text = "Alex met Alex.";
    actor_statement.anchor = {document.id,document.representation_uri,0,14,0,"Alex met Alex.",true};
    actor_case.statements.push_back(actor_statement);
    auto mention_anchor = actor_statement.anchor;
    mention_anchor.byte_length = 4; mention_anchor.quote = "Alex";
    actor_case.actor_mentions.push_back({"mention-a",actor_statement.id,"Alex","person",mention_anchor});
    mention_anchor.byte_offset = 9;
    actor_case.actor_mentions.push_back({"mention-b",actor_statement.id,"Alex","person",mention_anchor});
    actor_case.actors = {{"actor-a","Alex",{"mention-a"}},{"actor-b","Alex",{"mention-b"}}};
    check(common_investigation_case_validate(actor_case,error) && actor_case.actors.size() == 2,
        "equal names at distinct source positions must remain separate actor candidates");
    actor_case.evidence_links.push_back({"identity-proposal","possible_same_actor","actor-a","actor-b",{actor_statement.id},"Explicit comparison remains provisional.","model_proposal"});
    actor_case.evidence_links.push_back({"event-proposal","possible_same_event",actor_statement.id,actor_case.events.front().id,{actor_statement.id},"Reported and recorded event may correspond.","model_proposal"});
    check(common_investigation_case_validate(actor_case,error) && actor_case.actors.size() == 2 && actor_case.events.size() == 2,
        "proposed identity/event links must preserve distinct candidates and original log observations");
    common_investigation_case_revision actor_reloaded;
    check(common_investigation_case_from_json(common_investigation_case_to_json(actor_case),actor_reloaded,error) &&
        actor_reloaded.actor_mentions.size() == 2 && actor_reloaded.evidence_links.size() == 2,
        "actor structure and proposed cross-source links must survive case serialization");
    const auto make_review = [](std::string id, std::string left, std::string right,
            std::string decision, uint64_t revision, std::string supersedes = std::string()) {
        common_investigation_association_review review;
        review.id=id; review.operation_id=id; review.parameters_fingerprint="sha256:review-fixture";
        review.left_actor_id=left; review.right_actor_id=right; review.decision=decision;
        review.reviewer_id="authenticated-session:local/test-user"; review.created_at_ms=1000;
        review.published_revision=revision; review.supersedes_id=supersedes; return review;
    };
    auto reviewed_actors = actor_reloaded;
    reviewed_actors.revision = 2;
    reviewed_actors.association_reviews.push_back(make_review("review-same","actor-a","actor-b","same_actor",2));
    nlohmann::ordered_json associations;
    check(common_investigation_case_validate(reviewed_actors,error) &&
        common_investigation_actor_associations(reviewed_actors,associations,error) && associations["groups"].size() == 1 &&
        associations["groups"][0]["assessment_origin"] == "analyst_confirmed_identity" &&
        reviewed_actors.actors.size() == 2 && reviewed_actors.actor_mentions.size() == 2,
        "user-confirmed identity must group the view without merging original candidates or mentions");
    common_investigation_case_revision reviewed_reload;
    check(common_investigation_case_from_json(common_investigation_case_to_json(reviewed_actors),reviewed_reload,error) &&
        reviewed_reload.association_reviews.front().reviewer_id == "authenticated-session:local/test-user",
        "review provenance must survive immutable revision roundtrip");
    reviewed_actors.revision = 3;
    reviewed_actors.association_reviews.push_back(make_review("review-withdraw","actor-a","actor-b","unresolved",3,"review-same"));
    check(common_investigation_case_validate(reviewed_actors,error) &&
        common_investigation_actor_associations(reviewed_actors,associations,error) && associations["groups"].size() == 2 &&
        reviewed_actors.association_reviews.size() == 2,
        "withdrawing a confirmation must restore separate groups while preserving decision history");
    reviewed_actors.revision = 4;
    reviewed_actors.association_reviews.push_back(make_review("review-different","actor-a","actor-b","different_actors",4,"review-withdraw"));
    check(common_investigation_case_validate(reviewed_actors,error),"an analyst can distinguish same-name candidates after reconsideration");
    auto relation_review = make_review("review-relation","actor-a","actor-b","related",4);
    relation_review.kind="relationship"; relation_review.relation_type="employed_by";
    reviewed_actors.association_reviews.push_back(relation_review);
    check(common_investigation_case_validate(reviewed_actors,error) &&
        common_investigation_actor_associations(reviewed_actors,associations,error) && associations["relationships"].size() == 1 &&
        associations["relationships"][0]["source"] == "actor-a" && associations["relationships"][0]["target"] == "actor-b",
        "reviewed directional relationships must remain separate from identity decisions");
    auto ownership = relation_review;
    ownership.id="review-ownership"; ownership.operation_id=ownership.id; ownership.relation_type="owns";
    reviewed_actors.association_reviews.push_back(ownership);
    check(common_investigation_case_validate(reviewed_actors,error) &&
        common_investigation_actor_associations(reviewed_actors,associations,error) && associations["relationships"].size() == 2,
        "different reviewed relationship types must coexist rather than silently overwrite each other");
    auto broken_chain = reviewed_actors;
    broken_chain.association_reviews[1].supersedes_id="nonexistent";
    check(!common_investigation_case_validate(broken_chain,error),"review history must not skip or fabricate predecessor decisions");
    auto transitive = actor_reloaded;
    transitive.revision=4;
    auto third_mention=transitive.actor_mentions.back(); third_mention.id="mention-c";
    transitive.actor_mentions.push_back(third_mention); transitive.actors.push_back({"actor-c","Alex",{"mention-c"}});
    transitive.association_reviews = {make_review("same-ab","actor-a","actor-b","same_actor",2),
        make_review("same-bc","actor-b","actor-c","same_actor",3),
        make_review("different-ac","actor-a","actor-c","different_actors",4)};
    check(!common_investigation_case_validate(transitive,error),
        "transitive identity confirmations must not hide an explicit different-actor decision");
    actor_case.actor_mentions[1].anchor.byte_offset = 8;
    check(!common_investigation_case_validate(actor_case,error),"actor anchors must match the exact named substring inside the statement");
    actor_case = actor_reloaded;
    actor_case.evidence_links.front().assessment_origin = "analyst_review";
    check(!common_investigation_case_validate(actor_case,error),"model proposal records cannot become analyst decisions through an origin flag");
    actor_case = actor_reloaded;
    common_investigation_claim document_hypothesis;
    document_hypothesis.id = "hypothesis-document"; document_hypothesis.text = "A testable explanation";
    document_hypothesis.supporting_statement_ids = {actor_statement.id}; actor_case.claims.push_back(document_hypothesis);
    check(common_investigation_case_from_json(common_investigation_case_to_json(actor_case),actor_reloaded,error) &&
        actor_reloaded.claims.back().supporting_statement_ids == document_hypothesis.supporting_statement_ids,
        "hypothesis statement evidence must remain separate from log event evidence across reload");
    common_investigation_source another = document; another.id = "document-b";
    actor_case.sources.push_back(another);
    nlohmann::ordered_json followup;
    common_investigation_analysis_budget followup_budget;
    check(common_investigation_source_followup(actor_case,{document.id,another.id},followup_budget,followup,error) &&
        followup["complete"] == true && followup["steps"].size() == 2,
        "new selected sources should offer extraction and comparison work without inferring semantic matches");
    followup_budget.max_expansions = 1;
    check(common_investigation_source_followup(actor_case,{document.id,another.id},followup_budget,followup,error) &&
        followup["complete"] == false && followup["explored_states"] == 1,
        "follow-up work must obey a scan budget even before suggestion generation");
    check(!common_investigation_source_followup(actor_case,{"unselected-source"},followup_budget,followup,error),
        "follow-up cannot manufacture access to a source outside the case");
    check(failures == 0, "all investigation contract checks must pass");
    return failures == 0 ? 0 : 1;
}
