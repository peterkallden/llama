#include "investigation-contracts.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace {
bool valid_id(const std::string & value) { return !value.empty() && value.size() <= 512; }
const char * entity_kind(common_investigation_entity_kind v) {
    switch(v) { case common_investigation_entity_kind::user:return "user"; case common_investigation_entity_kind::host:return "host"; case common_investigation_entity_kind::process:return "process"; case common_investigation_entity_kind::ip:return "ip"; case common_investigation_entity_kind::file:return "file"; case common_investigation_entity_kind::hash:return "hash"; case common_investigation_entity_kind::service:return "service"; case common_investigation_entity_kind::trace:return "trace"; case common_investigation_entity_kind::message:return "message"; case common_investigation_entity_kind::event:return "event"; default:return "unknown"; }
}
const char * relation_kind(common_investigation_relation_kind v) {
    switch(v) { case common_investigation_relation_kind::authenticated_to:return "authenticated_to"; case common_investigation_relation_kind::communicated_with:return "communicated_with"; case common_investigation_relation_kind::spawned:return "spawned"; case common_investigation_relation_kind::accessed:return "accessed"; case common_investigation_relation_kind::modified:return "modified"; case common_investigation_relation_kind::participates_in_trace:return "participates_in_trace"; case common_investigation_relation_kind::follows:return "follows"; case common_investigation_relation_kind::correlated_with:return "correlated_with"; default:return "unknown"; }
}
template<class E> bool enum_from(const std::string & s, E & out) {
    for (int i = 0; i < static_cast<int>(E::unknown); ++i) {
        const auto v = static_cast<E>(i);
        if constexpr (std::is_same_v<E, common_investigation_entity_kind>) { if (s == entity_kind(v)) { out=v; return true; } }
        else { if (s == relation_kind(v)) { out=v; return true; } }
    }
    out=E::unknown; return s=="unknown";
}
}

const char * common_investigation_relation_kind_name(common_investigation_relation_kind value) {
    return relation_kind(value);
}

nlohmann::ordered_json common_investigation_case_to_json(const common_investigation_case_revision & c) {
    using json = nlohmann::ordered_json;
    json entities=json::array(),events=json::array(),relations=json::array(),claims=json::array();
    for(const auto & e:c.entities) entities.push_back({{"id",e.id},{"source_id",e.source_id},{"kind",entity_kind(e.kind)},{"value",e.value},{"display_name",e.display_name},{"host_identity",e.host_identity},{"valid_from_ms",e.valid_from_ms},{"valid_to_ms",e.valid_to_ms}});
    for(const auto & e:c.events) events.push_back({{"id",e.id},{"source_id",e.source_id},{"source_index",e.source_index},{"source_document_id",e.source_document_id},{"event_type",e.event_type},{"time",{{"event_time_ms",e.time.event_time_ms},{"collected_time_ms",e.time.collected_time_ms},{"uncertainty_ms",e.time.uncertainty_ms},{"has_event_time",e.time.has_event_time}}},{"entity_ids",e.entity_ids},{"entity_roles",e.entity_roles},{"raw_resource_uri",e.raw_resource_uri},{"normalized_resource_uri",e.normalized_resource_uri}});
    for(const auto & r:c.relations) relations.push_back({{"id",r.id},{"source_id",r.source_id},{"from_entity_id",r.from_entity_id},{"to_entity_id",r.to_entity_id},{"kind",relation_kind(r.kind)},{"first_seen_ms",r.first_seen_ms},{"last_seen_ms",r.last_seen_ms},{"has_time",r.has_time},{"event_count",r.event_count},{"rule_id",r.rule_id},{"rule_revision",r.rule_revision},{"confidence_label",r.confidence_label},{"evidence_event_ids",r.evidence_event_ids}});
    for(const auto & c1:c.claims) claims.push_back({{"id",c1.id},{"text",c1.text},{"state",static_cast<int>(c1.state)},{"supporting_event_ids",c1.supporting_event_ids},{"contradicting_event_ids",c1.contradicting_event_ids},{"limitations",c1.limitations}});
    return {{"schema_version",c.schema_version},{"case_id",c.case_id},{"revision",c.revision},{"previous_revision_uri",c.previous_revision_uri},{"title",c.title},{"question",c.question},{"scope",{{"namespace_id",c.scope.namespace_id},{"session_id",c.scope.session_id},{"project_id",c.scope.project_id},{"turn_id",c.scope.turn_id}}},{"source_id",c.source_id},{"window_start_ms",c.window_start_ms},{"window_end_ms",c.window_end_ms},{"entities",std::move(entities)},{"events",std::move(events)},{"relations",std::move(relations)},{"claims",std::move(claims)},{"gaps",c.gaps},{"fingerprint",c.fingerprint}};
}

bool common_investigation_case_from_json(const nlohmann::ordered_json & j, common_investigation_case_revision & c, std::string & error) {
    c={};
    try {
        if(!j.is_object()||!j.contains("scope")||!j["scope"].is_object()) throw std::runtime_error("case and scope must be objects");
        c.schema_version=j.at("schema_version").get<uint32_t>(); c.case_id=j.at("case_id").get<std::string>(); c.revision=j.at("revision").get<uint64_t>(); c.previous_revision_uri=j.value("previous_revision_uri",std::string{}); c.title=j.at("title").get<std::string>(); c.question=j.value("question",std::string{});
        const auto & s=j.at("scope"); c.scope.namespace_id=s.value("namespace_id",std::string("local")); c.scope.session_id=s.value("session_id",std::string("default")); c.scope.project_id=s.value("project_id",std::string{}); c.scope.turn_id=s.value("turn_id",std::string{}); c.source_id=j.at("source_id").get<std::string>(); c.window_start_ms=j.at("window_start_ms").get<int64_t>(); c.window_end_ms=j.at("window_end_ms").get<int64_t>(); c.fingerprint=j.value("fingerprint",std::string{}); c.gaps=j.value("gaps",std::vector<std::string>{});
        for(const auto & x:j.at("entities")){common_investigation_entity e;e.id=x.at("id").get<std::string>();e.source_id=x.at("source_id").get<std::string>();if(!enum_from(x.at("kind").get<std::string>(),e.kind))throw std::runtime_error("unknown entity kind");e.value=x.at("value").get<std::string>();e.display_name=x.value("display_name",std::string{});e.host_identity=x.value("host_identity",std::string{});e.valid_from_ms=x.value("valid_from_ms",int64_t(0));e.valid_to_ms=x.value("valid_to_ms",int64_t(0));c.entities.push_back(std::move(e));}
        for(const auto & x:j.at("events")){common_investigation_event e;e.id=x.at("id").get<std::string>();e.source_id=x.at("source_id").get<std::string>();e.source_index=x.value("source_index",std::string{});e.source_document_id=x.at("source_document_id").get<std::string>();e.event_type=x.value("event_type",std::string{});const auto & t=x.at("time");e.time.event_time_ms=t.value("event_time_ms",int64_t(0));e.time.collected_time_ms=t.value("collected_time_ms",int64_t(0));e.time.uncertainty_ms=t.value("uncertainty_ms",uint64_t(0));e.time.has_event_time=t.value("has_event_time",false);e.entity_ids=x.value("entity_ids",std::vector<std::string>{});e.entity_roles=x.value("entity_roles",std::map<std::string,std::string>{});e.raw_resource_uri=x.value("raw_resource_uri",std::string{});e.normalized_resource_uri=x.value("normalized_resource_uri",std::string{});c.events.push_back(std::move(e));}
        for(const auto & x:j.at("relations")){common_investigation_relation r;r.id=x.at("id").get<std::string>();r.source_id=x.at("source_id").get<std::string>();r.from_entity_id=x.at("from_entity_id").get<std::string>();r.to_entity_id=x.at("to_entity_id").get<std::string>();if(!enum_from(x.at("kind").get<std::string>(),r.kind))throw std::runtime_error("unknown relation kind");r.first_seen_ms=x.value("first_seen_ms",int64_t(0));r.last_seen_ms=x.value("last_seen_ms",int64_t(0));r.has_time=x.value("has_time",false);r.event_count=x.value("event_count",uint64_t(0));r.rule_id=x.value("rule_id",std::string{});r.rule_revision=x.value("rule_revision",std::string{});r.confidence_label=x.value("confidence_label",std::string("observed"));r.evidence_event_ids=x.value("evidence_event_ids",std::vector<std::string>{});c.relations.push_back(std::move(r));}
        for(const auto & x:j.at("claims")){common_investigation_claim v;v.id=x.at("id").get<std::string>();v.text=x.at("text").get<std::string>();const int state=x.value("state",2);if(state<0||state>3)throw std::runtime_error("unknown claim state");v.state=static_cast<common_investigation_claim_state>(state);v.supporting_event_ids=x.value("supporting_event_ids",std::vector<std::string>{});v.contradicting_event_ids=x.value("contradicting_event_ids",std::vector<std::string>{});v.limitations=x.value("limitations",std::vector<std::string>{});c.claims.push_back(std::move(v));}
    } catch(const std::exception & e){error=std::string("invalid investigation case JSON: ")+e.what();return false;}
    return common_investigation_case_validate(c,error);
}

bool common_investigation_project_context(
        const common_investigation_case_revision & value,
        const common_investigation_context_binding & binding,
        nlohmann::ordered_json & projection, std::string & error) {
    if (!common_investigation_case_validate(value, error)) return false;
    if (binding.case_id != value.case_id || binding.revision != value.revision) {
        error = "investigation context case or revision is stale";
        return false;
    }
    projection = common_investigation_case_to_json(value);
    const auto select = [&](const std::vector<std::string> & refs, const char * selected_key,
            const char * source_key) {
        if (refs.size() > 64) {
            error = std::string("investigation context has too many ") + selected_key;
            return false;
        }
        auto & output = projection[selected_key];
        output = nlohmann::ordered_json::array();
        const auto & source = projection[source_key];
        std::set<std::string> unique;
        for (const auto & ref : refs) {
            if (ref.empty() || ref.size() > 512 || !unique.insert(ref).second) {
                error = std::string("investigation context has invalid ") + selected_key + " references";
                return false;
            }
            const auto found = std::find_if(source.begin(), source.end(), [&](const auto & item) {
                return item.is_object() && item.value("id", std::string{}) == ref;
            });
            if (found == source.end()) {
                error = std::string("investigation context contains an unknown ") + selected_key + " reference";
                return false;
            }
            output.push_back(*found);
        }
        return true;
    };
    if (!select(binding.entity_ids, "selected_entities", "entities") ||
            !select(binding.relation_ids, "selected_relations", "relations") ||
            !select(binding.event_ids, "selected_events", "events")) return false;
    projection.erase("entities");
    projection.erase("relations");
    projection.erase("events");
    projection.erase("claims");
    projection.erase("fingerprint");
    projection.erase("previous_revision_uri");
    projection.erase("scope");
    error.clear();
    return true;
}

bool common_investigation_case_validate(
        const common_investigation_case_revision & value, std::string & error) {
    error.clear();
    if (value.schema_version != 1 || !valid_id(value.case_id) || value.revision == 0 ||
            value.title.empty() || value.title.size() > 512 || value.question.size() > 8192 ||
            value.source_id.empty() || value.source_id.size() > 256 ||
            value.scope.namespace_id.empty() || value.scope.session_id.empty() ||
            value.window_start_ms >= value.window_end_ms ||
            value.entities.size() > 100000 || value.events.size() > 500000 ||
            value.relations.size() > 500000 || value.claims.size() > 10000) {
        error = "investigation case identity, time window, or bounded collection is invalid";
        return false;
    }
    std::unordered_set<std::string> entities, events, relations;
    for (const auto & entity : value.entities) {
        if (!valid_id(entity.id) || entity.source_id != value.source_id || entity.kind == common_investigation_entity_kind::unknown || entity.value.empty() ||
                entity.value.size() > 2048 || !entities.insert(entity.id).second ||
                (entity.valid_to_ms && entity.valid_from_ms >= entity.valid_to_ms)) {
            error = "investigation entity has invalid source identity or validity interval";
            return false;
        }
    }
    for (const auto & event : value.events) {
        if (!valid_id(event.id) || event.source_id != value.source_id ||
                event.source_document_id.empty() || event.source_document_id.size() > 512 || event.source_index.size() > 512 ||
                event.entity_ids.size() > 64 || !events.insert(event.id).second) {
            error = "investigation event has invalid identity or exceeds bounds";
            return false;
        }
        for (const auto & id : event.entity_ids) if (!entities.count(id)) {
            error = "investigation event references an unknown entity";
            return false;
        }
        for (const auto & role : event.entity_roles) if (role.first.empty() || role.first.size() > 64 ||
                !entities.count(role.second) || std::find(event.entity_ids.begin(), event.entity_ids.end(), role.second) == event.entity_ids.end()) {
            error = "investigation event role references an unknown or unlisted entity";
            return false;
        }
    }
    for (const auto & relation : value.relations) {
        if (!valid_id(relation.id) || relation.source_id != value.source_id || relation.kind == common_investigation_relation_kind::unknown ||
                !entities.count(relation.from_entity_id) || !entities.count(relation.to_entity_id) ||
                relation.rule_id.empty() || relation.rule_revision.empty() ||
                relation.event_count == 0 || relation.evidence_event_ids.empty() ||
                relation.evidence_event_ids.size() > 10000 ||
                !relations.insert(relation.id).second ||
                (relation.has_time && relation.first_seen_ms > relation.last_seen_ms)) {
            error = "investigation relation has invalid endpoints, provenance, time, or bounds";
            return false;
        }
        for (const auto & id : relation.evidence_event_ids) if (!events.count(id)) {
            error = "investigation relation references an unknown evidence event";
            return false;
        }
    }
    std::unordered_set<std::string> claims;
    for (const auto & claim : value.claims) {
        if (!valid_id(claim.id) || claim.text.empty() || claim.text.size() > 8192 ||
                (claim.state == common_investigation_claim_state::reviewed_conclusion && claim.supporting_event_ids.empty()) ||
                !claims.insert(claim.id).second) {
            error = "investigation claim identity or text is invalid";
            return false;
        }
        for (const auto & id : claim.supporting_event_ids) if (!events.count(id)) {
            error = "investigation claim references unknown supporting evidence";
            return false;
        }
        for (const auto & id : claim.contradicting_event_ids) if (!events.count(id)) {
            error = "investigation claim references unknown contradicting evidence";
            return false;
        }
    }
    return true;
}

bool common_investigation_graph_expand(
        const common_investigation_case_revision & value,
        const common_investigation_graph_query & query,
        common_investigation_graph_result & result, std::string & error) {
    result = {};
    if (!common_investigation_case_validate(value, error)) return false;
    if (query.case_id != value.case_id || query.case_revision != value.revision ||
            query.max_depth == 0 || query.max_depth > 8 || query.max_nodes == 0 ||
            query.max_nodes > 1000 || query.max_edges == 0 || query.max_edges > 4000 ||
            (query.window_start_ms && query.window_end_ms && query.window_start_ms >= query.window_end_ms)) {
        error = "investigation graph query is stale or outside supported bounds";
        return false;
    }
    const auto start = std::find_if(value.entities.begin(), value.entities.end(), [&](const auto & entity) {
        return entity.id == query.center_entity_id;
    });
    if (start == value.entities.end()) { error = "investigation graph center is unavailable in this case"; return false; }
    std::unordered_map<std::string, const common_investigation_entity *> entity_by_id;
    for (const auto & entity : value.entities) entity_by_id.emplace(entity.id, &entity);
    std::unordered_set<std::string> visible{query.center_entity_id};
    std::unordered_set<std::string> emitted_relations;
    std::deque<std::pair<std::string, size_t>> queue{{query.center_entity_id, 0}};
    while (!queue.empty()) {
        const auto current = queue.front(); queue.pop_front();
        if (current.second >= query.max_depth) continue;
        for (const auto & relation : value.relations) {
            if (relation.from_entity_id != current.first && relation.to_entity_id != current.first) continue;
            if (!query.relation_kinds.empty() && std::find(query.relation_kinds.begin(), query.relation_kinds.end(), relation.kind) == query.relation_kinds.end()) continue;
            if (query.window_start_ms && relation.has_time && relation.last_seen_ms < query.window_start_ms) continue;
            if (query.window_end_ms && relation.has_time && relation.first_seen_ms >= query.window_end_ms) continue;
            if (emitted_relations.insert(relation.id).second) {
                if (result.edges.size() >= query.max_edges) { result.truncated = true; break; }
                result.edges.push_back(relation);
            }
            const auto next = relation.from_entity_id == current.first ? relation.to_entity_id : relation.from_entity_id;
            if (visible.insert(next).second) {
                if (visible.size() > query.max_nodes) { visible.erase(next); result.truncated = true; break; }
                queue.emplace_back(next, current.second + 1);
            }
        }
        if (result.truncated) break;
    }
    result.case_revision = value.revision;
    for (const auto & entity : value.entities) if (visible.count(entity.id)) result.nodes.push_back(entity);
    if (result.truncated) result.limitations.push_back("graph expansion reached a configured node or relation bound");
    if (query.window_start_ms || query.window_end_ms) {
        result.limitations.push_back("relations without event time cannot be excluded by the time filter");
    }
    error.clear();
    return true;
}

bool common_investigation_graph_shortest_path(
        const common_investigation_case_revision & value, const std::string & start_id,
        const std::string & goal_id, const common_investigation_analysis_budget & budget,
        std::vector<std::string> & relation_ids, bool & complete, std::string & error) {
    relation_ids.clear(); complete = false;
    if (!common_investigation_case_validate(value, error)) return false;
    if (budget.max_expansions == 0 || budget.max_expansions > 100000 ||
            budget.max_path_length == 0 || budget.max_path_length > 64 ||
            std::none_of(value.entities.begin(), value.entities.end(), [&](const auto & e) { return e.id == start_id; }) ||
            std::none_of(value.entities.begin(), value.entities.end(), [&](const auto & e) { return e.id == goal_id; })) {
        error = "investigation path request has unknown endpoints or invalid budget";
        return false;
    }
    struct parent { std::string previous, relation; size_t depth = 0; };
    std::unordered_map<std::string, parent> visited;
    std::deque<std::string> queue{start_id};
    visited.emplace(start_id, parent{});
    size_t expanded = 0;
    bool found = start_id == goal_id;
    while (!queue.empty() && !found && expanded < budget.max_expansions) {
        auto current = std::move(queue.front()); queue.pop_front(); ++expanded;
        const auto depth = visited.at(current).depth;
        if (depth >= budget.max_path_length) continue;
        for (const auto & relation : value.relations) {
            std::string next;
            if (relation.from_entity_id == current) next = relation.to_entity_id;
            else if (relation.to_entity_id == current) next = relation.from_entity_id;
            else continue;
            if (visited.count(next)) continue;
            visited.emplace(next, parent{current, relation.id, depth + 1});
            if (next == goal_id) { found = true; break; }
            queue.push_back(std::move(next));
        }
    }
    complete = found || queue.empty();
    if (found) {
        for (auto at = goal_id; at != start_id; at = visited.at(at).previous) relation_ids.push_back(visited.at(at).relation);
        std::reverse(relation_ids.begin(), relation_ids.end());
    }
    error.clear();
    return true;
}

bool common_investigation_graph_analyze(
        const common_investigation_case_revision & value,
        const common_investigation_analysis_budget & budget,
        common_investigation_graph_metrics & result, std::string & error) {
    result = {};
    if (!common_investigation_case_validate(value, error)) return false;
    if (value.entities.size() > 512 || value.relations.size() > 4000 ||
            budget.max_expansions == 0 || budget.max_expansions > 100000) {
        error = "graph analysis exceeds the supported node, edge or expansion bound";
        return false;
    }
    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < value.entities.size(); ++i) index.emplace(value.entities[i].id, i);
    std::vector<std::vector<size_t>> adjacency(value.entities.size());
    for (size_t i = 0; i < value.relations.size(); ++i) {
        const auto & relation = value.relations[i];
        const auto from = index.find(relation.from_entity_id), to = index.find(relation.to_entity_id);
        if (from == index.end() || to == index.end()) {
            error = "graph analysis encountered a relation with an unknown endpoint";
            return false;
        }
        adjacency[from->second].push_back(to->second);
        if (from->second != to->second) adjacency[to->second].push_back(from->second);
    }
    for (auto & neighbors : adjacency) std::sort(neighbors.begin(), neighbors.end());
    result.case_revision = value.revision;
    result.nodes.resize(value.entities.size());
    for (size_t i = 0; i < value.entities.size(); ++i) {
        result.nodes[i].entity_id = value.entities[i].id;
        result.nodes[i].degree = adjacency[i].size();
    }
    size_t component = 0;
    std::vector<bool> assigned(value.entities.size(), false);
    for (size_t root = 0; root < value.entities.size(); ++root) {
        if (assigned[root]) continue;
        std::deque<size_t> queue{root}; assigned[root] = true;
        while (!queue.empty()) {
            const size_t current = queue.front(); queue.pop_front();
            result.nodes[current].component = component;
            for (const size_t next : adjacency[current]) {
                if (++result.explored_states > budget.max_expansions) {
                    result.complete = false; error.clear(); return true;
                }
                if (!assigned[next]) { assigned[next] = true; queue.push_back(next); }
            }
        }
        ++component;
    }

    std::vector<double> centrality(value.entities.size(), 0.0);
    bool exhausted = false;
    for (size_t source = 0; source < value.entities.size() && !exhausted; ++source) {
        std::vector<std::vector<size_t>> predecessors(value.entities.size());
        std::vector<double> paths(value.entities.size(), 0.0);
        std::vector<int64_t> distance(value.entities.size(), -1);
        std::vector<size_t> stack;
        std::deque<size_t> queue{source};
        paths[source] = 1.0; distance[source] = 0;
        while (!queue.empty() && !exhausted) {
            const size_t current = queue.front(); queue.pop_front(); stack.push_back(current);
            for (const size_t next : adjacency[current]) {
                if (++result.explored_states > budget.max_expansions) { exhausted = true; break; }
                if (distance[next] < 0) {
                    distance[next] = distance[current] + 1;
                    queue.push_back(next);
                }
                if (distance[next] == distance[current] + 1) {
                    paths[next] += paths[current];
                    predecessors[next].push_back(current);
                }
            }
        }
        if (exhausted) break;
        std::vector<double> dependency(value.entities.size(), 0.0);
        while (!stack.empty()) {
            const size_t node = stack.back(); stack.pop_back();
            for (const size_t previous : predecessors[node]) {
                if (paths[node] > 0.0) dependency[previous] +=
                    (paths[previous] / paths[node]) * (1.0 + dependency[node]);
            }
            if (node != source) centrality[node] += dependency[node];
        }
    }
    result.complete = !exhausted;
    if (result.complete) {
        const double normalization = value.entities.size() > 2
            ? 1.0 / (static_cast<double>(value.entities.size() - 1) *
                static_cast<double>(value.entities.size() - 2)) : 0.0;
        for (size_t i = 0; i < centrality.size(); ++i) result.nodes[i].betweenness = centrality[i] * normalization;
        result.betweenness_normalized = true;
    }
    error.clear();
    return true;
}

bool common_investigation_temporal_path(
        const common_investigation_case_revision & value, const std::string & start_id,
        const std::string & goal_id, int64_t not_before_ms,
        const common_investigation_analysis_budget & budget,
        std::vector<std::string> & relation_ids, bool & complete, std::string & error) {
    relation_ids.clear(); complete = false;
    if (!common_investigation_case_validate(value, error)) return false;
    if (budget.max_expansions == 0 || budget.max_expansions > 100000 ||
            budget.max_path_length == 0 || budget.max_path_length > 64 ||
            std::none_of(value.entities.begin(), value.entities.end(), [&](const auto & entity) { return entity.id == start_id; }) ||
            std::none_of(value.entities.begin(), value.entities.end(), [&](const auto & entity) { return entity.id == goal_id; })) {
        error = "temporal path request has unknown endpoints or invalid budget";
        return false;
    }
    struct state { std::string entity; int64_t time = 0; size_t depth = 0; };
    const size_t no_parent = std::numeric_limits<size_t>::max();
    struct parent { size_t previous = std::numeric_limits<size_t>::max(); std::string relation; };
    std::vector<state> states{{start_id, not_before_ms, 0}};
    std::vector<parent> parents(1);
    std::deque<size_t> queue{0};
    std::set<std::pair<std::string, int64_t>> visited{{start_id, not_before_ms}};
    size_t found = start_id == goal_id ? 0 : no_parent;
    size_t expanded = 0;
    while (!queue.empty() && found == no_parent && expanded < budget.max_expansions) {
        const size_t current_index = queue.front(); queue.pop_front(); ++expanded;
        const auto current = states[current_index];
        if (current.depth >= budget.max_path_length) continue;
        for (const auto & relation : value.relations) {
            if (!relation.has_time || relation.first_seen_ms < current.time) continue;
            std::string next;
            if (relation.from_entity_id == current.entity) next = relation.to_entity_id;
            else continue;
            const int64_t next_time = std::max(current.time, relation.last_seen_ms);
            if (!visited.emplace(next, next_time).second) continue;
            const size_t next_index = states.size();
            states.push_back({next, next_time, current.depth + 1});
            parents.push_back({current_index, relation.id});
            if (next == goal_id) { found = next_index; break; }
            queue.push_back(next_index);
        }
    }
    complete = found != no_parent || queue.empty();
    if (found != no_parent) {
        for (size_t at = found; at != 0; at = parents[at].previous) relation_ids.push_back(parents[at].relation);
        std::reverse(relation_ids.begin(), relation_ids.end());
    }
    error.clear();
    return true;
}

bool common_investigation_build_relations(
        const std::string & source_id,
        const std::vector<common_investigation_entity> & entities,
        const std::vector<common_investigation_event> & events,
        const std::vector<common_investigation_relation_rule> & rules,
        std::vector<common_investigation_relation> & relations,
        std::vector<std::string> & limitations,
        std::string & error) {
    using key_type = std::tuple<std::string, std::string, std::string,
        common_investigation_relation_kind, std::string>;
    relations.clear();
    limitations.clear();
    error.clear();
    if (source_id.empty() || source_id.size() > 256 || entities.size() > 100000 ||
            events.size() > 500000 || rules.size() > 256) {
        error = "relation build input exceeds supported identity or size bounds";
        return false;
    }
    std::unordered_map<std::string, const common_investigation_entity *> by_id;
    for (const auto & entity : entities) {
        if (!valid_id(entity.id) || entity.source_id != source_id ||
                entity.kind == common_investigation_entity_kind::unknown ||
                !by_id.emplace(entity.id, &entity).second) {
            error = "relation build contains duplicate or invalid source-scoped entities";
            return false;
        }
    }
    std::unordered_set<std::string> event_ids;
    for (const auto & event : events) {
        if (!valid_id(event.id) || event.source_id != source_id ||
                event.event_type.empty() || event.event_type.size() > 256 ||
                !event_ids.insert(event.id).second) {
            error = "relation build contains duplicate or invalid source-scoped events";
            return false;
        }
        for (const auto & id : event.entity_ids) if (!by_id.count(id)) {
            error = "relation build event references an unknown entity";
            return false;
        }
        for (const auto & role : event.entity_roles) {
            if (role.first.empty() || role.first.size() > 64 || !by_id.count(role.second) ||
                    std::find(event.entity_ids.begin(), event.entity_ids.end(), role.second) == event.entity_ids.end()) {
                error = "relation build event role references an unknown or unlisted entity";
                return false;
            }
        }
    }
    std::unordered_set<std::string> rule_ids;
    for (const auto & rule : rules) {
        if (!valid_id(rule.id) || rule.event_type.empty() || rule.event_type.size() > 256 ||
                rule.from_kind == common_investigation_entity_kind::unknown ||
                rule.to_kind == common_investigation_entity_kind::unknown ||
                rule.relation_kind == common_investigation_relation_kind::unknown ||
                (rule.from_role.empty() != rule.to_role.empty()) ||
                rule.from_role.size() > 64 || rule.to_role.size() > 64 ||
                !rule_ids.insert(rule.id).second) {
            error = "relation build rule is invalid or duplicated";
            return false;
        }
    }
    std::map<key_type, common_investigation_relation> aggregated;
    bool missing_time = false;
    for (const auto & event : events) {
        for (const auto & rule : rules) {
            if (rule.event_type != event.event_type) continue;
            std::vector<const common_investigation_entity *> from, to;
            if (!rule.from_role.empty() || !rule.to_role.empty()) {
                const auto from_role = event.entity_roles.find(rule.from_role);
                const auto to_role = event.entity_roles.find(rule.to_role);
                if (from_role == event.entity_roles.end() || to_role == event.entity_roles.end()) continue;
                const auto * left = by_id.at(from_role->second);
                const auto * right = by_id.at(to_role->second);
                if (left->kind == rule.from_kind && right->kind == rule.to_kind) {
                    from.push_back(left); to.push_back(right);
                }
            } else for (const auto & id : event.entity_ids) {
                const auto * entity = by_id.at(id);
                if (entity->kind == rule.from_kind) from.push_back(entity);
                if (entity->kind == rule.to_kind) to.push_back(entity);
            }
            for (const auto * left : from) for (const auto * right : to) {
                if (left->id == right->id) continue;
                const key_type key{source_id, rule.id, left->id, rule.relation_kind, right->id};
                auto [it, inserted] = aggregated.emplace(key, common_investigation_relation{});
                auto & relation = it->second;
                if (inserted) {
                    relation.source_id = source_id;
                    relation.from_entity_id = left->id;
                    relation.to_entity_id = right->id;
                    relation.kind = rule.relation_kind;
                    relation.rule_id = rule.id;
                    relation.rule_revision = "1";
                    relation.confidence_label = "observed";
                }
                if (std::find(relation.evidence_event_ids.begin(), relation.evidence_event_ids.end(), event.id) == relation.evidence_event_ids.end()) {
                    relation.evidence_event_ids.push_back(event.id);
                    ++relation.event_count;
                }
                if (event.time.has_event_time) {
                    if (!relation.has_time) {
                        relation.first_seen_ms = event.time.event_time_ms;
                        relation.last_seen_ms = event.time.event_time_ms;
                        relation.has_time = true;
                    } else {
                        relation.first_seen_ms = std::min(relation.first_seen_ms, event.time.event_time_ms);
                        relation.last_seen_ms = std::max(relation.last_seen_ms, event.time.event_time_ms);
                    }
                } else {
                    missing_time = true;
                }
            }
        }
    }
    std::unordered_set<std::string> relation_ids;
    for (auto & [key, relation] : aggregated) {
        std::string identity;
        for (const auto & part : {std::get<0>(key), std::get<1>(key), std::get<2>(key),
                std::string(relation_kind(std::get<3>(key))), std::get<4>(key)}) {
            identity += std::to_string(part.size()) + ":" + part + ";";
        }
        uint64_t hash = 1469598103934665603ULL;
        for (unsigned char byte : identity) { hash ^= byte; hash *= 1099511628211ULL; }
        static const char hex[] = "0123456789abcdef";
        std::string suffix(16, '0');
        for (int i = 15; i >= 0; --i) { suffix[static_cast<size_t>(i)] = hex[hash & 0xf]; hash >>= 4; }
        relation.id = "rel:" + suffix;
        if (!relation_ids.insert(relation.id).second) {
            error = "stable relation identifier collision";
            relations.clear();
            return false;
        }
        std::sort(relation.evidence_event_ids.begin(), relation.evidence_event_ids.end());
        relations.push_back(std::move(relation));
    }
    if (missing_time) limitations.push_back("one_or_more_relation_events_have_unknown_event_time");
    return true;
}
