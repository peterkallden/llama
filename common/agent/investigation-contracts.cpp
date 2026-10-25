#include "investigation-contracts.h"
#include "investigation-sources.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace {
bool valid_id(const std::string & value) { return !value.empty() && value.size() <= 512; }
const char * source_kind(common_investigation_source_kind value) {
    switch (value) {
        case common_investigation_source_kind::search_connector: return "search_connector";
        case common_investigation_source_kind::document: return "document";
        case common_investigation_source_kind::log_file: return "log_file";
    }
    return "document";
}
const char * statement_kind(common_investigation_statement_kind value) {
    switch (value) {
        case common_investigation_statement_kind::statement: return "statement";
        case common_investigation_statement_kind::reported_event: return "reported_event";
        case common_investigation_statement_kind::definition: return "definition";
        case common_investigation_statement_kind::interpretation: return "interpretation";
    }
    return "statement";
}
const char * comparison_kind(common_investigation_comparison_kind value) {
    switch (value) {
        case common_investigation_comparison_kind::supports: return "supports";
        case common_investigation_comparison_kind::contradicts: return "contradicts";
        case common_investigation_comparison_kind::qualifies: return "qualifies";
        case common_investigation_comparison_kind::same_topic: return "same_topic";
        case common_investigation_comparison_kind::source_dependency: return "source_dependency";
        case common_investigation_comparison_kind::inconclusive: return "inconclusive";
    }
    return "inconclusive";
}
template<class T> bool enum_named(const std::string & value, T & out) {
    constexpr int count = std::is_same_v<T, common_investigation_source_kind> ? 3 :
        std::is_same_v<T, common_investigation_statement_kind> ? 4 : 6;
    for (int i = 0; i < count; ++i) {
        const auto candidate = static_cast<T>(i);
        const char * name;
        if constexpr (std::is_same_v<T, common_investigation_source_kind>) name = source_kind(candidate);
        else if constexpr (std::is_same_v<T, common_investigation_statement_kind>) name = statement_kind(candidate);
        else name = comparison_kind(candidate);
        if (value == name) { out = candidate; return true; }
    }
    return false;
}
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

const char * common_investigation_entity_kind_name(common_investigation_entity_kind value) {
    return entity_kind(value);
}

nlohmann::ordered_json common_investigation_case_to_json(const common_investigation_case_revision & c) {
    using json = nlohmann::ordered_json;
    json entities=json::array(),events=json::array(),relations=json::array(),claims=json::array();
    json sources=json::array(),statements=json::array(),comparisons=json::array(),analyses=json::array();
    json actor_mentions=json::array(), actors=json::array(), evidence_links=json::array(), association_reviews=json::array();
    for(const auto & e:c.entities) entities.push_back({{"id",e.id},{"source_id",e.source_id},{"kind",entity_kind(e.kind)},{"value",e.value},{"display_name",e.display_name},{"host_identity",e.host_identity},{"valid_from_ms",e.valid_from_ms},{"valid_to_ms",e.valid_to_ms}});
    for (const auto & event : c.events) {
        events.push_back({{"id",event.id},{"source_id",event.source_id},{"source_index",event.source_index},
            {"source_byte_offset",event.source_byte_offset},{"source_byte_length",event.source_byte_length},
            {"source_line",event.source_line},{"parser_version",event.parser_version},
            {"source_document_id",event.source_document_id},{"event_type",event.event_type},
            {"time",{{"event_time_ms",event.time.event_time_ms},{"collected_time_ms",event.time.collected_time_ms},
                {"uncertainty_ms",event.time.uncertainty_ms},{"has_event_time",event.time.has_event_time}}},
            {"entity_ids",event.entity_ids},{"entity_roles",event.entity_roles},
            {"raw_resource_uri",event.raw_resource_uri},{"normalized_resource_uri",event.normalized_resource_uri}});
        // Keep pre-outcome revision serialization unchanged; never invent legacy success.
        if (!event.outcome.empty()) events.back()["outcome"] = event.outcome;
    }
    for(const auto & r:c.relations) relations.push_back({{"id",r.id},{"source_id",r.source_id},{"from_entity_id",r.from_entity_id},{"to_entity_id",r.to_entity_id},{"kind",relation_kind(r.kind)},{"first_seen_ms",r.first_seen_ms},{"last_seen_ms",r.last_seen_ms},{"has_time",r.has_time},{"event_count",r.event_count},{"rule_id",r.rule_id},{"rule_revision",r.rule_revision},{"confidence_label",r.confidence_label},{"evidence_event_ids",r.evidence_event_ids}});
    for(const auto & c1:c.claims) claims.push_back({{"id",c1.id},{"text",c1.text},{"state",static_cast<int>(c1.state)},{"supporting_event_ids",c1.supporting_event_ids},{"contradicting_event_ids",c1.contradicting_event_ids},{"limitations",c1.limitations},{"supporting_statement_ids",c1.supporting_statement_ids},{"contradicting_statement_ids",c1.contradicting_statement_ids}});
    for (const auto & s : c.sources) sources.push_back({{"id",s.id},{"kind",source_kind(s.kind)},
        {"resource_uri",s.resource_uri},{"representation_uri",s.representation_uri},{"provider_id",s.provider_id},
        {"display_name",s.display_name},{"mime_type",s.mime_type},{"content_hash",s.content_hash},
        {"representation_hash",s.representation_hash},{"original_uri",s.original_uri},
        {"original_parent_uri",s.original_parent_uri},{"processor_version",s.processor_version},
        {"processing_status",s.processing_status},{"processing_limitations",s.processing_limitations},
        {"source_revision",s.source_revision},{"complete",s.complete},
        {"log_parser_profile_json",s.log_parser_profile_json},{"log_import_next_record",s.log_import_next_record},
        {"log_import_rejected_records",s.log_import_rejected_records},{"log_import_exhausted",s.log_import_exhausted}});
    for (const auto & s : c.statements) statements.push_back({{"id",s.id},{"kind",statement_kind(s.kind)},
        {"text",s.text},{"anchor",{{"source_id",s.anchor.source_id},{"resource_uri",s.anchor.resource_uri},
        {"byte_offset",s.anchor.byte_offset},{"byte_length",s.anchor.byte_length},{"page_number",s.anchor.page_number},
        {"quote",s.anchor.quote},{"quote_verified",s.anchor.quote_verified}}},{"actor_ids",s.actor_ids},
        {"reported_time_text",s.reported_time_text},{"extraction_origin",s.extraction_origin}});
    for (const auto & c1 : c.comparisons) comparisons.push_back({{"id",c1.id},
        {"left_statement_id",c1.left_statement_id},{"right_statement_id",c1.right_statement_id},
        {"kind",comparison_kind(c1.kind)},{"rationale",c1.rationale},{"source_ids",c1.source_ids},
        {"assessment_origin",c1.assessment_origin}});
    for (const auto & a : c.analyses) analyses.push_back({{"id",a.id},{"kind",a.kind},
        {"algorithm_version",a.algorithm_version},{"published_revision",a.published_revision},{"source_ids",a.source_ids},
        {"parameters_fingerprint",a.parameters_fingerprint},{"result_json",a.result_json},{"completeness",a.completeness},
        {"limitations",a.limitations}});
    for (const auto & m : c.actor_mentions) actor_mentions.push_back({{"id",m.id},
        {"statement_id",m.statement_id},{"name",m.name},{"kind",m.kind},
        {"anchor",{{"source_id",m.anchor.source_id},{"resource_uri",m.anchor.resource_uri},
        {"byte_offset",m.anchor.byte_offset},{"byte_length",m.anchor.byte_length},
        {"page_number",m.anchor.page_number},{"quote",m.anchor.quote},{"quote_verified",m.anchor.quote_verified}}}});
    for (const auto & a : c.actors) actors.push_back({{"id",a.id},{"display_name",a.display_name},{"mention_ids",a.mention_ids}});
    for (const auto & l : c.evidence_links) evidence_links.push_back({{"id",l.id},{"kind",l.kind},
        {"left_id",l.left_id},{"right_id",l.right_id},{"supporting_statement_ids",l.supporting_statement_ids},
        {"rationale",l.rationale},{"assessment_origin",l.assessment_origin}});
    for (const auto & r : c.association_reviews) association_reviews.push_back({{"id",r.id},
        {"operation_id",r.operation_id},{"parameters_fingerprint",r.parameters_fingerprint},
        {"left_actor_id",r.left_actor_id},{"right_actor_id",r.right_actor_id},{"kind",r.kind},
        {"decision",r.decision},{"relation_type",r.relation_type},{"comment",r.comment},
        {"reviewer_id",r.reviewer_id},{"created_at_ms",r.created_at_ms},
        {"published_revision",r.published_revision},{"supersedes_id",r.supersedes_id}});
    return {{"schema_version",c.schema_version},{"case_id",c.case_id},{"revision",c.revision},{"previous_revision_uri",c.previous_revision_uri},{"title",c.title},{"question",c.question},{"scope",{{"namespace_id",c.scope.namespace_id},{"session_id",c.scope.session_id},{"project_id",c.scope.project_id},{"turn_id",c.scope.turn_id}}},{"source_id",c.source_id},{"window_start_ms",c.window_start_ms},{"window_end_ms",c.window_end_ms},{"entities",std::move(entities)},{"events",std::move(events)},{"relations",std::move(relations)},{"claims",std::move(claims)},{"sources",std::move(sources)},{"statements",std::move(statements)},{"comparisons",std::move(comparisons)},{"analyses",std::move(analyses)},{"actor_mentions",std::move(actor_mentions)},{"actors",std::move(actors)},{"evidence_links",std::move(evidence_links)},{"association_reviews",std::move(association_reviews)},{"gaps",c.gaps},{"fingerprint",c.fingerprint}};
}

bool common_investigation_case_from_json(const nlohmann::ordered_json & j, common_investigation_case_revision & c, std::string & error) {
    c={};
    try {
        if(!j.is_object()||!j.contains("scope")||!j["scope"].is_object()) throw std::runtime_error("case and scope must be objects");
        c.schema_version=j.at("schema_version").get<uint32_t>();
        if (c.schema_version == 1) c.schema_version = 2;
        c.case_id=j.at("case_id").get<std::string>(); c.revision=j.at("revision").get<uint64_t>(); c.previous_revision_uri=j.value("previous_revision_uri",std::string{}); c.title=j.at("title").get<std::string>(); c.question=j.value("question",std::string{});
        const auto & s=j.at("scope"); c.scope.namespace_id=s.value("namespace_id",std::string("local")); c.scope.session_id=s.value("session_id",std::string("default")); c.scope.project_id=s.value("project_id",std::string{}); c.scope.turn_id=s.value("turn_id",std::string{}); c.source_id=j.at("source_id").get<std::string>(); c.window_start_ms=j.at("window_start_ms").get<int64_t>(); c.window_end_ms=j.at("window_end_ms").get<int64_t>(); c.fingerprint=j.value("fingerprint",std::string{}); c.gaps=j.value("gaps",std::vector<std::string>{});
        for(const auto & x:j.at("entities")){common_investigation_entity e;e.id=x.at("id").get<std::string>();e.source_id=x.at("source_id").get<std::string>();if(!enum_from(x.at("kind").get<std::string>(),e.kind))throw std::runtime_error("unknown entity kind");e.value=x.at("value").get<std::string>();e.display_name=x.value("display_name",std::string{});e.host_identity=x.value("host_identity",std::string{});e.valid_from_ms=x.value("valid_from_ms",int64_t(0));e.valid_to_ms=x.value("valid_to_ms",int64_t(0));c.entities.push_back(std::move(e));}
        for(const auto & x:j.at("events")){common_investigation_event e;e.id=x.at("id").get<std::string>();e.source_id=x.at("source_id").get<std::string>();e.source_index=x.value("source_index",std::string{});e.source_byte_offset=x.value("source_byte_offset",uint64_t(0));e.source_byte_length=x.value("source_byte_length",uint64_t(0));e.source_line=x.value("source_line",uint64_t(0));e.parser_version=x.value("parser_version",std::string{});e.source_document_id=x.at("source_document_id").get<std::string>();e.event_type=x.value("event_type",std::string{});e.outcome=x.value("outcome",std::string{});const auto & t=x.at("time");e.time.event_time_ms=t.value("event_time_ms",int64_t(0));e.time.collected_time_ms=t.value("collected_time_ms",int64_t(0));e.time.uncertainty_ms=t.value("uncertainty_ms",uint64_t(0));e.time.has_event_time=t.value("has_event_time",false);e.entity_ids=x.value("entity_ids",std::vector<std::string>{});e.entity_roles=x.value("entity_roles",std::map<std::string,std::string>{});e.raw_resource_uri=x.value("raw_resource_uri",std::string{});e.normalized_resource_uri=x.value("normalized_resource_uri",std::string{});c.events.push_back(std::move(e));}
        for(const auto & x:j.at("relations")){common_investigation_relation r;r.id=x.at("id").get<std::string>();r.source_id=x.at("source_id").get<std::string>();r.from_entity_id=x.at("from_entity_id").get<std::string>();r.to_entity_id=x.at("to_entity_id").get<std::string>();if(!enum_from(x.at("kind").get<std::string>(),r.kind))throw std::runtime_error("unknown relation kind");r.first_seen_ms=x.value("first_seen_ms",int64_t(0));r.last_seen_ms=x.value("last_seen_ms",int64_t(0));r.has_time=x.value("has_time",false);r.event_count=x.value("event_count",uint64_t(0));r.rule_id=x.value("rule_id",std::string{});r.rule_revision=x.value("rule_revision",std::string{});r.confidence_label=x.value("confidence_label",std::string("observed"));r.evidence_event_ids=x.value("evidence_event_ids",std::vector<std::string>{});c.relations.push_back(std::move(r));}
        for(const auto & x:j.at("claims")){common_investigation_claim v;v.id=x.at("id").get<std::string>();v.text=x.at("text").get<std::string>();const int state=x.value("state",2);if(state<0||state>3)throw std::runtime_error("unknown claim state");v.state=static_cast<common_investigation_claim_state>(state);v.supporting_event_ids=x.value("supporting_event_ids",std::vector<std::string>{});v.contradicting_event_ids=x.value("contradicting_event_ids",std::vector<std::string>{});v.limitations=x.value("limitations",std::vector<std::string>{});v.supporting_statement_ids=x.value("supporting_statement_ids",std::vector<std::string>{});v.contradicting_statement_ids=x.value("contradicting_statement_ids",std::vector<std::string>{});c.claims.push_back(std::move(v));}
        if (j.contains("sources")) for (const auto & x : j.at("sources")) {
            common_investigation_source s; s.id=x.at("id").get<std::string>();
            if (!enum_named(x.at("kind").get<std::string>(),s.kind)) throw std::runtime_error("unknown investigation source kind");
            s.resource_uri=x.value("resource_uri",std::string{}); s.representation_uri=x.value("representation_uri",std::string{});
            s.provider_id=x.value("provider_id",std::string{}); s.display_name=x.value("display_name",std::string{});
            s.mime_type=x.value("mime_type",std::string{}); s.content_hash=x.value("content_hash",std::string{});
            s.representation_hash=x.value("representation_hash",std::string{});
            s.original_uri=x.value("original_uri",std::string{});
            s.original_parent_uri=x.value("original_parent_uri",std::string{});
            s.processor_version=x.value("processor_version",std::string{});
            s.processing_status=x.value("processing_status",std::string("ready"));
            s.processing_limitations=x.value("processing_limitations",std::vector<std::string>{});
            s.source_revision=x.value("source_revision",uint64_t(1));
            s.complete=x.value("complete",true);
            s.log_parser_profile_json=x.value("log_parser_profile_json",std::string{});
            s.log_import_next_record=x.value("log_import_next_record",uint64_t(0));
            s.log_import_rejected_records=x.value("log_import_rejected_records",uint64_t(0));
            s.log_import_exhausted=x.value("log_import_exhausted",false); c.sources.push_back(std::move(s));
        }
        if (j.contains("statements")) for (const auto & x : j.at("statements")) {
            common_investigation_statement s; s.id=x.at("id").get<std::string>();
            if (!enum_named(x.at("kind").get<std::string>(),s.kind)) throw std::runtime_error("unknown investigation statement kind");
            s.text=x.at("text").get<std::string>(); const auto & a=x.at("anchor");
            s.anchor.source_id=a.at("source_id").get<std::string>(); s.anchor.resource_uri=a.at("resource_uri").get<std::string>();
            s.anchor.byte_offset=a.at("byte_offset").get<uint64_t>(); s.anchor.byte_length=a.at("byte_length").get<uint64_t>();
            s.anchor.page_number=a.value("page_number",int64_t(0)); s.anchor.quote=a.at("quote").get<std::string>();
            s.anchor.quote_verified=a.value("quote_verified",false); s.actor_ids=x.value("actor_ids",std::vector<std::string>{});
            s.reported_time_text=x.value("reported_time_text",std::string{}); s.extraction_origin=x.value("extraction_origin",std::string("model_proposal"));
            c.statements.push_back(std::move(s));
        }
        if (j.contains("comparisons")) for (const auto & x : j.at("comparisons")) {
            common_investigation_comparison v; v.id=x.at("id").get<std::string>();
            v.left_statement_id=x.at("left_statement_id").get<std::string>(); v.right_statement_id=x.at("right_statement_id").get<std::string>();
            if (!enum_named(x.at("kind").get<std::string>(),v.kind)) throw std::runtime_error("unknown investigation comparison kind");
            v.rationale=x.value("rationale",std::string{}); v.source_ids=x.value("source_ids",std::vector<std::string>{});
            v.assessment_origin=x.value("assessment_origin",std::string("model_proposal")); c.comparisons.push_back(std::move(v));
        }
        if (j.contains("actor_mentions")) for (const auto & x : j.at("actor_mentions")) {
            common_investigation_actor_mention m;
            m.id=x.at("id").get<std::string>(); m.statement_id=x.at("statement_id").get<std::string>();
            m.name=x.at("name").get<std::string>(); m.kind=x.value("kind",std::string("unknown"));
            const auto & a=x.at("anchor"); m.anchor.source_id=a.at("source_id").get<std::string>();
            m.anchor.resource_uri=a.at("resource_uri").get<std::string>();
            m.anchor.byte_offset=a.at("byte_offset").get<uint64_t>(); m.anchor.byte_length=a.at("byte_length").get<uint64_t>();
            m.anchor.page_number=a.value("page_number",int64_t(0)); m.anchor.quote=a.at("quote").get<std::string>();
            m.anchor.quote_verified=a.value("quote_verified",false); c.actor_mentions.push_back(std::move(m));
        }
        if (j.contains("actors")) for (const auto & x : j.at("actors")) {
            common_investigation_actor_candidate a; a.id=x.at("id").get<std::string>();
            a.display_name=x.at("display_name").get<std::string>(); a.mention_ids=x.at("mention_ids").get<std::vector<std::string>>();
            c.actors.push_back(std::move(a));
        }
        if (j.contains("evidence_links")) for (const auto & x : j.at("evidence_links")) {
            common_investigation_evidence_link l; l.id=x.at("id").get<std::string>(); l.kind=x.at("kind").get<std::string>();
            l.left_id=x.at("left_id").get<std::string>(); l.right_id=x.at("right_id").get<std::string>();
            l.supporting_statement_ids=x.at("supporting_statement_ids").get<std::vector<std::string>>();
            l.rationale=x.value("rationale",std::string{}); l.assessment_origin=x.value("assessment_origin",std::string("model_proposal"));
            c.evidence_links.push_back(std::move(l));
        }
        if (j.contains("association_reviews")) for (const auto & x : j.at("association_reviews")) {
            common_investigation_association_review r;
            r.id=x.at("id").get<std::string>(); r.operation_id=x.at("operation_id").get<std::string>();
            r.parameters_fingerprint=x.at("parameters_fingerprint").get<std::string>();
            r.left_actor_id=x.at("left_actor_id").get<std::string>(); r.right_actor_id=x.at("right_actor_id").get<std::string>();
            r.kind=x.at("kind").get<std::string>(); r.decision=x.at("decision").get<std::string>();
            r.relation_type=x.value("relation_type",std::string{}); r.comment=x.value("comment",std::string{});
            r.reviewer_id=x.at("reviewer_id").get<std::string>(); r.created_at_ms=x.at("created_at_ms").get<int64_t>();
            r.published_revision=x.at("published_revision").get<uint64_t>(); r.supersedes_id=x.value("supersedes_id",std::string{});
            c.association_reviews.push_back(std::move(r));
        }
        if (j.contains("analyses")) for (const auto & x : j.at("analyses")) {
            common_investigation_analysis_record a; a.id=x.at("id").get<std::string>(); a.kind=x.at("kind").get<std::string>();
            a.algorithm_version=x.at("algorithm_version").get<std::string>(); a.source_ids=x.value("source_ids",std::vector<std::string>{});
            a.published_revision=x.value("published_revision",c.revision);
            a.parameters_fingerprint=x.value("parameters_fingerprint",std::string{}); a.result_json=x.value("result_json",std::string{}); a.completeness=x.value("completeness",std::string("complete"));
            a.limitations=x.value("limitations",std::vector<std::string>{}); c.analyses.push_back(std::move(a));
        }
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
    if (binding.source_ids.size() > 32) {
        error = "investigation context has too many selected sources";
        return false;
    }
    {
        std::set<std::string> unique;
        auto & output = projection["selected_sources"];
        output = nlohmann::ordered_json::array();
        const auto & all_sources = projection["sources"];
        for (const auto & ref : binding.source_ids) {
            if (ref.empty() || ref.size() > 512 || !unique.insert(ref).second) {
                error = "investigation context has invalid selected source references";
                return false;
            }
            const auto found = std::find_if(all_sources.begin(), all_sources.end(), [&](const auto & item) {
                return item.is_object() && (item.value("id", std::string{}) == ref ||
                    item.value("provider_id", std::string{}) == ref);
            });
            if (found == all_sources.end() && ref == value.source_id && !value.source_id.empty()) {
                output.push_back({{"id", value.source_id}, {"kind", "search_connector"},
                    {"provider_id", value.source_id}, {"display_name", value.source_id}});
                continue;
            }
            if (found == all_sources.end()) {
                error = "investigation context contains an unknown selected source";
                return false;
            }
            output.push_back(*found);
        }
    }
    static const std::set<std::string> allowed_profiles = {"identity_analysis", "communication_analysis", "actor_relationship", "narrative_comparison",
        "document_provenance", "hypothesis_testing", "event_chain", "integration_trace", "report_composition", "source_followup"};
    if (!binding.analysis_profile.empty() && !allowed_profiles.count(binding.analysis_profile)) {
        error = "investigation context analysis profile is not registered";
        return false;
    }
    for (auto & source : projection["selected_sources"]) {
        source["has_log_parser_profile"] = !source.value("log_parser_profile_json",std::string{}).empty();
        source.erase("log_parser_profile_json");
    }
    projection["analysis_profile"] = binding.analysis_profile;
    projection.erase("entities");
    projection.erase("relations");
    projection.erase("events");
    projection.erase("claims");
    projection.erase("sources");
    projection.erase("statements");
    projection.erase("comparisons");
    projection.erase("analyses");
    projection.erase("actors");
    projection.erase("actor_mentions");
    projection.erase("evidence_links");
    projection.erase("association_reviews");
    projection.erase("fingerprint");
    projection.erase("previous_revision_uri");
    projection.erase("scope");
    error.clear();
    return true;
}

std::vector<std::string> common_investigation_search_providers(
        const common_investigation_case_revision & value) {
    std::set<std::string> providers;
    if (!value.source_id.empty()) providers.insert(value.source_id);
    for (const auto & source : value.sources)
        if (source.kind == common_investigation_source_kind::search_connector && !source.provider_id.empty())
            providers.insert(source.provider_id);
    return {providers.begin(), providers.end()};
}

bool common_investigation_source_complete(const common_investigation_source & source) {
    return source.complete && source.processing_limitations.empty() &&
        (source.kind != common_investigation_source_kind::log_file ||
            (!source.log_parser_profile_json.empty() && source.log_import_exhausted)) &&
        (source.kind == common_investigation_source_kind::search_connector
            ? source.processing_status == "query_complete" : source.processing_status == "ready");
}

bool common_investigation_case_validate(
        const common_investigation_case_revision & value, std::string & error) {
    error.clear();
    if (value.schema_version != 2 || !valid_id(value.case_id) || value.revision == 0 ||
            value.title.empty() || value.title.size() > 512 || value.question.size() > 8192 ||
            value.source_id.size() > 256 || (value.source_id.empty() && value.sources.empty()) ||
            value.scope.namespace_id.empty() || value.scope.session_id.empty() ||
            ((!value.source_id.empty() || value.window_start_ms || value.window_end_ms) &&
                value.window_start_ms >= value.window_end_ms) ||
            value.entities.size() > 100000 || value.events.size() > 500000 ||
            value.relations.size() > 500000 || value.claims.size() > 10000 ||
            value.sources.size() > 32 || value.statements.size() > 10000 ||
            value.comparisons.size() > 10000 || value.analyses.size() > 1000 ||
            value.actor_mentions.size() > 10000 || value.actors.size() > 10000 || value.evidence_links.size() > 10000 || value.association_reviews.size() > 10000) {
        error = "investigation case identity, time window, or bounded collection is invalid";
        return false;
    }
    std::unordered_set<std::string> source_ids, source_record_ids, providers, entities, events, relations, statements;
    if (!value.source_id.empty()) source_ids.insert(value.source_id);
    bool has_search_source = !value.source_id.empty();
    for (const auto & source : value.sources) {
        if (!valid_id(source.id) || !source_record_ids.insert(source.id).second || source.source_revision == 0 ||
                source.resource_uri.size() > 512 || source.representation_uri.size() > 512 ||
                source.provider_id.size() > 256 || source.display_name.size() > 512 ||
            source.mime_type.size() > 128 || source.content_hash.size() > 256 ||
                source.representation_hash.size() > 256 || source.original_uri.size() > 512 ||
                source.original_parent_uri.size() > 512 || source.processor_version.size() > 256 ||
                source.processing_status.empty() || source.processing_status.size() > 64 ||
                source.processing_limitations.size() > 64 || source.log_parser_profile_json.size() > 16384 ||
                ((!source.log_parser_profile_json.empty() || source.log_import_next_record || source.log_import_rejected_records || source.log_import_exhausted) &&
                    source.kind != common_investigation_source_kind::log_file) ||
                ((source.kind == common_investigation_source_kind::document ||
                    source.kind == common_investigation_source_kind::log_file) && source.resource_uri.empty()) ||
                (source.kind == common_investigation_source_kind::search_connector &&
                    (source.provider_id.empty() || !providers.insert(source.provider_id).second))) {
            error = "investigation source identity or provenance is invalid";
            return false;
        }
        if (!source.log_parser_profile_json.empty()) {
            const auto profile = nlohmann::ordered_json::parse(source.log_parser_profile_json,nullptr,false);
            if (!profile.is_object() || !profile.contains("schema") || !profile["schema"].is_string() ||
                    profile["schema"] != "investigation-log-parser.v1" || !profile.contains("source_id") ||
                    !profile["source_id"].is_string() || profile["source_id"] != source.id || !profile.contains("fields") || !profile["fields"].is_object() ||
                    source.log_import_rejected_records > source.log_import_next_record) {
                error = "saved log parser profile or import progress is invalid"; return false;
            }
        } else if (source.log_import_next_record || source.log_import_rejected_records || source.log_import_exhausted) {
            error = "log import progress requires a published parser profile"; return false;
        }
        source_ids.insert(source.id);
        if (source.kind == common_investigation_source_kind::search_connector) source_ids.insert(source.provider_id);
        for (const auto & limitation : source.processing_limitations) if (limitation.empty() || limitation.size() > 1024) {
            error = "investigation source processing limitation is invalid";
            return false;
        }
        has_search_source = has_search_source || source.kind == common_investigation_source_kind::search_connector;
    }
    for (const auto & source : value.sources) if (source.kind == common_investigation_source_kind::search_connector &&
            source.provider_id != source.id && source_record_ids.count(source.provider_id)) {
        error = "investigation provider domain collides with another source record identity";
        return false;
    }
    if (!has_search_source && (value.window_start_ms || value.window_end_ms)) {
        error = "document-only investigation cases must not imply a search time window";
        return false;
    }
    for (const auto & entity : value.entities) {
        if (!valid_id(entity.id) || !source_ids.count(entity.source_id) || entity.kind == common_investigation_entity_kind::unknown || entity.value.empty() ||
                entity.value.size() > 2048 || !entities.insert(entity.id).second ||
                (entity.valid_to_ms && entity.valid_from_ms >= entity.valid_to_ms)) {
            error = "investigation entity has invalid source identity or validity interval";
            return false;
        }
    }
    for (const auto & event : value.events) {
        if (!valid_id(event.id) || !source_ids.count(event.source_id) ||
                event.source_document_id.empty() || event.source_document_id.size() > 512 || event.source_index.size() > 512 ||
                event.entity_ids.size() > 64 || event.outcome.size() > 256 || event.parser_version.size() > 128 ||
                event.source_byte_length > 65536 || (!event.parser_version.empty() && (!event.source_byte_length || !event.source_line)) || !events.insert(event.id).second) {
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
        if (!valid_id(relation.id) || !source_ids.count(relation.source_id) || relation.kind == common_investigation_relation_kind::unknown ||
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
    for (const auto & statement : value.statements) {
        const auto source = std::find_if(value.sources.begin(), value.sources.end(), [&](const auto & item) {
            return item.id == statement.anchor.source_id;
        });
        if (!valid_id(statement.id) || statement.text.empty() || statement.text.size() > 8192 ||
                !statements.insert(statement.id).second || source == value.sources.end() ||
                (statement.anchor.resource_uri != source->resource_uri &&
                    statement.anchor.resource_uri != source->representation_uri) ||
                statement.anchor.quote.empty() || statement.anchor.quote.size() > 4096 ||
                statement.anchor.byte_length != statement.anchor.quote.size() ||
                !statement.anchor.quote_verified || statement.anchor.page_number < 0 ||
                (statement.extraction_origin != "model_proposal" && statement.extraction_origin != "host_rule")) {
            error = "investigation statement lacks a verified bounded source anchor";
            return false;
        }
        if (statement.actor_ids.size() > 32 || statement.reported_time_text.size() > 512) {
            error = "investigation statement actor or reported-time fields exceed bounds";
            return false;
        }
        std::unordered_set<std::string> actor_ids;
        for (const auto & actor_id : statement.actor_ids) if (!valid_id(actor_id) || !actor_ids.insert(actor_id).second) {
            error = "investigation statement contains an invalid or duplicate actor mention reference";
            return false;
        }
    }
    for (const auto & claim : value.claims) {
        if (claim.supporting_statement_ids.size() > 64 || claim.contradicting_statement_ids.size() > 64) {
            error = "claim statement evidence exceeds bounds"; return false;
        }
        std::unordered_set<std::string> support, contradiction;
        for (const auto & id : claim.supporting_statement_ids) if (!statements.count(id) || !support.insert(id).second) {
            error = "claim references unknown or duplicate supporting statements"; return false;
        }
        for (const auto & id : claim.contradicting_statement_ids) if (!statements.count(id) || !contradiction.insert(id).second) {
            error = "claim references unknown or duplicate contradicting statements"; return false;
        }
    }
    std::unordered_set<std::string> mention_ids, actor_ids, link_ids;
    for (const auto & m : value.actor_mentions) {
        const auto statement = std::find_if(value.statements.begin(), value.statements.end(),
            [&](const auto & s) { return s.id == m.statement_id; });
        if (!valid_id(m.id) || !mention_ids.insert(m.id).second || statement == value.statements.end() ||
                m.name.empty() || m.name.size() > 512 ||
                (m.kind != "person" && m.kind != "organization" && m.kind != "system" && m.kind != "place" && m.kind != "unknown") ||
                !m.anchor.quote_verified || m.anchor.quote != m.name || m.anchor.byte_length != m.name.size() ||
                m.anchor.source_id != statement->anchor.source_id || m.anchor.resource_uri != statement->anchor.resource_uri ||
                m.anchor.byte_offset < statement->anchor.byte_offset || m.anchor.byte_length > statement->anchor.byte_length ||
                m.anchor.byte_offset - statement->anchor.byte_offset > statement->anchor.byte_length - m.anchor.byte_length ||
                statement->anchor.quote.substr(m.anchor.byte_offset - statement->anchor.byte_offset, m.anchor.byte_length) != m.name) {
            error = "actor mention is not anchored within its source statement"; return false;
        }
    }
    std::unordered_set<std::string> assigned_mentions;
    for (const auto & a : value.actors) {
        if (!valid_id(a.id) || !actor_ids.insert(a.id).second || a.display_name.empty() || a.display_name.size() > 512 ||
                a.mention_ids.empty() || a.mention_ids.size() > 10000) {
            error = "actor candidate identity or mention bounds are invalid"; return false;
        }
        for (const auto & id : a.mention_ids) if (!mention_ids.count(id) || !assigned_mentions.insert(id).second) {
            error = "actor candidate references an unknown or multiply assigned mention"; return false;
        }
    }
    if (assigned_mentions.size() != mention_ids.size()) { error = "actor mention lacks a candidate"; return false; }
    for (const auto & l : value.evidence_links) {
        const bool identity = l.kind == "possible_same_actor";
        const bool reported_event = l.kind == "possible_same_event";
        if (!valid_id(l.id) || !link_ids.insert(l.id).second || l.left_id == l.right_id ||
                (!identity && !reported_event) || l.assessment_origin != "model_proposal" ||
                l.rationale.empty() || l.rationale.size() > 4096 || l.supporting_statement_ids.empty() || l.supporting_statement_ids.size() > 32 ||
                (identity && (!actor_ids.count(l.left_id) || !actor_ids.count(l.right_id))) ||
                (reported_event && (!statements.count(l.left_id) || !events.count(l.right_id)))) {
            error = "evidence link has invalid endpoints, evidence, or assessment origin"; return false;
        }
        std::unordered_set<std::string> support;
        for (const auto & id : l.supporting_statement_ids) if (!statements.count(id) || !support.insert(id).second) {
            error = "evidence link references unknown or duplicate supporting statements"; return false;
        }
        if (identity) for (const auto & actor_id : {l.left_id,l.right_id}) {
            const auto actor = std::find_if(value.actors.begin(),value.actors.end(),[&](const auto & a) { return a.id == actor_id; });
            bool covered = false;
            for (const auto & id : actor->mention_ids) {
                const auto mention = std::find_if(value.actor_mentions.begin(),value.actor_mentions.end(),[&](const auto & m) { return m.id == id; });
                covered = covered || support.count(mention->statement_id);
            }
            if (!covered) { error = "identity proposal must cite statements containing both actor candidates"; return false; }
        }
        if (reported_event) {
            const auto found = std::find_if(value.statements.begin(), value.statements.end(), [&](const auto & s) { return s.id == l.left_id; });
            if (found->kind != common_investigation_statement_kind::reported_event || !support.count(l.left_id)) {
                error = "event link requires the anchored reported event as supporting evidence"; return false;
            }
        }
    }
    std::unordered_set<std::string> comparison_ids, analysis_ids;
    for (const auto & comparison : value.comparisons) {
        if (!valid_id(comparison.id) || !comparison_ids.insert(comparison.id).second ||
                comparison.left_statement_id == comparison.right_statement_id ||
                !statements.count(comparison.left_statement_id) || !statements.count(comparison.right_statement_id) ||
                comparison.rationale.size() > 4096 ||
                (comparison.assessment_origin != "model_proposal" && comparison.assessment_origin != "analyst_review")) {
            error = "investigation comparison references invalid statements or origin";
            return false;
        }
        for (const auto & source_id : comparison.source_ids) if (!source_ids.count(source_id)) {
            error = "investigation comparison references an unknown source";
            return false;
        }
    }
    if (!value.association_reviews.empty()) {
        nlohmann::ordered_json associations;
        if (!common_investigation_actor_associations(value,associations,error)) return false;
    }
    for (const auto & analysis : value.analyses) {
        if (!valid_id(analysis.id) || !analysis_ids.insert(analysis.id).second || analysis.kind.empty() ||
                analysis.kind.size() > 64 || analysis.algorithm_version.empty() ||
                analysis.algorithm_version.size() > 128 || analysis.published_revision == 0 ||
                analysis.published_revision > value.revision || analysis.parameters_fingerprint.size() > 256 || analysis.result_json.size() > 32768 ||
                analysis.source_ids.size() > 32 || analysis.limitations.size() > 96 ||
                (analysis.completeness != "complete" && analysis.completeness != "partial" && analysis.completeness != "unknown")) {
            error = "investigation analysis provenance is invalid";
            return false;
        }
        for (const auto & source_id : analysis.source_ids) if (!source_ids.count(source_id)) {
            error = "investigation analysis references an unknown source";
            return false;
        }
        for (const auto & limitation : analysis.limitations) if (limitation.empty() || limitation.size() > 1024) {
            error = "investigation analysis limitation is invalid";
            return false;
        }
    }
    return true;
}

bool common_investigation_anchor_validate(const common_investigation_source_anchor & anchor,
        const std::string & source_text, std::string & error) {
    if (anchor.source_id.empty() || anchor.resource_uri.empty() || anchor.quote.empty() ||
            anchor.byte_length != anchor.quote.size() || anchor.byte_offset > source_text.size() ||
            anchor.byte_length > source_text.size() - static_cast<size_t>(anchor.byte_offset) ||
            source_text.compare(static_cast<size_t>(anchor.byte_offset),
                static_cast<size_t>(anchor.byte_length), anchor.quote) != 0) {
        error = "source anchor quote does not exactly match the retained UTF-8 text at the declared byte position";
        return false;
    }
    error.clear();
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
            const auto next = relation.from_entity_id == current.first ? relation.to_entity_id : relation.from_entity_id;
            if (!visible.count(next) && visible.size() >= query.max_nodes) {
                result.truncated = true;
                break;
            }
            if (emitted_relations.insert(relation.id).second) {
                if (result.edges.size() >= query.max_edges) { result.truncated = true; break; }
                result.edges.push_back(relation);
            }
            if (visible.insert(next).second) {
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
                event.event_type.empty() || event.event_type.size() > 256 || event.outcome.size() > 256 ||
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
    bool missing_endpoints = false;
    bool unknown_authentication_outcome = false;
    for (const auto & event : events) {
        for (const auto & rule : rules) {
            if (rule.event_type != event.event_type) continue;
            // An authentication category or endpoint match does not establish success.
            // Preserve unsuccessful/unknown events without promoting them to access.
            if (rule.relation_kind == common_investigation_relation_kind::authenticated_to &&
                    event.outcome != "success") {
                if (event.outcome != "failure") unknown_authentication_outcome = true;
                continue;
            }
            std::vector<const common_investigation_entity *> from, to;
            if (!rule.from_role.empty() || !rule.to_role.empty()) {
                const auto from_role = event.entity_roles.find(rule.from_role);
                const auto to_role = event.entity_roles.find(rule.to_role);
                // Only legacy authentication events have an unambiguous role migration.
                // Missing roles in newer or other event profiles must remain missing.
                const bool legacy_authentication = event.entity_roles.empty() &&
                    rule.from_role == "user" && rule.to_role == "host" &&
                    rule.from_kind == common_investigation_entity_kind::user &&
                    rule.to_kind == common_investigation_entity_kind::host &&
                    rule.relation_kind == common_investigation_relation_kind::authenticated_to;
                if (!legacy_authentication &&
                        (from_role == event.entity_roles.end() || to_role == event.entity_roles.end())) {
                    missing_endpoints = true;
                    continue;
                }
                const auto unique_kind_entity = [&](common_investigation_entity_kind kind)
                        -> const common_investigation_entity * {
                    const common_investigation_entity * match = nullptr;
                    for (const auto & id : event.entity_ids) {
                        const auto * candidate = by_id.at(id);
                        if (candidate->kind != kind) continue;
                        if (match != nullptr) return nullptr;
                        match = candidate;
                    }
                    return match;
                };
                const auto * left = from_role == event.entity_roles.end()
                    ? unique_kind_entity(rule.from_kind) : by_id.at(from_role->second);
                const auto * right = to_role == event.entity_roles.end()
                    ? unique_kind_entity(rule.to_kind) : by_id.at(to_role->second);
                if (left == nullptr || right == nullptr) {
                    missing_endpoints = true;
                    continue;
                }
                if (left->kind == rule.from_kind && right->kind == rule.to_kind) {
                    from.push_back(left); to.push_back(right);
                }
            } else for (const auto & id : event.entity_ids) {
                const auto * entity = by_id.at(id);
                if (entity->kind == rule.from_kind) from.push_back(entity);
                if (entity->kind == rule.to_kind) to.push_back(entity);
            }
            if (from.empty() || to.empty()) missing_endpoints = true;
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
                    relation.rule_revision = rule.relation_kind == common_investigation_relation_kind::authenticated_to ? "2" : "1";
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
    if (unknown_authentication_outcome) limitations.push_back("authentication_events_without_explicit_success_outcome_remain_unlinked");
    if (missing_time) limitations.push_back("one_or_more_relation_events_have_unknown_event_time");
    if (missing_endpoints) limitations.push_back("one_or_more_relation_profiles_have_missing_or_ambiguous_endpoints");
    return true;
}

bool common_investigation_source_followup(const common_investigation_case_revision & value,
        const std::vector<std::string> & source_ids, const common_investigation_analysis_budget & budget,
        nlohmann::ordered_json & result, std::string & error) {
    using json = nlohmann::ordered_json;
    error.clear(); result = json::object();
    const std::set<std::string> selected(source_ids.begin(), source_ids.end());
    if (selected.empty() || selected.size() != source_ids.size() || selected.size() > 32 ||
            !budget.max_expansions || budget.max_expansions > 100000) {
        error = "follow-up requires unique selected sources and a bounded expansion budget"; return false;
    }
    for (const auto & id : selected) if (std::none_of(value.sources.begin(),value.sources.end(),
            [&](const auto & source) { return source.id == id; })) {
        error = "follow-up references a source outside this case"; return false;
    }
    size_t visited = 0;
    bool complete = true;
    json steps = json::array();
    const auto visit = [&]() { if (visited >= budget.max_expansions) { complete = false; return false; } ++visited; return true; };
    const auto append = [&](const json & step) {
        if (steps.size() >= 32) { complete = false; return; } steps.push_back(step);
    };
    std::set<std::string> statement_sources;
    std::set<std::pair<std::string,std::string>> compared;
    for (const auto & statement : value.statements) {
        if (!visit()) break;
        statement_sources.insert(statement.anchor.source_id);
    }
    if (complete) for (const auto & analysis : value.analyses) {
        if (!visit()) break;
        if (analysis.completeness != "complete" || (analysis.kind != "narrative_comparison" && analysis.kind != "mixed_event_chain")) continue;
        for (const auto & a : analysis.source_ids) for (const auto & b : analysis.source_ids) if (a < b) {
            if (!visit()) break;
            compared.emplace(a,b);
        }
        if (!complete) break;
    }
    if (complete) for (const auto & source : value.sources) if (selected.count(source.id)) {
        if (!visit()) break;
        if (source.kind != common_investigation_source_kind::search_connector && source.representation_uri.empty())
            append({{"operation","investigation.document.prepare"},{"source_id",source.id},{"reason","source_text_not_prepared"}});
        else if (source.kind == common_investigation_source_kind::log_file && !source.log_import_exhausted)
            append({{"operation","investigation.log.import"},{"source_id",source.id},
                {"reason",source.log_parser_profile_json.empty() ? "explicit_parser_profile_required" : "log_import_has_remaining_records"}});
        else if (source.kind == common_investigation_source_kind::document && !statement_sources.count(source.id))
            append({{"operation","extract_anchored_statements"},{"source_id",source.id},{"reason","source_has_no_saved_statements"}});
        for (const auto & other : selected) if (source.id < other) {
            if (!visit()) break;
            if (!compared.count({source.id,other})) append({{"operation","compare_selected_sources"},
                {"source_ids",{source.id,other}},{"reason","no_complete_published_pair_analysis"},{"semantic_match",false}});
        }
        if (!complete) break;
    }
    result = {{"case_revision",value.revision},{"analysis","source_followup"},{"complete",complete},
        {"steps",steps},{"explored_states",visited},
        {"limitations",{"Structural suggestions only; no identity, contradiction or causality is inferred."}}};
    return true;
}

bool common_investigation_actor_associations(const common_investigation_case_revision & value,
        nlohmann::ordered_json & result, std::string & error) {
    using json = nlohmann::ordered_json;
    error.clear(); result = json::object();
    if (value.actors.size() > 10000 || value.association_reviews.size() > 10000) {
        error = "actor association projection exceeds its bounds"; return false;
    }
    std::map<std::string,std::string> parent;
    std::map<std::string,const common_investigation_actor_candidate *> actors;
    for (const auto & actor : value.actors) {
        if (!actors.emplace(actor.id,&actor).second) { error = "duplicate actor candidate"; return false; }
        parent[actor.id] = actor.id;
    }
    const auto root = [&](std::string id) {
        std::string found = id;
        while (parent.at(found) != found) found = parent.at(found);
        while (parent.at(id) != id) { const auto next = parent.at(id); parent[id] = found; id = next; }
        return found;
    };
    const auto pair_key = [](const common_investigation_association_review & review) {
        if (review.kind == "identity") return std::make_tuple(review.kind,std::string(),
            std::min(review.left_actor_id,review.right_actor_id),std::max(review.left_actor_id,review.right_actor_id));
        return std::make_tuple(review.kind,review.relation_type,review.left_actor_id,review.right_actor_id);
    };
    std::map<std::tuple<std::string,std::string,std::string,std::string>,const common_investigation_association_review *> latest;
    std::set<std::string> ids;
    std::map<std::string,std::tuple<std::string,std::string,uint64_t>> operations;
    uint64_t last_revision = 0;
    const std::set<std::string> relation_types = {"associated_with","employed_by","owns","member_of","communicated_with","other"};
    for (const auto & review : value.association_reviews) {
        const bool identity = review.kind == "identity";
        const bool relationship = review.kind == "relationship";
        if (!valid_id(review.id) || !ids.insert(review.id).second || review.operation_id.empty() || review.operation_id.size() > 128 ||
                review.parameters_fingerprint.empty() || review.parameters_fingerprint.size() > 128 ||
                !actors.count(review.left_actor_id) || !actors.count(review.right_actor_id) || review.left_actor_id == review.right_actor_id ||
                (!identity && !relationship) || review.comment.size() > 4096 || review.reviewer_id.empty() || review.reviewer_id.size() > 512 ||
                review.created_at_ms <= 0 || !review.published_revision || review.published_revision > value.revision ||
                review.published_revision < last_revision ||
                (identity && (review.decision != "same_actor" && review.decision != "different_actors" && review.decision != "unresolved")) ||
                (relationship && (review.decision != "related" && review.decision != "unresolved")) ||
                (identity && !review.relation_type.empty()) ||
                (relationship && (!relation_types.count(review.relation_type) ||
                    (review.relation_type == "other" && review.comment.empty())))) {
            error = "association review has invalid provenance, endpoints or decision"; return false;
        }
        const auto provenance = std::make_tuple(review.parameters_fingerprint,review.reviewer_id,review.published_revision);
        const auto operation = operations.emplace(review.operation_id,provenance);
        if (!operation.second && operation.first->second != provenance) {
            error = "association operation has inconsistent publication or reviewer provenance"; return false;
        }
        const auto key = pair_key(review);
        const auto old = latest.find(key);
        if (review.supersedes_id != (old == latest.end() ? std::string() : old->second->id)) {
            error = "association review does not supersede the current decision for this pair"; return false;
        }
        latest[key] = &review; last_revision = review.published_revision;
    }
    for (const auto & item : latest) {
        const auto & review = *item.second;
        if (review.kind != "identity" || review.decision != "same_actor") continue;
        const auto a = root(review.left_actor_id), b = root(review.right_actor_id);
        if (a != b) parent[std::max(a,b)] = std::min(a,b);
    }
    for (const auto & item : latest) {
        const auto & review = *item.second;
        if (review.kind == "identity" && review.decision == "different_actors" && root(review.left_actor_id) == root(review.right_actor_id)) {
            error = "different-actor decision conflicts with remaining confirmations; withdraw the conflicting same-actor decisions first";
            return false;
        }
    }
    std::map<std::string,json> groups;
    for (const auto & actor : value.actors) {
        const auto id = root(actor.id);
        if (!groups.count(id)) groups[id] = {{"id",id},{"display_name",actors.at(id)->display_name},
            {"actor_ids",json::array()},{"mention_ids",json::array()},{"assessment_origin","unreviewed_candidate"}};
        groups[id]["actor_ids"].push_back(actor.id);
        for (const auto & mention : actor.mention_ids) groups[id]["mention_ids"].push_back(mention);
    }
    std::set<std::string> distinguished;
    for (const auto & item : latest) if (item.second->kind == "identity" && item.second->decision == "different_actors") {
        distinguished.insert(root(item.second->left_actor_id)); distinguished.insert(root(item.second->right_actor_id));
    }
    json group_rows=json::array(), decisions=json::array(), relationships=json::array(), limitations=json::array();
    for (auto & group : groups) {
        if (group.second["actor_ids"].size() > 1) group.second["assessment_origin"] = "analyst_confirmed_identity";
        else if (distinguished.count(group.first)) group.second["assessment_origin"] = "analyst_distinguished_identity";
        group.second["actor_count"] = group.second["actor_ids"].size();
        group.second["mention_count"] = group.second["mention_ids"].size();
        const bool abbreviated = group.second["actor_ids"].size() > 128 || group.second["mention_ids"].size() > 128;
        group.second["members_truncated"] = abbreviated;
        for (const char * field : {"actor_ids","mention_ids"})
            while (group.second[field].size() > 128) group.second[field].erase(group.second[field].end() - 1);
        group_rows.push_back(std::move(group.second));
    }
    for (const auto & item : latest) {
        const auto & review = *item.second;
        decisions.push_back(review.id);
        if (review.kind == "relationship" && review.decision == "related") {
            const auto a=root(review.left_actor_id), b=root(review.right_actor_id);
            if (a == b && review.published_revision == value.revision) {
                error = "a new relationship requires two distinct actor groups"; return false;
            }
            if (a == b) { limitations.push_back("Reviewed relation " + review.id + " now falls within one actor group."); continue; }
            relationships.push_back({{"id",review.id},{"source",a},{"target",b},{"type",review.relation_type},
                {"assessment_origin","analyst_confirmed"},{"reviewer_id",review.reviewer_id}});
        }
    }
    result = {{"case_revision",value.revision},{"groups",group_rows},{"relationships",relationships},
        {"active_review_ids",decisions},{"limitations",limitations}};
    return true;
}
