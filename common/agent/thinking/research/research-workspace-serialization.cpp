#include "research-workspace-serialization.h"

#include <algorithm>
#include <stdexcept>

namespace {
using json = nlohmann::ordered_json;

json resource_ref_json(const common_runtime_resource_ref & ref) {
    return {{"uri", ref.uri}, {"name", ref.name}, {"description", ref.description},
        {"mime_type", ref.mime_type}, {"size_bytes", ref.size_bytes},
        {"scope", static_cast<int>(ref.scope)},
        {"metadata", {{"purpose", ref.metadata.purpose}, {"content_summary", ref.metadata.content_summary},
            {"usage_hint", ref.metadata.usage_hint}, {"limitations", ref.metadata.limitations},
            {"keywords", ref.metadata.keywords}, {"entities", ref.metadata.entities},
            {"processing_cache_key", ref.metadata.processing_cache_key},
            {"declared_language", ref.metadata.declared_language}, {"resolved_language", ref.metadata.resolved_language},
            {"language_confidence", ref.metadata.language_confidence}, {"language_source", ref.metadata.language_source},
            {"source_provider", ref.metadata.source_provider}, {"source_operation", ref.metadata.source_operation},
            {"source_request_json", ref.metadata.source_request_json}, {"retrieved_at", ref.metadata.retrieved_at},
            {"content_hash", ref.metadata.content_hash}}},
        {"lineage", {{"parent_uri", ref.lineage.parent_uri}, {"chunk_index", ref.lineage.chunk_index},
            {"chunk_count", ref.lineage.chunk_count}, {"byte_offset", ref.lineage.byte_offset},
            {"byte_length", ref.lineage.byte_length}, {"overlap_bytes", ref.lineage.overlap_bytes},
            {"derivation", ref.lineage.derivation}}}};
}

common_runtime_resource_ref parse_resource_ref(const json & value) {
    common_runtime_resource_ref ref;
    ref.uri = value.at("uri").get<std::string>();
    ref.name = value.value("name", std::string{});
    ref.description = value.value("description", std::string{});
    ref.mime_type = value.value("mime_type", std::string{});
    ref.size_bytes = value.value("size_bytes", size_t(0));
    const int scope = value.value("scope", 0);
    if (scope < 0 || scope > static_cast<int>(common_runtime_resource_scope::project))
        throw std::domain_error("resource scope is invalid");
    ref.scope = static_cast<common_runtime_resource_scope>(scope);
    if (value.contains("metadata") && value["metadata"].is_object()) {
        const auto & m = value["metadata"];
        ref.metadata.purpose = m.value("purpose", std::string{});
        ref.metadata.content_summary = m.value("content_summary", std::string{});
        ref.metadata.usage_hint = m.value("usage_hint", std::string{});
        ref.metadata.limitations = m.value("limitations", std::string{});
        ref.metadata.keywords = m.value("keywords", std::vector<std::string>{});
        ref.metadata.entities = m.value("entities", std::vector<std::string>{});
        ref.metadata.processing_cache_key = m.value("processing_cache_key", std::string{});
        ref.metadata.declared_language = m.value("declared_language", std::string{});
        ref.metadata.resolved_language = m.value("resolved_language", std::string{});
        ref.metadata.language_confidence = m.value("language_confidence", 0.0);
        ref.metadata.language_source = m.value("language_source", std::string{});
        ref.metadata.source_provider = m.value("source_provider", std::string{});
        ref.metadata.source_operation = m.value("source_operation", std::string{});
        ref.metadata.source_request_json = m.value("source_request_json", std::string{});
        ref.metadata.retrieved_at = m.value("retrieved_at", int64_t(0));
        ref.metadata.content_hash = m.value("content_hash", std::string{});
    }
    if (value.contains("lineage") && value["lineage"].is_object()) {
        const auto & l = value["lineage"];
        ref.lineage.parent_uri = l.value("parent_uri", std::string{});
        ref.lineage.chunk_index = l.value("chunk_index", size_t(0));
        ref.lineage.chunk_count = l.value("chunk_count", size_t(0));
        ref.lineage.byte_offset = l.value("byte_offset", size_t(0));
        ref.lineage.byte_length = l.value("byte_length", size_t(0));
        ref.lineage.overlap_bytes = l.value("overlap_bytes", size_t(0));
        ref.lineage.derivation = l.value("derivation", std::string{});
    }
    return ref;
}

bool bounded_enum(int value, int maximum) { return value >= 0 && value <= maximum; }
}

json common_agent_research_workspace_to_json(const common_agent_research_workspace & w) {
    json out = {{"schema", "research-workspace.v1"}, {"workspace_id", w.workspace_id},
        {"research_id", w.research_id}, {"draft_id", w.draft_id}, {"revision", w.revision},
        {"base_revision", w.base_revision}, {"request_id", w.request_id}, {"turn_id", w.turn_id},
        {"session_id", w.session_id}, {"plan_id", w.plan_id}, {"plan_step_id", w.plan_step_id},
        {"scope", {{"namespace_id", w.scope.namespace_id}, {"session_id", w.scope.session_id},
            {"project_id", w.scope.project_id}, {"turn_id", w.scope.turn_id}}},
        {"objective", {{"objective_id", w.objective.objective_id}, {"question", w.objective.question},
            {"purpose", w.objective.purpose}, {"expected_output", w.objective.expected_output},
            {"success_criteria", w.objective.success_criteria}, {"constraints", w.objective.constraints},
            {"excluded_topics", w.objective.excluded_topics}}},
        {"budget", {{"max_iterations", w.budget.max_iterations}, {"max_tasks", w.budget.max_tasks},
            {"max_tool_calls", w.budget.max_tool_calls}, {"max_sources", w.budget.max_sources},
            {"max_sources_per_gap", w.budget.max_sources_per_gap}, {"minimum_sources", w.budget.minimum_sources},
            {"max_tokens", w.budget.max_tokens}, {"max_elapsed_ms", w.budget.max_elapsed_ms},
            {"minimum_coverage", w.budget.minimum_coverage}, {"minimum_confidence", w.budget.minimum_confidence}}},
        {"coverage", {{"total_gaps", w.coverage.total_gaps}, {"answered_gaps", w.coverage.answered_gaps},
            {"blocked_gaps", w.coverage.blocked_gaps}, {"unresolved_critical_gaps", w.coverage.unresolved_critical_gaps},
            {"objective_coverage", w.coverage.objective_coverage}, {"evidence_quality", w.coverage.evidence_quality},
            {"source_diversity", w.coverage.source_diversity}}},
        {"iterations_completed", w.iterations_completed}, {"tool_calls", w.tool_calls},
        {"no_progress_iterations", w.no_progress_iterations}, {"gaps", json::array()}, {"tasks", json::array()},
        {"sources", json::array()}, {"evidence", json::array()}, {"comparisons", json::array()}};
    for (const auto & g : w.gaps) out["gaps"].push_back({{"id", g.gap_id}, {"question", g.question},
        {"reason", g.reason}, {"completion_criterion", g.completion_criterion}, {"priority", g.priority},
        {"status", static_cast<int>(g.status)}, {"assessment", static_cast<int>(g.assessment)},
        {"evidence_ids", g.evidence_ids}, {"child_gap_ids", g.child_gap_ids}});
    for (const auto & t : w.tasks) out["tasks"].push_back({{"id", t.task_id}, {"gap_id", t.gap_id},
        {"instruction", t.instruction}, {"kind", static_cast<int>(t.kind)},
        {"preferred_tools", t.preferred_tools}, {"dependency_ids", t.dependency_ids},
        {"priority", t.priority}, {"attempt", t.attempt}, {"max_attempts", t.max_attempts},
        {"status", static_cast<int>(t.status)}});
    for (const auto & s : w.sources) out["sources"].push_back({{"id", s.source_id}, {"title", s.title},
        {"origin", s.origin}, {"authority", s.authority}, {"retrieved_at", s.retrieved_at},
        {"content_hash", s.content_hash}, {"kind", static_cast<int>(s.kind)},
        {"resource_ref", s.resource_ref ? resource_ref_json(*s.resource_ref) : json(nullptr)},
        {"quality_score", s.quality_score}, {"primary_source", s.primary_source},
        {"memory_id", s.memory_id}, {"memory_kind", static_cast<int>(s.memory_kind)},
        {"availability", static_cast<int>(s.availability)}, {"role", s.role}, {"required", s.required}});
    for (const auto & e : w.evidence) out["evidence"].push_back({{"id", e.evidence_id}, {"source_id", e.source_id},
        {"claim_id", e.claim_id}, {"statement", e.statement}, {"source_location", e.source_location},
        {"relation", static_cast<int>(e.relation)}, {"origin", static_cast<int>(e.origin)},
        {"relevance", e.relevance}, {"confidence", e.confidence}, {"directly_observed", e.directly_observed},
        {"model_inferred", e.model_inferred}, {"supporting_excerpt", e.supporting_excerpt},
        {"supporting_excerpt_host_verified", e.supporting_excerpt_host_verified}});
    for (const auto & c : w.comparisons) out["comparisons"].push_back({{"id", c.comparison_id},
        {"gap_id", c.gap_id}, {"source_ids", c.source_ids}, {"evidence_ids", c.evidence_ids},
        {"relation", static_cast<int>(c.relation)}, {"summary", c.summary}});
    return out;
}

bool common_agent_research_workspace_from_json(const nlohmann::json & value,
        common_agent_research_workspace & out, std::string & error) {
    out = {};
    try {
        if (!value.is_object() || value.value("schema", std::string{}) != "research-workspace.v1")
            throw std::domain_error("unsupported research workspace schema");
        out.workspace_id = value.at("workspace_id").get<std::string>();
        out.research_id = value.value("research_id", std::string{});
        out.draft_id = value.value("draft_id", std::string{});
        out.revision = value.value("revision", uint64_t(0));
        out.base_revision = value.value("base_revision", uint64_t(0));
        out.request_id = value.value("request_id", std::string{});
        out.turn_id = value.value("turn_id", std::string{});
        out.session_id = value.value("session_id", std::string{});
        out.plan_id = value.value("plan_id", std::string{});
        out.plan_step_id = value.value("plan_step_id", std::string{});
        const auto & scope = value.at("scope");
        out.scope.namespace_id = scope.value("namespace_id", std::string{});
        out.scope.session_id = scope.value("session_id", std::string{});
        out.scope.project_id = scope.value("project_id", std::string{});
        out.scope.turn_id = scope.value("turn_id", std::string{});
        const auto & o = value.at("objective");
        out.objective.objective_id = o.value("objective_id", std::string{});
        out.objective.question = o.value("question", std::string{});
        out.objective.purpose = o.value("purpose", std::string{});
        out.objective.expected_output = o.value("expected_output", std::string{});
        out.objective.success_criteria = o.value("success_criteria", std::vector<std::string>{});
        out.objective.constraints = o.value("constraints", std::vector<std::string>{});
        out.objective.excluded_topics = o.value("excluded_topics", std::vector<std::string>{});
        const auto & b = value.at("budget");
        out.budget.max_iterations = b.value("max_iterations", 0); out.budget.max_tasks = b.value("max_tasks", 0);
        out.budget.max_tool_calls = b.value("max_tool_calls", 0); out.budget.max_sources = b.value("max_sources", 0);
        out.budget.max_sources_per_gap = b.value("max_sources_per_gap", 0); out.budget.minimum_sources = b.value("minimum_sources", 1);
        out.budget.max_tokens = b.value("max_tokens", int64_t(0)); out.budget.max_elapsed_ms = b.value("max_elapsed_ms", int64_t(0));
        out.budget.minimum_coverage = b.value("minimum_coverage", 1.0); out.budget.minimum_confidence = b.value("minimum_confidence", 0.0);
        const auto & cov = value.at("coverage");
        out.coverage.total_gaps = cov.value("total_gaps", 0); out.coverage.answered_gaps = cov.value("answered_gaps", 0);
        out.coverage.blocked_gaps = cov.value("blocked_gaps", 0); out.coverage.unresolved_critical_gaps = cov.value("unresolved_critical_gaps", 0);
        out.coverage.objective_coverage = cov.value("objective_coverage", 0.0); out.coverage.evidence_quality = cov.value("evidence_quality", 0.0);
        out.coverage.source_diversity = cov.value("source_diversity", 0.0);
        out.iterations_completed = value.value("iterations_completed", 0); out.tool_calls = value.value("tool_calls", 0);
        out.no_progress_iterations = value.value("no_progress_iterations", 0);
        for (const auto & x : value.at("gaps")) {
            common_agent_research_gap g; g.gap_id=x.at("id").get<std::string>(); g.question=x.at("question").get<std::string>();
            g.reason=x.value("reason",std::string{}); g.completion_criterion=x.at("completion_criterion").get<std::string>();
            g.priority=x.value("priority",0); const int status=x.value("status",0), assessment=x.value("assessment",0);
            if (!bounded_enum(status, static_cast<int>(common_agent_research_gap_status::abandoned)) ||
                    !bounded_enum(assessment, static_cast<int>(common_agent_research_gap_assessment::contradicted)))
                throw std::domain_error("research gap enum is invalid");
            g.status=static_cast<common_agent_research_gap_status>(status); g.assessment=static_cast<common_agent_research_gap_assessment>(assessment);
            g.evidence_ids=x.value("evidence_ids",std::vector<std::string>{}); g.child_gap_ids=x.value("child_gap_ids",std::vector<std::string>{});
            out.gaps.push_back(std::move(g));
        }
        for (const auto & x : value.at("tasks")) {
            common_agent_research_task t; t.task_id=x.at("id").get<std::string>(); t.gap_id=x.value("gap_id",std::string{});
            t.instruction=x.value("instruction",std::string{}); const int kind=x.value("kind",0), status=x.value("status",0);
            if (!bounded_enum(kind, static_cast<int>(common_agent_research_task_kind::summarize_material)) ||
                    !bounded_enum(status, static_cast<int>(common_agent_research_task_status::cancelled)))
                throw std::domain_error("research task enum is invalid");
            t.kind=static_cast<common_agent_research_task_kind>(kind); t.status=static_cast<common_agent_research_task_status>(status);
            t.preferred_tools=x.value("preferred_tools",std::vector<std::string>{}); t.dependency_ids=x.value("dependency_ids",std::vector<std::string>{});
            t.priority=x.value("priority",0); t.attempt=x.value("attempt",0); t.max_attempts=x.value("max_attempts",1); out.tasks.push_back(std::move(t));
        }
        for (const auto & x : value.at("sources")) {
            common_agent_research_source s; s.source_id=x.at("id").get<std::string>(); s.title=x.value("title",std::string{});
            s.origin=x.value("origin",std::string{}); s.authority=x.value("authority",std::string{}); s.retrieved_at=x.value("retrieved_at",std::string{});
            s.content_hash=x.value("content_hash",std::string{}); const int kind=x.value("kind",0), availability=x.value("availability",1), memory_kind=x.value("memory_kind",0);
            if (!bounded_enum(kind, static_cast<int>(common_agent_research_source_kind::remote_agent_result)) ||
                    !bounded_enum(availability, static_cast<int>(common_agent_research_source_availability::requires_acquisition)) ||
                    !bounded_enum(memory_kind, static_cast<int>(common_memory_kind::preference))) throw std::domain_error("research source enum is invalid");
            s.kind=static_cast<common_agent_research_source_kind>(kind); s.availability=static_cast<common_agent_research_source_availability>(availability);
            s.memory_kind=static_cast<common_memory_kind>(memory_kind); if (x.contains("resource_ref") && x["resource_ref"].is_object()) s.resource_ref=parse_resource_ref(x["resource_ref"]);
            s.quality_score=x.value("quality_score",0.0); s.primary_source=x.value("primary_source",false); s.memory_id=x.value("memory_id",std::string{});
            s.role=x.value("role",std::string("reference")); s.required=x.value("required",false); out.sources.push_back(std::move(s));
        }
        for (const auto & x : value.at("evidence")) {
            common_agent_research_evidence e; e.evidence_id=x.at("id").get<std::string>(); e.source_id=x.at("source_id").get<std::string>();
            e.claim_id=x.value("claim_id",std::string{}); e.statement=x.value("statement",std::string{}); e.source_location=x.value("source_location",std::string{});
            const int relation=x.value("relation",4), origin=x.value("origin",1);
            if (!bounded_enum(relation, static_cast<int>(common_agent_research_evidence_relation::inconclusive)) ||
                    !bounded_enum(origin, static_cast<int>(common_agent_research_evidence_origin::model_inference))) throw std::domain_error("research evidence enum is invalid");
            e.relation=static_cast<common_agent_research_evidence_relation>(relation); e.origin=static_cast<common_agent_research_evidence_origin>(origin);
            e.relevance=x.value("relevance",0.0); e.confidence=x.value("confidence",0.0); e.directly_observed=x.value("directly_observed",false);
            e.model_inferred=x.value("model_inferred",false); e.supporting_excerpt=x.value("supporting_excerpt",std::string{});
            e.supporting_excerpt_host_verified=x.value("supporting_excerpt_host_verified",false); out.evidence.push_back(std::move(e));
        }
        for (const auto & x : value.at("comparisons")) {
            common_agent_research_source_comparison c; c.comparison_id=x.at("id").get<std::string>(); c.gap_id=x.value("gap_id",std::string{});
            c.source_ids=x.value("source_ids",std::vector<std::string>{}); c.evidence_ids=x.value("evidence_ids",std::vector<std::string>{});
            const int relation=x.value("relation",0); if (!bounded_enum(relation,static_cast<int>(common_agent_research_comparison_relation::qualify))) throw std::domain_error("research comparison enum is invalid");
            c.relation=static_cast<common_agent_research_comparison_relation>(relation); c.summary=x.value("summary",std::string{}); out.comparisons.push_back(std::move(c));
        }
        if (!common_agent_research_workspace_validate(out, error)) { out={}; return false; }
        error.clear(); return true;
    } catch (const std::exception & e) { out={}; error=std::string("research workspace payload is invalid: ")+e.what(); return false; }
}
