#include "agent/adaptation/concept-input-routing.h"

#include "agent/contracts/agent-result.h"
#include "agent/thinking/research/research-contract.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <set>

namespace {

using json = nlohmann::ordered_json;

common_agent_concept_semantic_kind parse_kind(const std::string & value) {
    if (value == "fact") return common_agent_concept_semantic_kind::fact;
    if (value == "procedure") return common_agent_concept_semantic_kind::procedure;
    if (value == "decision_rule") return common_agent_concept_semantic_kind::decision_rule;
    if (value == "heuristic") return common_agent_concept_semantic_kind::heuristic;
    return common_agent_concept_semantic_kind::concept;
}

bool bounded(const std::string & value, size_t max_size = 1024) {
    return !value.empty() && value.size() <= max_size;
}

bool parse_string_array(
        const json & value,
        const char * key,
        std::vector<std::string> & output) {
    if (!value.contains(key)) return true;
    if (!value.at(key).is_array()) return false;
    output.clear();
    for (const auto & item : value.at(key)) {
        if (!item.is_string() || !bounded(item.get<std::string>())) return false;
        output.push_back(item.get<std::string>());
    }
    return true;
}

bool parse_hypothesis_envelope(
        const common_agent_research_evidence & evidence,
        const common_agent_scope & scope,
        common_agent_concept_hypothesis & hypothesis,
        std::string & error) {
    error.clear();
    json value;
    try {
        value = json::parse(evidence.statement);
    } catch (const std::exception &) {
        return false;
    }
    if (!value.is_object() || value.value("kind", "") != "concept_hypothesis" ||
            !value.contains("concept_key") || !value.contains("statement")) {
        return false;
    }
    const std::string concept_key = value.value("concept_key", "");
    const std::string statement = value.value("statement", "");
    if (!bounded(concept_key) || !bounded(statement)) return false;

    hypothesis = {};
    hypothesis.id = "research-hypothesis:" + evidence.evidence_id;
    hypothesis.concept_key = concept_key;
    hypothesis.statement = statement;
    hypothesis.canonical_statement_ref = value.value(
        "canonical_statement_ref", "research://evidence/" + evidence.evidence_id);
    hypothesis.semantic_kind = parse_kind(value.value("semantic_kind", "concept"));
    hypothesis.source_kind = common_agent_concept_source_kind::research;
    hypothesis.status = common_agent_concept_hypothesis_status::proposed;
    hypothesis.scope = scope;
    if (!parse_string_array(value, "preconditions", hypothesis.preconditions) ||
            !parse_string_array(value, "invariants", hypothesis.invariants) ||
            !parse_string_array(value, "counterexamples", hypothesis.counterexamples)) {
        error = "research concept hypothesis envelope contains an invalid string array";
        return false;
    }
    hypothesis.source_refs = {evidence.evidence_id};
    if (!evidence.source_id.empty()) hypothesis.source_refs.push_back(evidence.source_id);
    const float evidence_confidence = static_cast<float>(std::clamp(evidence.confidence, 0.0, 1.0));
    const float relevance = static_cast<float>(std::clamp(evidence.relevance, 0.0, 1.0));
    hypothesis.confidence = std::min(evidence_confidence, relevance);
    hypothesis.host_grounded = false;
    hypothesis.reusable = false;
    return common_agent_concept_hypothesis_validate(hypothesis, error);
}

bool research_evidence_admissible(const common_agent_research_evidence & evidence) {
    return evidence.relation == common_agent_research_evidence_relation::supports &&
        evidence.origin != common_agent_research_evidence_origin::model_inference &&
        !evidence.model_inferred && evidence.directly_observed &&
        bounded(evidence.evidence_id) && bounded(evidence.statement);
}

} // namespace

common_agent_concept_hypothesis_batch_provider
common_agent_make_research_concept_hypothesis_batch_provider() {
    return [](
            const common_agent_request & request,
            const common_plan_state &,
            const common_agent_result & result,
            const common_learning_transaction &,
            std::vector<common_agent_concept_hypothesis> & hypotheses,
            std::string & error) {
        error.clear();
        hypotheses.clear();
        if (!result.research_result || !result.research_result->complete ||
                !result.research_result->unresolved_claim_ids.empty() ||
                result.research_result->coverage.unresolved_critical_gaps > 0) {
            return true;
        }
        const common_agent_scope scope = common_agent_scope_from_request(request);
        std::set<std::string> concepts;
        for (const auto & evidence : result.research_result->evidence) {
            if (!research_evidence_admissible(evidence)) continue;
            common_agent_concept_hypothesis hypothesis;
            std::string parse_error;
            if (!parse_hypothesis_envelope(evidence, scope, hypothesis, parse_error)) {
                if (!parse_error.empty()) {
                    error = std::move(parse_error);
                    return false;
                }
                continue;
            }
            if (!concepts.insert(hypothesis.concept_key).second) continue;
            hypotheses.push_back(std::move(hypothesis));
        }
        return true;
    };
}

common_agent_concept_grounding_provider
common_agent_make_concept_grounding_provider(
        common_agent_concept_contrast_provider contrast_provider) {
    return [contrast_provider = std::move(contrast_provider)](
            const common_agent_request & request,
            const common_plan_state & plan,
            const common_agent_result & result,
            const common_learning_transaction & transaction,
            const common_agent_concept_hypothesis & hypothesis,
            common_agent_concept_grounding & grounding,
            std::vector<common_flydelta_teaching_relation> & relations,
            std::string & error) {
        error.clear();
        grounding = {};
        relations.clear();
        if (!contrast_provider) {
            error = "concept grounding requires a host contrast provider";
            return false;
        }
        std::vector<common_agent_concept_contrast> contrasts;
        if (!contrast_provider(request, plan, result, transaction, hypothesis, contrasts, error)) {
            return false;
        }
        if (contrasts.empty()) return true;

        grounding.id = "grounding:" + hypothesis.id;
        grounding.hypothesis_ref = hypothesis.id;
        grounding.behavior_key = contrasts.front().behavior_key;
        grounding.verifier_ref = contrasts.front().verifier_ref;
        grounding.changed_dimension = contrasts.front().changed_dimension;
        grounding.invariant_dimensions = contrasts.front().invariant_dimensions;
        grounding.scope = hypothesis.scope;
        grounding.minimum_contrasts = 2;
        grounding.host_approved = true;
        grounding.reusable = true;
        if (!common_agent_concept_grounding_validate(grounding, error)) return false;

        std::set<std::string> independent;
        for (const auto & contrast : contrasts) {
            if (!independent.insert(contrast.independent_key).second) {
                error = "host concept grounding contains duplicate independent contrasts";
                return false;
            }
            if (contrast.verifier_ref != grounding.verifier_ref ||
                    contrast.changed_dimension != grounding.changed_dimension ||
                    contrast.invariant_dimensions != grounding.invariant_dimensions) {
                error = "host concept contrasts disagree on verifier or changed dimension";
                return false;
            }
            auto grounded_contrast = contrast;
            grounded_contrast.grounding_ref = grounding.id;
            if (!common_agent_concept_contrast_validate(grounded_contrast, grounding, error)) return false;
            common_flydelta_teaching_relation relation;
            auto grounded_hypothesis = hypothesis;
            grounded_hypothesis.status = common_agent_concept_hypothesis_status::grounded;
            grounded_hypothesis.host_grounded = true;
            grounded_hypothesis.reusable = true;
            if (!common_agent_concept_contrast_to_teaching_relation(
                    grounded_hypothesis, grounding, grounded_contrast, relation, error)) return false;
            relations.push_back(std::move(relation));
        }
        return true;
    };
}
