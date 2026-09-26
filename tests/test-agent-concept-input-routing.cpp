#include "agent/adaptation/concept-input-routing.h"

#include <cassert>
#include <string>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

namespace {

common_agent_result research_result() {
    common_agent_result result;
    common_agent_research_result research;
    research.workspace_id = "research:dataset";
    research.complete = true;
    research.coverage.unresolved_critical_gaps = 0;
    common_agent_research_evidence evidence;
    evidence.evidence_id = "evidence:grouped-sum";
    evidence.source_id = "source:dataset-guide";
    evidence.relation = common_agent_research_evidence_relation::supports;
    evidence.origin = common_agent_research_evidence_origin::direct_source;
    evidence.directly_observed = true;
    evidence.relevance = 1.0;
    evidence.confidence = 0.9;
    evidence.statement = R"json({
        "kind":"concept_hypothesis",
        "concept_key":"dataset.grouped_sum",
        "statement":"Use grouped aggregation when totals are requested per region.",
        "semantic_kind":"decision_rule",
        "preconditions":["the request asks for totals per group"]
    })json";
    research.evidence.push_back(std::move(evidence));
    result.research_result = std::move(research);
    return result;
}

} // namespace

int main() {
    std::string error;
    common_agent_request request;
    request.namespace_id = "test";
    request.session_id = "session";
    request.project_id = "project";
    request.turn_id = "turn";
    common_plan_state plan;
    common_learning_transaction transaction;

    const auto hypothesis_provider =
        common_agent_make_research_concept_hypothesis_batch_provider();
    std::vector<common_agent_concept_hypothesis> hypotheses;
    auto result = research_result();
    CHECK(hypothesis_provider(request, plan, result, transaction, hypotheses, error));
    CHECK(hypotheses.size() == 1);
    CHECK(hypotheses.front().source_kind == common_agent_concept_source_kind::research);
    CHECK(hypotheses.front().status == common_agent_concept_hypothesis_status::proposed);
    CHECK(!hypotheses.front().host_grounded && !hypotheses.front().reusable);
    const auto valid_hypothesis = hypotheses.front();

    auto invalid_result = result;
    invalid_result.research_result->evidence.front().model_inferred = true;
    hypotheses.clear();
    CHECK(hypothesis_provider(request, plan, invalid_result, transaction, hypotheses, error));
    CHECK(hypotheses.empty());

    common_agent_concept_contrast_provider contrast_provider = [](
            const common_agent_request &,
            const common_plan_state &,
            const common_agent_result &,
            const common_learning_transaction &,
            const common_agent_concept_hypothesis & hypothesis,
            std::vector<common_agent_concept_contrast> & contrasts,
            std::string &) {
        contrasts.clear();
        for (int index = 0; index < 2; ++index) {
            common_agent_concept_contrast contrast;
            contrast.id = "contrast:" + std::to_string(index);
            contrast.behavior_key = hypothesis.concept_key;
            contrast.task_fingerprint = "task:" + std::to_string(index);
            contrast.baseline_ref = "baseline:" + std::to_string(index);
            contrast.conditioned_ref = "conditioned:" + std::to_string(index);
            contrast.control_ref = "control:" + std::to_string(index);
            contrast.verifier_ref = "oracle://dataset/v1";
            contrast.changed_dimension = "dataset_operation";
            contrast.independent_key = "independent:" + std::to_string(index);
            contrast.confidence = 0.95f;
            contrast.host_verified = true;
            contrasts.push_back(std::move(contrast));
        }
        return true;
    };

    const auto grounding_provider =
        common_agent_make_concept_grounding_provider(std::move(contrast_provider));
    common_agent_concept_grounding grounding;
    std::vector<common_flydelta_teaching_relation> relations;
    CHECK(grounding_provider(request, plan, result, transaction, valid_hypothesis,
        grounding, relations, error));
    CHECK(relations.size() == 2);
    CHECK(grounding.host_approved && grounding.reusable);
    CHECK(relations.front().status == common_flydelta_teaching_relation_status::resolved);
    CHECK(relations.front().host_approved);
    CHECK(common_agent_validate_concept_teaching_relations(
        [&]() {
            auto value = valid_hypothesis;
            value.status = common_agent_concept_hypothesis_status::grounded;
            value.host_grounded = true;
            value.reusable = true;
            return value;
        }(), grounding, relations, error));

    return 0;
}
