#pragma once

#include "agent/thinking/research/research-contract.h"
#include "memory/memory-types.h"
#include "plan/plan-types.h"

#include <algorithm>
#include <string>
#include <vector>

enum class common_agent_research_verification_decision {
    accept,
    revise_answer,
    gather_evidence,
    fail_with_uncertainty,
};

struct common_agent_research_verification {
    common_agent_research_verification_decision decision =
        common_agent_research_verification_decision::fail_with_uncertainty;
    bool answer_sufficient = false;
    bool revision_required = false;
    bool more_research_required = false;
    std::vector<std::string> unsupported_claims;
    std::vector<std::string> contradicted_claims;
    std::vector<std::string> unresolved_gaps;
    std::vector<std::string> evidence_ids;
    double confidence = 0.0;
    std::string summary;
};

// Ephemeral, host-built view used by verification. It does not become a new
// persistence layer and deliberately carries only references and metadata.
struct common_agent_research_verification_claim {
    std::string claim_id;
    std::string statement;
    std::vector<std::string> evidence_ids;
    std::vector<std::string> resource_uris;
    std::vector<std::string> memory_ids;
    double confidence = 0.0;
};

struct common_agent_research_verification_context {
    const common_plan_state * plan = nullptr;
    const std::vector<common_memory_hit> * memories = nullptr;
    std::vector<common_runtime_resource_ref> input_resources;
    std::vector<common_agent_research_verification_claim> claims;
};

class common_agent_research_answer_verifier {
public:
    virtual ~common_agent_research_answer_verifier() = default;

    virtual common_agent_research_verification verify(
            const common_agent_research_result & research,
            const std::string & draft,
            std::string & error) const = 0;

    virtual common_agent_research_verification verify(
            const common_agent_research_result & research,
            const std::string & draft,
            const common_agent_research_verification_context & context,
            std::string & error) const {
        (void) context;
        return verify(research, draft, error);
    }
};

// Bounded first-version assurance. This deliberately verifies the normalized
// research contract, not semantic claims in free-form draft text.
class common_agent_research_bounded_verifier final : public common_agent_research_answer_verifier {
public:
    common_agent_research_verification verify(
            const common_agent_research_result & research,
            const std::string & draft,
            std::string & error) const override {
        return verify(research, draft, {}, error);
    }

    common_agent_research_verification verify(
            const common_agent_research_result & research,
            const std::string & draft,
            const common_agent_research_verification_context & context,
            std::string & error) const override {
        error.clear();
        common_agent_research_verification result;
        result.evidence_ids = research.critical_evidence_ids;
        result.unresolved_gaps = research.unresolved_claim_ids;

        if (draft.empty()) {
            result.decision = common_agent_research_verification_decision::revise_answer;
            result.revision_required = true;
            result.summary = "research draft is empty";
            return result;
        }
        if (!research.complete || !research.unresolved_claim_ids.empty() ||
                research.coverage.unresolved_critical_gaps > 0) {
            result.decision = common_agent_research_verification_decision::gather_evidence;
            result.more_research_required = true;
            result.summary = "research result still has unresolved knowledge gaps";
            return result;
        }
        if (research.critical_evidence_ids.empty() ||
                research.coverage.evidence_quality <= 0.0) {
            result.decision = common_agent_research_verification_decision::fail_with_uncertainty;
            result.summary = "research result has no usable provenance-backed evidence";
            return result;
        }

        if (!context.claims.empty()) {
            std::vector<std::string> known_evidence = research.critical_evidence_ids;
            if (context.plan) {
                for (const auto & observation : context.plan->observations) {
                    known_evidence.insert(known_evidence.end(),
                        observation.evidence_ids.begin(), observation.evidence_ids.end());
                }
                for (const auto & assumption : context.plan->assumptions) {
                    known_evidence.insert(known_evidence.end(),
                        assumption.evidence_ids.begin(), assumption.evidence_ids.end());
                }
            }
            const auto has_evidence = [&](const std::string & id) {
                return std::find(known_evidence.begin(), known_evidence.end(), id) != known_evidence.end();
            };
            const auto evidence_for = [&](const std::string & id) -> const common_agent_research_evidence * {
                for (const auto & evidence : research.evidence) {
                    if (evidence.evidence_id == id) return &evidence;
                }
                return nullptr;
            };
            const auto has_resource = [&](const std::string & uri) {
                return std::any_of(context.input_resources.begin(), context.input_resources.end(),
                    [&](const common_runtime_resource_ref & resource) { return resource.uri == uri; });
            };
            const auto has_memory = [&](const std::string & id) {
                return context.memories && std::any_of(context.memories->begin(), context.memories->end(),
                    [&](const common_memory_hit & hit) { return hit.memory.id == id; });
            };
            for (const auto & claim : context.claims) {
                bool supported = false;
                bool contradicted = false;
                for (const auto & id : claim.evidence_ids) supported = supported || has_evidence(id);
                for (const auto & id : claim.evidence_ids) {
                    const auto * evidence = evidence_for(id);
                    contradicted = contradicted || (evidence &&
                        evidence->relation == common_agent_research_evidence_relation::contradicts);
                }
                for (const auto & uri : claim.resource_uris) supported = supported || has_resource(uri);
                for (const auto & id : claim.memory_ids) supported = supported || has_memory(id);
                if (contradicted) {
                    result.contradicted_claims.push_back(
                        claim.claim_id.empty() ? claim.statement : claim.claim_id);
                }
                if (!supported || claim.statement.empty()) {
                    result.unsupported_claims.push_back(
                        claim.claim_id.empty() ? claim.statement : claim.claim_id);
                }
            }
            if (!result.unsupported_claims.empty()) {
                result.decision = common_agent_research_verification_decision::revise_answer;
                result.revision_required = true;
                result.answer_sufficient = false;
                result.confidence = 0.0;
                result.summary = "one or more answer claims lack verified evidence, resource or memory references";
                return result;
            }
            if (!result.contradicted_claims.empty()) {
                result.decision = common_agent_research_verification_decision::revise_answer;
                result.revision_required = true;
                result.answer_sufficient = false;
                result.confidence = 0.0;
                result.summary = "one or more answer claims are contradicted by verified evidence";
                return result;
            }
        }

        result.decision = common_agent_research_verification_decision::accept;
        result.answer_sufficient = true;
        result.confidence = std::clamp(
            research.coverage.evidence_quality * research.coverage.objective_coverage,
            0.0,
            1.0);
        result.summary = "research draft passed bounded evidence and coverage assurance";
        return result;
    }
};
