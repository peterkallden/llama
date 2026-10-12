#include "agent/adaptation/semi-oracle-teaching.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <sstream>

namespace {
using json = nlohmann::ordered_json;

std::string state_fingerprint(const std::string & text) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : text) { hash ^= byte; hash *= 1099511628211ULL; }
    std::ostringstream out;
    out << "fnv1a64:" << std::hex << hash;
    return out.str();
}

bool action_proof_matches(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_placement_action_value & value) {
    const auto proof = json::parse(value.proof_json, nullptr, false);
    if (!proof.is_object() || proof.value("algorithm", std::string()) != "memoized_minimax_v1" ||
            proof.value("ruleset_id", std::string()) != ruleset.id ||
            proof.value("ruleset_revision", uint64_t(0)) != ruleset.revision ||
            proof.value("evaluator_ref", std::string()) != value.evaluator_ref ||
            proof.value("evaluator_revision", std::string()) != value.evaluator_revision ||
            proof.value("state_fingerprint", std::string()) != value.state_fingerprint ||
            proof.value("rule_fingerprint", std::string()) != ruleset.semantic_fingerprint ||
            proof.value("action_contract_fingerprint", std::string()) != ruleset.action_contract_fingerprint ||
            proof.value("player", std::string()) != value.player || !proof.contains("state") ||
            !proof["state"].is_object() ||
            state_fingerprint(proof["state"].dump()) != value.state_fingerprint ||
            !proof.contains("action") ||
            !proof["action"].is_object() || proof["action"].value("row", -1) != value.row ||
            proof["action"].value("column", -1) != value.column) return false;
    const char * outcome = value.value == common_semi_oracle_placement_value::win ? "win" :
        value.value == common_semi_oracle_placement_value::draw ? "draw" :
        value.value == common_semi_oracle_placement_value::loss ? "loss" : "unknown";
    if (proof.value("outcome", std::string()) != outcome) return false;
    std::vector<common_semi_oracle_placement_action_value> verified;
    std::string error;
    if (!common_semi_oracle_rank_placements(
            ruleset, proof["state"].dump(), {100000, 64}, verified, error)) return false;
    const auto match = std::find_if(verified.begin(), verified.end(), [&](const auto & candidate) {
        return candidate.row == value.row && candidate.column == value.column;
    });
    return match != verified.end() && match->value == value.value;
}
}

bool common_semi_oracle_make_teaching_relation(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_result & rejected,
        const common_semi_oracle_result & accepted,
        const common_agent_scope & scope,
        const std::string & relation_id,
        const std::string & task_fingerprint,
        const std::string & rejected_execution_ref,
        const std::string & accepted_execution_ref,
        const std::string & evidence_ref,
        common_flydelta_teaching_relation & relation,
        std::string & error) {
    error.clear();
    if (!common_semi_oracle_learning_authorized(ruleset, rejected) ||
            !common_semi_oracle_learning_authorized(ruleset, accepted) ||
            rejected.verdict != common_semi_oracle_verdict::violated ||
            accepted.verdict != common_semi_oracle_verdict::satisfied ||
            rejected.evaluator_ref != accepted.evaluator_ref ||
            rejected.evaluator_revision != accepted.evaluator_revision) {
        error = "semi-oracle teaching requires a verified rejected/accepted contrast from one evaluator and RuleSet revision";
        return false;
    }
    if (relation_id.empty() || task_fingerprint.empty() || rejected_execution_ref.empty() ||
            accepted_execution_ref.empty() || evidence_ref.empty() ||
            rejected_execution_ref == accepted_execution_ref ||
            scope.namespace_id.empty() || scope.session_id.empty()) {
        error = "semi-oracle teaching relation is missing host execution or scope references";
        return false;
    }

    std::ostringstream identity;
    identity << "semi-oracle://ruleset/" << ruleset.id << "@" << ruleset.revision
             << "/action/" << ruleset.action_contract_fingerprint;
    const auto key = identity.str();

    relation = {};
    relation.id = relation_id;
    relation.teaching_key = key;
    relation.source = common_adaptation_evidence_source::user_taught_concept;
    relation.behavior_key = key;
    relation.scope = scope;
    relation.task_fingerprint = task_fingerprint;
    relation.baseline_ref = rejected_execution_ref;
    relation.conditioned_ref = accepted_execution_ref;
    relation.verifier_ref = accepted.evaluator_ref;
    relation.oracle_revision = accepted.evaluator_revision + ":" +
        ruleset.semantic_fingerprint + ":" + ruleset.action_contract_fingerprint;
    relation.evidence_ref = evidence_ref;
    relation.contrast_ref = rejected_execution_ref + "<>" + accepted_execution_ref;
    relation.status = common_flydelta_teaching_relation_status::resolved;
    relation.baseline_origin = common_flydelta_teaching_origin::observed;
    relation.conditioned_origin = common_flydelta_teaching_origin::host_derived;
    relation.confidence = 1.0f;
    relation.host_approved = true;
    return common_flydelta_teaching_relation_validate(relation, error);
}

bool common_semi_oracle_make_preference_relation(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_placement_action_value & rejected,
        const common_semi_oracle_placement_action_value & preferred,
        const common_agent_scope & scope,
        const std::string & relation_id,
        const std::string & rejected_execution_ref,
        const std::string & preferred_execution_ref,
        const std::string & control_execution_ref,
        const std::string & evidence_ref,
        common_flydelta_teaching_relation & relation,
        std::string & error) {
    error.clear();
    const auto known_value = [](common_semi_oracle_placement_value value) {
        return value == common_semi_oracle_placement_value::loss ||
            value == common_semi_oracle_placement_value::draw ||
            value == common_semi_oracle_placement_value::win;
    };
    if (!common_semi_oracle_validate_ruleset(ruleset, error)) return false;
    const bool authorized_rules = std::all_of(ruleset.assertions.begin(), ruleset.assertions.end(),
        [](const common_semi_oracle_rule_assertion & assertion) {
            return assertion.authority != common_semi_oracle_rule_authority::inferred;
        });
    const bool same_evaluation = rejected.ruleset_id == ruleset.id &&
        preferred.ruleset_id == ruleset.id &&
        rejected.ruleset_revision == ruleset.revision &&
        preferred.ruleset_revision == ruleset.revision &&
        rejected.evaluator_ref == "semi-oracle://evaluator/placement_game" &&
        preferred.evaluator_ref == rejected.evaluator_ref &&
        rejected.evaluator_revision == "minimax-v1" &&
        preferred.evaluator_revision == rejected.evaluator_revision &&
        !rejected.state_fingerprint.empty() &&
        rejected.state_fingerprint == preferred.state_fingerprint &&
        !rejected.persisted_evaluation_ref.empty() &&
        rejected.persisted_evaluation_ref.rfind("semi-oracle://evaluation/", 0) == 0 &&
        rejected.persisted_evaluation_ref == preferred.persisted_evaluation_ref &&
        !rejected.player.empty() && rejected.player == preferred.player &&
        (rejected.row != preferred.row || rejected.column != preferred.column) &&
        known_value(rejected.value) && known_value(preferred.value) &&
        action_proof_matches(ruleset, rejected) && action_proof_matches(ruleset, preferred) &&
        static_cast<int>(preferred.value) > static_cast<int>(rejected.value);
    if (!authorized_rules || !same_evaluation) {
        error = "semi-oracle preference requires a proven, strictly better action pair from one state and RuleSet revision";
        return false;
    }
    if (ruleset.scope != common_semi_oracle_scope::session ||
            ruleset.scope_id != scope.session_id || scope.namespace_id.empty() ||
            scope.session_id.empty() || relation_id.empty() ||
            rejected_execution_ref.empty() || preferred_execution_ref.empty() ||
            control_execution_ref.empty() || evidence_ref.empty() ||
            evidence_ref.rfind(rejected.persisted_evaluation_ref, 0) != 0 ||
            rejected_execution_ref == preferred_execution_ref) {
        error = "semi-oracle preference relation is missing session-scoped execution, control or evidence references";
        return false;
    }

    std::ostringstream identity;
    identity << "semi-oracle://ruleset/" << ruleset.id << "@" << ruleset.revision
             << "/action/" << ruleset.action_contract_fingerprint;
    std::ostringstream contrast;
    contrast << "minimax-v1:" << rejected.state_fingerprint << ":"
             << rejected.player << "[" << rejected.row << "," << rejected.column
             << "]=" << static_cast<int>(rejected.value) << "<"
             << preferred.player << "[" << preferred.row << "," << preferred.column
             << "]=" << static_cast<int>(preferred.value);

    relation = {};
    relation.id = relation_id;
    relation.teaching_key = identity.str();
    relation.source = common_adaptation_evidence_source::user_taught_concept;
    relation.behavior_key = identity.str();
    relation.scope = scope;
    relation.task_fingerprint = rejected.state_fingerprint;
    relation.baseline_ref = rejected_execution_ref;
    relation.conditioned_ref = preferred_execution_ref;
    relation.control_ref = control_execution_ref;
    relation.verifier_ref = rejected.evaluator_ref;
    relation.oracle_revision = rejected.evaluator_revision + ":" +
        ruleset.semantic_fingerprint + ":" + ruleset.action_contract_fingerprint;
    relation.evidence_ref = evidence_ref;
    relation.contrast_ref = contrast.str();
    relation.status = common_flydelta_teaching_relation_status::resolved;
    relation.baseline_origin = common_flydelta_teaching_origin::observed;
    relation.conditioned_origin = common_flydelta_teaching_origin::host_derived;
    relation.control_origin = common_flydelta_teaching_origin::host_counterfactual;
    relation.confidence = 1.0f;
    relation.host_approved = true;
    return common_flydelta_teaching_relation_validate(relation, error);
}
