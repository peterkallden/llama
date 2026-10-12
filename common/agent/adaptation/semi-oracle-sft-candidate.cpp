#include "agent/adaptation/semi-oracle-sft-candidate.h"

#include <algorithm>
#include <cstdint>
#include <nlohmann/json.hpp>
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
            !proof["state"].is_object() || state_fingerprint(proof["state"].dump()) != value.state_fingerprint ||
            !proof.contains("action") || !proof["action"].is_object() ||
            proof["action"].value("row", -1) != value.row ||
            proof["action"].value("column", -1) != value.column) return false;
    const char * outcome = value.value == common_semi_oracle_placement_value::win ? "win" :
        value.value == common_semi_oracle_placement_value::draw ? "draw" :
        value.value == common_semi_oracle_placement_value::loss ? "loss" : "unknown";
    if (proof.value("outcome", std::string()) != outcome) return false;
    std::vector<common_semi_oracle_placement_action_value> verified;
    std::string error;
    if (!common_semi_oracle_rank_placements(ruleset, proof["state"].dump(), {100000, 64}, verified, error)) return false;
    const auto match = std::find_if(verified.begin(), verified.end(), [&](const auto & candidate) {
        return candidate.row == value.row && candidate.column == value.column;
    });
    return match != verified.end() && match->value == value.value;
}
}

bool common_semi_oracle_make_sft_candidate(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_placement_action_value & preferred,
        const std::string & persisted_evaluation_payload_json,
        const common_agent_scope & scope,
        const std::string & approved_prompt,
        const std::string & redaction_policy_id,
        const std::string & redaction_method,
        bool curator_approved,
        common_training_candidate & candidate,
        std::string & error) {
    error.clear();
    if (!common_semi_oracle_validate_ruleset(ruleset, error) ||
            ruleset.domain != common_semi_oracle_domain::placement_game ||
            !action_proof_matches(ruleset, preferred) ||
            preferred.value == common_semi_oracle_placement_value::unknown ||
            !preferred.ranking_complete || ruleset.scope != common_semi_oracle_scope::session ||
            scope.session_id.empty() || scope.session_id != ruleset.scope_id || scope.namespace_id.empty()) {
        if (error.empty()) error = "offline SFT demonstration requires a current, proven placement action";
        return false;
    }
    if (preferred.persisted_evaluation_ref.rfind("semi-oracle://evaluation/", 0) != 0 ||
            approved_prompt.empty() || approved_prompt.size() > 2048 ||
            redaction_policy_id.empty() || redaction_method.empty()) {
        error = "offline SFT demonstration requires a persisted proof reference, bounded prompt and redaction attestation";
        return false;
    }

    const auto evaluation = json::parse(persisted_evaluation_payload_json, nullptr, false);
    const auto ranking = evaluation.is_object() && evaluation.contains("proof_json") &&
        evaluation["proof_json"].is_string()
        ? json::parse(evaluation["proof_json"].get<std::string>(), nullptr, false) : json();
    if (!evaluation.is_object() || evaluation.value("ruleset_id", std::string()) != ruleset.id ||
            evaluation.value("ruleset_revision", uint64_t(0)) != ruleset.revision ||
            evaluation.value("verdict", std::string()) != "SATISFIED" ||
            evaluation.value("evaluator_ref", std::string()) != preferred.evaluator_ref ||
            evaluation.value("evaluator_revision", std::string()) != preferred.evaluator_revision ||
            !ranking.is_object() || ranking.value("algorithm", std::string()) != "memoized_minimax_v1" ||
            ranking.value("state_fingerprint", std::string()) != preferred.state_fingerprint ||
            ranking.value("action_contract_fingerprint", std::string()) != ruleset.action_contract_fingerprint ||
            !ranking.contains("actions") || !ranking["actions"].is_array() || ranking["actions"].empty()) {
        error = "offline SFT demonstration requires a persisted complete ranking for this RuleSet and state";
        return false;
    }
    bool found_preferred = false;
    for (const auto & action : ranking["actions"]) {
        if (!action.is_object() || !action.contains("proof_json") || !action["proof_json"].is_string() ||
                action["proof_json"].get<std::string>().empty() || !action.contains("value") ||
                !action["value"].is_number_integer() ||
                action["value"].get<int>() < static_cast<int>(common_semi_oracle_placement_value::loss) ||
                action["value"].get<int>() > static_cast<int>(common_semi_oracle_placement_value::win)) {
            error = "persisted placement ranking is incomplete or contains an unknown action value";
            return false;
        }
        if (action.value("player", std::string()) == preferred.player &&
                action.value("row", -1) == preferred.row && action.value("column", -1) == preferred.column &&
                action.value("value", 99) == static_cast<int>(preferred.value) &&
                action["proof_json"].get<std::string>() == preferred.proof_json) found_preferred = true;
    }
    if (!found_preferred) {
        error = "selected SFT action is not present in the persisted complete ranking";
        return false;
    }

    std::string split_group;
    const auto proof = json::parse(preferred.proof_json, nullptr, false);
    if (!proof.is_object() || !proof.contains("state") ||
            !common_semi_oracle_placement_state_group(ruleset, proof["state"].dump(), split_group, error)) return false;
    std::ostringstream action;
    action << "player: " << preferred.player << "\nrow: " << preferred.row
           << "\ncolumn: " << preferred.column;
    std::ostringstream key;
    key << "semi-oracle://ruleset/" << ruleset.id << "@" << ruleset.revision
        << "/action/" << ruleset.action_contract_fingerprint;
    const std::string persisted_proof_ref = preferred.persisted_evaluation_ref +
        "/action/" + std::to_string(preferred.row) + "/" + std::to_string(preferred.column);

    candidate = {};
    candidate.id = "learning://candidate/" +
        preferred.persisted_evaluation_ref.substr(std::string("semi-oracle://evaluation/").size()) +
        "/" + std::to_string(preferred.row) + "/" + std::to_string(preferred.column);
    candidate.provenance = common_training_candidate_provenance::host_verified_demonstration;
    candidate.source = common_adaptation_evidence_source::user_taught_concept;
    candidate.transaction_ids = {persisted_proof_ref};
    candidate.cause = common_learning_cause::host_contract;
    candidate.hypothesis = "Select a placement with the best proven minimax outcome under the declared RuleSet.";
    candidate.approved_prompt = approved_prompt;
    candidate.approved_target = action.str();
    candidate.teaching_key = key.str();
    candidate.behavior_key = key.str();
    candidate.learning_case_id = candidate.id;
    candidate.teaching_relation_id = "semi-oracle://position/" + preferred.state_fingerprint;
    candidate.evidence_ref = persisted_proof_ref;
    candidate.host_verifier_ref = preferred.evaluator_ref;
    candidate.host_verifier_revision = preferred.evaluator_revision + ":" +
        ruleset.semantic_fingerprint + ":" + ruleset.action_contract_fingerprint;
    candidate.host_proof_ref = persisted_proof_ref;
    candidate.split_group_key = split_group;
    candidate.verifier_ref = preferred.evaluator_ref;
    candidate.verifier_revision = candidate.host_verifier_revision;
    candidate.observed_occurrences = 0;
    candidate.verified_recoveries = 0;
    candidate.confidence = 1.0f;
    candidate.redaction_policy_id = redaction_policy_id;
    candidate.redaction_method = redaction_method;
    candidate.redaction_status = common_learning_redaction_status::caller_asserted;
    candidate.status = curator_approved ? common_training_candidate_status::approved
                                        : common_training_candidate_status::eligible;
    candidate.learning_domain = "placement_game";
    candidate.tool_family = "semi-oracle";
    candidate.provider_kind = "native";
    return true;
}
