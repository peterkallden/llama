#pragma once

#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"
#include "agent/adaptation/flydelta/flydelta-job.h"
#include "agent/adaptation/flydelta/oracles/contracts.h"
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

struct common_flydelta_evaluator_callbacks;

namespace agent_daemon_flydelta_internal {
struct daemon_flydelta_resource_provider;

bool daemon_flydelta_fixture_from_job(
        const common_flydelta_experiment_job & job,
        common_flydelta_experiment_fixture & fixture,
        std::string & error);

bool daemon_flydelta_parse_depth(
        const std::string & value, common_flydelta_search_depth & depth);

bool daemon_flydelta_parse_phase(
        const std::string & value, common_flydelta_experiment_phase & phase);

bool daemon_flydelta_parse_refinement(
        const std::string & value, common_flydelta_bootstrap_refinement_kind & kind);

nlohmann::ordered_json daemon_flydelta_orchestration_state_json(
        const common_flydelta_experiment_plan & plan,
        const common_flydelta_utility_history & history,
        const std::string & bootstrap_state_ref);

std::string daemon_flydelta_continuation_key(const common_flydelta_experiment_plan & plan);

std::string daemon_flydelta_bootstrap_key(const common_flydelta_bootstrap_zoom_state & state);

bool daemon_flydelta_orchestration_state_from_json(
        const std::string & text,
        common_flydelta_experiment_plan & plan,
        common_flydelta_utility_history & history,
        std::string & error);


nlohmann::ordered_json daemon_flydelta_anchor_observation_json(
        const common_flydelta_semantic_progress_observation & observation);
bool daemon_flydelta_anchor_observation_from_json(
        const nlohmann::ordered_json & value,
        common_flydelta_semantic_progress_observation & observation, std::string & error);
std::string daemon_flydelta_scope_fingerprint(const common_agent_scope & scope);
std::string daemon_flydelta_verifier_contract_revision(
        const common_flydelta_oracle_registry & registry, const std::string & verifier_ref);
bool daemon_flydelta_bind_state_callbacks(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        common_flydelta_evaluator_callbacks & callbacks, std::string & error);
} // namespace agent_daemon_flydelta_internal
