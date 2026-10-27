#pragma once

#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/oracles/contracts.h"
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

namespace agent_daemon_flydelta_internal {
struct daemon_flydelta_resource_provider;

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
bool daemon_flydelta_register_evaluator(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        common_flydelta_evaluator_config & config,
        common_flydelta_evaluator_callbacks & callbacks, std::string & error);
} // namespace agent_daemon_flydelta_internal
