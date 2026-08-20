#pragma once

#include "agent/agent-bootstrap.h"

#include <string>

// Stable, logical JSON representation used by package import/export. Persisted
// ids, embeddings and runtime state never appear in this format.
bool common_agent_package_parse_json(const std::string & text, common_agent_bootstrap_package & package, std::string & error);
bool common_agent_package_to_json(const common_agent_bootstrap_package & package, std::string & text, std::string & error, bool pretty = true);
