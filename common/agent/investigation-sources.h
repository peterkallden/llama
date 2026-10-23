#pragma once

#include "investigation-contracts.h"

// Provider IDs are identity domains; source record IDs remain distinct.
std::vector<std::string> common_investigation_search_providers(
    const common_investigation_case_revision &);
bool common_investigation_source_complete(const common_investigation_source &);
