#pragma once

#include "agent-data-store-cozo-aggregate.h"

bool agent_cozo_execute_native_join(
        const agent_cozo_aggregate_json & request,
        size_t max_scan_rows,
        size_t max_result_rows,
        const agent_cozo_query_runner & run,
        const agent_cozo_scan_metadata_reader & read_scan_metadata,
        std::string & result_json,
        std::string & error);
