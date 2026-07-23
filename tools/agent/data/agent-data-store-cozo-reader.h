#pragma once

#include "agent-data-store-cozo-aggregate.h"

#include <vector>

bool agent_cozo_read_dataset(
        const std::string & dataset,
        size_t max_scan_rows,
        std::vector<agent_cozo_aggregate_json> & rows,
        size_t & scanned_rows,
        bool & scan_truncated,
        const agent_cozo_query_runner & run,
        std::string & error);

bool agent_cozo_read_dataset_with_conditions(
        const std::string & dataset,
        const agent_cozo_aggregate_json & conditions,
        size_t max_scan_rows,
        std::vector<agent_cozo_aggregate_json> & rows,
        size_t & scanned_rows,
        bool & scan_truncated,
        const agent_cozo_query_runner & run,
        std::string & error);

bool agent_cozo_read_scan_metadata(
        const std::string & dataset,
        size_t max_scan_rows,
        size_t & scanned_rows,
        bool & scan_truncated,
        const agent_cozo_query_runner & run,
        std::string & error);
