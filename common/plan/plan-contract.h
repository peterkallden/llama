#pragma once

#include <string>
#include <vector>

// Shared semantic view of a JSON-schema object property. This is deliberately
// smaller than either the runtime schema or the model-facing renderer IR.
// It is the single extraction seam for typed dataflow and model projection.
struct common_plan_schema_field {
    std::string name;
    std::string semantic_type;
    std::string role;
    bool required = false;
    bool inferable = false;
};

bool common_plan_extract_schema_fields(
        const std::string & schema_json,
        std::vector<common_plan_schema_field> & fields,
        std::string & error);
