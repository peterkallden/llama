// Bounded JSON object contract validation shared by native invokers.
#pragma once

#include <string>

// Parses an object-shaped external payload, emits its canonical JSON form and
// validates the supported JSON-schema subset (required/properties/types,
// enum, scalar bounds and array bounds). It never supplies semantic defaults.
bool common_schema_normalize_and_validate_object(
    const std::string & input_json,
    const std::string & schema_json,
    std::string & normalized_json,
    std::string & error);
