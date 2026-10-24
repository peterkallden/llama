#pragma once

#include "chat.h"
#include <cstddef>
#include <string>
#include <vector>

enum class common_tool_dsl_grammar_form { action, plan, arguments };

// Sampling-time projection of authorized input schemas. Restricts tool/field
// names, scalar types/cardinality, required fields and bounded plan structure.
// Fields are generated in schema order; the codec remains order independent.
// Host validators still own semantic constraints, bindings, authorization,
// array index consistency, directive contents and execution semantics.
// Arguments form requires exactly one selected tool. Empty/invalid catalogs
// return an error instead of silently requesting unconstrained generation.
std::string common_build_tool_dsl_grammar(
    const std::vector<common_chat_tool> & tools,
    common_tool_dsl_grammar_form form,
    std::string & error,
    size_t max_steps = 5,
    size_t max_goal_chars = 256);
