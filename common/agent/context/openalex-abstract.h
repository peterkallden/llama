#pragma once
#include "inverted-index-text.h"

// The reconstructed text remains an abstract, not evidence from the full paper.
template<class Json>
std::string common_agent_openalex_abstract(const Json & index, bool & truncated) {
    return common_agent_inverted_index_text(index, truncated);
}
