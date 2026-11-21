#include "agent-resource-chunk-budget.h"

#include "agent-context-allocation.h"
#include "resource-chunk-extraction.h"

#include <algorithm>

bool common_agent_resolve_resource_chunk_policy(
        common_agent_inference & inference,
        const common_agent_context_budget_config & config,
        const std::string & task,
        common_runtime_resource_chunk_policy & policy,
        std::string & error) {
    error.clear();
    // Larger windows lose evidence with the current per-window extractor and
    // amplify remote token estimation errors. This is a quality bound, not a
    // claim that the model's context capacity is only this large.
    constexpr size_t quality_limit_bytes = 8192;
    constexpr size_t fallback_bytes = 4096;
    constexpr size_t output_tokens = 384;
    constexpr size_t template_and_repair_reserve = 256;
    constexpr size_t minimum_bytes = 4; // A complete UTF-8 code point.
    const size_t capacity = inference.context_capacity_tokens();
    size_t chosen = capacity ? quality_limit_bytes : fallback_bytes;
    if (config.resource_chunk_max_bytes != 0)
        chosen = std::min(chosen, config.resource_chunk_max_bytes);

    if (capacity) {
        common_agent_generation_resource_chunk empty;
        common_agent_generation_options options;
        options.n_predict = static_cast<int>(output_tokens);
        const auto request = common_agent_make_resource_chunk_extraction_request(
            task, empty, std::nullopt, options, "resource-chunk-budget");
        size_t mandatory = 0;
        for (const auto & message : request.messages) {
            const auto count = inference.count_context_text_tokens(message.content);
            // Without the server's tokenizer use one token per byte as a
            // conservative allowance, not an exact token count.
            const size_t part = count.value_or(message.content.size());
            if (part > capacity || mandatory > capacity - part) {
                error = "resource chunk instructions and task exceed model context";
                return false;
            }
            mandatory += part;
        }
        const auto allocation = common_agent_allocate_context(capacity, mandatory,
            output_tokens + template_and_repair_reserve, {{chosen, chosen}});
        if (!allocation.mandatory_fits || allocation.section_tokens.empty() ||
                allocation.section_tokens.front() < minimum_bytes) {
            error = "model context has insufficient room for a resource chunk";
            return false;
        }
        // A byte-per-token bound avoids assuming a token/character ratio for
        // arbitrary code, OCR, non-ASCII text or a remote tokenizer.
        chosen = std::min(chosen, allocation.section_tokens.front());
    }
    if (chosen < minimum_bytes) {
        error = "configured resource chunk maximum is too small";
        return false;
    }
    policy.max_bytes = chosen;
    policy.overlap_bytes = std::min(config.resource_chunk_overlap_bytes, chosen / 4);
    return true;
}
