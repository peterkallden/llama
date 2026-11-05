#include "resource-chunk-extraction.h"

// Shared host extraction path. Callers own scheduling, durable progress and
// the policy for unresolved slices; this function owns only model invocation
// and verification of the returned excerpts against the supplied text.
bool common_agent_extract_resource_chunk(
        common_agent_inference & inference, const std::string & task,
        const common_agent_generation_resource_chunk & chunk,
        const std::optional<common_agent_scope> & scope,
        common_agent_generation_options options, const std::string & trace_prefix,
        size_t max_attempts, common_agent_resource_chunk_extraction & extraction,
        int & decoded_tokens, std::string & error) {
    decoded_tokens = 0;
    for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
        const std::string correction = attempt == 0 ? "" :
            "the previous attempt failed because " + error +
            ". Return only complete evidence sentences copied from the visible text, or status: no_findings.";
        const auto request = common_agent_make_resource_chunk_extraction_request(
            task, chunk, scope, options, trace_prefix + ":attempt:" + std::to_string(attempt + 1), correction);
        const auto generated = inference.generate_result(request);
        decoded_tokens += generated.decoded_tokens > 0 ? generated.decoded_tokens : options.n_predict;
        if (!common_agent_generation_succeeded(generated)) {
            error = "resource evidence generation failed: " + generated.error_message;
            if (generated.error_message.empty()) error += " model did not return a completed result";
            if ((options.should_stop && options.should_stop()) || generated.error_message.find("context") != std::string::npos) return false;
            continue;
        }
        if (common_agent_parse_resource_chunk_extraction(generated.content, chunk, extraction, error)) {
            error.clear();
            return true;
        }
    }
    return false;
}
