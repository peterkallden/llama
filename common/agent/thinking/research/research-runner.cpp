#include "agent/thinking/research/research-runner.h"

#include "agent/thinking/research/research-workspace.h"
#include "agent/context/openalex-abstract.h"
#include "runtime/runtime-trace.h"
#include "hash/hash.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::ordered_json;

bool build_call(
        const std::string & tool_name,
        const std::string & instruction,
        const common_agent_research_workspace & workspace,
        common_agent_tool_call & call, size_t chunk_max_bytes) {
    call.name = tool_name;
    if (tool_name == "development.build" || tool_name == "development.test") {
        json arguments = {
            {"target", instruction},
            {"configuration", "Debug"},
        };
        if (tool_name == "development.test") arguments["timeout_ms"] = 120000;
        json resources = json::array();
        for (const auto & source : workspace.sources) {
            if (source.resource_ref && resources.size() < 32) {
                resources.push_back(source.resource_ref->uri);
            }
        }
        if (!resources.empty()) arguments["resource_refs"] = std::move(resources);
        call.arguments_json = std::move(arguments).dump();
        return true;
    }
    if (tool_name == "repository.search") {
        call.arguments_json = json{{"query", instruction}, {"max_results", 8}}.dump();
        return true;
    }
    if (tool_name == "web.search") {
        call.arguments_json = json{{"query", instruction}, {"limit", 5}}.dump();
        return true;
    }
    if (tool_name == "resource.read") {
        size_t max_bytes = chunk_max_bytes ? std::min<size_t>(chunk_max_bytes, 8192) : 2048;
        for (const auto & source : workspace.sources) {
            if (source.resource_ref && (source.kind == common_agent_research_source_kind::api_result ||
                    source.resource_ref->mime_type.find("json") != std::string::npos) &&
                    source.resource_ref->uri == instruction) {
                max_bytes = std::min<size_t>(10485760, std::max<size_t>(2048, source.resource_ref->size_bytes));
                break;
            }
        }
        call.arguments_json = json{{"uri", instruction}, {"max_bytes", max_bytes}}.dump();
        return true;
    }
    if (tool_name == "memory.get") {
        call.arguments_json = json{{"id", instruction}}.dump();
        return true;
    }
    if (tool_name == "web.fetch") {
        // Let the catalogued tool default (10 MiB) apply. Research must not
        // silently narrow this to a small preview-sized fetch.
        call.arguments_json = json{{"url", instruction}}.dump();
        return true;
    }
    return false;
}

std::string bounded_summary(const common_tool_execution_result & result, const std::string & tool_name,
        bool * normalized_abstract = nullptr, bool * incomplete_json = nullptr,
        bool * structured_metadata = nullptr) {
    if (normalized_abstract) *normalized_abstract = false;
    if (incomplete_json) *incomplete_json = false;
    if (structured_metadata) *structured_metadata = false;
    if (tool_name == "resource.read") {
        const auto payload = json::parse(result.output, nullptr, false);
        if (payload.is_object()) {
            const auto & value = payload.contains("result") && payload["result"].is_object()
                ? payload["result"] : payload;
            if (value.contains("content") && value["content"].is_string()) {
                const auto content = value["content"].get<std::string>();
                const auto record = json::parse(content, nullptr, false);
                if (record.is_discarded() && !content.empty() && content.front() == '{' && incomplete_json)
                    *incomplete_json = true;
                if (record.is_object() && record.contains("abstract_inverted_index")) {
                    bool truncated = false;
                    const auto abstract = common_agent_openalex_abstract(record["abstract_inverted_index"], truncated);
                    if (!abstract.empty()) {
                        if (normalized_abstract) *normalized_abstract = true;
                        std::string summary = "OpenAlex abstract: " + abstract;
                        if (truncated) summary += " [Abstract reconstruction truncated.]";
                        return summary;
                    }
                }
                if (record.is_object() || record.is_array()) {
                    if (structured_metadata) *structured_metadata = true;
                    size_t end = std::min<size_t>(content.size(), 4096);
                    while (end > 0 && end < content.size() &&
                            (static_cast<unsigned char>(content[end]) & 0xc0) == 0x80) --end;
                    return "Structured source metadata (not full text): " + content.substr(0, end) +
                        (end < content.size() ? " [Metadata projection truncated.]" : "");
                }
                return content;
            }
        }
    }
    std::string summary = !result.content_summary.empty()
        ? result.content_summary
        : (!result.safe_summary.empty() ? result.safe_summary : result.output);
    if (summary.size() > 2048) summary.resize(2048);
    return summary;
}

std::string existing_source_id_for_resource(
        const common_agent_research_workspace & workspace,
        const std::vector<common_runtime_resource_ref> & resources,
        const std::string & requested_uri) {
    for (const auto & source : workspace.sources) {
        if (source.resource_ref && source.resource_ref->uri == requested_uri) return source.source_id;
    }
    for (const auto & resource : resources) {
        for (const auto & source : workspace.sources) {
            if (source.resource_ref && source.resource_ref->uri == resource.uri) return source.source_id;
        }
    }
    return {};
}

std::string existing_source_id_for_memory(
        const common_agent_research_workspace & workspace,
        const std::string & memory_id) {
    for (const auto & source : workspace.sources) {
        if (source.kind == common_agent_research_source_kind::memory && source.memory_id == memory_id) {
            return source.source_id;
        }
    }
    return {};
}

} // namespace

common_agent_research_runtime_adapter::common_agent_research_runtime_adapter(
        const common_agent_tool_runtime & tools,
        const common_agent_research_assessor * assessor,
        std::function<bool()> should_stop, size_t chunk_max_bytes, size_t chunk_overlap_bytes,
        std::function<bool(const std::string &, const common_agent_generation_resource_chunk &,
            common_agent_resource_chunk_extraction &, int &, std::string &)> extract_chunk,
        std::function<bool(const common_agent_research_workspace &, std::string &)> progress_sink,
        std::function<bool(const std::string &, common_agent_resource_chunk_plan &, std::string &)> plan_chunks,
        decltype(common_agent_request::research_select_tool) select_tool,
        std::function<void(const std::string &, const std::string &)> selection_trace)
    : tools(tools), assessor(assessor), should_stop(std::move(should_stop)),
      chunk_max_bytes(chunk_max_bytes), chunk_overlap_bytes(chunk_overlap_bytes), extract_chunk(std::move(extract_chunk)), progress_sink(std::move(progress_sink)), plan_chunks(std::move(plan_chunks)), select_tool(std::move(select_tool)), selection_trace(std::move(selection_trace)),
      started_at(std::chrono::steady_clock::now()) {}

bool common_agent_research_runtime_adapter::execute(
        const common_agent_research_action & action,
        common_agent_research_workspace & workspace,
        common_agent_research_event & event,
        std::string & error) const {
    event = {};
    event.task_id = action.task_id;
    event.gap_id = action.gap_id;

    std::string last_failure_code;
    std::string last_failure_summary;
    bool last_failure_retryable = false;
    bool executed_failure = false;
    const bool semantic_search = select_tool &&
        action.preferred_tools != std::vector<std::string>{"resource.read"} &&
        action.preferred_tools != std::vector<std::string>{"web.fetch"} &&
        action.preferred_tools != std::vector<std::string>{"memory.get"};
    std::optional<common_agent_tool_call> selected_call;
    std::string decision_message;
    if (semantic_search) {
        if (!select_tool(action, workspace, tools, nullptr, {}, selected_call, decision_message, error)) return false;
        if (!selected_call) {
            if (selection_trace) selection_trace({}, "no_action");
            event.type = common_agent_research_event_type::no_action;
            event.failure_code = "research.no_action";
            event.failure_summary = decision_message.empty()
                ? "No authorized research action was selected; the question remains unresolved." : decision_message;
            for (const auto & requirement : workspace.source_requirements) {
                if (requirement.gap_id == action.gap_id && !requirement.searched)
                    event.failure_summary += " Requested source not searched: " + requirement.source_id + ".";
            }
            event.retryable = false;
            return true;
        }
        if (selection_trace) selection_trace(selected_call->name, "selected");
    }
    const auto candidates_to_run = selected_call
        ? std::vector<std::string>{selected_call->name} : action.preferred_tools;
    for (const auto & tool_name : candidates_to_run) {
        std::string resolved_tool_name = tool_name;
        std::vector<std::string> candidates;
        if (tools.resolve_tool_name(tool_name, resolved_tool_name, candidates)) {
            // Use the provider's canonical identity so route-envelope checks
            // see the same name as planning and model-facing tool catalogs.
        } else if (tools.has_execution_envelope()) {
            // Never invoke an unresolved legacy alias outside the active route.
            if (!executed_failure) {
                last_failure_code = "tool.unavailable";
                last_failure_summary = "Research tool is unavailable or outside the active policy: " + tool_name;
                last_failure_retryable = tool_name == "memory.get";
            }
            continue;
        }
        if (!tools.is_available(resolved_tool_name)) {
            if (!executed_failure) {
                last_failure_code = "tool.unavailable";
                last_failure_summary = "Research tool is unavailable or outside the active policy: " + resolved_tool_name;
                last_failure_retryable = resolved_tool_name == "memory.get";
            }
            continue;
        }
        common_agent_tool_call call;
        if (selected_call) {
            call = *selected_call;
            call.name = resolved_tool_name;
        } else if (!build_call(resolved_tool_name, action.instruction, workspace, call, chunk_max_bytes)) continue;
        if (resolved_tool_name == "resource.read") {
            const auto resume_source = std::find_if(workspace.sources.begin(), workspace.sources.end(),
                [&](const auto & item) { return item.resource_ref && item.resource_ref->uri == action.instruction; });
            if (resume_source != workspace.sources.end()) {
                if (plan_chunks && resume_source->kind != common_agent_research_source_kind::api_result &&
                        resume_source->resource_ref->mime_type.find("json") == std::string::npos) {
                    auto arguments = json::parse(call.arguments_json);
                    arguments["max_bytes"] = std::min<size_t>(8192, chunk_max_bytes + chunk_overlap_bytes);
                    call.arguments_json = arguments.dump();
                }
                const auto progress = std::find_if(workspace.resource_progress.begin(), workspace.resource_progress.end(),
                    [&](const auto & item) { return item.source_id == resume_source->source_id && item.gap_id == action.gap_id; });
                if (progress != workspace.resource_progress.end() && progress->position.status != common_resource_progress_status::complete) {
                    const auto gap = std::find_if(workspace.gaps.begin(), workspace.gaps.end(),
                        [&](const auto & item) { return item.gap_id == action.gap_id; });
                    const std::string task = gap == workspace.gaps.end() ? workspace.objective.question : gap->question + "\n" + gap->completion_criterion;
                    const auto task_hash = hash_sha256_hex(task.data(), task.size());
                    if (progress->position.status == common_resource_progress_status::stale ||
                            progress->position.unit != common_resource_position_unit::bytes ||
                            progress->position.resource_uri != action.instruction || progress->position.representation != "text" ||
                            progress->position.task_fingerprint != task_hash || progress->chunk_bytes != chunk_max_bytes ||
                            progress->overlap_bytes != chunk_overlap_bytes ||
                            (!resume_source->content_hash.empty() && resume_source->content_hash != progress->position.content_fingerprint)) {
                        progress->position.status = common_resource_progress_status::stale;
                        if (progress_sink && !progress_sink(workspace, error)) return false;
                        error = "research resource continuation is stale or incompatible with the current task or chunk policy";
                        return false;
                    }
                    auto arguments = json::parse(call.arguments_json);
                    arguments["offset"] = progress->position.next_offset;
                    arguments["max_bytes"] = chunk_max_bytes;
                    call.arguments_json = arguments.dump();
                }
            }
        }
        std::string validation_error;
        const bool valid = tools.validate(call, validation_error);
        if (selected_call && selection_trace) selection_trace(call.name, valid ? "validation_ok" : "validation_failed");
        if (!valid && selected_call) {
            std::optional<common_agent_tool_call> repaired;
            std::string ignored_decision;
            if (!select_tool(action, workspace, tools, &call, validation_error, repaired, ignored_decision, error)) return false;
            if (repaired && repaired->name == call.name) call = *repaired;
            validation_error.clear();
            if (!repaired || repaired->name != resolved_tool_name || !tools.validate(call, validation_error)) {
                if (selection_trace) selection_trace(call.name, "repair_failed");
                last_failure_code = "tool.invalid_arguments";
                last_failure_summary = validation_error.empty() ? "Research argument repair did not return the selected operation." : validation_error;
                break;
            }
            if (selection_trace) selection_trace(call.name, "repair_ok");
        } else if (!valid) {
            if (!executed_failure) {
                last_failure_code = "tool.invalid_arguments";
                last_failure_summary = validation_error;
                last_failure_retryable = false;
            }
            continue;
        }

        auto result = tools.execute(call);
        if (!result.ok) {
            executed_failure = true;
            last_failure_code = result.failure_code;
            last_failure_summary = result.safe_summary;
            last_failure_retryable = result.retryable;
            const bool candidate = resolved_tool_name == "web.fetch" && std::any_of(
                workspace.sources.begin(), workspace.sources.end(), [&](const auto & source) {
                    return source.role == "discovery_candidate" && source.origin == action.instruction;
                });
            if (candidate && (result.http.status == 401 || result.http.status == 403 ||
                    result.http.status == 404 || result.http.status == 410)) {
                // Move on to a different discovered candidate; do not
                // convert policy or generic network failures to success.
                last_failure_retryable = true;
            }
            continue;
        }

        if (selected_call) for (auto & requirement : workspace.source_requirements) {
            if (requirement.gap_id == action.gap_id &&
                    std::find(selected_call->research_source_ids.begin(), selected_call->research_source_ids.end(),
                        requirement.source_id) != selected_call->research_source_ids.end()) requirement.searched = true;
        }

        // Provider collections are discovery, not evidence. Retain bounded
        // candidate identities so a later model decision can select a record
        // without inventing an identifier. The provider result remains data.
        if (selected_call && !result.http.provider.empty() && resolved_tool_name != "web.search" &&
                resolved_tool_name != "web.fetch") {
            const auto payload = json::parse(result.output, nullptr, false);
            const auto & value = payload.is_object() && payload.contains("result") && payload["result"].is_object()
                ? payload["result"] : payload;
            const json * items = nullptr;
            if (!selected_call->collection_item_parameter.empty() && !selected_call->collection_items_pointer.empty()) {
                try {
                    const json::json_pointer pointer(selected_call->collection_items_pointer);
                    if (value.contains(pointer) && value.at(pointer).is_array()) items = &value.at(pointer);
                } catch (const json::exception &) {
                    // A malformed host projection must never become evidence.
                }
            }
            if (!selected_call->collection_item_parameter.empty() && !items) {
                event.type = common_agent_research_event_type::task_failed;
                event.failure_code = "research.collection_projection_unavailable";
                event.failure_summary = "The provider collection did not expose its declared candidate projection.";
                return true;
            }
            if (items) {
                for (const auto & item : *items) {
                    if (event.discovered_sources >= 5) break;
                    const auto & key = selected_call->collection_item_parameter;
                    if (!item.is_object() || !item.contains(key) || !item[key].is_string()) continue;
                    const auto id = item[key].get<std::string>();
                    if (id.empty() || id.size() > 2048 || std::any_of(workspace.sources.begin(), workspace.sources.end(),
                            [&](const auto & source) { return source.role == "api_discovery_candidate" && source.origin == id; })) continue;
                    if (common_agent_research_candidate_count(workspace) >=
                            static_cast<size_t>(workspace.budget.max_candidates)) break;
                    common_agent_research_source candidate;
                    candidate.source_id = action.task_id + ":candidate:" + std::to_string(event.discovered_sources + 1);
                    if (item.contains("display_name") && item["display_name"].is_string())
                        candidate.title = item["display_name"].get<std::string>();
                    else if (item.contains("title") && item["title"].is_string())
                        candidate.title = item["title"].get<std::string>();
                    if (candidate.title.empty()) candidate.title = id;
                    if (candidate.title.size() > 512) candidate.title.resize(512);
                    candidate.origin = id;
                    candidate.authority = result.http.provider;
                    candidate.kind = common_agent_research_source_kind::api_result;
                    candidate.role = "api_discovery_candidate";
                    if (!common_agent_research_add_source(workspace, std::move(candidate), error)) return false;
                    ++event.discovered_sources;
                }
                event.type = common_agent_research_event_type::task_completed;
                event.assessment_summary = "Provider search returned candidates; select and inspect a source before making claims.";
                error.clear();
                return true;
            }
            // A fetched API record is retained as a source for the existing
            // resource reader and chunk/checkpoint path, not as a claim.
            if (!result.resource_refs.empty()) {
                const auto arguments = json::parse(selected_call->arguments_json, nullptr, false);
                std::string selected_id;
                if (!selected_call->item_lookup_parameter.empty() && arguments.is_object() &&
                        arguments.contains(selected_call->item_lookup_parameter) &&
                        arguments[selected_call->item_lookup_parameter].is_string())
                    selected_id = arguments[selected_call->item_lookup_parameter].get<std::string>();
                const auto candidate = std::find_if(workspace.sources.begin(), workspace.sources.end(),
                    [&](const auto & item) {
                        if (item.role != "api_discovery_candidate" || item.authority != result.http.provider || selected_id.empty()) return false;
                        const auto slash = item.origin.find_last_of('/');
                        return item.origin == selected_id || (slash != std::string::npos && item.origin.substr(slash + 1) == selected_id);
                    });
                if (candidate != workspace.sources.end()) {
                    if (!common_agent_research_materialize_candidate(workspace, *candidate, result.resource_refs.front(), error)) return false;
                } else {
                    common_agent_research_source source;
                    source.source_id = action.task_id + ":source";
                    source.title = resolved_tool_name;
                    source.origin = result.http.requested_url.empty() ? resolved_tool_name :
                        common_runtime_trace_safe_url(result.http.requested_url);
                    source.authority = result.http.provider;
                    source.kind = common_agent_research_source_kind::api_result;
                    source.role = "discovered_source";
                    source.resource_ref = result.resource_refs.front();
                    source.content_hash = source.resource_ref->metadata.content_hash;
                    source.primary_source = false;
                    if (!common_agent_research_add_source(workspace, std::move(source), error)) return false;
                }
                event.type = common_agent_research_event_type::task_completed;
                event.discovered_sources = 1;
                event.assessment_summary = "Provider record acquired; its contents still require inspection.";
                error.clear();
                return true;
            }
        }

        // Structured API records must be parsed as a whole, not interpreted
        // as unrelated JSON fragments. Read continuation pages through the
        // same authorized tool and charge each call to the research budget.
        const auto structured_source = std::find_if(workspace.sources.begin(), workspace.sources.end(),
            [&](const auto & source) { return (source.kind == common_agent_research_source_kind::api_result ||
                    (source.resource_ref && source.resource_ref->mime_type.find("json") != std::string::npos)) &&
                source.resource_ref && source.resource_ref->uri == action.instruction; });
        if (resolved_tool_name == "resource.read" && structured_source != workspace.sources.end()) {
            auto payload = json::parse(result.output, nullptr, false);
            auto * value = &payload;
            if (payload.is_object() && payload.contains("result") && payload["result"].is_object()) value = &payload["result"];
            if (value->is_object() && value->contains("content") && (*value)["content"].is_string() &&
                    value->contains("range") && (*value)["range"].is_object()) {
                std::string assembled = (*value)["content"].get<std::string>();
                const std::string initial_hash = value->contains("resource") && (*value)["resource"].is_object()
                    ? (*value)["resource"].value("sha256", std::string{}) : std::string{};
                auto range = (*value)["range"];
                constexpr size_t max_structured_bytes = 10485760;
                size_t offset = 0;
                bool incomplete = !range.contains("resource_complete") || !range["resource_complete"].is_boolean() ||
                    range.value("offset", size_t(-1)) != 0 || range.value("bytes_read", size_t(-1)) != assembled.size() ||
                    assembled.size() > max_structured_bytes;
                // `has_more` and `remaining_bytes` describe only the requested
                // interval. Continue from the whole-resource completion state.
                while (!incomplete && !range.value("resource_complete", false)) {
                    if ((should_stop && should_stop()) || workspace.tool_calls >= workspace.budget.max_tool_calls ||
                            (workspace.budget.max_elapsed_ms > 0 &&
                             std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at).count() >= workspace.budget.max_elapsed_ms) ||
                            !range.contains("next_offset") || !range["next_offset"].is_number_unsigned()) {
                        incomplete = true;
                        break;
                    }
                    const auto next = range["next_offset"].get<size_t>();
                    if (next <= offset || next != assembled.size() || next >= max_structured_bytes) {
                        incomplete = true;
                        break;
                    }
                    const size_t remaining = std::min<size_t>(8192, max_structured_bytes - next);
                    common_agent_tool_call page{resolved_tool_name, json{{"uri", action.instruction},
                        {"offset", next}, {"max_bytes", remaining}}.dump()};
                    if (!tools.validate(page, validation_error)) { incomplete = true; break; }
                    ++workspace.tool_calls;
                    const auto read = tools.execute(page);
                    auto next_payload = json::parse(read.output, nullptr, false);
                    if (next_payload.is_object() && next_payload.contains("result") && next_payload["result"].is_object())
                        next_payload = json(next_payload["result"]);
                    if (!read.ok || !next_payload.is_object() || !next_payload.contains("content") ||
                            !next_payload["content"].is_string() || !next_payload.contains("range") ||
                            !next_payload["range"].is_object() ||
                            !next_payload["range"].contains("resource_complete") ||
                            !next_payload["range"]["resource_complete"].is_boolean()) { incomplete = true; break; }
                    if (!initial_hash.empty() && (!next_payload.contains("resource") || !next_payload["resource"].is_object() ||
                            next_payload["resource"].value("sha256", std::string{}) != initial_hash)) { incomplete = true; break; }
                    const auto page_content = next_payload["content"].get<std::string>();
                    if (page_content.empty() || page_content.size() > remaining ||
                            next_payload["range"].value("offset", size_t(-1)) != next ||
                            next_payload["range"].value("bytes_read", size_t(0)) != page_content.size()) {
                        incomplete = true;
                        break;
                    }
                    offset = next;
                    assembled += page_content;
                    range = next_payload["range"];
                }
                if (incomplete || !range.value("resource_complete", false)) {
                    event.type = common_agent_research_event_type::task_failed;
                    event.retryable = true;
                    event.failure_code = "research.structured_resource_incomplete";
                    event.failure_summary = "Structured resource was not fully read within the research limits; no JSON fields were treated as evidence. A later attempt must reread this record.";
                    event.assessment_summary = event.failure_summary;
                    event.assessment.status = common_agent_research_assessment_status::inconclusive;
                    error.clear();
                    return true;
                }
                (*value)["content"] = std::move(assembled);
                (*value)["range"] = std::move(range);
                result.output = payload.dump();
            }
        }

        // Research-acquired text did not pass through the workflow driver's
        // chunk queue. Use the same host extraction contract on each bounded
        // read, retaining only findings validated against their source slice.
        if (resolved_tool_name == "resource.read" && structured_source == workspace.sources.end() &&
                extract_chunk && chunk_max_bytes != 0) {
            const auto source = std::find_if(workspace.sources.begin(), workspace.sources.end(),
                [&](const auto & item) { return item.resource_ref && item.resource_ref->uri == action.instruction; });
            auto payload = json::parse(result.output, nullptr, false);
            if (payload.is_object() && payload.contains("result") && payload["result"].is_object()) payload = json(payload["result"]);
            if (source != workspace.sources.end() && payload.is_object() &&
                    payload.contains("content") && payload["content"].is_string() &&
                    payload.contains("range") && payload["range"].is_object()) {
                const auto gap = std::find_if(workspace.gaps.begin(), workspace.gaps.end(),
                    [&](const auto & item) { return item.gap_id == action.gap_id; });
                const std::string task = gap == workspace.gaps.end() ? workspace.objective.question :
                    gap->question + "\n" + gap->completion_criterion;
                const auto text_ref = result.resource_refs.empty() ? *source->resource_ref : result.resource_refs.front();
                if (!result.resource_refs.empty()) source->resource_ref = text_ref;
                std::vector<common_agent_research_evidence> findings;
                const auto task_hash = hash_sha256_hex(task.data(), task.size());
                auto progress = std::find_if(workspace.resource_progress.begin(), workspace.resource_progress.end(),
                    [&](const auto & item) { return item.source_id == source->source_id && item.gap_id == action.gap_id; });
                if (progress != workspace.resource_progress.end() && !progress_sink) {
                    error = "resuming research resource analysis requires a persistent checkpoint sink";
                    return false;
                }
                std::string source_hash;
                if (payload.contains("resource") && payload["resource"].is_object()) {
                    source_hash = payload["resource"].value("sha256", std::string{});
                    if (!source_hash.empty()) source_hash = "sha256:" + source_hash;
                }
                if (progress_sink && source_hash.empty()) {
                    error = "resumable research text requires a host-provided content fingerprint";
                    return false;
                }
                if (progress != workspace.resource_progress.end() && !source_hash.empty() &&
                        progress->position.content_fingerprint != source_hash) {
                    progress->position.status = common_resource_progress_status::stale;
                    if (progress_sink && !progress_sink(workspace, error)) return false;
                    error = "research resource content changed since its checkpoint";
                    return false;
                }
                common_agent_resource_chunk_plan chunk_plan;
                if (plan_chunks) {
                    if (!plan_chunks(text_ref.uri, chunk_plan, error)) return false;
                    if (chunk_plan.source_uri != text_ref.uri || chunk_plan.content_fingerprint != source_hash) {
                        error = "host chunk plan does not match the authorized text resource";
                        return false;
                    }
                    size_t covered = 0;
                    for (const auto & planned : chunk_plan.ranges) {
                        if (planned.byte_length == 0 || planned.byte_length > 8192 ||
                                planned.overlap_bytes >= planned.byte_length || planned.byte_offset > chunk_plan.source_bytes ||
                                planned.byte_length > chunk_plan.source_bytes - planned.byte_offset ||
                                planned.byte_offset + planned.overlap_bytes != covered) {
                            error = "host chunk plan contains invalid or discontinuous ranges";
                            return false;
                        }
                        covered = planned.byte_offset + planned.byte_length;
                    }
                    if (covered != chunk_plan.source_bytes) {
                        error = "host chunk plan does not cover the complete text resource";
                        return false;
                    }
                    if (progress != workspace.resource_progress.end() && progress->position.next_offset != 0 &&
                            progress->position.query_fingerprint != "host-text-window-v2") {
                        progress->position.status = common_resource_progress_status::stale;
                        if (progress_sink && !progress_sink(workspace, error)) return false;
                        error = "research checkpoint uses incompatible chunk boundaries; start a new analysis";
                        return false;
                    }
                    if (progress != workspace.resource_progress.end() && progress->position.next_offset == 0)
                        progress->position.query_fingerprint = "host-text-window-v2";
                }
                if (progress_sink && progress == workspace.resource_progress.end()) {
                    common_agent_research_resource_progress created;
                    created.source_id = source->source_id;
                    created.gap_id = action.gap_id;
                    created.chunk_bytes = chunk_max_bytes;
                    created.overlap_bytes = chunk_overlap_bytes;
                    created.position.resource_uri = text_ref.uri;
                    created.position.representation = "text";
                    created.position.content_fingerprint = source_hash;
                    created.position.task_fingerprint = task_hash;
                    if (plan_chunks) created.position.query_fingerprint = "host-text-window-v2";
                    workspace.resource_progress.push_back(std::move(created));
                    progress = std::prev(workspace.resource_progress.end());
                }
                if (progress != workspace.resource_progress.end()) {
                    for (const auto & item : workspace.evidence)
                        if (item.source_id == source->source_id && item.claim_id == action.gap_id &&
                                item.supporting_excerpt_host_verified) findings.push_back(item);
                }
                size_t expected_offset = progress == workspace.resource_progress.end() ? 0 : progress->position.next_offset;
                size_t page_index = 0;
                if (plan_chunks && expected_offset > chunk_plan.source_bytes) {
                    if (progress != workspace.resource_progress.end()) progress->position.status = common_resource_progress_status::stale;
                    if (progress_sink && !progress_sink(workspace, error)) return false;
                    error = "research checkpoint is outside the host-planned resource";
                    return false;
                }
                if (plan_chunks && expected_offset == chunk_plan.source_bytes) page_index = chunk_plan.ranges.size();
                if (plan_chunks && expected_offset < chunk_plan.source_bytes) {
                    while (page_index < chunk_plan.ranges.size() &&
                            chunk_plan.ranges[page_index].byte_offset + chunk_plan.ranges[page_index].overlap_bytes != expected_offset) ++page_index;
                    if (page_index == chunk_plan.ranges.size()) {
                        if (progress != workspace.resource_progress.end()) progress->position.status = common_resource_progress_status::stale;
                        if (progress_sink && !progress_sink(workspace, error)) return false;
                        error = "research checkpoint is not on a host-planned chunk boundary";
                        return false;
                    }
                }
                bool complete = false;
                bool unresolved = false;
                std::string page_error;
                std::string previous_text = progress == workspace.resource_progress.end() ? std::string{} : progress->overlap_text;
                while (true) {
                    // A failed durable write must not leave an advanced cursor
                    // in the final turn checkpoint. Findings and position are
                    // rolled back together to the last committed page.
                    std::optional<common_agent_research_workspace> before_page;
                    if (progress != workspace.resource_progress.end()) before_page = workspace;
                    if (plan_chunks && page_index < chunk_plan.ranges.size()) {
                        const auto & planned = chunk_plan.ranges[page_index];
                        if (payload["range"].value("offset", size_t(-1)) == planned.byte_offset &&
                                payload["content"].get_ref<const std::string &>().size() >= planned.byte_length) {
                            auto & content = payload["content"].get_ref<std::string &>();
                            content.resize(planned.byte_length);
                            payload["range"]["bytes_read"] = planned.byte_length;
                            payload["range"]["resource_complete"] = planned.byte_offset + planned.byte_length >= chunk_plan.source_bytes;
                        }
                        if (payload["range"].value("offset", size_t(-1)) != planned.byte_offset ||
                                payload["range"].value("bytes_read", size_t(-1)) != planned.byte_length) {
                            if (workspace.tool_calls >= workspace.budget.max_tool_calls || (should_stop && should_stop())) { unresolved = true; break; }
                            common_agent_tool_call chunk_call{resolved_tool_name, json{{"uri", text_ref.uri},
                                {"offset", planned.byte_offset}, {"max_bytes", planned.byte_length}}.dump()};
                            if (!tools.validate(chunk_call, page_error)) { unresolved = true; break; }
                            ++workspace.tool_calls;
                            const auto read = tools.execute(chunk_call);
                            payload = json::parse(read.output, nullptr, false);
                            if (payload.is_object() && payload.contains("result") && payload["result"].is_object()) payload = json(payload["result"]);
                            if (!read.ok || !payload.is_object() || !payload.contains("range") || !payload["range"].is_object() ||
                                    !payload.contains("content") || !payload["content"].is_string() ||
                                    !payload.contains("resource") || !payload["resource"].is_object() ||
                                    "sha256:" + payload["resource"].value("sha256", std::string{}) != source_hash) { unresolved = true; break; }
                        }
                    }
                    const auto & range = payload["range"];
                    const auto text = payload["content"].get<std::string>();
                    const size_t planned_overlap = plan_chunks && page_index < chunk_plan.ranges.size() ? chunk_plan.ranges[page_index].overlap_bytes : 0;
                    const size_t page_offset = plan_chunks && page_index < chunk_plan.ranges.size() ? chunk_plan.ranges[page_index].byte_offset : expected_offset;
                    if (range.value("offset", size_t(-1)) != page_offset ||
                            range.value("bytes_read", size_t(-1)) != text.size() ||
                            planned_overlap > text.size() || page_offset + planned_overlap != expected_offset ||
                            (plan_chunks && page_index < chunk_plan.ranges.size() && text.size() != chunk_plan.ranges[page_index].byte_length) ||
                            (text.empty() && !range.value("resource_complete", false))) {
                        unresolved = true;
                        break;
                    }
                    if (!text.empty()) {
                        if ((should_stop && should_stop()) ||
                                (workspace.budget.max_elapsed_ms > 0 &&
                                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at).count() >= workspace.budget.max_elapsed_ms) ||
                                (workspace.budget.max_tokens > 0 && workspace.generation_tokens_reserved + 384 > workspace.budget.max_tokens)) {
                            unresolved = true;
                            break;
                        }
                        common_agent_generation_resource_chunk chunk;
                        size_t overlap = plan_chunks ? 0 : std::min(chunk_overlap_bytes, previous_text.size());
                        size_t overlap_start = previous_text.size() - overlap;
                        while (overlap_start < previous_text.size() &&
                                (static_cast<unsigned char>(previous_text[overlap_start]) & 0xc0) == 0x80) ++overlap_start;
                        overlap = previous_text.size() - overlap_start;
                        chunk.text = previous_text.substr(overlap_start) + text;
                        chunk.source = text_ref;
                        chunk.chunk_index = page_index;
                        chunk.byte_offset = plan_chunks ? page_offset : expected_offset - overlap;
                        chunk.byte_length = chunk.text.size();
                        if (plan_chunks) {
                            common_runtime_resource_text_view view;
                            if (!common_runtime_resource_make_text_view(chunk.text, view, page_error)) { unresolved = true; break; }
                            chunk.text = common_runtime_resource_render_text_segments(view, 0, view.source_offsets.back(), chunk.segments);
                            for (auto & segment : chunk.segments) segment.byte_offset += chunk.byte_offset;
                            if (chunk.segments.empty()) { unresolved = true; break; }
                        } else {
                            chunk.segments.push_back({"P1", chunk.text, chunk.byte_offset, chunk.text.size()});
                        }
                        // Charge a conservative reservation before inference,
                        // including failed or interrupted extraction attempts.
                        workspace.generation_tokens_reserved += 384;
                        if (progress_sink && !progress_sink(workspace, page_error)) {
                            // Do not make the model call unless the pre-call
                            // checkpoint is durable. Otherwise a daemon crash
                            // can leave work that was never checkpointed, or
                            // let an unpersisted reservation advance the task.
                            if (before_page) workspace = std::move(*before_page);
                            error = page_error.empty()
                                ? "could not persist research checkpoint before chunk extraction"
                                : page_error;
                            return false;
                        }
                        common_agent_resource_chunk_extraction extraction;
                        int used = 0;
                        const bool extracted = extract_chunk(task, chunk, extraction, used, page_error);
                        if (!extracted || extraction.unresolved) { unresolved = true; break; }
                        for (const auto & item : extraction.findings) {
                            if (std::any_of(findings.begin(), findings.end(), [&](const auto & previous) {
                                    return previous.supporting_excerpt == item.evidence;
                                })) continue;
                            common_agent_research_evidence evidence;
                            evidence.evidence_id = action.task_id + ":page:" + std::to_string(page_index) +
                                ":finding:" + std::to_string(findings.size());
                            evidence.source_id = source->source_id;
                            evidence.claim_id = action.gap_id;
                            // The parser verifies the excerpt against the slice;
                            // a model-authored finding is not itself a quotation.
                            evidence.statement = item.evidence;
                            evidence.source_location = text_ref.uri + "#bytes=" + std::to_string(item.evidence_byte_offset) +
                                "-" + std::to_string(item.evidence_byte_offset + item.evidence_byte_length);
                            evidence.relation = common_agent_research_evidence_relation::supports;
                            evidence.origin = common_agent_research_evidence_origin::direct_source;
                            evidence.relevance = 0.75;
                            evidence.confidence = 0.9;
                            evidence.directly_observed = true;
                            evidence.supporting_excerpt = item.evidence;
                            evidence.supporting_excerpt_host_verified = true;
                            if (progress != workspace.resource_progress.end()) {
                                const auto key = source->source_id + "\n" + action.gap_id + "\n" + item.evidence;
                                evidence.evidence_id = "research-chunk:" + hash_sha256_hex(key.data(), key.size());
                                evidence.relation = common_agent_research_evidence_relation::contextualizes;
                                if (!common_agent_research_record_evidence(workspace, evidence, error)) return false;
                                if (gap != workspace.gaps.end()) gap->evidence_ids.push_back(evidence.evidence_id);
                            }
                            findings.push_back(std::move(evidence));
                        }
                        previous_text = plan_chunks ? text : chunk.text;
                    }
                    expected_offset += text.size() - planned_overlap;
                    if (progress != workspace.resource_progress.end()) {
                        progress->position.next_offset = expected_offset;
                        progress->position.more = range.value("resource_complete", false) ? common_resource_more::no : common_resource_more::yes;
                        size_t tail = previous_text.size() > chunk_overlap_bytes ? previous_text.size() - chunk_overlap_bytes : 0;
                        while (tail < previous_text.size() && (static_cast<unsigned char>(previous_text[tail]) & 0xc0) == 0x80) ++tail;
                        progress->overlap_text = previous_text.substr(tail);
                        if (!progress_sink(workspace, error)) {
                            workspace = std::move(*before_page);
                            return false;
                        }
                    }
                    if (range.value("resource_complete", false)) { complete = true; break; }
                    if (workspace.tool_calls >= workspace.budget.max_tool_calls || (should_stop && should_stop()) ||
                            (workspace.budget.max_elapsed_ms > 0 &&
                             std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at).count() >= workspace.budget.max_elapsed_ms)) {
                        unresolved = true;
                        break;
                    }
                    const size_t next_page_index = page_index + 1;
                    if (plan_chunks && next_page_index >= chunk_plan.ranges.size()) { unresolved = true; break; }
                    const size_t next_offset = plan_chunks ? chunk_plan.ranges[next_page_index].byte_offset : expected_offset;
                    const size_t next_size = plan_chunks ? chunk_plan.ranges[next_page_index].byte_length : std::min<size_t>(chunk_max_bytes, 8192);
                    common_agent_tool_call page{resolved_tool_name, json{{"uri", text_ref.uri},
                        {"offset", next_offset}, {"max_bytes", next_size}}.dump()};
                    if (!tools.validate(page, page_error)) { unresolved = true; break; }
                    ++workspace.tool_calls;
                    const auto read = tools.execute(page);
                    payload = json::parse(read.output, nullptr, false);
                    if (payload.is_object() && payload.contains("result") && payload["result"].is_object()) payload = json(payload["result"]);
                    if (!read.ok || !payload.is_object() || !payload.contains("content") ||
                            !payload["content"].is_string() || !payload.contains("range") || !payload["range"].is_object()) {
                        unresolved = true;
                        break;
                    }
                    if (!source_hash.empty() && (!payload.contains("resource") || !payload["resource"].is_object() ||
                            "sha256:" + payload["resource"].value("sha256", std::string{}) != source_hash)) {
                        unresolved = true;
                        page_error = "resource content changed while reading research pages";
                        break;
                    }
                    ++page_index;
                }
                common_agent_research_assessment best;
                bool sufficient = false;
                std::optional<common_agent_research_workspace> before_finalization;
                if (progress != workspace.resource_progress.end()) before_finalization = workspace;
                for (auto & evidence : findings) {
                    if (!complete || unresolved) {
                        evidence.relation = common_agent_research_evidence_relation::contextualizes;
                        if (evidence.statement.rfind("[Partial resource analysis.] ", 0) != 0)
                            evidence.statement = "[Partial resource analysis.] " + evidence.statement;
                    } else if (assessor && gap != workspace.gaps.end()) {
                        evidence.relation = common_agent_research_evidence_relation::contextualizes;
                        const std::string partial_prefix = "[Partial resource analysis.] ";
                        if (evidence.statement.rfind(partial_prefix, 0) == 0) evidence.statement.erase(0, partial_prefix.size());
                        const auto assessment = assessor->assess(*gap, evidence);
                        if (assessment.status == common_agent_research_assessment_status::sufficient) {
                            evidence.relation = common_agent_research_evidence_relation::supports;
                            sufficient = true;
                            best = assessment;
                        }
                    }
                    event.evidence_ids.push_back(evidence.evidence_id);
                    const auto existing = std::find_if(workspace.evidence.begin(), workspace.evidence.end(),
                        [&](const auto & item) { return item.evidence_id == evidence.evidence_id; });
                    if (existing != workspace.evidence.end()) *existing = evidence;
                    else if (!common_agent_research_record_evidence(workspace, evidence, error)) return false;
                }
                if (progress != workspace.resource_progress.end()) {
                    progress->position.status = complete && !unresolved ? common_resource_progress_status::complete : common_resource_progress_status::unresolved;
                    if (!progress_sink(workspace, error)) {
                        workspace = std::move(*before_finalization);
                        return false;
                    }
                }
                event.type = common_agent_research_event_type::task_completed;
                event.evidence_count = static_cast<int>(event.evidence_ids.size());
                if (event.evidence_count == 0) event.discovered_sources = 1; // The source was inspected; try another.
                event.gap_sufficiently_answered = complete && !unresolved && sufficient;
                event.gap_confidence = event.gap_sufficiently_answered ? best.confidence : 0.0;
                event.assessment = event.gap_sufficiently_answered ? best : common_agent_research_assessment{
                    common_agent_research_assessment_status::inconclusive, 0.0,
                    complete && !unresolved ? "The analyzed resource has not established sufficient findings." :
                        "Resource analysis is incomplete; later pages or findings remain unresolved."};
                event.assessment_summary = event.assessment.summary;
                error.clear();
                return true;
            }
        }

        // Search results are discovery candidates, not evidence that resolves
        // the research question. In particular, a generic “N candidates”
        // summary must never become a supporting claim or a primary source.
        // A later fetch/read can materialize source-backed evidence.
        if (resolved_tool_name == "web.search") {
            const auto payload = json::parse(result.output, nullptr, false);
            const auto & value = payload.is_object() && payload.contains("result") && payload["result"].is_object()
                ? payload["result"] : payload;
            if (value.is_object() && value.contains("results") && value["results"].is_array()) {
                for (const auto & candidate : value["results"]) {
                    if (!candidate.is_object() || !candidate.contains("url") || !candidate["url"].is_string()) continue;
                    const auto url = candidate["url"].get<std::string>();
                    // Persist only public-looking URLs without credentials or
                    // query secrets; web.fetch still enforces host policy.
                    if (url.rfind("https://", 0) != 0 || url.size() > 2048 ||
                            common_runtime_trace_safe_url(url) != url) continue;
                    if (std::any_of(workspace.sources.begin(), workspace.sources.end(),
                            [&](const auto & source) { return source.origin == url; })) continue;
                    if (common_agent_research_candidate_count(workspace) >=
                            static_cast<size_t>(workspace.budget.max_candidates)) break;
                    common_agent_research_source source;
                    source.source_id = action.task_id + ":candidate:" + std::to_string(event.discovered_sources + 1);
                    source.title = candidate.contains("title") && candidate["title"].is_string()
                        ? candidate["title"].get<std::string>().substr(0, 512) : url;
                    if (source.title.empty()) source.title = url;
                    source.origin = url;
                    source.authority = "search-candidate";
                    source.kind = common_agent_research_source_kind::web_page;
                    source.role = "discovery_candidate";
                    if (!common_agent_research_add_source(workspace, std::move(source), error)) return false;
                    ++event.discovered_sources;
                }
            }
            event.type = common_agent_research_event_type::task_completed;
            event.gap_sufficiently_answered = false;
            event.gap_confidence = 0.0;
            event.assessment_summary = "web search found candidates but did not read a source; the research gap remains open";
            event.assessment = {common_agent_research_assessment_status::insufficient, 0.0,
                event.assessment_summary};
            error.clear();
            return true;
        }

        if (resolved_tool_name == "web.fetch" && !result.resource_refs.empty()) {
            const auto candidate = std::find_if(workspace.sources.begin(), workspace.sources.end(),
                [&](const auto & source) { return source.role == "discovery_candidate" && source.origin == action.instruction; });
            if (candidate != workspace.sources.end()) {
                if (!common_agent_research_materialize_candidate(workspace, *candidate, result.resource_refs.front(), error)) return false;
                event.type = common_agent_research_event_type::task_completed;
                event.discovered_sources = 1;
                event.assessment_summary = "Fetched a candidate; its contents still require inspection.";
                error.clear();
                return true;
            }
        }

        const std::string existing_source_id = existing_source_id_for_resource(
            workspace, result.resource_refs, action.instruction);
        const std::string memory_source_id = (resolved_tool_name == "memory.get")
            ? existing_source_id_for_memory(workspace, action.instruction)
            : std::string();
        const std::string source_id = !memory_source_id.empty()
            ? memory_source_id
            : (existing_source_id.empty() ? action.task_id + ":source" : existing_source_id);
        if (!existing_source_id.empty() || !memory_source_id.empty()) {
            common_agent_research_evidence evidence;
            evidence.evidence_id = action.task_id + ":evidence";
            evidence.source_id = source_id;
            evidence.claim_id = action.gap_id;
            bool normalized_abstract = false;
            bool incomplete_json = false;
            bool structured_metadata = false;
            evidence.statement = bounded_summary(result, resolved_tool_name, &normalized_abstract, &incomplete_json, &structured_metadata);
            if (incomplete_json) {
                event.type = common_agent_research_event_type::task_completed;
                event.assessment_summary = "Structured resource was only partially read; no JSON fields were treated as evidence.";
                event.assessment.status = common_agent_research_assessment_status::insufficient;
                error.clear();
                return true;
            }
            if (evidence.statement.empty()) evidence.statement = "Research resource was read successfully.";
            bool partial_resource = false;
            if (resolved_tool_name == "resource.read") {
                const auto payload = json::parse(result.output, nullptr, false);
                const auto & value = payload.is_object() && payload.contains("result") && payload["result"].is_object()
                    ? payload["result"] : payload;
                partial_resource = value.is_object() && value.contains("range") && value["range"].is_object() &&
                    !value["range"].value("resource_complete", false);
            }
            evidence.source_location = result.resource_refs.empty()
                ? action.instruction
                : result.resource_refs.front().uri;
            // A successful read provides context; it does not establish that
            // the content supports the research question.
            evidence.relation = common_agent_research_evidence_relation::contextualizes;
            evidence.origin = resolved_tool_name == "resource.read"
                ? common_agent_research_evidence_origin::direct_source
                : common_agent_research_evidence_origin::normalized_tool_result;
            if (normalized_abstract) {
                evidence.source_location = action.instruction + "#/abstract_inverted_index";
                evidence.origin = common_agent_research_evidence_origin::normalized_tool_result;
                evidence.relation = common_agent_research_evidence_relation::contextualizes;
            } else if (resolved_tool_name == "resource.read" && !structured_metadata && !evidence.statement.empty()) {
                evidence.supporting_excerpt = evidence.statement;
                evidence.supporting_excerpt_host_verified = true;
                evidence.directly_observed = true;
            }
            if (partial_resource) {
                evidence.statement = "[Partial resource read; further pages were not analyzed.] " + evidence.statement;
                evidence.relation = common_agent_research_evidence_relation::contextualizes;
            }
            evidence.relevance = 0.75;
            evidence.confidence = resolved_tool_name == "memory.get" ? 0.0 : 0.9;
            if (!common_agent_research_record_evidence(workspace, evidence, error)) return false;
            const auto gap = std::find_if(workspace.gaps.begin(), workspace.gaps.end(),
                [&](const auto & candidate) { return candidate.gap_id == action.gap_id; });
            event.type = common_agent_research_event_type::task_completed;
            event.evidence_count = 1;
            event.evidence_ids = {evidence.evidence_id};
            event.gap_sufficiently_answered = false;
            event.gap_confidence = evidence.confidence;
            event.assessment_summary = "The resource contents have not been semantically assessed.";
            if (resolved_tool_name == "memory.get") {
                event.gap_sufficiently_answered = false;
                event.gap_confidence = 0.0;
                event.assessment_summary = "Scoped memory was retrieved as context; its claims require independent assessment.";
                event.assessment = {common_agent_research_assessment_status::insufficient, 0.0,
                    event.assessment_summary};
            } else if (assessor && gap != workspace.gaps.end()) {
                event.assessment = assessor->assess(*gap, evidence);
                event.gap_confidence = event.assessment.confidence;
                event.gap_sufficiently_answered =
                    event.assessment.status == common_agent_research_assessment_status::sufficient;
                event.assessment_summary = event.assessment.summary;
            }
            if (normalized_abstract) {
                // A bibliographic abstract is useful evidence but lexical
                // overlap cannot establish full research coverage by itself.
                event.gap_sufficiently_answered = false;
                event.assessment = {common_agent_research_assessment_status::insufficient,
                    evidence.confidence, "Abstract evidence requires further coverage assessment."};
                event.assessment_summary = event.assessment.summary;
            }
            if (structured_metadata) {
                event.gap_sufficiently_answered = false;
                event.assessment = {common_agent_research_assessment_status::inconclusive,
                    0.0, "Structured source metadata does not establish full-text analysis."};
                event.assessment_summary = event.assessment.summary;
            }
            if (partial_resource) {
                event.gap_sufficiently_answered = false;
                event.assessment = {common_agent_research_assessment_status::insufficient,
                    evidence.confidence, "Only part of the resource was inspected."};
                event.assessment_summary = event.assessment.summary;
            }
            if (event.gap_sufficiently_answered) {
                const auto stored = std::find_if(workspace.evidence.begin(), workspace.evidence.end(),
                    [&](const auto & item) { return item.evidence_id == evidence.evidence_id; });
                if (stored != workspace.evidence.end()) {
                    stored->relation = common_agent_research_evidence_relation::supports;
                    stored->confidence = event.assessment.confidence;
                }
            }
            event.assessment = {
                event.assessment.status,
                event.assessment.confidence,
                event.assessment.summary.empty() ? event.assessment_summary : event.assessment.summary};
            error.clear();
            return true;
        }
        common_agent_research_source source;
        source.source_id = source_id;
        source.title = resolved_tool_name;
        source.origin = resolved_tool_name;
        if(!result.http.requested_url.empty())source.origin=common_runtime_trace_safe_url(result.http.requested_url);
        source.authority = "host-tool-runtime";
        source.kind = (resolved_tool_name == "web.search" || resolved_tool_name == "web.fetch")
            ? common_agent_research_source_kind::web_page
            : (!result.resource_refs.empty() &&
                result.resource_refs.front().uri.rfind("artifact://", 0) == 0
                ? common_agent_research_source_kind::remote_agent_result
                : common_agent_research_source_kind::repository_file);
        source.quality_score = result.resource_refs.empty() ? 0.5 : 0.75;
        source.primary_source = true;
        if (!result.resource_refs.empty()) {
            source.resource_ref = result.resource_refs.front();
            source.content_hash = result.resource_refs.front().metadata.content_hash;
            if(result.resource_refs.front().metadata.retrieved_at>0)
                source.retrieved_at=std::to_string(result.resource_refs.front().metadata.retrieved_at);
        }else if(!result.output.empty()){
            source.content_hash="sha256:"+hash_sha256_hex(result.output.data(),result.output.size());
        }
        if (!common_agent_research_add_source(workspace, source, error)) return false;

        common_agent_research_evidence evidence;
        evidence.evidence_id = action.task_id + ":evidence";
        evidence.source_id = source_id;
        evidence.claim_id = action.gap_id;
        evidence.statement = bounded_summary(result, resolved_tool_name);
        if (evidence.statement.empty()) evidence.statement = "Research tool returned a successful result.";
        evidence.source_location = result.resource_refs.empty()
            ? action.instruction
            : result.resource_refs.front().uri;
        evidence.relation = common_agent_research_evidence_relation::contextualizes;
        evidence.origin = common_agent_research_evidence_origin::normalized_tool_result;
        evidence.relevance = 0.75;
        evidence.confidence = source.quality_score;
        if (!common_agent_research_record_evidence(workspace, evidence, error)) return false;

        event.type = common_agent_research_event_type::task_completed;
        event.evidence_count = 1;
        event.evidence_ids = {evidence.evidence_id};
        const auto gap = std::find_if(workspace.gaps.begin(), workspace.gaps.end(),
            [&](const auto & candidate) { return candidate.gap_id == action.gap_id; });
        event.gap_sufficiently_answered = false;
        event.gap_confidence = evidence.confidence;
        event.assessment_summary = "The tool result has not been semantically assessed.";
        if (assessor && gap != workspace.gaps.end()) {
            event.assessment = assessor->assess(*gap, evidence);
            event.gap_confidence = event.assessment.confidence;
            event.gap_sufficiently_answered =
                event.assessment.status == common_agent_research_assessment_status::sufficient;
            event.assessment_summary = event.assessment.summary;
        }
        if (event.gap_sufficiently_answered) {
            const auto stored = std::find_if(workspace.evidence.begin(), workspace.evidence.end(),
                [&](const auto & item) { return item.evidence_id == evidence.evidence_id; });
            if (stored != workspace.evidence.end()) {
                stored->relation = common_agent_research_evidence_relation::supports;
                stored->confidence = event.assessment.confidence;
            }
        }
        event.assessment = {
            event.assessment.status,
            event.assessment.confidence,
            event.assessment.summary.empty() ? event.assessment_summary : event.assessment.summary};
        error.clear();
        return true;
    }

    event.type = common_agent_research_event_type::task_failed;
    event.retryable = last_failure_retryable;
    event.failure_code = (!executed_failure && last_failure_code.empty())
        ? "research.acquisition_failed" : last_failure_code;
    event.failure_summary = last_failure_summary.empty()
        ? "All host-approved acquisition tools failed."
        : last_failure_summary;
    error.clear();
    return true;
}

common_agent_research_result common_agent_research_runner::run(
        common_agent_research_workspace & workspace,
        const common_agent_research_tool_executor & executor,
        std::string & error,
        const std::function<bool()> & should_stop,
        const std::function<common_agent_research_stop_reason()> & stop_reason,
        const common_agent_research_lifecycle_sink & lifecycle_sink,
        const std::function<bool(const common_agent_research_workspace &, std::string &)> & progress_sink) const {
    common_agent_research_controller controller(executor.evidence_assessor());
    const auto emit_lifecycle = [&](common_agent_research_lifecycle_event event) {
        if (lifecycle_sink) lifecycle_sink(event);
    };
    for (const auto & gap : workspace.gaps) {
        emit_lifecycle({common_agent_research_lifecycle_event_type::gap_opened, gap.gap_id, {}, 0});
    }
    const auto finalize_with_reason = [&](common_agent_research_stop_reason reason) {
        auto result = controller.finalize(workspace);
        result.stop_reason = reason;
        result.complete = false;
        return result;
    };
    const auto started_at = std::chrono::steady_clock::now();
    std::string terminal_task_failure;
    std::string no_action_message;
    auto action = controller.begin(workspace, error);
    if (!error.empty()) return controller.finalize(workspace);

    int iterations = 0;
    while (action.kind == common_agent_research_action_kind::schedule_task) {
        if (should_stop && should_stop()) {
            return finalize_with_reason(
                stop_reason ? stop_reason() : common_agent_research_stop_reason::cancelled);
        }
        if (workspace.iterations_completed >= workspace.budget.max_iterations) {
            auto result = controller.finalize(workspace);
            result.stop_reason = common_agent_research_stop_reason::budget_exhausted;
            result.complete = false;
            return result;
        }
        if (workspace.tool_calls >= workspace.budget.max_tool_calls) {
            auto result = controller.finalize(workspace);
            result.stop_reason = common_agent_research_stop_reason::budget_exhausted;
            result.complete = false;
            return result;
        }
        if (workspace.budget.max_elapsed_ms > 0 &&
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - started_at).count() >= workspace.budget.max_elapsed_ms) {
            auto result = controller.finalize(workspace);
            result.stop_reason = common_agent_research_stop_reason::deadline_exceeded;
            result.complete = false;
            return result;
        }
        ++iterations;
        bool retry = false;
        for (const auto & task : workspace.tasks) {
            if (task.task_id == action.task_id) {
                retry = task.attempt > 0;
                break;
            }
        }
        emit_lifecycle({common_agent_research_lifecycle_event_type::task_scheduled,
            action.gap_id, action.task_id, iterations, retry});
        emit_lifecycle({common_agent_research_lifecycle_event_type::task_started,
            action.gap_id, action.task_id, iterations});
        common_agent_research_event event;
        bool dependencies_ready = true;
        for (const auto & dependency_id : action.dependency_ids) {
            const auto dependency = std::find_if(workspace.tasks.begin(), workspace.tasks.end(),
                [&](const auto & task) { return task.task_id == dependency_id; });
            if (dependency == workspace.tasks.end() ||
                    dependency->status != common_agent_research_task_status::completed) {
                dependencies_ready = false;
                break;
            }
        }
        if (!dependencies_ready) {
            event.type = common_agent_research_event_type::task_failed;
            event.failure_code = "research.dependency_not_ready";
            event.failure_summary = "Research task dependency is missing or incomplete.";
            event.retryable = false;
        }
        ++workspace.tool_calls;
        if (dependencies_ready && !executor.execute(action, workspace, event, error)) return controller.finalize(workspace);
        if (event.type == common_agent_research_event_type::no_action) {
            --workspace.tool_calls;
            no_action_message = event.failure_summary;
        }
        if (event.type == common_agent_research_event_type::task_failed && !event.retryable) {
            terminal_task_failure = event.failure_code.empty()
                ? event.failure_summary
                : event.failure_code + ": " + event.failure_summary;
        }
        emit_lifecycle({
            event.type == common_agent_research_event_type::task_failed
                ? common_agent_research_lifecycle_event_type::task_failed
                : common_agent_research_lifecycle_event_type::task_completed,
            action.gap_id,
            action.task_id,
            iterations});
        emit_lifecycle({common_agent_research_lifecycle_event_type::iteration_completed,
            action.gap_id, action.task_id, iterations});
        ++workspace.iterations_completed;
        const size_t comparisons_before = workspace.comparisons.size();
        action = controller.advance(workspace, event, error);
        if (workspace.comparisons.size() > comparisons_before) {
            emit_lifecycle({common_agent_research_lifecycle_event_type::sources_compared,
                action.gap_id.empty() ? event.gap_id : action.gap_id,
                action.task_id.empty() ? event.task_id : action.task_id,
                iterations});
        }
        if (!error.empty()) return controller.finalize(workspace);
        // Persist the completed observation and next scheduled state together.
        // The host sink turns an active task into a recoverable interrupted
        // task before writing a durable checkpoint.
        if (progress_sink && !progress_sink(workspace, error)) return controller.finalize(workspace);
        if (should_stop && should_stop()) {
            return finalize_with_reason(
                stop_reason ? stop_reason() : common_agent_research_stop_reason::cancelled);
        }
    }

    auto result = controller.finalize(workspace);
    result.stop_reason = action.stop_reason;
    if (!no_action_message.empty()) {
        result.no_action_message = no_action_message;
        result.synthesis_context += "Research selection: " + no_action_message + "\n";
    }
    result.complete = action.stop_reason == common_agent_research_stop_reason::sufficient_coverage ||
        action.stop_reason == common_agent_research_stop_reason::success_criteria_met;
    if (!result.complete && error.empty() && !terminal_task_failure.empty()) {
        error = std::move(terminal_task_failure);
    }
    return result;
}
