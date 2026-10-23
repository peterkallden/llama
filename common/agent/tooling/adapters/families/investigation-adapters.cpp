#include "agent/tooling/adapters/families/investigation-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/artifact/artifact-contract.h"
#include "agent/investigation-access.h"
#include "agent/investigation-log-parser.h"
#include "hash/hash.h"
#include "resource/resource-retention.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
using json = common_adapter_json;

std::string investigation_page_cursor(uint64_t revision, size_t offset, const json & binding) {
    const auto serialized = binding.dump();
    const std::string fingerprint = hash_sha256_hex(serialized.data(), serialized.size());
    return std::to_string(revision) + ":" + std::to_string(offset) + ":" + fingerprint;
}

bool investigation_page_offset(const std::string & cursor, uint64_t revision,
        const json & binding, size_t & offset) {
    offset = 0;
    if (cursor.empty()) return true;
    const auto first = cursor.find(':');
    const auto second = first == std::string::npos ? first : cursor.find(':', first + 1);
    if (first == std::string::npos || second == std::string::npos || cursor.find(':', second + 1) != std::string::npos) return false;
    if (first == 0 || second == first + 1 ||
            !std::all_of(cursor.begin(), cursor.begin() + first, [](unsigned char c) { return c >= '0' && c <= '9'; }) ||
            !std::all_of(cursor.begin() + first + 1, cursor.begin() + second, [](unsigned char c) { return c >= '0' && c <= '9'; })) return false;
    uint64_t cursor_revision = 0;
    size_t cursor_offset = 0;
    try {
        size_t used_revision = 0, used_offset = 0;
        cursor_revision = std::stoull(cursor.substr(0, first), &used_revision);
        const auto parsed_offset = std::stoull(cursor.substr(first + 1, second - first - 1), &used_offset);
        if (parsed_offset > std::numeric_limits<size_t>::max()) return false;
        cursor_offset = static_cast<size_t>(parsed_offset);
        if (used_revision != first || used_offset != second - first - 1) return false;
    } catch (...) { return false; }
    const auto serialized = binding.dump();
    const std::string fingerprint = hash_sha256_hex(serialized.data(), serialized.size());
    if (cursor_revision != revision || cursor.substr(second + 1) != fingerprint) return false;
    offset = cursor_offset;
    return true;
}


bool parse_investigation_time(const json & value, int64_t & out) {
    if (value.is_number_unsigned()) {
        const auto v = value.get<uint64_t>();
        if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return false;
        out = static_cast<int64_t>(v); return true;
    }
    if (value.is_number_integer()) { out = value.get<int64_t>(); return true; }
    if (value.is_number_float()) return false;
    if (!value.is_string()) return false;
    const std::string text = value.get<std::string>();
    char * end = nullptr;
    errno = 0;
    const auto integer = std::strtoll(text.c_str(), &end, 10);
    if (errno != ERANGE && end != text.c_str() && *end == '\0') { out = integer; return true; }
    if (text.size() < 20) return false;
    std::tm parsed{};
    std::istringstream input(text.substr(0, 19));
    input >> std::get_time(&parsed, "%Y-%m-%dT%H:%M:%S");
    if (input.fail()) return false;
    const std::tm original = parsed;
    const time_t seconds = timegm(&parsed);
    if (parsed.tm_year != original.tm_year || parsed.tm_mon != original.tm_mon || parsed.tm_mday != original.tm_mday ||
            parsed.tm_hour != original.tm_hour || parsed.tm_min != original.tm_min || parsed.tm_sec != original.tm_sec) return false;
    int64_t millis = static_cast<int64_t>(seconds) * 1000;
    size_t cursor = 19;
    if (cursor < text.size() && text[cursor] == '.') {
        ++cursor;
        int fraction = 0, digits = 0;
        while (cursor < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor]))) {
            if (digits < 3) fraction = fraction * 10 + (text[cursor] - '0');
            ++digits; ++cursor;
        }
        if (!digits) return false;
        while (digits < 3) { fraction *= 10; ++digits; }
        millis += fraction;
    }
    if (cursor < text.size() && text[cursor] == 'Z') ++cursor;
    else if (cursor + 6 == text.size() && (text[cursor] == '+' || text[cursor] == '-') && text[cursor + 3] == ':') {
        for (size_t i : {cursor + 1,cursor + 2,cursor + 4,cursor + 5})
            if (text[i] < '0' || text[i] > '9') return false;
        const int hours = std::atoi(text.substr(cursor + 1, 2).c_str());
        const int minutes = std::atoi(text.substr(cursor + 4, 2).c_str());
        if (hours > 23 || minutes > 59) return false;
        const int64_t offset_ms = static_cast<int64_t>(hours * 60 + minutes) * 60 * 1000;
        millis += text[cursor] == '+' ? -offset_ms : offset_ms;
        cursor = text.size();
    }
    if (cursor != text.size()) return false;
    out = millis;
    return true;
}

std::string normalize_investigation_event_type(std::string value) {
    std::string normalized;
    normalized.reserve(value.size());
    bool separator = false;
    for (const unsigned char c : value) {
        if (std::isalnum(c)) {
            if (separator && !normalized.empty()) normalized.push_back('_');
            normalized.push_back(static_cast<char>(std::tolower(c)));
            separator = false;
        } else {
            separator = true;
        }
    }
    return normalized;
}

bool investigation_entity_kind_from_name(const std::string & value,
        common_investigation_entity_kind & out) {
    for (const auto kind : {common_investigation_entity_kind::user, common_investigation_entity_kind::host,
            common_investigation_entity_kind::process, common_investigation_entity_kind::ip,
            common_investigation_entity_kind::file, common_investigation_entity_kind::hash,
            common_investigation_entity_kind::service, common_investigation_entity_kind::trace,
            common_investigation_entity_kind::message, common_investigation_entity_kind::event}) {
        if (value == common_investigation_entity_kind_name(kind)) { out = kind; return true; }
    }
    return false;
}

bool investigation_relation_kind_from_name(const std::string & value,
        common_investigation_relation_kind & out) {
    for (const auto kind : {common_investigation_relation_kind::authenticated_to,
            common_investigation_relation_kind::communicated_with, common_investigation_relation_kind::spawned,
            common_investigation_relation_kind::accessed, common_investigation_relation_kind::modified,
            common_investigation_relation_kind::participates_in_trace, common_investigation_relation_kind::follows,
            common_investigation_relation_kind::correlated_with}) {
        if (value == common_investigation_relation_kind_name(kind)) { out = kind; return true; }
    }
    return false;
}

bool investigation_statement_kind_from_name(const std::string & value,
        common_investigation_statement_kind & out) {
    if (value == "statement") out = common_investigation_statement_kind::statement;
    else if (value == "reported_event") out = common_investigation_statement_kind::reported_event;
    else if (value == "definition") out = common_investigation_statement_kind::definition;
    else if (value == "interpretation") out = common_investigation_statement_kind::interpretation;
    else return false;
    return true;
}

bool investigation_comparison_kind_from_name(const std::string & value,
        common_investigation_comparison_kind & out) {
    if (value == "supports") out = common_investigation_comparison_kind::supports;
    else if (value == "contradicts") out = common_investigation_comparison_kind::contradicts;
    else if (value == "qualifies") out = common_investigation_comparison_kind::qualifies;
    else if (value == "same_topic") out = common_investigation_comparison_kind::same_topic;
    else if (value == "source_dependency") out = common_investigation_comparison_kind::source_dependency;
    else if (value == "inconclusive") out = common_investigation_comparison_kind::inconclusive;
    else return false;
    return true;
}

bool investigation_retain_resource(const common_native_tool_bindings & bindings,
        const agent_resource_descriptor & original, const std::string & source_tool,
        agent_resource_descriptor & retained, std::string & error) {
    constexpr size_t max_bytes = 4 * 1024 * 1024;
    if (!bindings.resource_runtime.store || original.size_bytes > max_bytes) {
        error = "investigation resource exceeds its retention limit or storage is unavailable";
        return false;
    }
    const auto authority = make_agent_resource_read_authority(bindings.resource_runtime,
        static_cast<int64_t>(std::time(nullptr)));
    agent_resource_retention_target target;
    target.scope = bindings.tool_scope.project_id.empty()
        ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
    target.namespace_id = bindings.tool_scope.namespace_id;
    target.session_id = bindings.tool_scope.session_id;
    target.project_id = bindings.tool_scope.project_id;
    target.turn_id = bindings.tool_scope.turn_id;
    target.source_tool = source_tool;
    target.max_bytes = max_bytes;
    bool copied = false;
    return retain_agent_resource(*bindings.resource_runtime.store,
        static_cast<const common_runtime_resource_ref &>(original), authority, target, retained, copied, error);
}

bool investigation_make_resource_source(const common_native_tool_bindings & bindings,
        const std::string & case_id, const std::string & uri, const std::string & kind,
        common_investigation_source & source, std::string & error) {
    if (bindings.resource_runtime.store == nullptr || (kind != "document" && kind != "log_file")) {
        error = "investigation resource source is unavailable or has an unsupported type";
        return false;
    }
    const auto authority = make_agent_resource_read_authority(bindings.resource_runtime,
        static_cast<int64_t>(std::time(nullptr)));
    agent_resource_descriptor original;
    if (!bindings.resource_runtime.store->stat(uri, authority, original, error)) return false;
    if (original.size_bytes > 4 * 1024 * 1024) {
        error = "investigation source exceeds the existing 4 MiB per-resource retention limit";
        return false;
    }
    agent_resource_descriptor retained;
    if (!investigation_retain_resource(bindings, original, "investigation.source.attach", retained, error)) return false;
    const std::string identity = case_id + "\n" + uri + "\n" + original.sha256 + "\n" + kind;
    source = {};
    source.id = "source-" + hash_sha256_hex(identity.data(), identity.size()).substr(0, 40);
    source.kind = kind == "log_file" ? common_investigation_source_kind::log_file
                                      : common_investigation_source_kind::document;
    source.resource_uri = retained.uri;
    source.representation_uri = common_resource_media_type_is_text_like(retained.mime_type) ? retained.uri : std::string();
    source.display_name = retained.name;
    source.mime_type = retained.mime_type;
    source.content_hash = retained.sha256.empty() ? std::string() : "sha256:" + retained.sha256;
    source.representation_hash = source.representation_uri.empty() ? std::string() : source.content_hash;
    source.original_uri = original.uri;
    source.original_parent_uri = original.lineage.parent_uri;
    source.processing_status = source.representation_uri.empty() ? "pending" : "ready";
    if (!retained.metadata.limitations.empty()) source.processing_limitations.push_back(retained.metadata.limitations);
    source.complete = source.processing_status == "ready" && source.processing_limitations.empty();
    if (!source.representation_uri.empty() && !source.complete) source.processing_status = "ready_with_limitations";
    if (source.representation_uri.empty()) source.processor_version = "pending-host-processor";
    error.clear();
    return true;
}

bool investigation_authorize_sources(const common_native_tool_bindings & bindings,
        const common_investigation_case_revision & value, std::string & error) {
    return common_investigation_authorize_sources(value, bindings.tool_scope, *bindings.data_store,
        bindings.resource_runtime.store, make_agent_resource_read_authority(bindings.resource_runtime,
            static_cast<int64_t>(std::time(nullptr))), error);
}

bool investigation_publish_revision(const common_native_tool_bindings & bindings,
        common_investigation_case_revision value, uint64_t expected_revision,
        common_investigation_case_revision & published, std::string & error) {
    value.revision = expected_revision + 1;
    const auto revision_uri = "investigation-case://" + value.case_id + "/revision/" + std::to_string(value.revision);
    return bindings.data_store->publish_investigation_case(value, expected_revision,
        revision_uri, published, error);
}


common_tool_execution_result execute_investigation_tool(
        const common_native_tool_bindings & bindings, const std::string & operation,
        const json & arguments) {
    if (operation == "investigation.association.review" && bindings.investigation_reviewer_id.empty())
        return common_adapter_failure("tool.investigation.user_review_required",common_tool_failure_class::policy,false,
            "Association decisions require the authenticated Investigation Workspace user path.",
            "ordinary native/model execution cannot authorize analyst decisions");
    if (!bindings.data_store) return common_adapter_execution_failure(
        "tool.investigation.store_unavailable", "investigation persistence is unavailable",
        "Investigation storage is unavailable.");
    std::string error;
    if (operation == "investigation.source.list") {
        if (arguments.contains("limit") && !arguments["limit"].is_number_integer())
            return common_adapter_validation_failure("tool.investigation.source_list_invalid_arguments",
                "limit must be an integer");
        const size_t limit = arguments.value("limit", size_t(50));
        if (limit == 0 || limit > 100) return common_adapter_validation_failure(
            "tool.investigation.source_list_limit_invalid", "source listing limit must be from 1 to 100");
        std::vector<common_agent_dataset_descriptor> datasets;
        if (!bindings.data_store->list_dataset_descriptors(datasets, error)) return common_adapter_execution_failure(
            "tool.investigation.sources_unavailable", std::move(error), "Configured investigation sources are unavailable.");
        std::map<std::string, std::string> configured;
        for (const auto & dataset : datasets) {
            if (dataset.ref.storage_mode != common_agent_dataset_storage_mode::virtual_ ||
                    dataset.ref.source_provider.empty()) continue;
            std::string authorization_error;
            if (!bindings.data_store->authorize_investigation_source(dataset.ref.source_provider,
                    bindings.tool_scope, authorization_error)) continue;
            configured.emplace(dataset.ref.source_provider, dataset.ref.source_provider);
        }
        json sources = json::array();
        bool truncated = false;
        for (const auto & item : configured) {
            if (sources.size() == limit) { truncated = true; break; }
            sources.push_back({{"source_id", item.first}, {"display_name", item.second}});
        }
        return common_adapter_success_json({{"sources", std::move(sources)}, {"truncated", truncated}});
    }
    if (operation == "investigation.case.create") {
        if (!arguments.is_object() || !arguments.contains("title") || !arguments["title"].is_string() ||
                !arguments.contains("question") || !arguments["question"].is_string() ||
                (arguments.contains("source_id") && !arguments["source_id"].is_string()) ||
                (arguments.contains("resource_uri") && !arguments["resource_uri"].is_string()) ||
                (arguments.contains("resource_kind") && !arguments["resource_kind"].is_string()) ||
                (arguments.contains("window_start_ms") && !arguments["window_start_ms"].is_number_integer()) ||
                (arguments.contains("window_end_ms") && !arguments["window_end_ms"].is_number_integer())) {
            return common_adapter_validation_failure("tool.investigation.case_invalid_arguments",
                "case title, question, optional source fields, and time bounds have invalid types");
        }
        const auto source_id = arguments.value("source_id", std::string());
        const auto resource_uri = arguments.value("resource_uri", std::string());
        if (source_id.empty() && resource_uri.empty()) return common_adapter_validation_failure(
            "tool.investigation.case_source_required", "case creation requires a configured search source or an authorized resource",
            "Choose at least one configured source or document resource.");
        if (!source_id.empty()) {
            std::vector<common_agent_dataset_descriptor> datasets;
            if (!bindings.data_store->list_dataset_descriptors(datasets, error)) {
                return common_adapter_execution_failure("tool.investigation.sources_unavailable",
                    std::move(error), "Configured investigation sources are unavailable.");
            }
            if (!bindings.data_store->authorize_investigation_source(source_id, bindings.tool_scope, error))
                return common_adapter_not_found_failure("tool.investigation.source_unavailable",
                    std::move(error), "Choose a source currently authorized for this scope.");
            const bool source_exists = std::any_of(datasets.begin(), datasets.end(), [&](const auto & descriptor) {
                return descriptor.ref.storage_mode == common_agent_dataset_storage_mode::virtual_ &&
                    descriptor.ref.source_provider == source_id;
            });
            if (!source_exists) return common_adapter_not_found_failure(
                "tool.investigation.source_unavailable", "case source is not configured",
                "Choose a source available to the host.");
        }
        common_investigation_case_revision value;
        static std::atomic<uint64_t> sequence{1};
        value.case_id = "case-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(sequence.fetch_add(1));
        value.title = arguments.value("title", std::string());
        value.question = arguments.value("question", std::string());
        value.source_id = source_id;
        value.window_start_ms = source_id.empty() ? 0 : arguments.value("window_start_ms", int64_t(0));
        value.window_end_ms = source_id.empty() ? 0 : arguments.value("window_end_ms", int64_t(0));
        value.scope = bindings.tool_scope;
        value.revision = 1;
        if (!source_id.empty()) {
            common_investigation_source search_source;
            search_source.id = "provider-" + source_id;
            search_source.kind = common_investigation_source_kind::search_connector;
            search_source.provider_id = source_id;
            search_source.display_name = source_id;
            search_source.processing_status = "not_collected";
            search_source.complete = false;
            value.sources.push_back(std::move(search_source));
        }
        if (!resource_uri.empty()) {
            common_investigation_source resource_source;
            if (!investigation_make_resource_source(bindings, value.case_id, resource_uri,
                    arguments.value("resource_kind", std::string("document")), resource_source, error))
                return common_adapter_not_found_failure("tool.investigation.resource_unavailable",
                    std::move(error), "The selected resource cannot be added to this case.");
            value.sources.push_back(std::move(resource_source));
        }
        common_investigation_case_revision published;
        const auto revision_uri = "investigation-case://" + value.case_id + "/revision/1";
        if (!bindings.data_store->publish_investigation_case(value, 0, revision_uri, published, error)) {
            return common_adapter_execution_failure("tool.investigation.case_create_failed",
                std::move(error), "The investigation case could not be created.");
        }
        return common_adapter_success_json({{"case_id", published.case_id}, {"revision", published.revision},
            {"scope", {{"namespace_id", published.scope.namespace_id},
                {"project_id", published.scope.project_id}, {"session_id", published.scope.session_id}}}});
    }
    if (operation == "investigation.case.list") {
        const size_t limit = arguments.value("limit", size_t(50));
        std::vector<common_investigation_case_summary> cases;
        std::string next_cursor;
        if (!bindings.data_store->list_investigation_cases(bindings.tool_scope, limit,
                arguments.value("after_case_id", std::string()), cases, next_cursor, error)) {
            return common_adapter_execution_failure("tool.investigation.case_list_failed",
                std::move(error), "Cases could not be listed in the current scope.");
        }
        json rows = json::array();
        for (const auto & item : cases) {
            common_investigation_case_revision scoped_case;
            std::string authorization_error;
            if (!bindings.data_store->get_investigation_case(item.case_id, bindings.tool_scope, scoped_case, authorization_error) ||
                    !investigation_authorize_sources(bindings, scoped_case, authorization_error)) continue;
            rows.push_back({{"case_id", item.case_id},
            {"revision", item.revision}, {"title", item.title}, {"question", item.question},
            {"source_id", item.source_id}, {"window_start_ms", item.window_start_ms},
            {"window_end_ms", item.window_end_ms}});
        }
        return common_adapter_success_json({{"cases", std::move(rows)}, {"next_cursor", next_cursor}});
    }
    const auto case_id = arguments.value("case_id", std::string());
    common_investigation_case_revision value;
    if (!bindings.data_store->get_investigation_case(case_id, bindings.tool_scope, value, error)) {
        return common_adapter_not_found_failure("tool.investigation.case_unavailable",
            std::move(error), "The case is unavailable in the current scope.");
    }
    if (!investigation_authorize_sources(bindings, value, error))
        return common_adapter_not_found_failure("tool.investigation.case_unavailable",
            "case source is no longer authorized", "The case is unavailable in the current scope.");
    if (operation == "investigation.source.attach") {
        if (!arguments.contains("expected_revision") || !arguments["expected_revision"].is_number_integer() ||
                (arguments.contains("resource_uri") && !arguments["resource_uri"].is_string()) ||
                (arguments.contains("provider_id") && !arguments["provider_id"].is_string()) ||
                (arguments.contains("kind") && !arguments["kind"].is_string()))
            return common_adapter_validation_failure("tool.investigation.source_invalid_arguments",
                "expected_revision, optional resource/provider reference and kind have invalid types");
        const bool has_resource = arguments.contains("resource_uri") && !arguments.at("resource_uri").get<std::string>().empty();
        const bool has_provider = arguments.contains("provider_id") && !arguments.at("provider_id").get<std::string>().empty();
        if (has_resource == has_provider) return common_adapter_validation_failure(
            "tool.investigation.source_invalid_arguments", "attach exactly one resource_uri or provider_id");
        const auto expected = arguments.value("expected_revision", uint64_t(0));
        common_investigation_source source;
        if (has_resource) {
            if (!investigation_make_resource_source(bindings, value.case_id,
                    arguments.at("resource_uri").get<std::string>(),
                    arguments.value("kind", std::string("document")), source, error))
                return common_adapter_not_found_failure("tool.investigation.resource_unavailable",
                    std::move(error), "The selected resource cannot be added to this case.");
        } else {
            const auto provider_id = arguments.at("provider_id").get<std::string>();
            std::vector<common_agent_dataset_descriptor> datasets;
            if (!bindings.data_store->list_dataset_descriptors(datasets, error)) return common_adapter_execution_failure(
                "tool.investigation.sources_unavailable", std::move(error), "Configured investigation sources are unavailable.");
            const bool configured = std::any_of(datasets.begin(), datasets.end(), [&](const auto & descriptor) {
                return descriptor.ref.storage_mode == common_agent_dataset_storage_mode::virtual_ &&
                    descriptor.ref.source_provider == provider_id;
            });
            if (!configured || !bindings.data_store->authorize_investigation_source(provider_id, bindings.tool_scope, error))
                return common_adapter_not_found_failure("tool.investigation.source_unavailable",
                    error.empty() ? "search source is not configured" : std::move(error),
                    "Choose a search source currently authorized for this scope.");
            source.id = "provider-" + provider_id;
            source.kind = common_investigation_source_kind::search_connector;
            source.provider_id = provider_id;
            source.display_name = provider_id;
            source.processing_status = "not_collected";
            source.complete = false;
        }
        const auto prior = std::find_if(value.sources.begin(), value.sources.end(), [&](const auto & item) {
            return item.id == source.id || (has_provider && item.kind == common_investigation_source_kind::search_connector &&
                item.provider_id == source.provider_id);
        });
        if (prior != value.sources.end()) return common_adapter_success_json({
            {"case_id", value.case_id}, {"revision", value.revision}, {"source_id", prior->id},
            {"resource_uri", prior->resource_uri}, {"processing_status", prior->processing_status}, {"reused", true}});
        if (value.sources.size() >= 32) return common_adapter_limit_failure(
            "tool.investigation.case_source_count_limit", "case already contains the maximum 32 sources",
            "Remove a source or create another case before adding more sources.");
        if (expected != value.revision) return common_adapter_execution_failure(
            "tool.investigation.revision_conflict", "case revision changed before source attachment",
            "Reload the case before adding another source.");
        if (has_provider) {
            const bool has_window_start = arguments.contains("window_start_ms") && arguments["window_start_ms"].is_number_integer();
            const bool has_window_end = arguments.contains("window_end_ms") && arguments["window_end_ms"].is_number_integer();
            if (arguments.contains("window_start_ms") != has_window_start || arguments.contains("window_end_ms") != has_window_end ||
                    has_window_start != has_window_end)
                return common_adapter_validation_failure("tool.investigation.source_window_invalid",
                    "search source time bounds must be supplied together as integers");
            if (value.source_id.empty()) {
                if (!has_window_start || arguments.at("window_start_ms").get<int64_t>() >= arguments.at("window_end_ms").get<int64_t>())
                    return common_adapter_validation_failure("tool.investigation.source_window_required",
                        "attaching the first search connector requires a non-empty time window");
                value.source_id = source.provider_id;
                value.window_start_ms = arguments.at("window_start_ms").get<int64_t>();
                value.window_end_ms = arguments.at("window_end_ms").get<int64_t>();
            } else if (has_window_start && (arguments.at("window_start_ms").get<int64_t>() != value.window_start_ms ||
                    arguments.at("window_end_ms").get<int64_t>() != value.window_end_ms)) {
                return common_adapter_validation_failure("tool.investigation.source_window_mismatch",
                    "all configured search sources in a case must share the case time window");
            }
        }
        if (has_resource) {
        size_t retained_total = 0;
        const auto authority = make_agent_resource_read_authority(bindings.resource_runtime,
            static_cast<int64_t>(std::time(nullptr)));
        for (const auto & item : value.sources) if (!item.resource_uri.empty()) {
            agent_resource_descriptor descriptor;
            if (!bindings.resource_runtime.store->stat(item.resource_uri, authority, descriptor, error))
                return common_adapter_not_found_failure("tool.investigation.source_unavailable",
                    std::move(error), "An existing case source is no longer available.");
            retained_total += descriptor.size_bytes;
        }
        agent_resource_descriptor descriptor;
        if (!bindings.resource_runtime.store->stat(source.resource_uri, authority, descriptor, error) ||
                retained_total + descriptor.size_bytes > 16 * 1024 * 1024)
            return common_adapter_limit_failure("tool.investigation.case_source_limit",
                "case resources exceed the existing 16 MiB project-retention limit",
                "This resource would exceed the case source retention limit.");
        }
        value.sources.push_back(std::move(source));
        common_investigation_case_revision published;
        if (!investigation_publish_revision(bindings, std::move(value), expected, published, error))
            return common_adapter_execution_failure("tool.investigation.source_publish_failed",
                std::move(error), "The source was retained but was not published in the case revision.");
        const auto & added = published.sources.back();
        return common_adapter_success_json({{"case_id", published.case_id}, {"revision", published.revision},
            {"source_id", added.id}, {"resource_uri", added.resource_uri}, {"processing_status", added.processing_status},
            {"reused", false}});
    }
    if (operation == "investigation.document.prepare") {
        if (!arguments.contains("expected_revision") || !arguments["expected_revision"].is_number_integer() ||
                !arguments.contains("source_id") || !arguments["source_id"].is_string())
            return common_adapter_validation_failure("tool.investigation.document_invalid_arguments",
                "expected_revision and source_id are required");
        const auto expected = arguments.value("expected_revision", uint64_t(0));
        auto source = std::find_if(value.sources.begin(), value.sources.end(), [&](const auto & item) {
            return item.id == arguments.at("source_id").get<std::string>();
        });
        if (source == value.sources.end() || source->kind == common_investigation_source_kind::search_connector)
            return common_adapter_not_found_failure("tool.investigation.document_unavailable",
                "selected document source is not part of this case", "Choose a document or log-file source in the open case.");
        if (!source->representation_uri.empty()) return common_adapter_success_json({
            {"case_id", value.case_id}, {"revision", value.revision}, {"source_id", source->id},
            {"resource_uri", source->representation_uri}, {"processor_version", source->processor_version}, {"reused", true}});
        if (expected != value.revision) return common_adapter_execution_failure(
            "tool.investigation.revision_conflict", "case revision changed before document preparation",
            "Reload the case before preparing the document source.");
        if (bindings.resource_runtime.store == nullptr ||
                (bindings.resource_processing_service == nullptr && !bindings.resource_processing_provider_factory))
            return common_adapter_not_found_failure("tool.investigation.processor_unavailable",
                "no configured processor can produce readable text for this resource",
                "This source has no available text representation or configured OCR processor.");
        const auto authority = make_agent_resource_read_authority(bindings.resource_runtime,
            static_cast<int64_t>(std::time(nullptr)));
        agent_resource_descriptor original;
        if (!bindings.resource_runtime.store->stat(source->resource_uri, authority, original, error))
            return common_adapter_not_found_failure("tool.investigation.document_unavailable",
                std::move(error), "The selected document resource is unavailable.");
        agent_resource_processing_binding_request request;
        request.source_uri = original.uri;
        request.operation_id = "investigation-document/" + value.case_id + "/" + source->id + "/" +
            std::to_string(value.revision);
        request.authority = authority;
        request.media_type.declared_type = original.mime_type;
        request.target_representation = "text";
        request.purpose = agent_resource_processing_purpose::normalization;
        std::shared_ptr<agent_resource_processing_provider> owned;
        auto * provider = bindings.resource_processing_service;
        if (bindings.resource_processing_provider_factory) {
            owned = bindings.resource_processing_provider_factory(request);
            provider = owned.get();
        }
        if (provider == nullptr) return common_adapter_execution_failure(
            "tool.investigation.processor_unavailable", "document processor could not be initialized",
            "The selected document could not be prepared.");
        const auto processed = provider->process(request);
        if (!processed.success || processed.resources.empty()) return common_adapter_not_found_failure(
            "tool.investigation.processing_failed", processed.safe_summary,
            "No readable text representation could be produced. The original resource is unchanged.");
        agent_resource_descriptor representation;
        if (!bindings.resource_runtime.store->stat(processed.resources.front().uri, authority, representation, error) ||
                !common_resource_media_type_is_text_like(representation.mime_type))
            return common_adapter_execution_failure("tool.investigation.processing_invalid",
                "processor output is not an authorized readable text resource", "Document processing produced an invalid representation.");
        agent_resource_descriptor retained_representation;
        if (!investigation_retain_resource(bindings, representation, "investigation.document.prepare",
                retained_representation, error)) return common_adapter_execution_failure(
            "tool.investigation.processing_retention_failed", std::move(error),
            "The derived text could not be retained for this case's scope.");
        representation = std::move(retained_representation);
        source->representation_uri = representation.uri;
        source->representation_hash = representation.sha256.empty() ? std::string() : "sha256:" + representation.sha256;
        source->processor_version = processed.processor_id;
        if (!representation.metadata.limitations.empty())
            source->processing_limitations.push_back(representation.metadata.limitations);
        source->complete = source->processing_limitations.empty();
        source->processing_status = source->complete ? "ready" : "ready_with_limitations";
        const std::string prepared_source_id = source->id;
        const std::string prepared_resource_uri = source->representation_uri;
        const std::string prepared_processor_version = source->processor_version;
        common_investigation_case_revision published;
        if (!investigation_publish_revision(bindings, std::move(value), expected, published, error))
            return common_adapter_execution_failure("tool.investigation.processing_publish_failed",
                std::move(error), "The derived text was created but was not published in the case revision.");
        return common_adapter_success_json({{"case_id", published.case_id}, {"revision", published.revision},
            {"source_id", prepared_source_id}, {"resource_uri", prepared_resource_uri},
            {"processor_version", prepared_processor_version}, {"reused", false}});
    }
    if (operation == "investigation.statement.record") {
        if (!arguments.contains("expected_revision") || !arguments["expected_revision"].is_number_integer() ||
                !arguments.contains("idempotency_key") || !arguments["idempotency_key"].is_string() ||
                !arguments.contains("analysis_kind") || !arguments["analysis_kind"].is_string() ||
                !arguments.contains("source_ids") || !arguments["source_ids"].is_array() ||
                !arguments.contains("statements") || !arguments["statements"].is_array() ||
                !arguments.contains("comparisons") || !arguments["comparisons"].is_array())
            return common_adapter_validation_failure("tool.investigation.analysis_invalid_arguments",
                "analysis kind, idempotency key, source IDs, statements and comparisons are required");
        const auto expected = arguments.value("expected_revision", uint64_t(0));
        const auto idempotency_key = arguments.at("idempotency_key").get<std::string>();
        const auto analysis_kind = arguments.at("analysis_kind").get<std::string>();
        static const std::set<std::string> analysis_kinds = {"actor_relationship", "narrative_comparison",
            "document_provenance", "hypothesis_testing", "mixed_event_chain"};
        if (idempotency_key.empty() || idempotency_key.size() > 128 || !analysis_kinds.count(analysis_kind) ||
                arguments["statements"].size() > 64 || arguments["comparisons"].size() > 64 ||
                arguments["source_ids"].empty() || arguments["source_ids"].size() > 32)
            return common_adapter_validation_failure("tool.investigation.analysis_out_of_bounds",
                "analysis operation exceeds its supported type or size limits");
        json request_body = arguments;
        request_body.erase("expected_revision"); request_body.erase("case_id");
        request_body.erase("idempotency_key");
        const std::string parameters = nlohmann::json(request_body).dump();
        const std::string parameters_fingerprint = "sha256:" + hash_sha256_hex(parameters.data(), parameters.size());
        const std::string analysis_identity = value.case_id + "\n" + idempotency_key;
        const std::string analysis_id = "analysis-" + hash_sha256_hex(
            analysis_identity.data(), analysis_identity.size()).substr(0, 40);
        const auto existing = std::find_if(value.analyses.begin(), value.analyses.end(), [&](const auto & item) {
            return item.id == analysis_id;
        });
        if (existing != value.analyses.end()) {
            if (existing->parameters_fingerprint != parameters_fingerprint)
                return common_adapter_failure("tool.investigation.idempotency_conflict", common_tool_failure_class::conflict,
                    false, "This analysis key was already used for different results.", "analysis idempotency payload mismatch");
            return common_adapter_success_json({{"case_id", value.case_id}, {"revision", value.revision},
                {"published_revision", existing->published_revision}, {"analysis_id", analysis_id}, {"reused", true}});
        }
        if (expected != value.revision) return common_adapter_execution_failure(
            "tool.investigation.revision_conflict", "case revision changed before analysis publication",
            "Reload the case and retry the analysis against its current revision.");
        std::set<std::string> analysis_sources;
        for (const auto & source_ref : arguments["source_ids"]) {
            if (!source_ref.is_string() || !analysis_sources.insert(source_ref.get<std::string>()).second ||
                    std::none_of(value.sources.begin(), value.sources.end(), [&](const auto & source) { return source.id == source_ref.get<std::string>(); }))
                return common_adapter_validation_failure("tool.investigation.analysis_source_invalid",
                    "analysis references a duplicate or unselected source");
        }
        const auto authority = make_agent_resource_read_authority(bindings.resource_runtime,
            static_cast<int64_t>(std::time(nullptr)));
        std::map<std::string, std::string> source_texts;
        for (const auto & source_ref : analysis_sources) {
            const auto source = std::find_if(value.sources.begin(), value.sources.end(), [&](const auto & item) { return item.id == source_ref; });
            if (source != value.sources.end() && source->kind == common_investigation_source_kind::search_connector) continue;
            if (source == value.sources.end() || source->representation_uri.empty())
                return common_adapter_not_found_failure("tool.investigation.analysis_text_unavailable",
                    "selected source has no prepared text representation", "Prepare the selected document sources before recording extracted statements.");
            std::string text;
            if (!bindings.resource_runtime.store->read_text(source->representation_uri, authority, 4 * 1024 * 1024, text, error))
                return common_adapter_not_found_failure("tool.investigation.analysis_text_unavailable",
                    std::move(error), "The selected source text is no longer available.");
            source_texts.emplace(source_ref, std::move(text));
        }
        for (const auto & item : arguments["statements"]) {
            common_investigation_statement statement;
            if (!item.is_object() || !item.contains("id") || !item["id"].is_string() ||
                    !item.contains("kind") || !item["kind"].is_string() ||
                    !item.contains("text") || !item["text"].is_string() || !item.contains("anchor") || !item["anchor"].is_object())
                return common_adapter_validation_failure("tool.investigation.statement_invalid",
                    "each statement requires bounded text, kind and source anchor fields");
            statement.id = item["id"].get<std::string>(); statement.text = item["text"].get<std::string>();
            if (!investigation_statement_kind_from_name(item["kind"].get<std::string>(), statement.kind)) return common_adapter_validation_failure(
                "tool.investigation.statement_invalid", "statement kind is not supported");
            const auto & anchor = item["anchor"];
            const auto nonnegative_integer = [&](const char * field) {
                return anchor.contains(field) && anchor[field].is_number_integer() &&
                    (anchor[field].is_number_unsigned() || anchor[field].get<int64_t>() >= 0);
            };
            if (!nonnegative_integer("byte_offset") || !nonnegative_integer("byte_length") ||
                    (anchor.contains("page_number") && !nonnegative_integer("page_number")))
                return common_adapter_validation_failure("tool.investigation.statement_anchor_invalid",
                    "source anchor offsets, lengths and optional page numbers must be nonnegative integers");
            try {
                statement.anchor.source_id = anchor.at("source_id").get<std::string>();
                statement.anchor.resource_uri = anchor.at("resource_uri").get<std::string>();
                statement.anchor.byte_offset = anchor.at("byte_offset").get<uint64_t>();
                statement.anchor.byte_length = anchor.at("byte_length").get<uint64_t>();
                statement.anchor.page_number = anchor.value("page_number", int64_t(0));
                statement.anchor.quote = anchor.at("quote").get<std::string>();
            } catch (const std::exception & exception) { return common_adapter_validation_failure(
                "tool.investigation.statement_anchor_invalid", exception.what()); }
            if (!source_texts.count(statement.anchor.source_id) ||
                    !common_investigation_anchor_validate(statement.anchor,
                        source_texts.at(statement.anchor.source_id), error))
                return common_adapter_validation_failure("tool.investigation.statement_anchor_unverified", error);
            const auto selected_source = std::find_if(value.sources.begin(), value.sources.end(), [&](const auto & source) {
                return source.id == statement.anchor.source_id;
            });
            if (selected_source == value.sources.end() || selected_source->representation_uri != statement.anchor.resource_uri)
                return common_adapter_validation_failure("tool.investigation.statement_anchor_source_mismatch",
                    "statement anchor must point to the selected immutable text representation");
            statement.anchor.quote_verified = true;
            statement.actor_ids = item.value("actor_ids", std::vector<std::string>{});
            statement.reported_time_text = item.value("reported_time_text", std::string{});
            statement.extraction_origin = "model_proposal";
            if (item.contains("mentions")) {
                if (!item["mentions"].is_array() || item["mentions"].size() > 32)
                    return common_adapter_validation_failure("tool.investigation.mention_invalid", "at most 32 anchored actor mentions are allowed per statement");
                for (const auto & mention : item["mentions"]) {
                    if (!mention.is_object() || !mention.contains("name") || !mention["name"].is_string() ||
                            !mention.contains("byte_offset") || !mention["byte_offset"].is_number_integer() ||
                            (!mention["byte_offset"].is_number_unsigned() && mention["byte_offset"].get<int64_t>() < 0) ||
                            (mention.contains("kind") && !mention["kind"].is_string()))
                        return common_adapter_validation_failure("tool.investigation.mention_invalid", "mention name, kind and absolute byte offset have invalid types");
                    common_investigation_actor_mention m;
                    m.statement_id = statement.id;
                    m.name = mention["name"].get<std::string>();
                    m.kind = mention.value("kind",std::string("unknown"));
                    m.anchor = statement.anchor;
                    m.anchor.byte_offset = mention["byte_offset"].get<uint64_t>();
                    m.anchor.byte_length = m.name.size(); m.anchor.quote = m.name;
                    if (!common_investigation_anchor_validate(m.anchor, source_texts.at(m.anchor.source_id), error))
                        return common_adapter_validation_failure("tool.investigation.mention_unverified", error);
                    const std::string identity = value.case_id + "\n" + statement.id + "\n" + m.anchor.source_id + "\n" +
                        std::to_string(m.anchor.byte_offset) + "\n" + m.name;
                    m.id = "mention-" + hash_sha256_hex(identity.data(), identity.size()).substr(0,40);
                    const std::string actor_id = "actor-" + hash_sha256_hex(m.id.data(), m.id.size()).substr(0,40);
                    if (std::find(statement.actor_ids.begin(), statement.actor_ids.end(), actor_id) == statement.actor_ids.end())
                        statement.actor_ids.push_back(actor_id);
                    value.actors.push_back({actor_id,m.name,{m.id}});
                    value.actor_mentions.push_back(std::move(m));
                }
            }
            value.statements.push_back(std::move(statement));
        }
        for (const auto & item : arguments["comparisons"]) {
            if (!item.is_object() || !item.contains("id") || !item["id"].is_string() ||
                    !item.contains("left_statement_id") || !item["left_statement_id"].is_string() ||
                    !item.contains("right_statement_id") || !item["right_statement_id"].is_string() ||
                    !item.contains("kind") || !item["kind"].is_string())
                return common_adapter_validation_failure("tool.investigation.comparison_invalid",
                    "comparison requires an ID, two saved statement IDs and a declared relation");
            common_investigation_comparison comparison;
            comparison.id = item["id"].get<std::string>();
            comparison.left_statement_id = item["left_statement_id"].get<std::string>();
            comparison.right_statement_id = item["right_statement_id"].get<std::string>();
            if (!investigation_comparison_kind_from_name(item["kind"].get<std::string>(), comparison.kind)) return common_adapter_validation_failure(
                "tool.investigation.comparison_invalid", "comparison relation is not supported");
            comparison.rationale = item.value("rationale", std::string{});
            comparison.assessment_origin = "model_proposal";
            const auto left = std::find_if(value.statements.begin(), value.statements.end(), [&](const auto & s) { return s.id == comparison.left_statement_id; });
            const auto right = std::find_if(value.statements.begin(), value.statements.end(), [&](const auto & s) { return s.id == comparison.right_statement_id; });
            if (left == value.statements.end() || right == value.statements.end()) return common_adapter_validation_failure(
                "tool.investigation.comparison_statement_missing", "comparison statements must be present in the case revision");
            if (!analysis_sources.count(left->anchor.source_id) || !analysis_sources.count(right->anchor.source_id))
                return common_adapter_validation_failure("tool.investigation.comparison_source_unselected",
                    "comparison statements must belong to the explicitly selected analysis sources");
            comparison.source_ids = {left->anchor.source_id, right->anchor.source_id};
            value.comparisons.push_back(std::move(comparison));
        }
        if (arguments.contains("evidence_links")) {
            if (!arguments["evidence_links"].is_array() || arguments["evidence_links"].size() > 64)
                return common_adapter_validation_failure("tool.investigation.evidence_link_invalid", "at most 64 proposed evidence links are allowed");
            for (const auto & item : arguments["evidence_links"]) {
                common_investigation_evidence_link link;
                try {
                    link.id = item.at("id").get<std::string>(); link.kind = item.at("kind").get<std::string>();
                    link.left_id = item.at("left_id").get<std::string>(); link.right_id = item.at("right_id").get<std::string>();
                    link.supporting_statement_ids = item.at("supporting_statement_ids").get<std::vector<std::string>>();
                    link.rationale = item.at("rationale").get<std::string>();
                } catch (const std::exception &) {
                    return common_adapter_validation_failure("tool.investigation.evidence_link_invalid", "evidence link fields have invalid types");
                }
                for (const auto & id : link.supporting_statement_ids) {
                    const auto statement = std::find_if(value.statements.begin(),value.statements.end(),[&](const auto & v) { return v.id == id; });
                    if (statement == value.statements.end() || !analysis_sources.count(statement->anchor.source_id))
                        return common_adapter_validation_failure("tool.investigation.evidence_link_unselected", "link evidence must belong to the explicitly selected sources");
                }
                if (link.kind == "possible_same_actor") {
                    for (const auto & actor_id : {link.left_id,link.right_id}) {
                        const auto actor = std::find_if(value.actors.begin(),value.actors.end(),[&](const auto & a) { return a.id == actor_id; });
                        if (actor == value.actors.end()) return common_adapter_validation_failure("tool.investigation.actor_missing", "link references an unknown actor candidate");
                        for (const auto & mention_id : actor->mention_ids) {
                            const auto m = std::find_if(value.actor_mentions.begin(),value.actor_mentions.end(),[&](const auto & v) { return v.id == mention_id; });
                            if (m == value.actor_mentions.end() || !analysis_sources.count(m->anchor.source_id))
                                return common_adapter_validation_failure("tool.investigation.actor_unselected", "linked actors must belong to the selected sources");
                        }
                    }
                } else if (link.kind == "possible_same_event") {
                    const auto event = std::find_if(value.events.begin(),value.events.end(),[&](const auto & v) { return v.id == link.right_id; });
                    if (event == value.events.end() || std::none_of(value.sources.begin(),value.sources.end(),[&](const auto & v) {
                            return analysis_sources.count(v.id) && (v.id == event->source_id ||
                                (v.kind == common_investigation_source_kind::search_connector && v.provider_id == event->source_id)); }))
                        return common_adapter_validation_failure("tool.investigation.event_unselected", "linked log event must belong to a selected source");
                }
                // All links use the model-proposal origin. User review remains
                // an independent host-authorized operation.
                value.evidence_links.push_back(std::move(link));
            }
        }
        common_investigation_analysis_record record;
        record.id = analysis_id; record.kind = analysis_kind; record.algorithm_version = "evidence-anchor-validation-v1";
        record.published_revision = expected + 1;
        record.source_ids.assign(analysis_sources.begin(), analysis_sources.end());
        record.parameters_fingerprint = parameters_fingerprint;
        record.completeness = arguments.value("completeness", std::string("complete"));
        record.limitations = arguments.value("limitations", std::vector<std::string>{});
        for (const auto & source : value.sources) if (analysis_sources.count(source.id) &&
                !common_investigation_source_complete(source)) {
            if (record.completeness == "complete") record.completeness = "partial";
            record.limitations.push_back("Selected source " + source.id + " has incomplete or qualified coverage.");
        }
        value.analyses.push_back(std::move(record));
        common_investigation_case_revision published;
        if (!investigation_publish_revision(bindings, std::move(value), expected, published, error))
            return common_adapter_execution_failure("tool.investigation.analysis_publish_failed",
                std::move(error), "The analysis candidate was not published to the case.");
        return common_adapter_success_json({{"case_id", published.case_id}, {"revision", published.revision},
            {"published_revision", published.revision},
            {"analysis_id", analysis_id}, {"statement_count", published.statements.size()},
            {"comparison_count", published.comparisons.size()}, {"reused", false}});
    }
    if (operation == "investigation.association.review") {
        if (!arguments.contains("expected_revision") || !arguments["expected_revision"].is_number_integer() ||
                (!arguments["expected_revision"].is_number_unsigned() && arguments["expected_revision"].get<int64_t>() < 1) ||
                !arguments.contains("idempotency_key") || !arguments["idempotency_key"].is_string() ||
                !arguments.contains("actor_ids") || !arguments["actor_ids"].is_array() ||
                !arguments.contains("kind") || !arguments["kind"].is_string() ||
                !arguments.contains("decision") || !arguments["decision"].is_string() ||
                (arguments.contains("comment") && !arguments["comment"].is_string()) ||
                (arguments.contains("relation_type") && !arguments["relation_type"].is_string()))
            return common_adapter_validation_failure("tool.investigation.review_invalid", "review fields have invalid types");
        const auto expected = arguments["expected_revision"].get<uint64_t>();
        const auto key = arguments["idempotency_key"].get<std::string>();
        const auto kind = arguments["kind"].get<std::string>(), decision = arguments["decision"].get<std::string>();
        const auto relation_type = arguments.value("relation_type",std::string{}), comment = arguments.value("comment",std::string{});
        if (key.empty() || key.size() > 128 || comment.size() > 4096 || arguments["actor_ids"].size() < 2 ||
                arguments["actor_ids"].size() > 32 || (decision != "same_actor" && arguments["actor_ids"].size() != 2) ||
                (kind != "identity" && kind != "relationship"))
            return common_adapter_validation_failure("tool.investigation.review_invalid", "review selection or operation exceeds its bounds");
        std::vector<std::string> selected;
        std::set<std::string> unique;
        for (const auto & id : arguments["actor_ids"]) {
            if (!id.is_string() || !unique.insert(id.get<std::string>()).second ||
                    std::none_of(value.actors.begin(),value.actors.end(),[&](const auto & actor) { return actor.id == id.get<std::string>(); }))
                return common_adapter_validation_failure("tool.investigation.review_actor_invalid", "choose distinct actor candidates in the current case");
            selected.push_back(id.get<std::string>());
        }
        nlohmann::json payload = arguments; payload.erase("expected_revision"); payload.erase("idempotency_key");
        const auto serialized = payload.dump();
        const auto fingerprint = "sha256:" + hash_sha256_hex(serialized.data(),serialized.size());
        const auto operation_identity = json::array({value.case_id,bindings.investigation_reviewer_id,key}).dump();
        const auto operation_id = "review-op-" + hash_sha256_hex(operation_identity.data(),operation_identity.size()).substr(0,40);
        json previous_ids = json::array();
        uint64_t published_revision = 0;
        for (const auto & review : value.association_reviews) if (review.operation_id == operation_id) {
            if (review.parameters_fingerprint != fingerprint) return common_adapter_failure(
                "tool.investigation.idempotency_conflict", common_tool_failure_class::conflict, false,
                "This review key already belongs to a different decision.","association review idempotency mismatch");
            previous_ids.push_back(review.id); published_revision = review.published_revision;
        }
        if (!previous_ids.empty()) return common_adapter_success_json({{"case_id",value.case_id},{"revision",value.revision},
            {"published_revision",published_revision},{"review_ids",previous_ids},{"reused",true}});
        if (expected != value.revision) return common_adapter_failure("tool.investigation.revision_conflict",
            common_tool_failure_class::conflict,false,"The case changed. Reload and review its current evidence.","stale association review");
        json review_ids = json::array();
        for (size_t i = 1; i < selected.size(); ++i) {
            common_investigation_association_review review;
            const auto identity = operation_id + ":" + std::to_string(i);
            review.id = "review-" + hash_sha256_hex(identity.data(),identity.size()).substr(0,40);
            review.operation_id = operation_id; review.parameters_fingerprint = fingerprint;
            review.left_actor_id = selected.front(); review.right_actor_id = selected[i];
            review.kind = kind; review.decision = decision; review.relation_type = relation_type; review.comment = comment;
            review.reviewer_id = bindings.investigation_reviewer_id;
            review.created_at_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            review.published_revision = expected + 1;
            for (auto old = value.association_reviews.rbegin(); old != value.association_reviews.rend(); ++old)
                if (old->kind == kind &&
                        ((kind == "identity" && ((old->left_actor_id == review.left_actor_id && old->right_actor_id == review.right_actor_id) ||
                            (old->left_actor_id == review.right_actor_id && old->right_actor_id == review.left_actor_id))) ||
                         (kind == "relationship" && old->relation_type == relation_type &&
                            old->left_actor_id == review.left_actor_id && old->right_actor_id == review.right_actor_id))) {
                    review.supersedes_id = old->id; break;
                }
            review_ids.push_back(review.id); value.association_reviews.push_back(std::move(review));
        }
        auto proposed = value; proposed.revision = expected + 1;
        json associations;
        if (!common_investigation_actor_associations(proposed,associations,error))
            return common_adapter_validation_failure("tool.investigation.review_conflict",error);
        common_investigation_case_revision published;
        if (!investigation_publish_revision(bindings,std::move(value),expected,published,error))
            return common_adapter_failure("tool.investigation.review_publish_failed",common_tool_failure_class::conflict,false,
                "The decision was not published. Reload before retrying.",error);
        return common_adapter_success_json({{"case_id",published.case_id},{"revision",published.revision},
            {"published_revision",published.revision},{"review_ids",review_ids},{"reused",false}});
    }
    if (operation == "investigation.case.open") {
        const std::string first_entity_id = value.entities.empty() ? std::string() : value.entities.front().id;
        return common_adapter_success_json({{"case", {{"case_id", value.case_id}, {"revision", value.revision},
            {"title", value.title}, {"question", value.question}, {"source_id", value.source_id},
            {"window_start_ms", value.window_start_ms}, {"window_end_ms", value.window_end_ms},
            {"counts", {{"entities", value.entities.size()}, {"events", value.events.size()},
                {"relations", value.relations.size()}, {"claims", value.claims.size()}, {"gaps", value.gaps.size()},
                {"sources", value.sources.size()}, {"statements", value.statements.size()},
                {"comparisons", value.comparisons.size()}, {"analyses", value.analyses.size()},
                {"actors",value.actors.size()},{"actor_mentions",value.actor_mentions.size()},{"evidence_links",value.evidence_links.size()},
                {"association_reviews",value.association_reviews.size()}}},
            {"first_entity_id", first_entity_id}, {"status", value.gaps.empty() ? "open" : "qualified"}}}});
    }
    if (operation == "investigation.case.items") {
        if (!arguments.contains("case_revision") || !arguments["case_revision"].is_number_integer() ||
                !arguments.contains("type") || !arguments["type"].is_string() ||
                (arguments.contains("limit") && !arguments["limit"].is_number_integer()) ||
                (arguments.contains("cursor") && !arguments["cursor"].is_string()))
            return common_adapter_validation_failure("tool.investigation.items_invalid_arguments",
                "case_revision and limit must be integers, type and cursor must be strings");
        const uint64_t requested_revision = arguments.value("case_revision", uint64_t(0));
        if (requested_revision != value.revision) return common_adapter_execution_failure(
            "tool.investigation.stale_cursor", "requested case revision is no longer current",
            "Reload the case before continuing this page.");
        const std::string item_type = arguments.value("type", std::string());
        const size_t limit = arguments.value("limit", size_t(50));
        if (limit == 0 || limit > 100) return common_adapter_validation_failure(
            "tool.investigation.page_limit_invalid", "case item page size must be from 1 to 100");
        const json binding = {{"case_id", value.case_id}, {"type", item_type},
            {"namespace_id", bindings.tool_scope.namespace_id}, {"project_id", bindings.tool_scope.project_id},
            {"session_id", bindings.tool_scope.project_id.empty() ? bindings.tool_scope.session_id : std::string()}};
        size_t offset = 0;
        if (!investigation_page_offset(arguments.value("cursor", std::string()), value.revision, binding, offset))
            return common_adapter_execution_failure("tool.investigation.cursor_invalid",
                "case item cursor does not match this case revision and filter",
                "The page cursor is invalid or stale; reopen the case and start again.");
        json source;
        json projection_limitations = json::array();
        if (item_type == "reports") {
            if (!bindings.data_store) return common_adapter_execution_failure(
                "tool.investigation.report_index_unavailable", "investigation report index is unavailable",
                "Frozen reports cannot be listed in this runtime.");
            std::vector<common_investigation_report_index_entry> reports;
            if (!bindings.data_store->list_investigation_reports(bindings.tool_scope, value.case_id,
                    requested_revision, reports, error)) return common_adapter_execution_failure(
                "tool.investigation.report_index_failed", std::move(error),
                "Frozen reports could not be read from the case index.");
            source = json::array();
            for (const auto & report : reports) source.push_back({
                {"case_id", report.case_id}, {"case_revision", report.case_revision},
                {"source_id", report.source_id}, {"case_fingerprint", report.case_fingerprint},
                {"content_fingerprint", report.content_fingerprint}, {"report_uri", report.report_uri},
                {"markdown_uri", report.markdown_uri}, {"source_manifest_uri", report.source_manifest_uri},
                {"visualization_uris", report.visualization_uris}, {"limitations", report.limitations},
                {"complete", report.complete}, {"created_at_ms", report.created_at_ms}});
        } else {
            const auto serialized = common_investigation_case_to_json(value);
            const char * collection = nullptr;
            if (item_type == "actor_groups" || item_type == "actor_relationships") {
                json associations;
                if (!common_investigation_actor_associations(value,associations,error))
                    return common_adapter_validation_failure("tool.investigation.associations_invalid",error);
                source = associations.at(item_type == "actor_groups" ? "groups" : "relationships");
                projection_limitations = associations.at("limitations");
            }
            else if (item_type == "sources") collection = "sources";
            else if (item_type == "statements") collection = "statements";
            else if (item_type == "comparisons") collection = "comparisons";
            else if (item_type == "analyses") collection = "analyses";
            else if (item_type == "actors") collection = "actors";
            else if (item_type == "actor_mentions") collection = "actor_mentions";
            else if (item_type == "evidence_links") collection = "evidence_links";
            else if (item_type == "association_reviews") collection = "association_reviews";
            else if (item_type == "entities") collection = "entities";
            else if (item_type == "events") collection = "events";
            else if (item_type == "relations") collection = "relations";
            else if (item_type == "claims") collection = "claims";
            else if (item_type == "gaps") collection = "gaps";
            else return common_adapter_validation_failure("tool.investigation.items_type_invalid",
                "type must be sources, statements, comparisons, analyses, actors, actor_mentions, evidence_links, actor_groups, actor_relationships, association_reviews, entities, events, relations, claims, gaps or reports");
            if (collection) source = serialized.at(collection);
            if (item_type == "association_reviews") std::reverse(source.begin(),source.end());
        }
        json items = json::array();
        if (offset > source.size()) return common_adapter_execution_failure(
            "tool.investigation.cursor_invalid", "case item cursor offset exceeds the collection size",
            "The page cursor is invalid or stale; reopen the case and start again.");
        constexpr size_t max_page_bytes = 40 * 1024;
        size_t end = offset, page_bytes = 0;
        while (end < source.size() && items.size() < limit) {
            const size_t item_bytes = source[end].dump().size();
            if (!items.empty() && page_bytes + item_bytes > max_page_bytes) break;
            if (items.empty() && item_bytes > max_page_bytes) return common_adapter_execution_failure(
                "tool.investigation.item_too_large", "one case item exceeds the bounded page size",
                "Read its referenced resource through the resource tools.");
            page_bytes += item_bytes;
            items.push_back(source[end++]);
        }
        const bool has_more = end < source.size();
        const std::string next_cursor = has_more ? investigation_page_cursor(value.revision, end, binding) : std::string();
        return common_adapter_success_json({{"case_id", value.case_id}, {"case_revision", value.revision},
            {"type", item_type}, {"items", std::move(items)}, {"next_cursor", next_cursor},
            {"has_more", has_more}, {"total", source.size()}, {"limitations",projection_limitations}});
    }
    if (operation == "investigation.report.compose") {
        for (auto it = arguments.begin(); it != arguments.end(); ++it) {
            if (it.key() != "case_id" && it.key() != "expected_revision" && it.key() != "visualization_uris")
                return common_adapter_validation_failure("tool.investigation.report_invalid_arguments",
                    "unsupported report argument: " + it.key());
        }
        if (arguments.value("expected_revision", uint64_t(0)) != value.revision) return
            common_adapter_execution_failure("tool.investigation.revision_conflict",
                "case revision changed before report composition", "Reload the case before composing a report.");
        if (!bindings.resource_runtime.store) return common_adapter_execution_failure(
            "tool.investigation.report_store_unavailable", "resource store is unavailable",
            "A durable report resource cannot be created.");
        json attached_snapshots = json::array();
        if (arguments.contains("visualization_uris")) {
            if (!arguments["visualization_uris"].is_array() || arguments["visualization_uris"].size() > 8)
                return common_adapter_validation_failure("tool.investigation.report_snapshot_list_invalid",
                    "visualization_uris must be an array with at most eight snapshot references");
            const auto authority = make_agent_resource_read_authority(
                bindings.resource_runtime, static_cast<int64_t>(std::time(nullptr)));
            std::set<std::string> seen;
            for (const auto & item : arguments["visualization_uris"]) {
                if (!item.is_string() || item.get<std::string>().empty() || item.get<std::string>().size() > 512)
                    return common_adapter_validation_failure("tool.investigation.report_snapshot_list_invalid",
                        "each snapshot reference must be a non-empty resource URI of at most 512 bytes");
                const auto uri = item.get<std::string>();
                if (!seen.insert(uri).second) continue;
                agent_resource_descriptor snapshot;
                if (!bindings.resource_runtime.store->stat(uri, authority, snapshot, error))
                    return common_adapter_not_found_failure("tool.investigation.report_snapshot_unavailable",
                        std::move(error), "A selected snapshot is unavailable in the current runtime scope.");
                if (snapshot.mime_type != "image/png" || snapshot.source_tool != "visualization.snapshot")
                    return common_adapter_validation_failure("tool.investigation.report_snapshot_invalid",
                        "selected resource is not a persisted visualization PNG snapshot");
                std::string image;
                if (!bindings.resource_runtime.store->read_bytes(snapshot.uri, authority, 4 * 1024 * 1024, image, error))
                    return common_adapter_execution_failure("tool.investigation.report_snapshot_read_failed",
                        std::move(error), "A selected snapshot could not be verified.");
                static constexpr unsigned char signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
                if (image.size() < 45 ||
                        !std::equal(std::begin(signature), std::end(signature),
                            reinterpret_cast<const unsigned char *>(image.data())) ||
                        image.compare(12, 4, "IHDR") != 0 || image.compare(image.size() - 8, 4, "IEND") != 0 ||
                        snapshot.sha256 != hash_sha256_hex(image.data(), image.size()))
                    return common_adapter_execution_failure("tool.investigation.report_snapshot_invalid",
                        "snapshot bytes failed PNG signature or resource hash verification",
                        "A selected snapshot failed integrity validation.");
                attached_snapshots.push_back({{"uri", snapshot.uri}, {"sha256", snapshot.sha256},
                    {"size_bytes", snapshot.size_bytes}, {"visualization_uri", snapshot.lineage.parent_uri}});
            }
        }
        const auto safe_text = [](std::string text) {
            for (auto & c : text) {
                if (c == '\n' || c == '\r' || c == '\t') c = ' ';
                else if (static_cast<unsigned char>(c) < 0x20) c = '?';
            }
            std::string out;
            out.reserve(text.size());
            for (const char c : text) {
                if (c == '`') out += "'";
                else if (c == '<') out += "&lt;";
                else if (c == '>') out += "&gt;";
                else out += c;
            }
            return out;
        };
        std::string markdown = "# Investigation report: " + safe_text(value.title) + "\n\n";
        markdown += "- Case: `" + safe_text(value.case_id) + "`\n- Frozen revision: `" + std::to_string(value.revision) + "`\n";
        if (!value.source_id.empty()) markdown += "- Primary search source: `" + safe_text(value.source_id) + "`\n";
        if (!value.sources.empty()) markdown += "- Selected sources: " + std::to_string(value.sources.size()) + "\n";
        if (!value.source_id.empty()) markdown += "- Search time window (epoch ms): `" +
            std::to_string(value.window_start_ms) + "` to `" + std::to_string(value.window_end_ms) + "`\n";
        markdown += "\n";
        markdown += "## Investigation question\n\n" + safe_text(value.question) + "\n\n";
        markdown += "## Scope and coverage\n\n";
        std::vector<std::string> report_limitations = value.gaps;
        for (const auto & source : value.sources) {
            if (!common_investigation_source_complete(source))
                report_limitations.push_back("Source " + source.id + " is incomplete or not fully processed (" + source.processing_status + ").");
            for (const auto & limitation : source.processing_limitations)
                report_limitations.push_back("Source " + source.id + " processing limitation: " + limitation);
        }
        for (const auto & analysis : value.analyses) if (analysis.completeness != "complete")
            report_limitations.push_back("Analysis " + analysis.id + " is " + analysis.completeness + ".");
        const bool report_complete = report_limitations.empty();
        markdown += report_complete ? "No recorded collection or analysis gaps. This does not establish exhaustive source coverage.\n\n"
            : "Coverage is partial or qualified; see the listed gaps.\n\n";
        markdown += "### Selected sources\n\n";
        if (value.sources.empty()) markdown += "No explicit resource sources are attached; legacy source reference: `" + safe_text(value.source_id) + "`.\n\n";
        for (const auto & source : value.sources) {
            const char * source_type = source.kind == common_investigation_source_kind::search_connector ? "search_connector" :
                source.kind == common_investigation_source_kind::log_file ? "log_file" : "document";
            markdown += "- `" + safe_text(source.id) + "` (" + source_type + "): " + safe_text(source.display_name) +
                "; original `" + safe_text(source.original_uri) + "`; retained `" + safe_text(source.resource_uri) +
                "`; SHA-256 `" + safe_text(source.content_hash) + "`; status `" + safe_text(source.processing_status) +
                "`; processor `" + safe_text(source.processor_version) + "`";
            if (!source.representation_uri.empty()) markdown += "; text representation `" + safe_text(source.representation_uri) +
                "` (SHA-256 `" + safe_text(source.representation_hash) + "`)";
            for (const auto & limitation : source.processing_limitations)
                markdown += "; processing limitation: " + safe_text(limitation);
            if (!source.original_parent_uri.empty()) markdown += "; original parent `" + safe_text(source.original_parent_uri) + "`";
            markdown += "\n";
        }
        if (!value.sources.empty()) markdown += "\n";
        std::vector<const common_investigation_event *> ordered_events;
        ordered_events.reserve(value.events.size());
        for (const auto & event : value.events) ordered_events.push_back(&event);
        std::stable_sort(ordered_events.begin(), ordered_events.end(), [](const auto * a, const auto * b) {
            if (a->time.has_event_time != b->time.has_event_time) return a->time.has_event_time;
            if (!a->time.has_event_time) return a->id < b->id;
            if (a->time.event_time_ms != b->time.event_time_ms) return a->time.event_time_ms < b->time.event_time_ms;
            return a->id < b->id;
        });
        markdown += "## Timeline\n\n";
        if (ordered_events.empty()) markdown += "No source events are retained in this revision.\n\n";
        for (const auto * event : ordered_events) {
            markdown += "- " + (event->time.has_event_time ? std::to_string(event->time.event_time_ms) : std::string("time unknown")) +
                " — " + safe_text(event->event_type) + " (`" + safe_text(event->id) + "`; document `" +
                safe_text(event->source_document_id) + "`" + (event->source_index.empty() ? std::string() : "; index `" + safe_text(event->source_index) + "`") +
                ") [raw: " + safe_text(event->raw_resource_uri) + "]\n";
        }
        markdown += "\n## Relations\n\n";
        if (value.relations.empty()) markdown += "No relations are derived in this revision.\n\n";
        for (const auto & relation : value.relations) {
            markdown += "- `" + safe_text(relation.from_entity_id) + "` → `" + safe_text(relation.to_entity_id) +
                "` (" + common_investigation_relation_kind_name(relation.kind) + ", " + safe_text(relation.confidence_label) +
                ", rule `" + safe_text(relation.rule_id) + "`); events: ";
            for (size_t i = 0; i < relation.evidence_event_ids.size(); ++i) {
                if (i) markdown += ", ";
                markdown += "`" + safe_text(relation.evidence_event_ids[i]) + "`";
            }
            markdown += "\n";
        }
        markdown += "\n## Claims\n\n";
        if (value.claims.empty()) markdown += "No claims are recorded.\n\n";
        for (const auto & claim : value.claims) {
            const char * state = claim.state == common_investigation_claim_state::hypothesis ? "Hypothesis" :
                claim.state == common_investigation_claim_state::reviewed_conclusion ? "Reviewed conclusion" :
                claim.state == common_investigation_claim_state::correlation ? "Correlation" : "Observation";
            markdown += "### " + std::string(state) + ": " + safe_text(claim.text) + "\n\n";
            for (const auto & id : claim.supporting_event_ids) markdown += "- Supports: `" + safe_text(id) + "`\n";
            for (const auto & id : claim.contradicting_event_ids) markdown += "- Contradicts: `" + safe_text(id) + "`\n";
            for (const auto & id : claim.supporting_statement_ids) markdown += "- Supporting statement: `" + safe_text(id) + "`\n";
            for (const auto & id : claim.contradicting_statement_ids) markdown += "- Contradicting statement: `" + safe_text(id) + "`\n";
            for (const auto & limitation : claim.limitations) markdown += "- Limitation: " + safe_text(limitation) + "\n";
            markdown += "\n";
        }
        markdown += "## Actor candidates and proposed links\n\n";
        for (const auto & actor : value.actors) {
            markdown += "- Candidate `" + safe_text(actor.id) + "`: " + safe_text(actor.display_name) + "; mentions:";
            for (const auto & id : actor.mention_ids) markdown += " `" + safe_text(id) + "`";
            markdown += "\n";
        }
        for (const auto & mention : value.actor_mentions) markdown += "- Mention `" + safe_text(mention.id) + "`: " +
            safe_text(mention.name) + " in statement `" + safe_text(mention.statement_id) + "`, resource `" +
            safe_text(mention.anchor.resource_uri) + "`, byte " + std::to_string(mention.anchor.byte_offset) + "\n";
        for (const auto & link : value.evidence_links) {
            markdown += "- Proposed " + safe_text(link.kind) + ": `" + safe_text(link.left_id) + "` ↔ `" + safe_text(link.right_id) +
                "` — " + safe_text(link.rationale) + "; statements:";
            for (const auto & id : link.supporting_statement_ids) markdown += " `" + safe_text(id) + "`";
            markdown += "\n";
        }
        markdown += "\n### Analyst association decisions\n\n";
        for (const auto & review : value.association_reviews) markdown += "- `" + safe_text(review.id) + "`: " +
            safe_text(review.decision) + " (`" + safe_text(review.left_actor_id) + "`, `" + safe_text(review.right_actor_id) +
            "`), relation `" + safe_text(review.relation_type) + "`; authenticated reviewer `" + safe_text(review.reviewer_id) +
            "`; revision " + std::to_string(review.published_revision) + "; supersedes `" + safe_text(review.supersedes_id) +
            "`; " + safe_text(review.comment) + ". Analyst decision, not machine-verified causality.\n";
        markdown += "\n## Source statements\n\n";
        if (value.statements.empty()) markdown += "No source-anchored statements are recorded.\n\n";
        for (const auto & statement : value.statements) {
            const char * kind = statement.kind == common_investigation_statement_kind::reported_event ? "reported_event" :
                statement.kind == common_investigation_statement_kind::definition ? "definition" :
                statement.kind == common_investigation_statement_kind::interpretation ? "interpretation" : "statement";
            markdown += "- `" + safe_text(statement.id) + "` (" + kind + "): " + safe_text(statement.text) +
                " — source `" + safe_text(statement.anchor.source_id) + "`, resource `" + safe_text(statement.anchor.resource_uri) +
                "`, byte offset `" + std::to_string(statement.anchor.byte_offset) + "`, quote `" + safe_text(statement.anchor.quote) + "`";
            if (statement.anchor.page_number > 0) markdown += ", page " + std::to_string(statement.anchor.page_number);
            if (!statement.reported_time_text.empty()) markdown += ", reported time `" + safe_text(statement.reported_time_text) + "`";
            markdown += "\n";
        }
        if (!value.statements.empty()) markdown += "\n";
        markdown += "## Comparisons\n\n";
        if (value.comparisons.empty()) markdown += "No pairwise source comparisons are recorded.\n\n";
        for (const auto & comparison : value.comparisons) {
            const char * kind = comparison.kind == common_investigation_comparison_kind::supports ? "supports" :
                comparison.kind == common_investigation_comparison_kind::contradicts ? "contradicts" :
                comparison.kind == common_investigation_comparison_kind::qualifies ? "qualifies" :
                comparison.kind == common_investigation_comparison_kind::same_topic ? "same_topic" :
                comparison.kind == common_investigation_comparison_kind::source_dependency ? "source_dependency" : "inconclusive";
            markdown += "- `" + safe_text(comparison.id) + "`: `" + safe_text(comparison.left_statement_id) + "` " + kind +
                " `" + safe_text(comparison.right_statement_id) + "` (" + safe_text(comparison.assessment_origin) + ")";
            if (!comparison.rationale.empty()) markdown += " — " + safe_text(comparison.rationale);
            markdown += "\n";
        }
        if (!value.comparisons.empty()) markdown += "\n";
        markdown += "## Analysis provenance\n\n";
        if (value.analyses.empty()) markdown += "No explicit analysis records are attached.\n\n";
        for (const auto & analysis : value.analyses) {
            markdown += "- `" + safe_text(analysis.id) + "` — `" + safe_text(analysis.kind) + "` / `" +
                safe_text(analysis.algorithm_version) + "`, completeness `" + safe_text(analysis.completeness) + "`\n";
            for (const auto & limitation : analysis.limitations) markdown += "  - Limitation: " + safe_text(limitation) + "\n";
        }
        if (!value.analyses.empty()) markdown += "\n";
        markdown += "## Gaps and limitations\n\n";
        if (report_limitations.empty()) markdown += "No explicit gaps are recorded.\n";
        for (const auto & gap : report_limitations) markdown += "- " + safe_text(gap) + "\n";
        markdown += "\n## Evidence references\n\n";
        for (const auto & event : value.events) {
            markdown += "- Event `" + safe_text(event.id) + "`: raw `" + safe_text(event.raw_resource_uri) +
                "`; normalized `" + safe_text(event.normalized_resource_uri) + "`\n";
        }
        if (!attached_snapshots.empty()) {
            markdown += "\n## Visualization snapshots\n\n";
            markdown += "The following PNG snapshots were explicitly selected as report attachments. Their relationships to the case are not inferred; inspect their recorded visualization source and revision separately.\n\n";
            for (size_t i = 0; i < attached_snapshots.size(); ++i) {
                const auto & snapshot = attached_snapshots[i];
                markdown += "- [Snapshot " + std::to_string(i + 1) + "](" + safe_text(snapshot["uri"].get<std::string>()) +
                    ") — SHA-256 `" + safe_text(snapshot["sha256"].get<std::string>()) + "`; visualization `" +
                    safe_text(snapshot["visualization_uri"].get<std::string>()) + "`\n";
            }
        }
        if (markdown.size() > 2 * 1024 * 1024) return common_adapter_validation_failure(
            "tool.investigation.report_too_large", "frozen report exceeds the 2 MiB resource limit");
        const std::string fingerprint = "sha256:" + hash_sha256_hex(markdown.data(), markdown.size());
        json frozen_associations;
        if (!common_investigation_actor_associations(value,frozen_associations,error))
            return common_adapter_validation_failure("tool.investigation.associations_invalid",error);
        const auto frozen_case_json = common_investigation_case_to_json(value);
        const json frozen_source{{"schema", "investigation-report-source.v1"}, {"case_id", value.case_id},
            {"case_revision", value.revision}, {"source_id", value.source_id},
            {"window_start_ms", value.window_start_ms}, {"window_end_ms", value.window_end_ms},
            {"case_fingerprint", value.fingerprint},
            {"content_fingerprint", fingerprint}, {"visualization_snapshots", attached_snapshots},
            {"event_count", value.events.size()}, {"relation_count", value.relations.size()},
            {"claim_count", value.claims.size()}, {"statement_count", value.statements.size()},
            {"comparison_count", value.comparisons.size()}, {"analysis_count", value.analyses.size()},
            {"sources", frozen_case_json.at("sources")},
            {"statements", frozen_case_json.at("statements")},
            {"comparisons", frozen_case_json.at("comparisons")},
            {"analyses", frozen_case_json.at("analyses")},
            {"actors", frozen_case_json.at("actors")},{"actor_mentions", frozen_case_json.at("actor_mentions")},
            {"evidence_links", frozen_case_json.at("evidence_links")},
            {"association_reviews", frozen_case_json.at("association_reviews")},{"actor_associations",frozen_associations},
            {"events", frozen_case_json.at("events")},{"claims",frozen_case_json.at("claims")},
            {"gaps", report_limitations}};
        const auto source_json = frozen_source.dump();
        const std::string source_fingerprint = "sha256:" + hash_sha256_hex(source_json.data(), source_json.size());
        agent_resource_put_request source_put;
        source_put.name = value.case_id + "-report-r" + std::to_string(value.revision) + "-sources.json";
        source_put.description = "Frozen source and revision manifest for an investigation report.";
        source_put.mime_type = "application/json"; source_put.text = source_json;
        source_put.scope = bindings.tool_scope.project_id.empty() ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
        source_put.source_provider = value.source_id; source_put.source_tool = "investigation.report.compose";
        apply_agent_resource_runtime(bindings.resource_runtime, source_put);
        source_put.metadata.purpose = "Immutable case revision and explicitly selected snapshot references.";
        source_put.metadata.content_summary = "Case " + value.case_id + " revision " + std::to_string(value.revision);
        source_put.metadata.source_request_json = json{{"case_id", value.case_id},
            {"case_revision", value.revision}, {"case_fingerprint", value.fingerprint},
            {"report_fingerprint", fingerprint}, {"visualization_snapshots", attached_snapshots}}.dump();
        source_put.metadata.source_operation = "investigation.report.compose";
        source_put.metadata.content_hash = source_fingerprint;
        agent_resource_descriptor source_manifest;
        if (!bindings.resource_runtime.store->put_text(source_put, source_manifest, error)) return common_adapter_execution_failure(
            "tool.investigation.report_source_store_failed", std::move(error), "The frozen report source manifest could not be persisted.");

        common_agent_artifact_document document;
        document.schema_version = 3;
        document.kind = common_agent_artifact_document_kind::document;
        document.id = "investigation-report-" + hash_sha256_hex(fingerprint.data(), fingerprint.size()).substr(0, 24);
        document.title = "Investigation report: " + safe_text(value.title);
        document.purpose = "Frozen report for case " + value.case_id + " revision " + std::to_string(value.revision);
        document.revision = value.revision;
        document.snapshot_uri = source_manifest.uri;
        document.snapshot_fingerprint = source_fingerprint;
        document.source_resource_uri = source_manifest.uri;
        document.source_content_fingerprint = source_fingerprint;
        document.importer_version = "investigation-report-projection-v1";
        document.coverage_complete = false;
        document.unresolved_gaps = report_limitations;
        if (document.unresolved_gaps.empty())
            document.unresolved_gaps.push_back("Exhaustive source coverage was not established.");
        common_agent_artifact_section report_section;
        report_section.id = "investigation-report";
        report_section.title = "Investigation report";
        common_agent_artifact_block report_block;
        report_block.id = "frozen-report-body";
        report_block.kind = common_agent_artifact_block_kind::observation;
        report_block.text = markdown;
        report_block.origin = "investigation.report.compose";
        report_block.author = common_agent_artifact_author::host;
        report_block.review_state = common_agent_artifact_review_state::unreviewed;
        report_block.validation = common_agent_artifact_validation::unvalidated;
        document.sections.push_back(std::move(report_section));
        document.sections.back().blocks.push_back(std::move(report_block));
        std::string editable_markdown;
        if (!common_agent_artifact_render_editable_markdown(document, editable_markdown, error))
            return common_adapter_execution_failure("tool.investigation.report_render_failed", std::move(error),
                "The structured report could not be prepared.");
        document.content_fingerprint = "sha256:" + hash_sha256_hex(editable_markdown.data(), editable_markdown.size());
        agent_resource_put_request markdown_put;
        markdown_put.name = value.case_id + "-report-r" + std::to_string(value.revision) + ".md";
        markdown_put.description = "Rendered Markdown for frozen investigation report revision " + std::to_string(value.revision);
        markdown_put.mime_type = "text/markdown"; markdown_put.text = markdown;
        markdown_put.scope = bindings.tool_scope.project_id.empty() ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
        markdown_put.source_provider = value.source_id; markdown_put.source_tool = "investigation.report.compose";
        apply_agent_resource_runtime(bindings.resource_runtime, markdown_put);
        markdown_put.metadata.purpose = "Rendered view of the frozen investigation report.";
        markdown_put.metadata.source_operation = "investigation.report.compose";
        markdown_put.metadata.content_hash = fingerprint;
        markdown_put.lineage.parent_uri = source_manifest.uri; markdown_put.lineage.chunk_count = 1;
        markdown_put.lineage.byte_length = markdown.size(); markdown_put.lineage.derivation = "investigation.report.compose:markdown-v1";
        agent_resource_descriptor markdown_resource;
        if (!bindings.resource_runtime.store->put_text(markdown_put, markdown_resource, error))
            return common_adapter_execution_failure("tool.investigation.report_markdown_store_failed", std::move(error),
                "The frozen Markdown rendering could not be persisted.");
        document.editor_markdown_uri = markdown_resource.uri;
        std::string document_json;
        if (!common_agent_artifact_document_to_json(document, document_json, error))
            return common_adapter_execution_failure("tool.investigation.report_serialize_failed", std::move(error),
                "The structured report could not be validated.");
        agent_resource_put_request report_put;
        report_put.name = value.case_id + "-report-r" + std::to_string(value.revision) + ".json";
        report_put.description = "Structured frozen investigation report for case revision " + std::to_string(value.revision);
        report_put.mime_type = "application/vnd.llama-agent.document+json"; report_put.text = document_json;
        report_put.scope = bindings.tool_scope.project_id.empty() ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
        report_put.source_provider = value.source_id; report_put.source_tool = "investigation.report.compose";
        apply_agent_resource_runtime(bindings.resource_runtime, report_put);
        report_put.metadata.purpose = "Frozen investigation report using the shared document and export contract.";
        report_put.metadata.content_summary = "Case " + value.case_id + " revision " + std::to_string(value.revision);
        report_put.metadata.source_provider = value.source_id;
        report_put.metadata.source_operation = "investigation.report.compose";
        report_put.metadata.source_request_json = json{{"case_id", value.case_id},
            {"case_revision", value.revision}, {"content_fingerprint", fingerprint},
            {"source_manifest_uri", source_manifest.uri}, {"visualization_snapshots", attached_snapshots}}.dump();
        report_put.metadata.content_hash = document.content_fingerprint;
        report_put.metadata.keywords = {"investigation", "report", value.case_id};
        report_put.metadata.limitations = value.gaps.empty() ?
            "No recorded gaps; this does not establish exhaustive source coverage." :
            [&]() { std::string text; for (const auto & gap : value.gaps) { if (!text.empty()) text += "; "; text += gap; } return text; }();
        report_put.lineage.parent_uri = source_manifest.uri;
        report_put.lineage.chunk_count = 1; report_put.lineage.byte_length = document_json.size();
        report_put.lineage.derivation = "investigation.report.compose:structured-document-v1";
        agent_resource_descriptor report;
        if (!bindings.resource_runtime.store->put_text(report_put, report, error)) return common_adapter_execution_failure(
            "tool.investigation.report_store_failed", std::move(error), "The frozen structured report could not be persisted.");
        const bool complete = report_complete;
        if (!bindings.data_store) return common_adapter_execution_failure(
            "tool.investigation.report_index_unavailable", "investigation report index is unavailable",
            "The report resources were stored, but cannot be attached to the case for later discovery.");
        common_investigation_report_index_entry report_index;
        report_index.case_id = value.case_id;
        report_index.case_revision = value.revision;
        report_index.source_id = value.source_id;
        report_index.case_fingerprint = value.fingerprint;
        report_index.content_fingerprint = fingerprint;
        report_index.report_uri = report.uri;
        report_index.markdown_uri = markdown_resource.uri;
        report_index.source_manifest_uri = source_manifest.uri;
        report_index.limitations = report_limitations;
        report_index.complete = complete;
        report_index.created_at_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        for (const auto & snapshot : attached_snapshots)
            report_index.visualization_uris.push_back(snapshot.at("uri").get<std::string>());
        common_investigation_report_index_entry published_report;
        if (!bindings.data_store->publish_investigation_report(bindings.tool_scope, report_index,
                published_report, error)) return common_adapter_execution_failure(
            "tool.investigation.report_index_publish_failed", std::move(error),
            "The frozen report was created but was not published in the case report index.");
        auto result = common_adapter_success_json({{"case_id", value.case_id}, {"case_revision", value.revision},
            {"report_uri", published_report.report_uri}, {"markdown_uri", published_report.markdown_uri},
            {"source_manifest_uri", published_report.source_manifest_uri},
            {"content_fingerprint", published_report.content_fingerprint}, {"complete", published_report.complete},
            {"limitations", published_report.limitations}, {"visualization_uris", [&]() {
                json uris = json::array(); for (const auto & snapshot : attached_snapshots) uris.push_back(snapshot["uri"]); return uris;
            }()}});
        result.resource_refs = {source_manifest, markdown_resource, report};
        if (published_report.report_uri != report.uri || published_report.markdown_uri != markdown_resource.uri ||
                published_report.source_manifest_uri != source_manifest.uri) {
            result.resource_refs.clear();
            const auto authority = make_agent_resource_read_authority(
                bindings.resource_runtime, static_cast<int64_t>(std::time(nullptr)));
            for (const auto & uri : {published_report.source_manifest_uri, published_report.markdown_uri, published_report.report_uri}) {
                agent_resource_descriptor descriptor;
                if (bindings.resource_runtime.store->stat(uri, authority, descriptor, error)) result.resource_refs.push_back(std::move(descriptor));
                else error.clear();
            }
        }
        return result;
    }
    if (operation == "investigation.collect" || operation == "investigation.log.import") {
        const bool file_import = operation == "investigation.log.import";
        if (!arguments.contains("expected_revision") || !arguments["expected_revision"].is_number_integer() ||
                (!arguments["expected_revision"].is_number_unsigned() && arguments["expected_revision"].get<int64_t>() < 1))
            return common_adapter_validation_failure("tool.investigation.collect_invalid_arguments", "expected_revision must be a positive integer");
        std::string import_analysis_id, import_parameters;
        if (file_import) {
            if (!arguments.contains("expected_revision") || !arguments["expected_revision"].is_number_integer() ||
                    !arguments.contains("idempotency_key") || !arguments["idempotency_key"].is_string() ||
                    arguments["idempotency_key"].get<std::string>().empty() || arguments["idempotency_key"].get<std::string>().size() > 128)
                return common_adapter_validation_failure("tool.investigation.log_invalid_arguments", "a positive revision and bounded idempotency key are required");
            const auto identity = "log-import\n" + value.case_id + "\n" + arguments["idempotency_key"].get<std::string>();
            import_analysis_id = "analysis-" + hash_sha256_hex(identity.data(),identity.size()).substr(0,40);
            nlohmann::json canonical = arguments;
            canonical.erase("expected_revision"); canonical.erase("idempotency_key");
            const auto payload = canonical.dump();
            import_parameters = "sha256:" + hash_sha256_hex(payload.data(),payload.size());
            const auto existing = std::find_if(value.analyses.begin(),value.analyses.end(),[&](const auto & a) { return a.id == import_analysis_id; });
            if (existing != value.analyses.end()) {
                if (existing->parameters_fingerprint != import_parameters)
                    return common_adapter_failure("tool.investigation.idempotency_conflict",common_tool_failure_class::conflict,
                        false,"This log import key was already used for a different page or profile.","log import idempotency payload mismatch");
                auto replay = json::parse(existing->result_json,nullptr,false);
                if (!replay.is_object()) return common_adapter_execution_failure("tool.investigation.import_result_invalid",
                    "saved import response is invalid","The import result cannot be replayed.");
                replay["reused"] = true; replay["current_revision"] = value.revision;
                return common_adapter_success_json(replay);
            }
        }
        if (arguments.value("expected_revision", uint64_t(0)) != value.revision) {
            return common_adapter_execution_failure("tool.investigation.revision_conflict",
                "case revision changed before collection", "Reload the case before collecting more evidence.");
        }
        if (!bindings.resource_runtime.store) return common_adapter_execution_failure(
            "tool.investigation.evidence_store_unavailable", "resource store is unavailable",
            "Evidence collection is unavailable because persistent resource storage is not configured.");
        if (!arguments.contains("expected_revision") || !arguments["expected_revision"].is_number_integer() ||
                (arguments.contains("limit") && !arguments["limit"].is_number_unsigned() &&
                    (!arguments["limit"].is_number_integer() || arguments["limit"].get<int64_t>() < 1)))
            return common_adapter_validation_failure("tool.investigation.collect_invalid_arguments", "revision and limit must be positive integers");
        const size_t limit = arguments.value("limit", size_t(200));
        if (limit == 0 || limit > 500) return common_adapter_validation_failure(
            "tool.investigation.collect_limit_invalid", "collection page size must be from 1 to 500");
        std::string collection_source_id;
        const auto dataset_uri = arguments.value("dataset", std::string());
        std::vector<common_investigation_relation_rule> configured_relation_profiles;
        json page, query;
        common_investigation_log_page imported;
        std::string parser_fingerprint;
        json import_binding;
        size_t start_record = 0;
        if (file_import) {
            if (!arguments.contains("source_id") || !arguments["source_id"].is_string() ||
                    !arguments.contains("format") || !arguments["format"].is_string() ||
                    !arguments.contains("fields") || !arguments["fields"].is_object() ||
                    (arguments.contains("cursor") && !arguments["cursor"].is_string()))
                return common_adapter_validation_failure("tool.investigation.log_profile_invalid", "source, format, fields and optional cursor have invalid types");
            collection_source_id = arguments["source_id"].get<std::string>();
            const auto source = std::find_if(value.sources.begin(), value.sources.end(), [&](const auto & item) {
                return item.id == collection_source_id && item.kind == common_investigation_source_kind::log_file;
            });
            if (source == value.sources.end() || source->representation_uri.empty())
                return common_adapter_validation_failure("tool.investigation.log_source_unprepared", "attach and prepare the selected log file before importing events");
            std::map<std::string, std::string> fields;
            for (auto it = arguments["fields"].begin(); it != arguments["fields"].end(); ++it) {
                if (!it.value().is_string()) return common_adapter_validation_failure(
                    "tool.investigation.log_profile_invalid", "every mapped source field must be a string");
                fields.emplace(it.key(), it.value().get<std::string>());
            }
            query = {{"schema","investigation-log-parser.v1"},{"format",arguments["format"]},{"fields",fields},
                {"source_id",source->id},{"source_revision",source->source_revision},{"representation_hash",source->representation_hash}};
            const auto profile_json = nlohmann::json(query).dump();
            parser_fingerprint = hash_sha256_hex(profile_json.data(), profile_json.size());
            import_binding = {{"case_id",value.case_id},{"scope",{{"namespace",bindings.tool_scope.namespace_id},
                {"project",bindings.tool_scope.project_id},{"session",bindings.tool_scope.session_id}}},{"profile",parser_fingerprint}};
            start_record = source->log_import_next_record;
            size_t requested_record = start_record;
            if (arguments.contains("cursor") && !arguments["cursor"].get<std::string>().empty() &&
                    (!investigation_page_offset(arguments["cursor"].get<std::string>(), value.revision, import_binding, requested_record) ||
                        requested_record != start_record))
                return common_adapter_validation_failure("tool.investigation.cursor_invalid", "log cursor is stale or belongs to another source, profile or scope");
            const auto authority = make_agent_resource_read_authority(bindings.resource_runtime, static_cast<int64_t>(std::time(nullptr)));
            agent_resource_descriptor descriptor;
            std::string text;
            if (!bindings.resource_runtime.store->stat(source->representation_uri, authority, descriptor, error) ||
                    descriptor.size_bytes > 4 * 1024 * 1024 ||
                    !bindings.resource_runtime.store->read_text(source->representation_uri, authority, descriptor.size_bytes + 1, text, error) ||
                    text.size() != descriptor.size_bytes)
                return common_adapter_execution_failure("tool.investigation.log_read_failed", error.empty() ? "log representation is oversized or incomplete" : error,
                    "The retained log text could not be read completely.");
            if (!common_investigation_parse_log(text, arguments["format"].get<std::string>(), fields,
                    start_record, limit, imported, error))
                return common_adapter_validation_failure("tool.investigation.log_parse_failed", error);
            // One source uses one declared parser profile. A different profile is
            // a new source/analysis, never a silent replacement of old evidence.
            if (!source->log_parser_profile_json.empty() && source->log_parser_profile_json != profile_json)
                return common_adapter_validation_failure("tool.investigation.log_profile_conflict", "this source already has a different published parser profile");
            json names = json::array();
            for (const auto & row : imported.rows) for (auto it = row.begin(); it != row.end(); ++it)
                if (std::find(names.begin(), names.end(), json(it.key())) == names.end()) names.push_back(it.key());
            json rows = json::array();
            for (auto row : imported.rows) {
                row["@source.index"] = parser_fingerprint;
                if (std::find(names.begin(), names.end(), json("@source.index")) == names.end()) names.push_back("@source.index");
                json cells = json::array();
                for (const auto & name : names) cells.push_back(row.contains(name.get<std::string>()) ? row.at(name.get<std::string>()) : json());
                rows.push_back(std::move(cells));
            }
            page = {{"columns",names},{"rows",rows},{"complete",imported.exhausted && imported.rejected_count == 0},
                {"rejected",imported.rejected},{"rejected_count",imported.rejected_count},{"exhausted",imported.exhausted},
                {"record_start",start_record},{"record_end",imported.next_record},{"representation_uri",source->representation_uri}};
        } else {
            common_agent_dataset_descriptor dataset;
            if (!bindings.data_store->get_dataset_descriptor(dataset_uri, dataset, error) ||
                    dataset.ref.storage_mode != common_agent_dataset_storage_mode::virtual_) {
                return common_adapter_not_found_failure("tool.investigation.dataset_unavailable",
                    error.empty() ? "dataset is not a configured virtual search dataset" : std::move(error),
                    "Choose a configured virtual search dataset.");
            }
            collection_source_id = dataset.ref.source_provider;
            auto attached_source = std::find_if(value.sources.begin(), value.sources.end(), [&](const auto & source) {
                return source.kind == common_investigation_source_kind::search_connector &&
                    source.provider_id == collection_source_id;
            });
            if (collection_source_id.empty() ||
                    (attached_source == value.sources.end() && collection_source_id != value.source_id)) {
                return common_adapter_not_found_failure("tool.investigation.dataset_unavailable",
                    "dataset provider is not attached to this case", "Attach an authorized search source before collecting it.");
            }
            if (attached_source == value.sources.end()) {
                common_investigation_source legacy_source;
                legacy_source.id = "provider-" + collection_source_id;
                legacy_source.kind = common_investigation_source_kind::search_connector;
                legacy_source.provider_id = collection_source_id;
                legacy_source.display_name = collection_source_id;
                legacy_source.processing_status = "not_collected";
                legacy_source.complete = false;
                value.sources.push_back(std::move(legacy_source));
            }
            if (!dataset.ref.source_request_json.empty()) {
                const auto source_request = json::parse(dataset.ref.source_request_json, nullptr, false);
                if (!source_request.is_object() ||
                        (source_request.contains("relation_profiles") &&
                            (!source_request["relation_profiles"].is_array() || source_request["relation_profiles"].size() > 256))) {
                    return common_adapter_validation_failure("tool.investigation.source_descriptor_invalid",
                        "configured source relation profiles are malformed");
                }
                if (source_request.contains("relation_profiles")) {
                    for (const auto & profile : source_request["relation_profiles"]) {
                        if (!profile.is_object() || !profile.contains("id") || !profile["id"].is_string() ||
                                !profile.contains("event_type") || !profile["event_type"].is_string() ||
                                !profile.contains("from_kind") || !profile["from_kind"].is_string() ||
                                !profile.contains("to_kind") || !profile["to_kind"].is_string() ||
                                !profile.contains("relation_kind") || !profile["relation_kind"].is_string()) {
                            return common_adapter_validation_failure("tool.investigation.source_descriptor_invalid",
                                "configured source relation profile is incomplete");
                        }
                        common_investigation_relation_rule rule;
                        rule.id = profile["id"].get<std::string>();
                        rule.event_type = profile["event_type"].get<std::string>();
                        if (!investigation_entity_kind_from_name(profile["from_kind"].get<std::string>(), rule.from_kind) ||
                                !investigation_entity_kind_from_name(profile["to_kind"].get<std::string>(), rule.to_kind) ||
                                !investigation_relation_kind_from_name(profile["relation_kind"].get<std::string>(), rule.relation_kind) ||
                                (profile.contains("from_role") && !profile["from_role"].is_string()) ||
                                (profile.contains("to_role") && !profile["to_role"].is_string())) {
                            return common_adapter_validation_failure("tool.investigation.source_descriptor_invalid",
                                "configured source relation profile contains an invalid kind or role");
                        }
                        rule.from_role = profile.value("from_role", std::string());
                        rule.to_role = profile.value("to_role", std::string());
                        configured_relation_profiles.push_back(std::move(rule));
                    }
                }
            }
            json selected = json::array();
            for (const auto & column : dataset.columns) selected.push_back(column.name);
            json predicates = json::array({{{"field", arguments.value("identifier_field", std::string())},
                {"operator", "="}, {"value", arguments.value("identifier_value", std::string())}}});
            const auto has_column = [&](const std::string & name) {
                return std::any_of(dataset.columns.begin(), dataset.columns.end(),
                    [&](const auto & column) { return column.name == name; });
            };
            const auto first_column = [&](std::initializer_list<const char *> candidates) {
                for (const char * candidate : candidates) if (has_column(candidate)) return std::string(candidate);
                return std::string();
            };
            const auto time_field = first_column({"@timestamp", "event_time", "event.time"});
            if (time_field.empty()) return common_adapter_validation_failure(
                "tool.investigation.timestamp_unmapped",
                "the configured source has no supported event-time field; collection cannot enforce this case window");
            predicates.push_back({{"field", time_field}, {"operator", ">="}, {"value", value.window_start_ms}});
            predicates.push_back({{"field", time_field}, {"operator", "<"}, {"value", value.window_end_ms}});
            query = {{"dataset", dataset_uri}, {"select", selected}, {"where", predicates},
                {"limit", limit}, {"max_result_rows", limit}, {"max_scan_rows", limit}};
            std::string response;
            if (!bindings.data_store->execute_scoped("data.query", query.dump(), bindings.tool_scope,
                    bindings.allow_network, response, error)) {
                return common_adapter_execution_failure("tool.investigation.collect_failed", std::move(error),
                    "The configured source could not return this evidence page.");
            }
            page = json::parse(response, nullptr, false);
            if (!page.is_object() || !page.contains("columns") || !page["columns"].is_array() ||
                    !page.contains("rows") || !page["rows"].is_array()) return common_adapter_execution_failure(
                "tool.investigation.collect_invalid_result", "virtual dataset returned an invalid row page",
                "The source result could not be normalized.");
        }
        const bool query_complete = page.value("complete", false) &&
            !page.value("scan_truncated", false) && !page.value("result_truncated", false);
        std::vector<std::string> columns;
        for (const auto & column : page["columns"]) if (column.is_string()) columns.push_back(column.get<std::string>());
        const auto first_column = [&](std::initializer_list<const char *> candidates) {
            for (const char * candidate : candidates)
                if (std::find(columns.begin(), columns.end(), candidate) != columns.end()) return std::string(candidate);
            return std::string();
        };
        std::vector<json> rows;
        for (const auto & row : page["rows"]) {
            if (!row.is_array() || row.size() != columns.size()) continue;
            json object = json::object();
            for (size_t i = 0; i < columns.size(); ++i) object[columns[i]] = row[i];
            rows.push_back(std::move(object));
        }
        const auto get_string = [](const json & row, const std::string & field) {
            if (field.empty() || !row.contains(field) || row[field].is_null()) return std::string();
            if (row[field].is_string()) return row[field].get<std::string>();
            return row[field].dump();
        };
        const auto get_alias = [&](const json & row, std::initializer_list<const char *> candidates) {
            for (const char * candidate : candidates) {
                const auto result = get_string(row, candidate);
                if (!result.empty()) return result;
            }
            return std::string();
        };
        std::map<std::string, common_investigation_entity> entities_by_id;
        std::unordered_set<std::string> existing_events;
        for (const auto & event : value.events) existing_events.insert(event.id);
        std::vector<common_investigation_event> collected_events;
        std::vector<std::string> limitations;
        if (!file_import && (!page.value("complete", false) || page.value("scan_truncated", false) ||
                page.value("result_truncated", false))) {
            limitations.push_back("source page is partial or truncated; case coverage does not represent a complete search");
        }
        const auto entity_id = [&](common_investigation_entity_kind kind, const std::string & raw,
                const std::string & host = std::string()) {
            if (raw.empty()) return std::string();
            const std::string kind_name = kind == common_investigation_entity_kind::user ? "user" :
                kind == common_investigation_entity_kind::host ? "host" :
                kind == common_investigation_entity_kind::process ? "process" :
                kind == common_investigation_entity_kind::ip ? "ip" :
                kind == common_investigation_entity_kind::file ? "file" :
                kind == common_investigation_entity_kind::hash ? "hash" :
                kind == common_investigation_entity_kind::service ? "service" :
                kind == common_investigation_entity_kind::trace ? "trace" : "message";
            std::string identity = collection_source_id + ":" + kind_name + ":" + (host.empty() ? "" : host + ":") + raw;
            if (identity.size() > 480) identity = collection_source_id + ":" + kind_name + ":sha256:" + hash_sha256_hex(identity.data(), identity.size());
            const std::string id = "entity:" + identity;
            if (!entities_by_id.count(id)) entities_by_id.emplace(id, common_investigation_entity{
                id, collection_source_id, kind, raw, raw, host, 0, 0});
            return id;
        };
        if (file_import && imported.rejected_count) limitations.push_back(
            "Rejected " + std::to_string(imported.rejected_count) + " invalid log records on this page; positions are preserved in raw evidence.");
        json normalized_rows = json::array();
        for (const auto & row : rows) {
            const std::string source_index = get_alias(row, {"@source.index"});
            const std::string source_document_id = get_alias(row, {"@source.id", "event.id", "event_id", "id"});
            const std::string serialized_row = row.dump();
            const std::string document_identity = source_index.empty() ? source_document_id : source_index + ":" + source_document_id;
            const std::string event_identity = source_document_id.empty()
                ? hash_sha256_hex(serialized_row.data(), serialized_row.size()) : document_identity;
            const std::string event_prefix = "event:" + collection_source_id + ":";
            const std::string event_id = event_prefix + (event_prefix.size() + event_identity.size() > 512
                ? "sha256:" + hash_sha256_hex(event_identity.data(), event_identity.size()) : event_identity);
            if (existing_events.count(event_id)) continue;
            const std::string event_type_value = get_alias(row, {"event.type", "event_type", "event.category"});
            const std::string event_type = event_type_value.empty() ? "unknown" : event_type_value;
            common_investigation_time event_time;
            const std::string timestamp_field = first_column({"@timestamp", "event_time", "event.time"});
            if (!timestamp_field.empty() && row.contains(timestamp_field))
                event_time.has_event_time = parse_investigation_time(row[timestamp_field], event_time.event_time_ms);
            if (!event_time.has_event_time) limitations.push_back("one or more source timestamps were unavailable or invalid; event time remains unknown");
            event_time.collected_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            const std::string user_id = entity_id(common_investigation_entity_kind::user, get_alias(row, {"user.id", "user_id", "user.name"}));
            const std::string host_value = get_alias(row, {"host.id", "host_id", "host.name", "host_name", "host"});
            const std::string host_id = entity_id(common_investigation_entity_kind::host, host_value);
            const std::string source_ip = entity_id(common_investigation_entity_kind::ip, get_alias(row, {"source.ip", "source_ip"}));
            const std::string destination_ip = entity_id(common_investigation_entity_kind::ip, get_alias(row, {"destination.ip", "destination_ip"}));
            const std::string process_value = get_alias(row, {"process.entity_id", "process_entity_id", "process.pid", "process_pid", "process_id"});
            const std::string process_start = get_alias(row, {"process.start", "process_start", "process.start_time", "process_start_time"});
            const std::string process_identity = process_value.empty() ? std::string() :
                process_start.empty() ? process_value + "@" + event_id : process_value + "@" + process_start;
            if (!process_value.empty() && process_start.empty()) limitations.push_back(
                "process start time was missing; the process identity is isolated to its source event");
            const std::string process_id = entity_id(common_investigation_entity_kind::process, process_identity, host_value);
            const std::string parent_process_value = get_alias(row, {"process.parent.entity_id", "process.parent.pid", "parent_process_id", "parent_pid"});
            const std::string parent_process_start = get_alias(row, {"process.parent.start", "parent_process_start"});
            const std::string parent_process_identity = parent_process_value.empty() ? std::string() :
                parent_process_start.empty() ? parent_process_value + "@" + event_id : parent_process_value + "@" + parent_process_start;
            const std::string parent_process_id = entity_id(common_investigation_entity_kind::process,
                parent_process_identity, host_value);
            const std::string file_hash = get_alias(row, {"file.hash.sha256", "file_hash"});
            const std::string file_value = file_hash.empty()
                ? get_alias(row, {"file.path", "file_path"}) : file_hash;
            const std::string file_id = entity_id(file_hash.empty()
                ? common_investigation_entity_kind::file : common_investigation_entity_kind::hash, file_value);
            const std::string trace_id = entity_id(common_investigation_entity_kind::trace, get_alias(row, {"trace.id", "trace_id"}));
            const std::string message_id = entity_id(common_investigation_entity_kind::message, get_alias(row, {"message.id", "message_id"}));
            std::vector<std::string> refs;
            for (const auto & id : {user_id, host_id, source_ip, destination_ip, process_id, parent_process_id,
                    file_id, trace_id, message_id}) if (!id.empty()) refs.push_back(id);
            collected_events.push_back({event_id, collection_source_id,
                source_document_id.empty() ? event_id : source_document_id, event_type, event_time,
                refs, {}, {}, {}, {}});
            collected_events.back().source_index = source_index;
            if (file_import) {
                auto & event = collected_events.back();
                event.source_byte_offset = row.value("@source.byte_offset", uint64_t(0));
                event.source_byte_length = row.value("@source.byte_length", uint64_t(0));
                event.source_line = row.value("@source.line", uint64_t(0));
                event.parser_version = "log-parser-v1:" + parser_fingerprint;
            }
            if (!user_id.empty()) collected_events.back().entity_roles["user"] = user_id;
            if (!host_id.empty()) collected_events.back().entity_roles["host"] = host_id;
            if (!source_ip.empty()) collected_events.back().entity_roles["source_ip"] = source_ip;
            if (!destination_ip.empty()) collected_events.back().entity_roles["destination_ip"] = destination_ip;
            if (!process_id.empty()) {
                collected_events.back().entity_roles["child_process"] = process_id;
                collected_events.back().entity_roles["process"] = process_id;
            }
            if (!parent_process_id.empty()) collected_events.back().entity_roles["parent_process"] = parent_process_id;
            if (!file_id.empty()) collected_events.back().entity_roles["file"] = file_id;
            if (!trace_id.empty()) collected_events.back().entity_roles["trace"] = trace_id;
            if (!message_id.empty()) collected_events.back().entity_roles["message"] = message_id;
            existing_events.insert(event_id);
            normalized_rows.push_back({{"event_id", event_id}, {"source_index", source_index},
                {"source_document_id", source_document_id}, {"event_type", event_type},
                {"source_byte_offset",collected_events.back().source_byte_offset},
                {"source_byte_length",collected_events.back().source_byte_length},
                {"source_line",collected_events.back().source_line},{"parser_version",collected_events.back().parser_version},
                {"event_time_ms", event_time.event_time_ms}, {"has_event_time", event_time.has_event_time},
                {"collected_time_ms", event_time.collected_time_ms},
                {"entity_ids", refs}, {"entity_roles", collected_events.back().entity_roles}});
        }
        std::sort(limitations.begin(),limitations.end());
        limitations.erase(std::unique(limitations.begin(),limitations.end()),limitations.end());
        json raw_page = {{"source_id", collection_source_id}, {"dataset", dataset_uri},
            {"query", query}, {"result", page}, {"collected_at_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()}};
        agent_resource_put_request raw_put;
        raw_put.name = value.case_id + "-source-page-r" + std::to_string(value.revision + 1) + ".json";
        raw_put.description = "Immutable raw search page collected for investigation case " + value.case_id;
        raw_put.mime_type = "application/json"; raw_put.text = raw_page.dump(2);
        raw_put.scope = bindings.tool_scope.project_id.empty() ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
        raw_put.source_provider = collection_source_id; raw_put.source_tool = operation;
        apply_agent_resource_runtime(bindings.resource_runtime, raw_put);
        agent_resource_descriptor raw_descriptor;
        if (!bindings.resource_runtime.store->put_text(raw_put, raw_descriptor, error)) return common_adapter_execution_failure(
            "tool.investigation.raw_evidence_store_failed", std::move(error), "The raw search result could not be preserved.");
        json normalized = {{"schema_version", 1}, {"case_id", value.case_id}, {"case_revision", value.revision + 1},
            {"source_id", collection_source_id}, {"events", normalized_rows}, {"limitations", limitations}};
        agent_resource_put_request normalized_put = raw_put;
        normalized_put.name = value.case_id + "-normalized-r" + std::to_string(value.revision + 1) + ".json";
        normalized_put.description = "Normalized investigation events and entity references";
        normalized_put.text = normalized.dump(2);
        agent_resource_descriptor normalized_descriptor;
        if (!bindings.resource_runtime.store->put_text(normalized_put, normalized_descriptor, error)) return common_adapter_execution_failure(
            "tool.investigation.normalized_evidence_store_failed", std::move(error), "The normalized evidence could not be preserved.");
        for (auto & event : collected_events) {
            event.raw_resource_uri = raw_descriptor.uri;
            event.normalized_resource_uri = normalized_descriptor.uri;
            value.events.push_back(event);
        }
        for (auto & item : entities_by_id) {
            if (std::none_of(value.entities.begin(), value.entities.end(), [&](const auto & old) { return old.id == item.first; }))
                value.entities.push_back(std::move(item.second));
        }
        std::vector<common_investigation_relation_rule> rules;
        std::unordered_set<std::string> newly_collected_event_ids;
        for (const auto & event : collected_events) newly_collected_event_ids.insert(event.id);
        bool unprofiled_event = false;
        if (!configured_relation_profiles.empty()) {
            rules = configured_relation_profiles;
            for (const auto & event : collected_events) {
                if (std::none_of(rules.begin(), rules.end(), [&](const auto & rule) {
                        return rule.event_type == event.event_type;
                    })) unprofiled_event = true;
            }
        } else for (const auto & event : value.events) {
            const auto normalized_type = normalize_investigation_event_type(event.event_type);
            const auto rule_id = [&](const char * profile) {
                return std::string("builtin-") + profile + "-v1:" +
                    hash_sha256_hex(event.event_type.data(), event.event_type.size());
            };
            static const std::set<std::string> authentication_types = {
                "auth", "authentication", "authentication_failure", "authentication_success",
                "login", "logon", "user_authentication", "user_login", "user_logon"};
            static const std::set<std::string> communication_types = {
                "connection", "network", "network_connection", "network_flow"};
            static const std::set<std::string> process_types = {
                "process_create", "process_created", "process_spawn", "process_start"};
            static const std::set<std::string> file_types = {
                "file_access", "file_create", "file_delete", "file_modify", "file_open", "file_read", "file_write"};
            static const std::set<std::string> trace_types = {
                "message_correlation", "span_correlation", "trace_correlation"};
            const bool is_authentication = authentication_types.count(normalized_type) != 0;
            const bool is_communication = communication_types.count(normalized_type) != 0;
            const bool is_process = process_types.count(normalized_type) != 0;
            const bool is_file = file_types.count(normalized_type) != 0;
            const bool is_trace = trace_types.count(normalized_type) != 0;
            if (newly_collected_event_ids.count(event.id) &&
                    !(is_authentication || is_communication || is_process || is_file || is_trace)) {
                unprofiled_event = true;
            }
            if (is_authentication)
                rules.push_back({rule_id("authentication"), event.event_type, common_investigation_entity_kind::user,
                    common_investigation_entity_kind::host, common_investigation_relation_kind::authenticated_to,
                    "user", "host"});
            if (is_communication)
                rules.push_back({rule_id("network"), event.event_type, common_investigation_entity_kind::ip,
                    common_investigation_entity_kind::ip, common_investigation_relation_kind::communicated_with,
                    "source_ip", "destination_ip"});
            if (is_process)
                rules.push_back({rule_id("process"), event.event_type, common_investigation_entity_kind::process,
                    common_investigation_entity_kind::process, common_investigation_relation_kind::spawned,
                    "parent_process", "child_process"});
            if (is_file)
                rules.push_back({rule_id("file"), event.event_type, common_investigation_entity_kind::process,
                    common_investigation_entity_kind::file, common_investigation_relation_kind::accessed,
                    "process", "file"});
            if (is_trace)
                rules.push_back({rule_id("trace"), event.event_type, common_investigation_entity_kind::trace,
                    common_investigation_entity_kind::message, common_investigation_relation_kind::correlated_with,
                    "trace", "message"});
        }
        if (unprofiled_event) limitations.push_back(
            "one_or_more_events_have_no_declared_relation_profile_and_remain_unlinked_observations");
        std::sort(rules.begin(), rules.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
        rules.erase(std::unique(rules.begin(), rules.end(), [](const auto & a, const auto & b) { return a.id == b.id && a.event_type == b.event_type; }), rules.end());
        std::vector<std::string> relation_limitations;
        std::vector<common_investigation_entity> source_entities;
        std::copy_if(value.entities.begin(), value.entities.end(), std::back_inserter(source_entities),
            [&](const auto & entity) { return entity.source_id == collection_source_id; });
        std::vector<common_investigation_event> source_events;
        std::copy_if(value.events.begin(), value.events.end(), std::back_inserter(source_events),
            [&](const auto & event) { return event.source_id == collection_source_id; });
        std::vector<common_investigation_relation> source_relations;
        if (!common_investigation_build_relations(collection_source_id, source_entities, source_events,
                rules, source_relations, relation_limitations, error)) return common_adapter_validation_failure(
            "tool.investigation.normalization_failed", std::move(error));
        limitations.insert(limitations.end(), relation_limitations.begin(), relation_limitations.end());
        value.relations.erase(std::remove_if(value.relations.begin(), value.relations.end(), [&](const auto & relation) {
            return relation.source_id == collection_source_id;
        }), value.relations.end());
        value.relations.insert(value.relations.end(), source_relations.begin(), source_relations.end());
        std::sort(limitations.begin(),limitations.end());
        limitations.erase(std::unique(limitations.begin(),limitations.end()),limitations.end());
        for (const auto & limitation : limitations)
            if (std::find(value.gaps.begin(),value.gaps.end(),limitation) == value.gaps.end()) value.gaps.push_back(limitation);
        for (auto & source : value.sources) if (source.kind == common_investigation_source_kind::search_connector &&
                source.provider_id == collection_source_id) {
            const bool first_collection = source.processing_status == "not_collected";
            source.complete = first_collection ? query_complete : source.complete && query_complete;
            source.processing_status = source.complete ? "query_complete" : "query_partial";
            static const std::string partial_reason =
                "At least one collected query page was partial, truncated, or did not report complete coverage.";
            if (!query_complete && std::find(source.processing_limitations.begin(), source.processing_limitations.end(),
                    partial_reason) == source.processing_limitations.end())
                source.processing_limitations.push_back(partial_reason);
        }
        bool imported_source_complete = false;
        if (file_import) for (auto & source : value.sources) if (source.id == collection_source_id) {
            source.log_parser_profile_json = nlohmann::json(query).dump();
            source.log_import_next_record = imported.next_record;
            source.log_import_rejected_records += imported.rejected_count;
            source.log_import_exhausted = imported.exhausted;
            source.processing_status = imported.exhausted ? "ready" : "log_import_partial";
            if (imported.rejected_count && std::find(source.processing_limitations.begin(), source.processing_limitations.end(),
                    "Some log records could not be parsed; see immutable import pages.") == source.processing_limitations.end()) {
                source.processing_limitations.push_back("Some log records could not be parsed; see immutable import pages.");
                source.complete = false;
            }
            imported_source_complete = common_investigation_source_complete(source);
        }
        json import_result;
        if (file_import) {
            import_result = {{"case_id",value.case_id},{"revision",value.revision + 1},
                {"collected_events",collected_events.size()},{"complete",query_complete},
                {"limitations",limitations},{"raw_resource_uri",raw_descriptor.uri},{"normalized_resource_uri",normalized_descriptor.uri},
                {"next_cursor",!imported.exhausted ? investigation_page_cursor(value.revision + 1,imported.next_record,import_binding) : std::string()},
                {"exhausted",imported.exhausted},{"rejected_count",imported.rejected_count},
                {"source_complete",imported_source_complete},{"reused",false}};
            common_investigation_analysis_record record;
            record.id = import_analysis_id; record.kind = "log_import"; record.algorithm_version = "log-parser-v1";
            record.published_revision = value.revision + 1; record.source_ids = {collection_source_id};
            record.parameters_fingerprint = import_parameters; record.result_json = import_result.dump();
            record.completeness = query_complete ? "complete" : "partial"; record.limitations = limitations;
            value.analyses.push_back(std::move(record));
        }
        const uint64_t expected = value.revision++;
        common_investigation_case_revision published;
        const auto revision_uri = "investigation-case://" + value.case_id + "/revision/" + std::to_string(value.revision);
        if (!bindings.data_store->publish_investigation_case(value, expected, revision_uri, published, error))
            return common_adapter_execution_failure("tool.investigation.revision_conflict", std::move(error),
                "The case changed while evidence was being collected; reload and retry.");
        if (file_import) return common_adapter_success_json(import_result);
        return common_adapter_success_json({{"case_id", published.case_id}, {"revision", published.revision},
            {"collected_events", collected_events.size()}, {"complete", page.value("complete", false)},
            {"limitations", limitations}, {"raw_resource_uri", raw_descriptor.uri},
            {"normalized_resource_uri", normalized_descriptor.uri},
            {"next_cursor", file_import && !imported.exhausted ? investigation_page_cursor(published.revision, imported.next_record, import_binding) : std::string()},
            {"exhausted",file_import ? imported.exhausted : query_complete},
            {"rejected_count",file_import ? imported.rejected_count : size_t(0)}});
    }
    if (operation == "investigation.analyze") {
        const auto analysis = arguments.value("analysis", std::string());
        common_investigation_analysis_budget budget;
        budget.max_expansions = arguments.value("max_expansions", budget.max_expansions);
        budget.max_path_length = arguments.value("max_path_length", budget.max_path_length);
        if (analysis == "source_followup") {
            if (!arguments.contains("source_ids") || !arguments["source_ids"].is_array())
                return common_adapter_validation_failure("tool.investigation.followup_invalid", "select source IDs for follow-up");
            std::vector<std::string> selected;
            for (const auto & id : arguments["source_ids"]) {
                if (!id.is_string()) return common_adapter_validation_failure("tool.investigation.followup_invalid", "source IDs must be strings");
                selected.push_back(id.get<std::string>());
            }
            json result;
            if (!common_investigation_source_followup(value, selected, budget, result, error))
                return common_adapter_validation_failure("tool.investigation.followup_invalid",error);
            return common_adapter_success_json(result);
        }
        if (analysis == "graph_metrics") {
            common_investigation_graph_metrics metrics;
            if (!common_investigation_graph_analyze(value, budget, metrics, error)) {
                return common_adapter_validation_failure("tool.investigation.analysis_invalid", std::move(error));
            }
            json nodes = json::array();
            for (const auto & node : metrics.nodes) nodes.push_back({{"entity_id", node.entity_id},
                {"degree", node.degree}, {"component", node.component},
                {"betweenness", node.betweenness}});
            return common_adapter_success_json({{"case_revision", metrics.case_revision},
                {"analysis", analysis}, {"complete", metrics.complete},
                {"nodes", nodes}, {"explored_states", metrics.explored_states},
                {"betweenness_normalized", metrics.betweenness_normalized},
                {"optimality", metrics.complete ? "exact_for_case_revision" : "budget_limited"}});
        }
        const auto start = arguments.value("start_entity_id", std::string());
        const auto goal = arguments.value("goal_entity_id", std::string());
        if (start.empty() || goal.empty()) return common_adapter_validation_failure(
            "tool.investigation.analysis_endpoints_required", "path analysis requires start_entity_id and goal_entity_id");
        std::vector<std::string> relation_ids;
        bool complete = false;
        const bool ok = analysis == "shortest_path"
            ? common_investigation_graph_shortest_path(value, start, goal, budget, relation_ids, complete, error)
            : analysis == "temporal_path"
                ? common_investigation_temporal_path(value, start, goal,
                    arguments.value("not_before_ms", value.window_start_ms), budget, relation_ids, complete, error)
                : false;
        if (!ok) return common_adapter_validation_failure("tool.investigation.analysis_invalid",
            error.empty() ? "analysis must be graph_metrics, shortest_path, or temporal_path" : std::move(error));
        std::set<std::string> evidence_ids;
        for (const auto & relation_id : relation_ids) {
            const auto relation = std::find_if(value.relations.begin(), value.relations.end(),
                [&](const auto & item) { return item.id == relation_id; });
            if (relation != value.relations.end()) evidence_ids.insert(relation->evidence_event_ids.begin(),
                relation->evidence_event_ids.end());
        }
        return common_adapter_success_json({{"case_revision", value.revision},
            {"analysis", analysis}, {"complete", complete}, {"relation_ids", relation_ids},
            {"evidence_refs", std::vector<std::string>(evidence_ids.begin(), evidence_ids.end())},
            {"optimality", complete ? (relation_ids.empty() ? "no_path_in_case" : "shortest_hop_count") : "budget_limited"}});
    }
    if (operation == "investigation.graph.expand") {
        common_investigation_graph_query query;
        query.case_id = case_id;
        query.case_revision = arguments.value("case_revision", uint64_t(0));
        query.center_entity_id = arguments.value("center_entity_id", std::string());
        query.max_depth = arguments.value("max_depth", size_t(1));
        query.max_nodes = arguments.value("max_nodes", size_t(100));
        query.max_edges = arguments.value("max_edges", size_t(200));
        query.window_start_ms = arguments.value("window_start_ms", value.window_start_ms);
        query.window_end_ms = arguments.value("window_end_ms", value.window_end_ms);
        common_investigation_graph_result graph;
        if (!common_investigation_graph_expand(value, query, graph, error)) {
            return common_adapter_validation_failure("tool.investigation.graph_invalid", std::move(error));
        }
        json nodes = json::array(), edges = json::array();
        for (const auto & entity : graph.nodes) nodes.push_back({{"id", entity.id},
            {"type", static_cast<int>(entity.kind)}, {"label", entity.display_name.empty() ? entity.value : entity.display_name}});
        for (const auto & relation : graph.edges) edges.push_back({{"id", relation.id},
            {"source", relation.from_entity_id}, {"target", relation.to_entity_id},
            {"type", static_cast<int>(relation.kind)}, {"first_seen_ms", relation.first_seen_ms},
            {"last_seen_ms", relation.last_seen_ms}, {"event_count", relation.event_count},
            {"evidence_refs", relation.evidence_event_ids}, {"classification", relation.confidence_label}});
        return common_adapter_success_json({{"case_revision", graph.case_revision}, {"nodes", nodes},
            {"edges", edges}, {"truncated", graph.truncated}, {"limitations", graph.limitations}});
    }
    if (operation == "investigation.events") {
        for (const char * field : {"case_revision", "limit", "window_start_ms", "window_end_ms"}) {
            if (arguments.contains(field) && !arguments[field].is_number_integer())
                return common_adapter_validation_failure("tool.investigation.events_invalid_arguments",
                    std::string(field) + " must be an integer");
        }
        for (const char * field : {"entity_id", "relation_id", "cursor"}) {
            if (arguments.contains(field) && !arguments[field].is_string())
                return common_adapter_validation_failure("tool.investigation.events_invalid_arguments",
                    std::string(field) + " must be a string");
        }
        if (!arguments.contains("case_revision")) return common_adapter_validation_failure(
            "tool.investigation.events_invalid_arguments", "case_revision is required");
        const auto entity_id = arguments.value("entity_id", std::string());
        const auto relation_id = arguments.value("relation_id", std::string());
        const uint64_t requested_revision = arguments.value("case_revision", uint64_t(0));
        if (requested_revision != value.revision) return common_adapter_execution_failure(
            "tool.investigation.stale_cursor", "requested case revision is no longer current",
            "Reload the case before continuing this event page.");
        if (!entity_id.empty() && !relation_id.empty()) return common_adapter_validation_failure(
            "tool.investigation.events_invalid", "events accepts at most one entity_id or relation_id filter");
        std::set<std::string> evidence_ids;
        if (!relation_id.empty()) {
            const auto relation = std::find_if(value.relations.begin(), value.relations.end(),
                [&](const auto & item) { return item.id == relation_id; });
            if (relation == value.relations.end()) return common_adapter_not_found_failure(
                "tool.investigation.relation_unavailable", "relation is not in this case",
                "The selected relation is unavailable in this case.");
            evidence_ids.insert(relation->evidence_event_ids.begin(), relation->evidence_event_ids.end());
        }
        int64_t window_start = arguments.value("window_start_ms", value.window_start_ms);
        int64_t window_end = arguments.value("window_end_ms", value.window_end_ms);
        if (window_start == 0 && window_end == 0) {
            window_start = std::numeric_limits<int64_t>::min();
            window_end = std::numeric_limits<int64_t>::max();
        }
        if (window_start >= window_end) return common_adapter_validation_failure(
            "tool.investigation.events_window_invalid", "event window start must precede its end");
        const json binding = {{"case_id", value.case_id}, {"entity_id", entity_id},
            {"relation_id", relation_id}, {"window_start_ms", window_start}, {"window_end_ms", window_end},
            {"namespace_id", bindings.tool_scope.namespace_id}, {"project_id", bindings.tool_scope.project_id},
            {"session_id", bindings.tool_scope.project_id.empty() ? bindings.tool_scope.session_id : std::string()}};
        size_t offset = 0;
        if (!investigation_page_offset(arguments.value("cursor", std::string()), value.revision, binding, offset))
            return common_adapter_execution_failure("tool.investigation.cursor_invalid",
                "event cursor does not match this case revision and filter",
                "The page cursor is invalid or stale; reopen the case and start again.");
        const size_t limit = arguments.value("limit", size_t(100));
        if (limit == 0 || limit > 200) return common_adapter_validation_failure(
            "tool.investigation.page_limit_invalid", "event page size must be from 1 to 200");
        std::vector<const common_investigation_event *> matched;
        for (const auto & event : value.events) {
            if (!entity_id.empty() && std::find(event.entity_ids.begin(), event.entity_ids.end(), entity_id) == event.entity_ids.end()) continue;
            if (!relation_id.empty() && !evidence_ids.count(event.id)) continue;
            if (event.time.has_event_time && (event.time.event_time_ms < window_start || event.time.event_time_ms >= window_end)) continue;
            matched.push_back(&event);
        }
        std::stable_sort(matched.begin(), matched.end(), [](const auto * a, const auto * b) {
            if (a->time.has_event_time != b->time.has_event_time) return a->time.has_event_time;
            if (a->time.has_event_time && a->time.event_time_ms != b->time.event_time_ms)
                return a->time.event_time_ms < b->time.event_time_ms;
            return a->id < b->id;
        });
        if (offset > matched.size()) return common_adapter_execution_failure(
            "tool.investigation.cursor_invalid", "event cursor offset exceeds the matching result set",
            "The page cursor is invalid or stale; reopen the case and start again.");
        json events = json::array();
        size_t next_offset = offset;
        size_t serialized_bytes = 0;
        constexpr size_t max_page_bytes = 48 * 1024;
        for (; next_offset < matched.size() && events.size() < limit; ++next_offset) {
            const auto & event = *matched[next_offset];
            json row = {{"id", event.id}, {"source_index", event.source_index},
                {"source_document_id", event.source_document_id},
                {"source_byte_offset",event.source_byte_offset},{"source_byte_length",event.source_byte_length},
                {"source_line",event.source_line},{"parser_version",event.parser_version},
                {"event_type", event.event_type}, {"time", {{"event_time_ms", event.time.event_time_ms},
                    {"collected_time_ms", event.time.collected_time_ms}, {"uncertainty_ms", event.time.uncertainty_ms},
                    {"has_event_time", event.time.has_event_time}}}, {"raw_resource_uri", event.raw_resource_uri},
                {"normalized_resource_uri", event.normalized_resource_uri}, {"entity_ids", event.entity_ids}};
            const auto source = std::find_if(value.sources.begin(),value.sources.end(),[&](const auto & v) { return v.id == event.source_id; });
            if (source != value.sources.end()) row["source_resource_uri"] = source->representation_uri;
            const size_t row_bytes = row.dump().size();
            if (!events.empty() && serialized_bytes + row_bytes > max_page_bytes) break;
            if (row_bytes > max_page_bytes) return common_adapter_execution_failure(
                "tool.investigation.item_too_large", "one event exceeds the bounded page size",
                "Read its referenced resource through the resource tools.");
            serialized_bytes += row_bytes;
            events.push_back(std::move(row));
        }
        const bool has_more = next_offset < matched.size();
        const std::string next_cursor = has_more ? investigation_page_cursor(value.revision, next_offset, binding) : std::string();
        return common_adapter_success_json({{"case_id", value.case_id}, {"case_revision", value.revision},
            {"events", std::move(events)}, {"next_cursor", next_cursor}, {"has_more", has_more},
            {"total_matching", matched.size()}, {"limitations", value.gaps}});
    }
    if (operation == "investigation.claim.hypothesize") {
        const auto claim_id = arguments.value("claim_id", std::string());
        if (std::any_of(value.claims.begin(), value.claims.end(),
                [&](const auto & claim) { return claim.id == claim_id; })) {
            return common_adapter_validation_failure("tool.investigation.claim_duplicate", "claim_id already exists");
        }
        common_investigation_claim claim;
        claim.id = claim_id;
        claim.text = arguments.value("text", std::string());
        claim.state = common_investigation_claim_state::hypothesis;
        claim.supporting_event_ids = arguments.value("supporting_event_ids", std::vector<std::string>{});
        claim.contradicting_event_ids = arguments.value("contradicting_event_ids", std::vector<std::string>{});
        claim.limitations = arguments.value("limitations", std::vector<std::string>{});
        claim.supporting_statement_ids = arguments.value("supporting_statement_ids", std::vector<std::string>{});
        claim.contradicting_statement_ids = arguments.value("contradicting_statement_ids", std::vector<std::string>{});
        value.claims.push_back(std::move(claim));
        const uint64_t expected = value.revision++;
        common_investigation_case_revision published;
        const auto revision_uri = "investigation-case://" + value.case_id + "/revision/" + std::to_string(value.revision);
        if (!bindings.data_store->publish_investigation_case(value, expected, revision_uri, published, error)) {
            return common_adapter_execution_failure("tool.investigation.revision_conflict",
                std::move(error), "The case changed; reload it before recording the hypothesis.");
        }
        return common_adapter_success_json({{"case_id", published.case_id}, {"revision", published.revision},
            {"claim_state", "hypothesis"}});
    }
    return common_adapter_validation_failure("tool.investigation.operation_unknown", "unknown investigation operation");
}

}

bool common_try_register_investigation_tool_adapter(
        const common_tool_definition & definition, const common_native_tool_bindings & bindings,
        common_tool_registry & registry, bool & installed, std::string & error) {
    installed = false;
    if (definition.executor_id.rfind("builtin.investigation.", 0) != 0) return false;
    const bool read_only = definition.risk_class == common_tool_risk_class::local_read ||
        definition.risk_class == common_tool_risk_class::network_read;
    const bool policy_gated = !read_only && definition.requires_confirmation;
    installed = common_adapter_register_definition(definition, registry,
        [bindings, operation = definition.name](const std::string & input) {
            std::string error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, error)) {
                return common_adapter_validation_failure(
                    "tool.investigation.invalid_arguments", std::move(error));
            }
            return execute_investigation_tool(bindings, operation, arguments);
        }, error, read_only, policy_gated);
    return true;
}
