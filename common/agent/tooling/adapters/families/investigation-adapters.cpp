#include "agent/tooling/adapters/families/investigation-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/artifact/artifact-contract.h"
#include "hash/hash.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
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
    if (value.is_number_integer()) { out = value.get<int64_t>(); return true; }
    if (value.is_number_float()) { out = static_cast<int64_t>(value.get<double>()); return true; }
    if (!value.is_string()) return false;
    const std::string text = value.get<std::string>();
    char * end = nullptr;
    const auto integer = std::strtoll(text.c_str(), &end, 10);
    if (end != text.c_str() && *end == '\0') { out = integer; return true; }
    if (text.size() < 20) return false;
    std::tm parsed{};
    std::istringstream input(text.substr(0, 19));
    input >> std::get_time(&parsed, "%Y-%m-%dT%H:%M:%S");
    if (input.fail()) return false;
    const time_t seconds = timegm(&parsed);
    if (seconds < 0) return false;
    int64_t millis = static_cast<int64_t>(seconds) * 1000;
    size_t cursor = 19;
    if (cursor < text.size() && text[cursor] == '.') {
        ++cursor;
        int fraction = 0, digits = 0;
        while (cursor < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor]))) {
            if (digits < 3) fraction = fraction * 10 + (text[cursor] - '0');
            ++digits; ++cursor;
        }
        while (digits < 3) { fraction *= 10; ++digits; }
        millis += fraction;
    }
    if (cursor < text.size() && text[cursor] == 'Z') ++cursor;
    else if (cursor + 6 == text.size() && (text[cursor] == '+' || text[cursor] == '-') && text[cursor + 3] == ':') {
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


common_tool_execution_result execute_investigation_tool(
        const common_native_tool_bindings & bindings, const std::string & operation,
        const json & arguments) {
    if (!bindings.data_store) return common_adapter_execution_failure(
        "tool.investigation.store_unavailable", "investigation persistence is unavailable",
        "Investigation storage is unavailable.");
    std::string error;
    if (operation == "investigation.case.create") {
        std::vector<common_agent_dataset_descriptor> datasets;
        if (!bindings.data_store->list_dataset_descriptors(datasets, error)) {
            return common_adapter_execution_failure("tool.investigation.sources_unavailable",
                std::move(error), "Configured investigation sources are unavailable.");
        }
        const auto source_id = arguments.value("source_id", std::string());
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
        common_investigation_case_revision value;
        static std::atomic<uint64_t> sequence{1};
        value.case_id = "case-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(sequence.fetch_add(1));
        value.title = arguments.value("title", std::string());
        value.question = arguments.value("question", std::string());
        value.source_id = source_id;
        value.window_start_ms = arguments.value("window_start_ms", int64_t(0));
        value.window_end_ms = arguments.value("window_end_ms", int64_t(0));
        value.scope = bindings.tool_scope;
        value.revision = 1;
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
            std::string authorization_error;
            if (!bindings.data_store->authorize_investigation_source(item.source_id, bindings.tool_scope, authorization_error)) continue;
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
    if (!bindings.data_store->authorize_investigation_source(value.source_id, bindings.tool_scope, error))
        return common_adapter_not_found_failure("tool.investigation.case_unavailable",
            "case source is no longer authorized", "The case is unavailable in the current scope.");
    if (operation == "investigation.case.open") {
        const std::string first_entity_id = value.entities.empty() ? std::string() : value.entities.front().id;
        return common_adapter_success_json({{"case", {{"case_id", value.case_id}, {"revision", value.revision},
            {"title", value.title}, {"question", value.question}, {"source_id", value.source_id},
            {"window_start_ms", value.window_start_ms}, {"window_end_ms", value.window_end_ms},
            {"counts", {{"entities", value.entities.size()}, {"events", value.events.size()},
                {"relations", value.relations.size()}, {"claims", value.claims.size()}, {"gaps", value.gaps.size()}}},
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
            if (item_type == "entities") collection = "entities";
            else if (item_type == "events") collection = "events";
            else if (item_type == "relations") collection = "relations";
            else if (item_type == "claims") collection = "claims";
            else if (item_type == "gaps") collection = "gaps";
            else return common_adapter_validation_failure("tool.investigation.items_type_invalid",
                "type must be entities, events, relations, claims, gaps or reports");
            source = serialized.at(collection);
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
            {"has_more", has_more}, {"total", source.size()}});
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
        markdown += "- Source: `" + safe_text(value.source_id) + "`\n- Time window (epoch ms): `" +
            std::to_string(value.window_start_ms) + "` to `" + std::to_string(value.window_end_ms) + "`\n\n";
        markdown += "## Investigation question\n\n" + safe_text(value.question) + "\n\n";
        markdown += "## Scope and coverage\n\n";
        markdown += value.gaps.empty() ? "No recorded collection or analysis gaps. This does not establish exhaustive source coverage.\n\n"
            : "Coverage is partial or qualified; see the listed gaps.\n\n";
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
            for (const auto & limitation : claim.limitations) markdown += "- Limitation: " + safe_text(limitation) + "\n";
            markdown += "\n";
        }
        markdown += "## Gaps and limitations\n\n";
        if (value.gaps.empty()) markdown += "No explicit gaps are recorded.\n";
        for (const auto & gap : value.gaps) markdown += "- " + safe_text(gap) + "\n";
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
        const json frozen_source{{"schema", "investigation-report-source.v1"}, {"case_id", value.case_id},
            {"case_revision", value.revision}, {"source_id", value.source_id},
            {"window_start_ms", value.window_start_ms}, {"window_end_ms", value.window_end_ms},
            {"case_fingerprint", value.fingerprint},
            {"content_fingerprint", fingerprint}, {"visualization_snapshots", attached_snapshots},
            {"event_count", value.events.size()}, {"relation_count", value.relations.size()},
            {"claim_count", value.claims.size()}, {"gaps", value.gaps}};
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
        document.unresolved_gaps = value.gaps;
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
        const bool complete = value.gaps.empty();
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
        report_index.limitations = value.gaps;
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
    if (operation == "investigation.collect") {
        if (arguments.value("expected_revision", uint64_t(0)) != value.revision) {
            return common_adapter_execution_failure("tool.investigation.revision_conflict",
                "case revision changed before collection", "Reload the case before collecting more evidence.");
        }
        if (!bindings.resource_runtime.store) return common_adapter_execution_failure(
            "tool.investigation.evidence_store_unavailable", "resource store is unavailable",
            "Evidence collection is unavailable because persistent resource storage is not configured.");
        common_agent_dataset_descriptor dataset;
        const auto dataset_uri = arguments.value("dataset", std::string());
        if (!bindings.data_store->get_dataset_descriptor(dataset_uri, dataset, error) ||
                dataset.ref.storage_mode != common_agent_dataset_storage_mode::virtual_ ||
                dataset.ref.source_provider != value.source_id) {
            return common_adapter_not_found_failure("tool.investigation.dataset_unavailable",
                error.empty() ? "dataset is not a configured virtual dataset for this case source" : std::move(error),
                "Choose a virtual dataset configured for this case source.");
        }
        std::vector<common_investigation_relation_rule> configured_relation_profiles;
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
        const size_t limit = arguments.value("limit", size_t(200));
        if (limit == 0 || limit > 500) return common_adapter_validation_failure(
            "tool.investigation.collect_limit_invalid", "collection page size must be from 1 to 500");
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
        const json query = {{"dataset", dataset_uri}, {"select", selected}, {"where", predicates},
            {"limit", limit}, {"max_result_rows", limit}, {"max_scan_rows", limit}};
        std::string response;
        if (!bindings.data_store->execute_scoped("data.query", query.dump(), bindings.tool_scope,
                bindings.allow_network, response, error)) {
            return common_adapter_execution_failure("tool.investigation.collect_failed", std::move(error),
                "The configured source could not return this evidence page.");
        }
        const auto page = json::parse(response, nullptr, false);
        if (!page.is_object() || !page.contains("columns") || !page["columns"].is_array() ||
                !page.contains("rows") || !page["rows"].is_array()) return common_adapter_execution_failure(
            "tool.investigation.collect_invalid_result", "virtual dataset returned an invalid row page",
            "The source result could not be normalized.");
        std::vector<std::string> columns;
        for (const auto & column : page["columns"]) if (column.is_string()) columns.push_back(column.get<std::string>());
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
        if (!page.value("complete", false) || page.value("scan_truncated", false) ||
                page.value("result_truncated", false)) {
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
            std::string identity = value.source_id + ":" + kind_name + ":" + (host.empty() ? "" : host + ":") + raw;
            if (identity.size() > 480) identity = value.source_id + ":" + kind_name + ":sha256:" + hash_sha256_hex(identity.data(), identity.size());
            const std::string id = "entity:" + identity;
            if (!entities_by_id.count(id)) entities_by_id.emplace(id, common_investigation_entity{
                id, value.source_id, kind, raw, raw, host, 0, 0});
            return id;
        };
        json normalized_rows = json::array();
        for (const auto & row : rows) {
            const std::string source_index = get_alias(row, {"@source.index"});
            const std::string source_document_id = get_alias(row, {"@source.id", "event.id", "event_id", "id"});
            const std::string serialized_row = row.dump();
            const std::string document_identity = source_index.empty() ? source_document_id : source_index + ":" + source_document_id;
            const std::string event_identity = source_document_id.empty()
                ? hash_sha256_hex(serialized_row.data(), serialized_row.size()) : document_identity;
            const std::string event_prefix = "event:" + value.source_id + ":";
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
            collected_events.push_back({event_id, value.source_id,
                source_document_id.empty() ? event_id : source_document_id, event_type, event_time,
                refs, {}, {}, {}, {}});
            collected_events.back().source_index = source_index;
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
                {"event_time_ms", event_time.event_time_ms}, {"has_event_time", event_time.has_event_time},
                {"collected_time_ms", event_time.collected_time_ms},
                {"entity_ids", refs}, {"entity_roles", collected_events.back().entity_roles}});
        }
        json raw_page = {{"source_id", value.source_id}, {"dataset", dataset_uri},
            {"query", query}, {"result", page}, {"collected_at_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()}};
        agent_resource_put_request raw_put;
        raw_put.name = value.case_id + "-source-page-r" + std::to_string(value.revision + 1) + ".json";
        raw_put.description = "Immutable raw search page collected for investigation case " + value.case_id;
        raw_put.mime_type = "application/json"; raw_put.text = raw_page.dump(2);
        raw_put.scope = bindings.tool_scope.project_id.empty() ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
        raw_put.source_provider = value.source_id; raw_put.source_tool = "investigation.collect";
        apply_agent_resource_runtime(bindings.resource_runtime, raw_put);
        agent_resource_descriptor raw_descriptor;
        if (!bindings.resource_runtime.store->put_text(raw_put, raw_descriptor, error)) return common_adapter_execution_failure(
            "tool.investigation.raw_evidence_store_failed", std::move(error), "The raw search result could not be preserved.");
        json normalized = {{"schema_version", 1}, {"case_id", value.case_id}, {"case_revision", value.revision + 1},
            {"source_id", value.source_id}, {"events", normalized_rows}, {"limitations", limitations}};
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
        std::vector<common_investigation_relation> relations;
        std::vector<std::string> relation_limitations;
        if (!common_investigation_build_relations(value.source_id, value.entities, value.events,
                rules, relations, relation_limitations, error)) return common_adapter_validation_failure(
            "tool.investigation.normalization_failed", std::move(error));
        limitations.insert(limitations.end(), relation_limitations.begin(), relation_limitations.end());
        value.relations = std::move(relations);
        value.gaps.insert(value.gaps.end(), limitations.begin(), limitations.end());
        const uint64_t expected = value.revision++;
        common_investigation_case_revision published;
        const auto revision_uri = "investigation-case://" + value.case_id + "/revision/" + std::to_string(value.revision);
        if (!bindings.data_store->publish_investigation_case(value, expected, revision_uri, published, error))
            return common_adapter_execution_failure("tool.investigation.revision_conflict", std::move(error),
                "The case changed while evidence was being collected; reload and retry.");
        return common_adapter_success_json({{"case_id", published.case_id}, {"revision", published.revision},
            {"collected_events", collected_events.size()}, {"complete", page.value("complete", false)},
            {"limitations", limitations}, {"raw_resource_uri", raw_descriptor.uri},
            {"normalized_resource_uri", normalized_descriptor.uri}});
    }
    if (operation == "investigation.analyze") {
        const auto analysis = arguments.value("analysis", std::string());
        common_investigation_analysis_budget budget;
        budget.max_expansions = arguments.value("max_expansions", budget.max_expansions);
        budget.max_path_length = arguments.value("max_path_length", budget.max_path_length);
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
        if (entity_id.empty() == relation_id.empty()) return common_adapter_validation_failure(
            "tool.investigation.events_invalid", "events requires exactly one entity_id or relation_id");
        std::set<std::string> evidence_ids;
        if (!relation_id.empty()) {
            const auto relation = std::find_if(value.relations.begin(), value.relations.end(),
                [&](const auto & item) { return item.id == relation_id; });
            if (relation == value.relations.end()) return common_adapter_not_found_failure(
                "tool.investigation.relation_unavailable", "relation is not in this case",
                "The selected relation is unavailable in this case.");
            evidence_ids.insert(relation->evidence_event_ids.begin(), relation->evidence_event_ids.end());
        }
        const int64_t window_start = arguments.value("window_start_ms", value.window_start_ms);
        const int64_t window_end = arguments.value("window_end_ms", value.window_end_ms);
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
                {"event_type", event.event_type}, {"time", {{"event_time_ms", event.time.event_time_ms},
                    {"collected_time_ms", event.time.collected_time_ms}, {"uncertainty_ms", event.time.uncertainty_ms},
                    {"has_event_time", event.time.has_event_time}}}, {"raw_resource_uri", event.raw_resource_uri},
                {"normalized_resource_uri", event.normalized_resource_uri}, {"entity_ids", event.entity_ids}};
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
