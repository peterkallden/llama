#include "agent/adaptation/lifecycle-store.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>

using json = nlohmann::ordered_json;

namespace {

bool bounded_nonempty(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool same_record(const common_learning_lifecycle_record & left,
        const common_learning_lifecycle_record & right) {
    return left.event_id == right.event_id && left.subject_id == right.subject_id &&
        left.kind == right.kind && left.status == right.status &&
        left.content_hash == right.content_hash && left.payload_json == right.payload_json;
}

bool write_text(const std::filesystem::path & path, const std::string & text, std::string & error) {
    std::ofstream output(path, std::ios::binary | std::ios::app);
    if (!output) { error = "could not open lifecycle store"; return false; }
    output << text;
    if (!output) { error = "could not append lifecycle record"; return false; }
    return true;
}

} // namespace

const char * common_learning_lifecycle_kind_name(common_learning_lifecycle_kind kind) {
    switch (kind) {
        case common_learning_lifecycle_kind::candidate: return "candidate";
        case common_learning_lifecycle_kind::corpus_revision: return "corpus_revision";
        case common_learning_lifecycle_kind::training_job: return "training_job";
        case common_learning_lifecycle_kind::training_result: return "training_result";
        case common_learning_lifecycle_kind::adapter: return "adapter";
    }
    return "candidate";
}

const char * common_learning_lifecycle_status_name(common_learning_lifecycle_status status) {
    switch (status) {
        case common_learning_lifecycle_status::observed: return "observed";
        case common_learning_lifecycle_status::eligible: return "eligible";
        case common_learning_lifecycle_status::approved: return "approved";
        case common_learning_lifecycle_status::rejected: return "rejected";
        case common_learning_lifecycle_status::revoked: return "revoked";
        case common_learning_lifecycle_status::queued: return "queued";
        case common_learning_lifecycle_status::running: return "running";
        case common_learning_lifecycle_status::succeeded: return "succeeded";
        case common_learning_lifecycle_status::failed: return "failed";
        case common_learning_lifecycle_status::cancelled: return "cancelled";
        case common_learning_lifecycle_status::canary: return "canary";
        case common_learning_lifecycle_status::active: return "active";
        case common_learning_lifecycle_status::retired: return "retired";
    }
    return "failed";
}

bool parse_common_learning_lifecycle_kind(
        const std::string & value,
        common_learning_lifecycle_kind & kind,
        std::string & error) {
    if (value == "candidate") kind = common_learning_lifecycle_kind::candidate;
    else if (value == "corpus_revision") kind = common_learning_lifecycle_kind::corpus_revision;
    else if (value == "training_job") kind = common_learning_lifecycle_kind::training_job;
    else if (value == "training_result") kind = common_learning_lifecycle_kind::training_result;
    else if (value == "adapter") kind = common_learning_lifecycle_kind::adapter;
    else { error = "unknown lifecycle record kind"; return false; }
    return true;
}

bool parse_common_learning_lifecycle_status(
        const std::string & value,
        common_learning_lifecycle_status & status,
        std::string & error) {
    if (value == "observed") status = common_learning_lifecycle_status::observed;
    else if (value == "eligible") status = common_learning_lifecycle_status::eligible;
    else if (value == "approved") status = common_learning_lifecycle_status::approved;
    else if (value == "rejected") status = common_learning_lifecycle_status::rejected;
    else if (value == "revoked") status = common_learning_lifecycle_status::revoked;
    else if (value == "queued") status = common_learning_lifecycle_status::queued;
    else if (value == "running") status = common_learning_lifecycle_status::running;
    else if (value == "succeeded") status = common_learning_lifecycle_status::succeeded;
    else if (value == "failed") status = common_learning_lifecycle_status::failed;
    else if (value == "cancelled") status = common_learning_lifecycle_status::cancelled;
    else if (value == "canary") status = common_learning_lifecycle_status::canary;
    else if (value == "active") status = common_learning_lifecycle_status::active;
    else if (value == "retired") status = common_learning_lifecycle_status::retired;
    else { error = "unknown lifecycle record status"; return false; }
    return true;
}

bool common_learning_lifecycle_validate(
        const common_learning_lifecycle_record & record,
        size_t max_payload_bytes,
        std::string & error) {
    error.clear();
    if (record.schema_version != 1) { error = "unsupported lifecycle record schema"; return false; }
    if (!bounded_nonempty(record.event_id) || !bounded_nonempty(record.subject_id) ||
            !bounded_nonempty(record.idempotency_key) || !bounded_nonempty(record.created_at) ||
            !bounded_nonempty(record.content_hash) || !bounded_nonempty(record.namespace_id) ||
            record.project_id.size() > 512 || record.session_id.size() > 512 ||
            record.source_id.size() > 512) {
        error = "lifecycle record identity is incomplete or unbounded";
        return false;
    }
    if (record.payload_json.empty() || record.payload_json.size() > max_payload_bytes) {
        error = "lifecycle record payload exceeds its bound";
        return false;
    }
    const auto payload = json::parse(record.payload_json, nullptr, false);
    if (payload.is_discarded() || !payload.is_object()) {
        error = "lifecycle record payload is not a JSON object";
        return false;
    }
    return true;
}

std::string common_learning_lifecycle_to_json(
        const common_learning_lifecycle_record & record) {
    return json{
        {"schema_version", record.schema_version},
        {"event_id", record.event_id},
        {"subject_id", record.subject_id},
        {"kind", common_learning_lifecycle_kind_name(record.kind)},
        {"status", common_learning_lifecycle_status_name(record.status)},
        {"idempotency_key", record.idempotency_key},
        {"source_id", record.source_id},
        {"scope", {
            {"namespace_id", record.namespace_id},
            {"project_id", record.project_id},
            {"session_id", record.session_id},
        }},
        {"content_hash", record.content_hash},
        {"created_at", record.created_at},
        {"payload", json::parse(record.payload_json, nullptr, false)},
    }.dump();
}

bool common_learning_lifecycle_from_json(
        const std::string & text,
        common_learning_lifecycle_record & record,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        const auto scope = value.value("scope", json::object());
        const auto payload = value.value("payload", json::object());
        if (!value.is_object() || !scope.is_object() || !payload.is_object()) {
            error = "lifecycle record envelope is invalid";
            return false;
        }
        record = {};
        record.schema_version = value.value("schema_version", 0);
        record.event_id = value.value("event_id", "");
        record.subject_id = value.value("subject_id", "");
        if (!parse_common_learning_lifecycle_kind(value.value("kind", ""), record.kind, error) ||
                !parse_common_learning_lifecycle_status(value.value("status", ""), record.status, error)) return false;
        record.idempotency_key = value.value("idempotency_key", "");
        record.source_id = value.value("source_id", "");
        record.namespace_id = scope.value("namespace_id", "local");
        record.project_id = scope.value("project_id", "");
        record.session_id = scope.value("session_id", "");
        record.content_hash = value.value("content_hash", "");
        record.created_at = value.value("created_at", "");
        record.payload_json = payload.dump();
        return true;
    } catch (const std::exception & exception) {
        error = std::string("invalid lifecycle record JSON: ") + exception.what();
        return false;
    }
}

bool common_learning_in_memory_lifecycle_store::append(
        const common_learning_lifecycle_record & record, std::string & error) {
    if (!common_learning_lifecycle_validate(record, 4 * 1024 * 1024, error)) return false;
    bool contains = false;
    if (!contains_idempotency(record.idempotency_key, contains, error)) return false;
    if (contains) {
        const auto match = std::find_if(records.begin(), records.end(), [&](const auto & item) {
            return item.idempotency_key == record.idempotency_key;
        });
        if (match != records.end() && !same_record(*match, record)) {
            error = "lifecycle idempotency key conflicts with existing record";
            return false;
        }
        return true;
    }
    records.push_back(record);
    return true;
}

bool common_learning_in_memory_lifecycle_store::contains_idempotency(
        const std::string & key, bool & contains, std::string & error) const {
    error.clear();
    contains = std::any_of(records.begin(), records.end(), [&](const auto & item) {
        return item.idempotency_key == key;
    });
    return true;
}

std::vector<common_learning_lifecycle_record> common_learning_in_memory_lifecycle_store::list(
        std::string & error) const {
    error.clear();
    return records;
}

bool common_learning_jsonl_lifecycle_store::open(
        const std::filesystem::path & value, std::string & error) {
    error.clear();
    if (value.empty()) { error = "lifecycle store requires a path"; return false; }
    path = value;
    std::error_code ec;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) { error = "could not create lifecycle store directory"; return false; }
    std::ofstream output(path, std::ios::app);
    if (!output) { error = "could not open lifecycle store"; return false; }
    return true;
}

std::vector<common_learning_lifecycle_record> common_learning_jsonl_lifecycle_store::list(
        std::string & error) const {
    error.clear();
    std::vector<common_learning_lifecycle_record> result;
    std::ifstream input(path);
    if (!input) { error = "could not read lifecycle store"; return result; }
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        common_learning_lifecycle_record record;
        if (!common_learning_lifecycle_from_json(line, record, error) ||
                !common_learning_lifecycle_validate(record, 4 * 1024 * 1024, error)) return {};
        result.push_back(std::move(record));
    }
    return result;
}

bool common_learning_jsonl_lifecycle_store::contains_idempotency(
        const std::string & key, bool & contains, std::string & error) const {
    contains = false;
    const auto values = list(error);
    if (!error.empty()) return false;
    contains = std::any_of(values.begin(), values.end(), [&](const auto & item) {
        return item.idempotency_key == key;
    });
    return true;
}

bool common_learning_jsonl_lifecycle_store::append(
        const common_learning_lifecycle_record & record, std::string & error) {
    if (!common_learning_lifecycle_validate(record, 4 * 1024 * 1024, error)) return false;
    const auto values = list(error);
    if (!error.empty()) return false;
    const auto match = std::find_if(values.begin(), values.end(), [&](const auto & item) {
        return item.idempotency_key == record.idempotency_key;
    });
    if (match != values.end()) {
        if (!same_record(*match, record)) error = "lifecycle idempotency key conflicts with existing record";
        return error.empty();
    }
    return write_text(path, common_learning_lifecycle_to_json(record) + "\n", error);
}
