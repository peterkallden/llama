#include "agent-openapi-dynamic-admission.h"

#include <algorithm>
#include <cctype>

namespace {

bool safe_server_url(const std::string & value) {
    if (value.rfind("https://", 0) != 0 || value.size() > 2048 ||
            value.find_first_of("?#{}\r\n") != std::string::npos) {
        return false;
    }
    const auto authority_start = std::string("https://").size();
    const auto path_start = value.find('/', authority_start);
    const auto authority = value.substr(
        authority_start,
        path_start == std::string::npos ? std::string::npos : path_start - authority_start);
    return !authority.empty() && authority.find('@') == std::string::npos &&
        authority.find_first_of("\\ \t") == std::string::npos;
}

std::string safe_slug(std::string value) {
    std::string result;
    result.reserve(std::min<size_t>(value.size(), 32));
    for (const unsigned char ch : value) {
        if (result.size() == 32) break;
        if (std::isalnum(ch)) result.push_back(static_cast<char>(std::tolower(ch)));
        else if (!result.empty() && result.back() != '-') result.push_back('-');
    }
    while (!result.empty() && result.back() == '-') result.pop_back();
    if (result.empty()) result = "api";
    return result;
}

} // namespace

bool agent_openapi_dynamic_registry::add(
        agent_openapi_dynamic_registration registration,
        std::string & error) {
    if (registration.config.id.empty() || registration.catalog.operations.empty()) {
        error = "dynamic OpenAPI registration must have an id and at least one operation";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = registrations_.find(registration.config.id);
    if (existing != registrations_.end()) {
        if (existing->second.source_content_hash == registration.source_content_hash) {
            error.clear();
            return true;
        }
        error = "dynamic OpenAPI provider id conflicts with another spec in this session";
        return false;
    }
    registrations_.emplace(registration.config.id, std::move(registration));
    error.clear();
    return true;
}

std::vector<agent_openapi_dynamic_registration> agent_openapi_dynamic_registry::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<agent_openapi_dynamic_registration> result;
    result.reserve(registrations_.size());
    for (const auto & item : registrations_) result.push_back(item.second);
    return result;
}

bool agent_openapi_dynamic_registry::contains(const std::string & provider_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return registrations_.count(provider_id) != 0;
}

void agent_openapi_dynamic_registry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    registrations_.clear();
}

bool admit_agent_dynamic_openapi_document(
        const nlohmann::json & document,
        const std::string & source_resource_uri,
        const std::string & source_content_hash,
        agent_openapi_dynamic_registration & registration,
        std::string & error) {
    if (!document.is_object() || !document.contains("servers") ||
            !document["servers"].is_array() || document["servers"].empty() ||
            !document["servers"][0].is_object()) {
        error = "dynamic OpenAPI spec must declare an explicit HTTPS server";
        return false;
    }
    const auto server_url_it = document["servers"][0].find("url");
    if (server_url_it == document["servers"][0].end() || !server_url_it->is_string()) {
        error = "dynamic OpenAPI server URL must be a string";
        return false;
    }
    const auto server_url = server_url_it->get<std::string>();
    if (!safe_server_url(server_url)) {
        error = "dynamic OpenAPI server must be a plain public HTTPS URL without credentials or variables";
        return false;
    }
    if (source_resource_uri.empty() || source_content_hash.empty()) {
        error = "dynamic OpenAPI admission requires a host-owned spec resource and content hash";
        return false;
    }

    std::string title = "api";
    const auto info_it = document.find("info");
    if (info_it != document.end() && info_it->is_object()) {
        const auto title_it = info_it->find("title");
        if (title_it != info_it->end() && title_it->is_string()) {
            title = title_it->get<std::string>();
        }
    }
    agent_openapi_dynamic_registration candidate;
    candidate.config.id = "dynamic-" + safe_slug(title) + "-" + source_content_hash.substr(0, 12);
    candidate.config.prefix = candidate.config.id;
    candidate.config.enabled = true;
    candidate.config.required = true;
    candidate.config.base_url = server_url;
    candidate.config.access = "read_only";
    candidate.config.exposure = "auto";
    candidate.config.allow_private_network = false;
    candidate.config.auth.type = "none";
    candidate.config.max_result_bytes = 1024 * 1024;
    candidate.document = document;
    candidate.source_resource_uri = source_resource_uri;
    candidate.source_content_hash = source_content_hash;

    if (!build_agent_openapi_catalog(
            candidate.document, candidate.config, candidate.catalog, error)) {
        return false;
    }
    auto & operations = candidate.catalog.operations;
    operations.erase(std::remove_if(operations.begin(), operations.end(),
        [](const agent_openapi_operation & operation) {
            return !operation.read_only || operation.auth_required;
        }), operations.end());
    if (operations.empty()) {
        error = "dynamic OpenAPI spec has no anonymous read-only operations";
        return false;
    }
    candidate.catalog.relations.erase(
        std::remove_if(candidate.catalog.relations.begin(), candidate.catalog.relations.end(),
            [&](const agent_openapi_relation & relation) {
                const auto has_operation = [&](const std::string & id) {
                    return std::any_of(operations.begin(), operations.end(),
                        [&](const agent_openapi_operation & operation) {
                            return operation.operation_id == id;
                        });
                };
                return !has_operation(relation.collection_operation_id) ||
                    !has_operation(relation.item_operation_id);
            }),
        candidate.catalog.relations.end());

    registration = std::move(candidate);
    error.clear();
    return true;
}
