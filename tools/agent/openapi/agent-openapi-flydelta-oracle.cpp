#include "agent-openapi-flydelta-oracle.h"

#include "agent/tooling/schema/tool-schema-compact.h"

namespace {

std::string operation_ref(const agent_openapi_catalog & catalog,
        const agent_openapi_operation & operation) {
    return "openapi://" + catalog.provider_id + "/operation/" + operation.operation_id;
}

std::string evaluator_ref(const agent_openapi_catalog & catalog,
        const agent_openapi_operation & operation) {
    return "flydelta://evaluator/openapi/" + catalog.provider_id + "/" +
        operation.operation_id;
}

} // namespace

bool make_agent_openapi_flydelta_oracle_contract(
        const agent_openapi_catalog & catalog,
        const agent_openapi_operation & operation,
        common_flydelta_openapi_operation_contract & contract,
        std::string & error) {
    error.clear();
    if (catalog.provider_id.empty() || operation.operation_id.empty()) {
        error = "OpenAPI FlyDelta Oracle requires provider and operation identities";
        return false;
    }

    std::string schema_error;
    const std::string model_schema = common_project_model_input_schema_required_parameters(
        operation.input_schema_json, operation.host_required_parameters, schema_error);
    if (!schema_error.empty()) {
        error = "OpenAPI FlyDelta model-facing schema projection failed for " +
            operation.operation_id + ": " + schema_error;
        return false;
    }

    contract = {};
    contract.provider_id = catalog.provider_id;
    contract.operation_id = operation.operation_id;
    contract.method = operation.method;
    contract.path = operation.path;
    contract.path_parameters = operation.path_parameters;
    contract.query_parameters = operation.query_parameters;
    contract.host_required_parameters = operation.host_required_parameters;
    contract.read_only = operation.read_only;
    contract.requires_confirmation = operation.requires_confirmation;
    contract.tool.provider_ref = catalog.provider_id;
    contract.tool.exposed_tool_name = agent_openapi_exposed_tool_name(catalog, operation);
    contract.tool.contract_ref = operation_ref(catalog, operation);
    contract.tool.contract_revision = "openapi-v1";
    contract.tool.contract_fingerprint = contract.tool.contract_ref + ":" +
        operation.method + ":" + operation.path;
    contract.tool.model_input_schema_json = model_schema;
    contract.tool.host_input_schema_json = operation.input_schema_json;
    contract.tool.read_only = operation.read_only;
    contract.tool.requires_confirmation = operation.requires_confirmation;
    contract.tool.uses_network = true;
    // Network permission is host policy, not an OpenAPI document fact. A
    // caller can tighten this descriptor before registration when required.
    contract.tool.host_allows_network = true;
    contract.tool.confirmation_satisfied = !operation.requires_confirmation;
    return true;
}

bool register_agent_openapi_flydelta_oracles(
        const agent_openapi_catalog & catalog,
        common_flydelta_oracle_registry & registry,
        const std::string & oracle_revision,
        std::string & error) {
    error.clear();
    if (oracle_revision.empty()) {
        error = "OpenAPI FlyDelta Oracle registration requires a revision";
        return false;
    }
    for (const auto & operation : catalog.operations) {
        common_flydelta_openapi_operation_contract contract;
        if (!make_agent_openapi_flydelta_oracle_contract(
                catalog, operation, contract, error)) {
            return false;
        }
        if (!common_flydelta_register_oracle_evaluator(registry, {
                common_flydelta_oracle_strength::host_supported,
                evaluator_ref(catalog, operation),
                oracle_revision,
                "openapi_operation",
                "openapi_operation_contract",
                contract.tool.contract_ref,
                contract.tool.contract_revision,
                common_flydelta_make_openapi_operation_oracle(std::move(contract))}, error)) {
            return false;
        }
    }
    return true;
}

